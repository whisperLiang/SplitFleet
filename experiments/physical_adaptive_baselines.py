"""Actual-device comparison of published placement-controller adaptations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

from experiments.baselines.adaptive_reference import AdaptiveReference, LayerScopedProvider
from experiments.physical_multitask import _make_placement_policy, run_client, run_server
from experiments.physical_placement import MeasuredRecordingSplit, PhysicalIndependentUCB
from splitfleet.server.placement.cosplit_ucb import CoSplitUCBPlacementPolicy
from splitfleet.server.placement.cosplit_ucb.types import PlacementFeedback
from splitfleet.server.placement.cosplit_ucb.rpc_timing import state_exchange_duration


POLICIES = ("CoSplit-UCB-full", "CoSplit-UCB-layer", "Independent-UCB-layer",
            "L2S-LinUCB-E", "Profile-Greedy", "FedAdapt-PPO", "FedAdapt-PPO-training")


def reference_feedback(round_id, results):
    """Retain actual whole-fit RPC costs, including model-state transport."""
    observations = []
    for client, response in results:
        metrics = response.metrics
        if float(metrics.get("coordinator_fit_rpc_ms", 0)) <= 0:
            raise ValueError("Reference feedback lacks an actual coordinator fit-RPC span")
        observations.append(PlacementFeedback(round_id=round_id, client_id=str(client.cid),
            boundary=metrics['boundary'],
            client_forward_ms=float(metrics['client_forward_mean_ms']),
            client_backward_ms=float(metrics['client_backward_mean_ms']),
            network_roundtrip_ms=float(metrics['network_roundtrip_mean_ms']),
            server_service_ms=float(metrics['server_service_mean_ms']),
            switch_ms=float(metrics.get('switch_ms', 0)),
            state_exchange_ms=state_exchange_duration(metrics),
            completion_ms=float(metrics['coordinator_fit_rpc_ms']),
            num_examples=response.num_examples, num_batches=int(metrics['num_batches'])))
    return observations


def scoped_initial_boundary(policy, boundary):
    """Bootstrap within the controller's actual catalog before round planning."""
    candidates = tuple(policy.candidate_provider.get_candidates(training=True))
    if boundary in {candidate.boundary for candidate in candidates}:
        return boundary
    return min(candidates, key=lambda candidate: (
        abs(candidate.graph_position_ratio - .5), candidate.boundary)).boundary


class AdaptiveRecordingSplit(MeasuredRecordingSplit):
    def __init__(self, *, experiment_args, **kwargs):
        kwargs['boundary'] = scoped_initial_boundary(kwargs['placement_policy'], kwargs['boundary'])
        super().__init__(experiment_args=experiment_args, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        return [(CoordinatorObservedFitProxy(client), instruction) for client, instruction in instructions]

    def _update_placement_policy_after_fit(self, server_round, results, failures):
        if not isinstance(self.placement_policy, AdaptiveReference):
            return super()._update_placement_policy_after_fit(server_round, results, failures)
        if failures:
            raise RuntimeError("Reference physical arm has failed fits; preserve and stop")
        self.placement_policy.observe_round(round_id=server_round,
            feedback=reference_feedback(server_round, results))

    def experiment_result(self):
        result = super().experiment_result()
        policy = getattr(self, "placement_policy", None)
        # The base strategy may use a private attribute in some versions.
        if policy is None:
            policy = getattr(self, "_placement_policy", None)
        if isinstance(policy, AdaptiveReference):
            result["reference_controller"] = {"algorithm": policy.algorithm,
                "decision_receipts": policy.decision_receipts, "ppo_updates": policy.ppo_updates,
                "decision_granularity": "federated_round", "fully_local_escape_available": False,
                "gradient_volume_features": "captured differentiable frontier proxy; actual buffers logged separately"}
        return result


class CoordinatorObservedFitProxy:
    """Observe all client completions on one clock without cross-host subtraction."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def fit(self, ins, timeout, group_id):
        started = time.perf_counter_ns()
        response = self.client.fit(ins, timeout=timeout, group_id=group_id)
        finished = time.perf_counter_ns()
        response.metrics['coordinator_fit_started_monotonic_ns'] = started
        response.metrics['coordinator_fit_finished_monotonic_ns'] = finished
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--boundary", default="50%")
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--learning-rate", type=float, default=.005)
    parser.add_argument("--min-residence-rounds", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-model", action="store_true", help="Retain full final task weights")
    parser.add_argument("--bind")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--network-control", type=Path)
    parser.add_argument("--change-round", type=int, default=6)
    parser.add_argument("--degraded-mbps", type=float, default=50)
    parser.add_argument("--server")
    parser.add_argument("--client-id")
    parser.add_argument("--client-index", type=int)
    parser.add_argument("--barrier")
    parser.add_argument("--ppo-source", type=Path)
    parser.add_argument("--actor-checkpoint", type=Path)
    args = parser.parse_args()
    args.method = "splitfleet"
    args.fixed_boundary = args.boundary
    args.split_state_exchange = "owned"
    if args.role == "client":
        run_client(args)
        return
    if not args.output or not args.bind or not args.rounds or not args.network_control:
        raise ValueError("Server requires output, bind, rounds and network-control")
    if args.change_round < 1 or args.degraded_mbps <= 0:
        raise ValueError("Invalid actual network schedule")
    if args.policy == "FedAdapt-PPO" and not args.actor_checkpoint:
        raise ValueError("FedAdapt deployment requires its genuinely trained frozen actor")

    def factory(**kwargs):
        provider = kwargs["provider"]
        if args.policy != "CoSplit-UCB-full":
            provider = LayerScopedProvider(provider)
        config = _make_placement_policy(**kwargs).config
        if args.policy.startswith("CoSplit-UCB"):
            return CoSplitUCBPlacementPolicy(candidate_provider=provider, config=config)
        if args.policy == "Independent-UCB-layer":
            return PhysicalIndependentUCB(candidate_provider=provider, config=config)
        return AdaptiveReference(candidate_provider=provider, config=config,
            algorithm=args.policy, horizon=args.rounds, output=args.output,
            ppo_source=args.ppo_source, actor_checkpoint=args.actor_checkpoint)

    run_server(args, placement_policy_factory=factory,
        recording_strategy_class=lambda **kwargs: AdaptiveRecordingSplit(experiment_args=args, **kwargs))


if __name__ == "__main__":
    main()
