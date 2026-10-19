"""Greedy (SplitFleet) and L2S physical controls using one shared runtime."""

from pathlib import Path
import time
from splitfleet.server.placement.greedy import PlacementFeedback
from splitfleet.server.placement.greedy.rpc_timing import state_exchange_duration
from experiments.physical_multitask import RecordingSplit

POLICIES = ("Greedy", "L2S")

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
            completion_scope="coordinator_fit_rpc",
            server_load=({key.removeprefix("server_load_"): float(value)
                          for key, value in metrics.items() if key.startswith("server_load_")} or None),
            num_examples=response.num_examples, num_batches=int(metrics['num_batches'])))
    return observations


def initial_admitted_boundary(policy, boundary):
    """Bootstrap within the controller's actual catalog before round planning."""
    candidates = tuple(policy.candidate_provider.get_candidates(training=True))
    if boundary in {candidate.boundary for candidate in candidates}:
        return boundary
    return min(candidates, key=lambda candidate: (
        abs(candidate.graph_position_ratio - .5), candidate.boundary)).boundary


class AdaptiveRecordingSplit(RecordingSplit):
    def __init__(self, *, experiment_args, **kwargs):
        kwargs["boundary"] = initial_admitted_boundary(kwargs["placement_policy"], kwargs["boundary"])
        self.experiment_args = experiment_args
        super().__init__(**kwargs)

    def experiment_result(self):
        return {"policy": self.placement_policy.algorithm,
                "reference_controller": {"decision_receipts": self.placement_policy.decision_receipts},
                "shared_sfl_pipeline": True}


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
    import sys
    from experiments.physical_multitask import main as run
    args = list(sys.argv[1:])
    if "--policy" in args:
        index = args.index("--policy")
        if args[0] == "client":
            del args[index:index + 2]
        else:
            args[index] = "--placement-controller"
    if "--method" not in args:
        args.extend(["--method", "splitfleet"])
    sys.argv[1:] = args
    run()


if __name__ == "__main__":
    main()
