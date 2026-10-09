"""Measured edge/cloud placement controls using the existing SFL runner.

Independent-UCB is an experimental comparator with private per-client learners.
The production placement policy and its default configuration remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from experiments.baselines.independent_ucb import IndependentUCB
from experiments.physical_multitask import RecordingSplit, _make_placement_policy, run_client, run_server
from splitfleet.server.placement.cosplit_ucb import CoSplitUCBPlacementPolicy
from splitfleet.server.placement.cosplit_ucb.calibration import CalibratedTelemetry


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class PhysicalIndependentUCB(IndependentUCB):
    """Bootstrap each private learner solely from that physical worker.

    One restored server calibration is measured once, as for CoSplit-UCB.
    Each client then gets its own server learner, edge learner and real RPC
    probes. No observations are copied from another client's learner.
    """

    def __init__(self, *, candidate_provider, config):
        self.state_file = Path(config.state_path) if config.state_path else None
        config = replace(config, state_path=None)
        super().__init__(candidate_provider=candidate_provider, config=config)
        self.bootstrap = CoSplitUCBPlacementPolicy(candidate_provider=candidate_provider, config=config)
        self.state_download_bytes = 0

    def prepare_calibration(self, *args, **kwargs):
        if self.telemetry_provider is not None and self.telemetry_provider.initialized:
            return
        self.bootstrap.prepare_calibration(*args, **kwargs)
        self.telemetry_provider = self.bootstrap.telemetry_provider

    def bind_clients(self, clients, round_id):
        template = self.telemetry_provider
        if not isinstance(template, CalibratedTelemetry):
            raise ValueError("Prepare the restored server calibration before binding workers")
        for client in clients:
            cid = str(client.cid)
            policy = self._client(cid)
            if policy.telemetry_provider is template:
                policy.telemetry_provider = CalibratedTelemetry(
                    initial_model_hash=template.initial_model_hash,
                    server_receipt=copy.deepcopy(template.server_receipt),
                    parameters=template.parameters, policy=policy,
                )
            policy.bind_clients([client], round_id)
            template.clients[cid] = policy.telemetry_provider.clients[cid]
        receipts = [policy.telemetry_provider.receipt for policy in self.clients.values()]
        if not receipts:
            raise ValueError("A physical independent cohort requires clients")
        receipt = copy.deepcopy(receipts[0])
        receipt["client_samples"] = [row for value in receipts for row in value["client_samples"]]
        receipt["client_calibration_elapsed_sec"] = {
            key: value for item in receipts for key, value in item["client_calibration_elapsed_sec"].items()
        }
        for key in ("transport_bootstrap", "state_exchange_bootstrap"):
            receipt[key]["samples"] = [row for value in receipts for row in value[key]["samples"]]
        receipt["transport_bootstrap"]["warmup_samples"] = [
            row for value in receipts for row in value["transport_bootstrap"]["warmup_samples"]
        ]
        receipt["transport_bootstrap"]["elapsed_sec"] = sum(
            value["transport_bootstrap"]["elapsed_sec"] for value in receipts
        )
        receipt["learner_isolation"] = "separate edge/server/network/switch statistics per client"
        receipt["cooperative_observation_sharing"] = False
        template.receipt = receipt
        template.initialized = True

    def plan_round(self, **kwargs):
        for policy in self.clients.values():
            policy.state_download_bytes = self.state_download_bytes
        return super().plan_round(**kwargs)

    def observe_round(self, **kwargs):
        super().observe_round(**kwargs)
        if self.state_file:
            save_json(self.state_file, {"experimental_policy": "Independent-UCB", "clients": self.state_dict()})

    @property
    def round_diagnostics(self):
        result = {}
        for cid, policy in self.clients.items():
            for round_id, diagnostic in policy.round_diagnostics.items():
                row = result.setdefault(round_id, {"experimental_policy": "Independent-UCB", "per_client": {}})
                row["per_client"][cid] = diagnostic
                for key in ("assignment", "estimates", "learner_update_counts"):
                    row.setdefault(key, {}).update(diagnostic.get(key, {}))
        return result

    @property
    def exploration_controller(self):
        return SimpleNamespace(round_records={
            cid: policy.exploration_controller.round_records for cid, policy in self.clients.items()
        })


class MeasuredServerLane:
    """A real common training semaphore with coordinator monotonic receipts."""

    def __init__(self, output):
        self.gate = threading.Semaphore(1)
        self.output = Path(output)
        self.records = []
        self.log_lock = threading.Lock()

    def wrap(self, model):
        original = model.train_tail

        def train_tail(batches):
            from splitfleet.transport import decode_boundary
            identity = decode_boundary(batches[0].data["boundary"])
            queued = time.perf_counter_ns()
            with self.gate:
                acquired = time.perf_counter_ns()
                responses = original(batches)
                finished = time.perf_counter_ns()
            row = {"round_id": identity.round_id, "cid": identity.client_id,
                   "step_id": identity.step_id, "boundary": identity.split_id,
                   "queued_monotonic_ns": queued, "acquired_monotonic_ns": acquired,
                   "finished_monotonic_ns": finished,
                   "server_queue_wait_ms": (acquired - queued) / 1e6,
                   "server_handler_excluding_queue_ms": (finished - acquired) / 1e6}
            for response in responses:
                metadata = json.loads(response.data["metadata"])
                metadata["server_queue_wait_ms"] = row["server_queue_wait_ms"]
                response.data["metadata"] = json.dumps(metadata, sort_keys=True).encode()
            with self.log_lock:
                self.records.append(row)
                with self.output.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            return responses

        model.train_tail = train_tail
        return model


class MeasuredRecordingSplit(RecordingSplit):
    def __init__(self, *, experiment_args, **kwargs):
        self.experiment_args = experiment_args
        self.round_records = []
        self.current_round = None
        self.server_lane = MeasuredServerLane(Path(experiment_args.output).with_suffix(".server_lane.jsonl"))
        super().__init__(**kwargs)

    def _make_server_model(self):
        return self.server_lane.wrap(super()._make_server_model())

    def configure_fit(self, server_round, parameters, client_manager):
        started = time.perf_counter_ns()
        phase = {"round_id": int(server_round), "cap_mbps": (
            self.experiment_args.degraded_mbps if server_round >= self.experiment_args.change_round else None
        )}
        save_json(self.experiment_args.network_control, phase)
        instructions = super().configure_fit(server_round, parameters, client_manager)
        self.current_round = {**phase, "configure_started_monotonic_ns": started,
                              "configured_monotonic_ns": time.perf_counter_ns()}
        return instructions

    def evaluate(self, server_round, client_parameters, server_parameters):
        training_finished = time.perf_counter_ns()
        value = super().evaluate(server_round, client_parameters, server_parameters)
        finished = time.perf_counter_ns()
        if self.current_round is None or self.current_round["round_id"] != server_round:
            raise ValueError("Physical round timing lacks matching configure_fit receipt")
        row = {**self.current_round, "training_finished_monotonic_ns": training_finished,
               "evaluation_finished_monotonic_ns": finished,
               "coordinator_round_training_ms": (training_finished - self.current_round["configure_started_monotonic_ns"]) / 1e6,
               "coordinator_dispatch_through_aggregation_ms": (training_finished - self.current_round["configured_monotonic_ns"]) / 1e6,
               "coordinator_evaluation_ms": (finished - training_finished) / 1e6}
        self.round_records.append(row)
        save_json(Path(self.experiment_args.output).with_suffix(".progress.json"), self.experiment_result())
        return value

    def experiment_result(self):
        return {"schema": "splitfleet.physical-placement-observations.v1",
                "policy": self.experiment_args.policy, "physical_measurement": True,
                "server_training_concurrency": 1, "round_records": self.round_records,
                "server_lane_step_receipts": self.server_lane.records,
                "network_shaping": "real shared TCP application stream pacing, dedicated endpoints",
                "network_residual_scope": "Existing production split-RPC RTT residual includes server queuing; explicit queue receipt is additionally retained."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--policy", choices=("CoSplit-UCB", "Independent-UCB", "Fixed"), required=True)
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
    parser.add_argument("--degraded-mbps", type=float, default=100)
    parser.add_argument("--server")
    parser.add_argument("--client-id")
    parser.add_argument("--client-index", type=int)
    parser.add_argument("--barrier")
    args = parser.parse_args()
    args.method = "splitfed_fixed" if args.policy == "Fixed" else "splitfleet"
    args.fixed_boundary = args.boundary
    args.split_state_exchange = "owned"
    if args.role == "client":
        run_client(args)
        return
    if not args.output or not args.bind or not args.rounds or not args.network_control:
        raise ValueError("Physical server requires output, bind, rounds and network-control")
    if args.degraded_mbps <= 0 or args.change_round < 1:
        raise ValueError("Network change round and cap must be positive")
    def factory(**kwargs):
        if args.policy == "CoSplit-UCB":
            return _make_placement_policy(**kwargs)
        config = _make_placement_policy(**kwargs).config
        return PhysicalIndependentUCB(candidate_provider=kwargs["provider"], config=config)
    run_server(args, placement_policy_factory=factory,
               recording_strategy_class=lambda **kwargs: MeasuredRecordingSplit(experiment_args=args, **kwargs))


if __name__ == "__main__":
    main()
