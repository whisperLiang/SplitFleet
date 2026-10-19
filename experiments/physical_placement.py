"""Common physical timing utilities; placement is implemented by Greedy."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from experiments.physical_multitask import RecordingSplit


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class MeasuredServerLane:
    """A real common training semaphore with coordinator monotonic receipts."""

    def __init__(self, output, *, concurrency=1):
        self.concurrency = int(concurrency)
        if self.concurrency < 1:
            raise ValueError("server concurrency must be positive")
        self.gate = threading.Semaphore(self.concurrency)
        self.output = Path(output)
        self.records = []
        self.log_lock = threading.Lock()

    def wrap(self, model):
        original = model.train_tail
        replica_lock = threading.Lock()

        def train_tail(batches):
            from splitfleet.transport import decode_boundary
            identity = decode_boundary(batches[0].data["boundary"])
            queued = time.perf_counter_ns()
            with self.gate:
                acquired = time.perf_counter_ns()
                # A single suffix replica owns optimizer/state and therefore
                # stays ordered even if a transport retries or pipelines RPCs.
                with replica_lock:
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
        self.server_lane = MeasuredServerLane(
            Path(experiment_args.output).with_suffix(".server_lane.jsonl"),
            concurrency=getattr(experiment_args, "server_concurrency", 1),
        )
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
                "server_training_concurrency": self.server_lane.concurrency,
                "round_records": self.round_records,
                "server_lane_step_receipts": self.server_lane.records,
                "network_shaping": "real shared TCP application stream pacing, dedicated endpoints",
                "network_residual_scope": "Existing production split-RPC RTT residual includes server queuing; explicit queue receipt is additionally retained."}
