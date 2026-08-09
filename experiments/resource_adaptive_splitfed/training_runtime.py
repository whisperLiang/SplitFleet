"""Real full-local and TorchLens split training for logical clients."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from splitfleet.autosplit.torchlens_backend import prepare_torchlens_runtime
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.transport.envelopes import (
    decode_boundary,
    decode_gradients,
    encode_boundary,
    encode_gradients,
)
from splitfleet.transport.split_wire import (
    boundary_to_envelope,
    envelope_to_boundary,
    envelope_to_gradients,
    gradients_to_envelope,
)

from .logical_state import LogicalClientModelState
from .metrics import timed_call
from .resource_emulator import ControlledNetworkLink
from .resource_monitor import EnergyMeter, PeakMemoryTracker, ServerJobPool
from .split_candidates import ExperimentSplitCandidate


@dataclass
class SwitchMeasurement:
    old_split_key: str
    new_split_key: str
    runtime_prepare_ms: float
    model_transfer_ms: float
    optimizer_state_transfer_ms: float
    total_switch_ms: float


@dataclass
class BatchMeasurement:
    loss: float
    num_examples: int
    client_forward_ms: float
    client_backward_ms: float
    server_forward_ms: float
    server_backward_ms: float
    network_upload_ms: float
    network_download_ms: float
    server_queue_ms: float
    boundary_forward_bytes: int
    boundary_gradient_bytes: int
    client_peak_memory_mb: float | None
    server_peak_memory_mb: float | None
    client_energy_j: float | None
    server_energy_j: float | None
    end_to_end_batch_ms: float
    measured_uplink_mbps: float | None
    measured_downlink_mbps: float | None


class LogicalClientRuntime:
    """Own a complete model and ABI-keyed TorchLens runtimes for one client."""

    def __init__(
        self,
        client_id: str,
        model_factory: Callable[[], torch.nn.Module],
        sample_inputs: torch.Tensor,
        candidates: Mapping[str, ExperimentSplitCandidate],
        *,
        device: str | torch.device = "cpu",
        learning_rate: float = 0.01,
        max_batch_size: int | None = None,
    ) -> None:
        self.client_id = str(client_id)
        self.device = torch.device(device)
        self.model = model_factory().to(self.device)
        self.sample_inputs = sample_inputs.to(self.device)
        self.candidates = dict(candidates)
        self.learning_rate = float(learning_rate)
        self.max_batch_size = max(1, int(max_batch_size or sample_inputs.shape[0]))
        self.current_split_key = "full_local"
        self._runtime_cache: dict[str, Any] = {}
        self.optimizer: torch.optim.Optimizer | None = None

    def activate(
        self,
        logical_state: LogicalClientModelState,
        split_key: str,
    ) -> tuple[Any | None, SwitchMeasurement]:
        logical_state.validate_stateless_sgd()
        started = time.perf_counter_ns()
        old = self.current_split_key
        _, model_transfer_ms = timed_call(lambda: logical_state.load_model(self.model), self.device)
        self.model.train()
        runtime_prepare_ms = 0.0
        runtime = None
        if split_key != "full_local":
            candidate = self.candidates[split_key]
            runtime = self._runtime_cache.get(split_key)
            if runtime is None:
                runtime, runtime_prepare_ms = timed_call(
                    lambda: prepare_torchlens_runtime(
                        self.model,
                        self.sample_inputs,
                        boundary=str(candidate.boundary),
                        trainable=True,
                        dynamic_batch=(1, max(2, self.max_batch_size)),
                        trace_batch_mode="batch_gt1",
                        model_name=self.model.__class__.__name__,
                    ),
                    self.device,
                )
                self._runtime_cache[split_key] = runtime
        optimizer_started = time.perf_counter_ns()
        # Only no-momentum SGD is supported, so there is no hidden state to lose
        # when ownership moves across the boundary.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.0)
        optimizer_ms = (time.perf_counter_ns() - optimizer_started) / 1_000_000.0
        total_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.current_split_key = split_key
        logical_state.split_key = split_key
        return runtime, SwitchMeasurement(
            old_split_key=old,
            new_split_key=split_key,
            runtime_prepare_ms=runtime_prepare_ms,
            model_transfer_ms=model_transfer_ms,
            optimizer_state_transfer_ms=optimizer_ms,
            total_switch_ms=total_ms if old != split_key else 0.0,
        )

    def train_batch(
        self,
        runtime: Any | None,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        round_id: int,
        loss_fn: Callable[[Any, Any], torch.Tensor],
        server_pool: ServerJobPool,
        network: ControlledNetworkLink,
        energy: EnergyMeter,
        deadline_ns: int | None = None,
    ) -> BatchMeasurement:
        _check_deadline(deadline_ns)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        if self.optimizer is None:
            raise RuntimeError("Logical client runtime must be activated before training.")
        started = time.perf_counter_ns()
        client_energy_start = energy.read_joules()
        client_tracker = PeakMemoryTracker(self.device)
        holder: dict[str, Any] = {}

        def execute() -> None:
            if runtime is None:
                self.optimizer.zero_grad(set_to_none=True)
                outputs, forward_ms = timed_call(lambda: self.model(inputs), self.device)
                _check_deadline(deadline_ns)
                loss = loss_fn(outputs, targets)

                def backward() -> None:
                    loss.backward()
                    self.optimizer.step()

                _, backward_ms = timed_call(backward, self.device)
                _check_deadline(deadline_ns)
                holder.update(
                    loss=float(loss.detach().cpu()),
                    client_forward_ms=forward_ms,
                    client_backward_ms=backward_ms,
                    server_forward_ms=0.0,
                    server_backward_ms=0.0,
                    network_upload_ms=0.0,
                    network_download_ms=0.0,
                    server_queue_ms=0.0,
                    boundary_forward_bytes=0,
                    boundary_gradient_bytes=0,
                    server_peak_memory_mb=0.0,
                    measured_uplink_mbps=None,
                    measured_downlink_mbps=None,
                    server_energy_j=None,
                )
                return
            self.optimizer.zero_grad(set_to_none=True)
            boundary, prefix_ms = timed_call(
                lambda: runtime.backend.run_prefix(inputs, training=True), self.device
            )
            _check_deadline(deadline_ns)
            contract = graph_contract_for_runtime_handle(runtime)
            envelope = boundary_to_envelope(
                boundary,
                round_id=round_id,
                client_id=self.client_id,
                step_id=uuid.uuid4().hex,
                plan_id=runtime.plan.plan_id,
                split_id=contract.split_id,
                canonical_graph_hash=contract.canonical_graph_hash,
                boundary_schema_hash=contract.boundary_schema_hash,
                model_version=round_id,
            )
            wire = encode_boundary(envelope)
            upload = network.transfer(wire, "uplink", deadline_ns=deadline_ns)
            restored_envelope = decode_boundary(wire)
            server_boundary = envelope_to_boundary(restored_envelope, runtime.runtime, self.device)
            server_phase: dict[str, float] = {}
            server_energy_start = energy.read_joules()
            server_tracker = PeakMemoryTracker(self.device)

            def suffix():
                return runtime.backend.train_suffix(
                    server_boundary,
                    targets,
                    loss_fn=loss_fn,
                    optimizer=self.optimizer,
                    measurements=server_phase,
                )

            def queued_suffix():
                return server_tracker.measure(suffix)

            ((loss, boundary_grads), server_peak), queue_ms = server_pool.execute(
                queued_suffix,
                deadline_ns=deadline_ns,
            )
            _check_deadline(deadline_ns)
            if "server_forward_ms" not in server_phase or "server_backward_ms" not in server_phase:
                # The cost model is fitted on the forward/backward split. A
                # runtime that can only report `server_total_ms` must not be
                # recorded as zero server compute.
                raise RuntimeError(
                    "The split runtime reported no server forward/backward phase split "
                    f"(measured keys: {sorted(server_phase)}); RA-SplitFed requires the "
                    "phase-separated torch suffix runtime."
                )
            server_energy_end = energy.read_joules()
            gradient_envelope = gradients_to_envelope(restored_envelope, boundary_grads)
            gradient_wire = encode_gradients(gradient_envelope)
            download = network.transfer(
                gradient_wire,
                "downlink",
                deadline_ns=deadline_ns,
            )
            restored_gradients = envelope_to_gradients(decode_gradients(gradient_wire), self.device)
            _, backward_ms = timed_call(
                lambda: runtime.backend.backward_prefix(
                    boundary,
                    boundary_grads=restored_gradients,
                    optimizer=self.optimizer,
                ),
                self.device,
            )
            _check_deadline(deadline_ns)
            holder.update(
                loss=float(loss.detach().cpu()),
                client_forward_ms=prefix_ms,
                client_backward_ms=backward_ms,
                server_forward_ms=float(server_phase.get("server_forward_ms", 0.0)),
                server_backward_ms=float(server_phase.get("server_backward_ms", 0.0)),
                network_upload_ms=upload.elapsed_ms,
                network_download_ms=download.elapsed_ms,
                server_queue_ms=queue_ms,
                boundary_forward_bytes=len(wire),
                boundary_gradient_bytes=len(gradient_wire),
                server_peak_memory_mb=server_peak,
                measured_uplink_mbps=upload.achieved_mbps,
                measured_downlink_mbps=download.achieved_mbps,
                server_energy_j=_energy_delta(server_energy_start, server_energy_end),
            )

        _, client_peak = client_tracker.measure(execute)
        _check_deadline(deadline_ns)
        client_energy_end = energy.read_joules()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return BatchMeasurement(
            num_examples=int(targets.shape[0]),
            client_peak_memory_mb=client_peak,
            client_energy_j=_energy_delta(client_energy_start, client_energy_end),
            end_to_end_batch_ms=elapsed_ms,
            **holder,
        )


def _energy_delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return end - start


def _check_deadline(deadline_ns: int | None) -> None:
    if deadline_ns is not None and time.perf_counter_ns() >= int(deadline_ns):
        raise TimeoutError("client round deadline exceeded")
