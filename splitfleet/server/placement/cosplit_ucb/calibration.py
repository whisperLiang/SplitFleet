"""State-preserving deployment calibrations for fresh online CoSplit-UCB.

The existing captured graph is reused. Calibration performs temporary stage-owned optimizer steps; model, gradients
and RNG are restored. Samples
initialize the existing discounted learners, never replace their predictions.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
import random
import time
from typing import Any

import numpy as np
import torch

from flwr.common import GetPropertiesIns

from splitfleet.common.constants import (
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY, AUTOSPLIT_MODE_CONFIG_KEY, AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
    COSPLIT_CALIBRATION_PARAMETERS_CONFIG_KEY,
)
from splitfleet.common.model_state import tensor_state_hash
from .types import ExecutionProfileKey


def calibration_boundaries(candidates):
    """Four structural anchors, independent of previously measured timings."""
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("Calibration requires an admitted training catalog")
    anchors = tuple(min(candidates, key=lambda c: (abs(c.graph_position_ratio - fraction), c.boundary))
                    for fraction in (0.0, .25, .75, 1.0))
    return tuple(dict.fromkeys(candidate.boundary for candidate in anchors))


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def calibrate_split(model, inputs, targets, *, boundaries, make_handle, loss_fn,
                    device, source, optimizer_fn):
    """Measure device costs and restore training state.

    Deployment callers supply the training optimizer factory. Offline profiles
    explicitly pass ``None`` to measure forward/backward without updates.
    """
    device = torch.device(device)
    boundaries = tuple(boundaries)
    if not boundaries or len(set(boundaries)) != len(boundaries):
        raise ValueError("Calibration requires distinct admitted boundaries")
    started = time.perf_counter()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_hash = tensor_state_hash(state)
    gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                 for name, p in model.named_parameters()}
    modes = {name: module.training for name, module in model.named_modules()}
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    records = []
    warmup, repeats = 1, 1
    optimizer_stage = "suffix" if source == "server_shape_matched_current_deployment" else "prefix"
    try:
        model.train()
        for cut in boundaries:
            # Buffers such as BatchNorm statistics must start from the same
            # state at every anchor.
            model.load_state_dict(state)
            handle = make_handle(cut)
            if handle.plan.boundary != cut:
                raise ValueError("Calibration must use an exact admitted canonical boundary")
            optimizer = None
            if optimizer_fn is not None:
                from splitfleet.autosplit.state_ownership import state_ownership
                ownership = state_ownership(handle, "calibration")
                owned_names = {name for name, owner in zip(ownership["names"], ownership["owners"])
                               if owner == optimizer_stage}
                parameters = [parameter for name, parameter in model.named_parameters()
                              if name in owned_names and parameter.requires_grad]
                if parameters:
                    optimizer = optimizer_fn(torch.nn.ParameterList(parameters))
            samples = []
            for batch_index in range(warmup + repeats):
                model.zero_grad(set_to_none=True)
                _synchronize(device)
                begin = time.perf_counter()
                boundary = handle.backend.run_prefix(*inputs.args, input_kwargs=dict(inputs.kwargs), training=True)
                _synchronize(device)
                forward_ms = (time.perf_counter() - begin) * 1000
                begin = time.perf_counter()
                loss, returned = handle.backend.train_suffix(boundary, targets, loss_fn=loss_fn,
                    optimizer=optimizer if optimizer_stage == "suffix" else None)
                _synchronize(device)
                local_service_ms = (time.perf_counter() - begin) * 1000
                begin = time.perf_counter()
                handle.backend.backward_prefix(boundary, boundary_grads=returned,
                    optimizer=optimizer if optimizer_stage == "prefix" else None)
                _synchronize(device)
                backward_ms = (time.perf_counter() - begin) * 1000
                if batch_index >= warmup:
                    samples.append((forward_ms, backward_ms, local_service_ms))
                del boundary, returned
            forward_ms, backward_ms, local_service_ms = np.median(samples, axis=0).tolist()
            if not bool(torch.isfinite(loss).all()):
                raise ValueError("Calibration loss is non-finite")
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Calibration gradients are non-finite")
            records.append({"boundary": cut, "graph_signature": handle.plan.graph_signature,
                "feature_abi_id": handle.plan.feature_abi_id,
                "client_forward_ms": forward_ms, "client_backward_ms": backward_ms,
                "local_tail_service_ms": local_service_ms, "loss": float(loss.detach()),
                "measured_batches": repeats,
                "optimizer_steps": warmup + repeats if optimizer is not None else 0,
                "optimizer_stage": optimizer_stage if optimizer_fn is not None else None})
            del loss, handle, optimizer
    finally:
        model.load_state_dict(state)
        for name, parameter in model.named_parameters():
            value = gradients[name]
            parameter.grad = None if value is None else value.to(parameter.device)
        for name, module in model.named_modules():
            module.training = modes[name]
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    final_hash = tensor_state_hash(model.state_dict())
    if final_hash != initial_hash or not torch.equal(cpu_rng, torch.get_rng_state()):
        raise ValueError("Calibration did not preserve the initial model / RNG")
    if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise ValueError("Calibration did not preserve CUDA RNG")
    return {"schema": "splitfleet.cosplit-calibration", "source": source,
        "device": str(device), "model_hash_before": initial_hash, "model_hash_after": final_hash,
        "state_and_torch_rng_preserved": True,
        "optimizer_steps": sum(row["optimizer_steps"] for row in records),
        "persistent_optimizer_steps": 0, "warmup_batches": warmup,
        "elapsed_sec": time.perf_counter() - started, "records": records,
        "interpretation": "Local split execution; temporary optimizer state discarded and model/RNG restored; not network or global round timing"}


def provider_handle(provider, boundary):
    """Materialize an anchor from the existing capture, without retracing."""
    provider.get_candidates(training=True)
    backend = copy.copy(provider._backends[True])
    backend.split(provider._candidates[True][boundary])
    return backend.make_handle()


def measure_transport_echo(client, *, size, round_id, reply_size=None):
    """Measure coordinator monotonic RTT without inspecting private tensors."""
    payload = bytes(size)
    config = {"cosplit_probe_payload": payload}
    if reply_size is not None:
        config["cosplit_probe_reply_bytes"] = reply_size
    begin = time.perf_counter_ns()
    result = client.get_properties(GetPropertiesIns(config=config),
                                   timeout=60, group_id=round_id)
    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
    expected = payload if reply_size is None or reply_size == size else bytes(reply_size)
    if result.properties.get("cosplit_probe_payload") != expected or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError("Current-deployment transport probe failed")
    return elapsed_ms


class ClientTelemetry:
    """Bind device metadata without measuring or changing learned costs."""

    def __init__(self, *, policy=None):
        self.clients: dict[str, dict[str, Any]] = {}
        self.policy = policy
        self.receipt = None

    def _bind_properties(self, clients, round_id: int, *, config=None, refresh=False) -> None:
        for client in clients:
            if str(client.cid) in self.clients and not refresh:
                continue
            result = client.get_properties(
                GetPropertiesIns(config=config or {}), timeout=300, group_id=round_id,
            )
            props = dict(result.properties)
            if not props.get("logical_client_id"):
                raise ValueError("Online placement requires a physical worker identity")
            for field in ("num_batches", "batch_size"):
                if int(props.get(field, 0)) < 1:
                    raise ValueError(f"Online placement requires positive {field}")
            self.clients[str(client.cid)] = props

    def bind_clients(self, clients, round_id):
        self._bind_properties(clients, round_id)

    def bind_evaluation_clients(self, clients, round_id):
        self._bind_properties(clients, round_id)

    def client_context(self, client_id: str) -> dict[str, Any]:
        props = self.clients.get(str(client_id))
        if props is None:
            raise ValueError(f"Online placement has no bound worker {client_id!r}")
        return {key: props[key] for key in (
            "num_batches", "batch_size", "device_type", "accelerator", "precision",
        ) if key in props}

    def server_context(self) -> dict[str, Any]:
        return {}

    def execution_profile(self, client_id: str) -> dict[str, str]:
        props = self.clients[str(client_id)]
        return {
            "framework_backend": str(props["framework_backend"]),
            "runtime_backend": str(props["runtime_backend"]),
            "device_type": str(props["device_type"]),
            "accelerator": str(props["accelerator"]),
            "precision": str(props["precision"]),
        }


class CalibratedTelemetry(ClientTelemetry):
    """Bind current device measurements and seed discounted learners once."""

    def __init__(self, *, initial_model_hash, server_receipt, parameters=None, policy=None):
        super().__init__(policy=policy)
        self.initial_model_hash = initial_model_hash
        self.server_receipt = server_receipt
        self.parameters = parameters
        self.initialized = False

    def bind_clients(self, clients, round_id):
        if self.policy is None or self.server_receipt is None:
            raise ValueError("Calibration must be ready before client binding")
        policy = self.policy
        catalog = {candidate.boundary: candidate for candidate in policy._catalog()}
        if self.initialized or policy._state_restored:
            super().bind_clients(clients, round_id)
            return
        if (any(key[1] for key in policy._round_assignments) or
                any(key[1] for key in policy._round_contexts) or policy._observed_actions or
                policy._last_boundary or policy.learners.has_observations):
            raise ValueError("Bootstrap only a fresh, unplanned learner")
        clients = tuple(clients)
        if not clients:
            raise ValueError("Calibration requires bound physical clients")
        provider = policy.candidate_provider
        config = {"cosplit_calibration_boundaries": json.dumps(calibration_boundaries(catalog.values()))}
        if getattr(provider, "dynamic_batch", None) is not None:
            config[AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY] = json.dumps(provider.dynamic_batch)
        config[AUTOSPLIT_MODE_CONFIG_KEY] = getattr(provider, "mode", "generated_eager")
        if getattr(provider, "trace_batch_mode", None):
            config[AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY] = provider.trace_batch_mode
        if self.parameters is not None:
            config[COSPLIT_CALIBRATION_PARAMETERS_CONFIG_KEY] = self.parameters
        # Evaluation may have bound these workers without calibration. Request
        # receipts only for the first training cohort, using its global model.
        self._bind_properties(clients, round_id, config=config, refresh=True)
        calibration_clients = tuple(str(client.cid) for client in clients)
        expected = set(calibration_boundaries(catalog.values()))
        steps, measured = 2, 1
        measurements = []
        # Validate every receipt before mutating any sufficient statistics.
        receipts = [(cid, json.loads(self.clients[cid]["online_calibration_receipt"]))
                    for cid in calibration_clients]
        receipts.append((None, self.server_receipt))
        for cid, receipt in receipts:
            stage = "prefix" if cid is not None else "suffix"
            expected_steps = {
                cut: 0 if catalog[cut].metadata.get(f"optimizer_{stage}_parameter_bytes") == 0 else steps
                for cut in expected
            }
            expected_source = ("client_private_sample_current_deployment" if cid is not None
                               else "server_shape_matched_current_deployment")
            expected_schema = "splitfleet.cosplit-calibration"
            if receipt.get("schema") != expected_schema or receipt.get("source") != expected_source:
                raise ValueError("Calibration must come from the current deployment")
            if cid is not None and str(receipt.get("device", "")).split(":", 1)[0] != self.clients[cid]["device_type"]:
                raise ValueError("Calibration device differs from the worker execution profile")
            if not math.isfinite(float(receipt.get("elapsed_sec", -1))) or float(receipt["elapsed_sec"]) < 0:
                raise ValueError("Calibration duration must be finite and nonnegative")
            if (receipt.get("model_hash_before") != self.initial_model_hash or
                receipt.get("model_hash_after") != self.initial_model_hash or
                receipt.get("state_and_torch_rng_preserved") is not True or
                receipt.get("persistent_optimizer_steps") != 0):
                raise ValueError("Calibration state does not match the frozen initial model")
            if (receipt.get("optimizer_steps") != sum(expected_steps.values())
                                  or receipt.get("warmup_batches") != 1):
                raise ValueError("Calibration optimizer scope mismatch")
            records = receipt["records"]
            if len(records) != len(expected) or {r["boundary"] for r in records} != expected:
                raise ValueError("Calibration anchors differ from the canonical catalog")
            for row in records:
                candidate = catalog[row["boundary"]]
                if row["graph_signature"] != candidate.graph_signature or row["feature_abi_id"] != candidate.feature_abi_id:
                    raise ValueError("Calibration graph / boundary ABI mismatch")
                names = ("client_forward_ms", "client_backward_ms") if cid is not None else ("local_tail_service_ms",)
                if (row.get("measured_batches") != measured or
                        row.get("optimizer_steps") != expected_steps[row["boundary"]] or
                        row.get("optimizer_stage") != stage):
                    raise ValueError("Calibration sample scope mismatch")
                if any(not math.isfinite(float(row[name])) or float(row[name]) < 0 for name in names):
                    raise ValueError("Calibration costs must be finite and nonnegative")
                measurements.append((cid, candidate, row))
        server_context = policy._server_telemetry()
        prepared, logical = [], []
        for cid, candidate, row in measurements:
            context_cid = cid if cid is not None else calibration_clients[0]
            props = self.client_context(context_cid)
            contexts = policy._contexts(client_id=context_cid, candidate=candidate,
                                       client_telemetry=props, server_telemetry=server_context)
            profile = ExecutionProfileKey.from_value(self.execution_profile(context_cid))
            prepared.append((cid, row, contexts, profile))
        transport_samples, transport_contexts, transport_warmup = [], [], []
        exchange_samples, exchange_contexts = [], []
        # Application-level echo is a bootstrap prior measured here, not
        # a one-way timing or an assertion that Flower and split RPCs
        # have identical serialization. Actual training RTT replaces it.
        template = next(iter(catalog.values()))
        probe_started = time.perf_counter()
        for client in clients:
            cid = str(client.cid)
            for repeat in range(2):
                elapsed_ms = measure_transport_echo(client, size=65536, round_id=round_id)
                transport_warmup.append({"logical_client_id": self.clients[cid]["logical_client_id"],
                    "payload_bytes": 65536, "repeat": repeat, "roundtrip_ms": elapsed_ms, "echo_verified": True})
            for size in (65536, 2097152):
                batch_size = int(self.client_context(cid)["batch_size"])
                candidate = replace(template, boundary_forward_bytes=size,
                                    boundary_gradient_bytes=None, boundary_tensor_count=1,
                                    metadata={**template.metadata, "payload_batch_size": batch_size,
                                              "boundary_forward_bytes_by_batch_size": {batch_size: size}})
                context = policy.context_encoder.network_context(candidate, self.client_context(cid), direction="upload")
                for repeat in range(2):
                    elapsed_ms = measure_transport_echo(client, size=size, round_id=round_id)
                    transport_samples.append({"logical_client_id": self.clients[cid]["logical_client_id"],
                        "payload_bytes": size, "repeat": repeat, "roundtrip_ms": elapsed_ms,
                        "echo_verified": True})
                    transport_contexts.append((cid, context, elapsed_ms))
            for request_size, reply_size in ((65536, 65536), (2097152, 65536), (65536, 2097152)):
                context = policy.context_encoder.exchange_context(download_bytes=request_size, upload_bytes=reply_size)
                for repeat in range(2):
                    elapsed_ms = measure_transport_echo(client, size=request_size, reply_size=reply_size, round_id=round_id)
                    exchange_samples.append({"logical_client_id": self.clients[cid]["logical_client_id"],
                        "request_bytes": request_size, "reply_bytes": reply_size, "repeat": repeat,
                        "roundtrip_ms": elapsed_ms, "reply_verified": True})
                    exchange_contexts.append((cid, context, elapsed_ms))
        probe_elapsed = time.perf_counter() - probe_started
        for cid, row, contexts, profile in prepared:
            if cid is not None:
                policy.learners.edge.update(profile, contexts.edge,
                    forward_ms=row["client_forward_ms"], backward_ms=row["client_backward_ms"], round_id=0)
                logical.append({"logical_client_id": self.clients[cid]["logical_client_id"], **row})
            else:
                policy.learners.server.update(contexts.server, row["local_tail_service_ms"], round_id=0)
        for cid, context, elapsed_ms in transport_contexts:
            policy.learners.network.update(cid, context, elapsed_ms, round_id=0)
        for cid, context, elapsed_ms in exchange_contexts:
            policy.learners.network.update_exchange(cid, context, elapsed_ms, round_id=0)
        self.receipt = {"source": "current_deployment_restored_training_cost_calibration",
            "initial_model_hash": self.initial_model_hash, "client_samples": logical,
            "server_calibration": self.server_receipt,
            "client_calibration_elapsed_sec": {self.clients[cid]["logical_client_id"]: receipt["elapsed_sec"] for cid, receipt in receipts if cid is not None},
            "network_initialized": True, "prediction_override": False,
            "discounted_online_updates_retained": True,
            "mean_exploration_budget_enabled": True}
        self.receipt["temporary_optimizer_updates"] = True
        self.receipt["transport_bootstrap"] = {
            "schema": "splitfleet.current-transport-probe", "elapsed_sec": probe_elapsed,
            "source": "current_deployment_flower_echo", "samples": transport_samples,
            "warmup_samples": transport_warmup,
            "interpretation": "Coordinator monotonic RTT for a payload and its echo; no cross-host timestamp subtraction. Bootstrap only; split-RPC RTT feedback remains active."}
        self.receipt["state_exchange_bootstrap"] = {
            "schema": "splitfleet.current-state-exchange-probe",
            "source": "current_deployment_controlled_request_reply_sizes", "samples": exchange_samples,
            "interpretation": "Whole property-RPC durations with independently varied request/reply sizes, not unmeasured one-way delays. Bootstrap only; actual fit RPC minus full client-handler duration updates the per-round model."}
        self.initialized = True
        self.parameters = None
