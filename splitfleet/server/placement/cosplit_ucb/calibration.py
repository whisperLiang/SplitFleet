"""Small current-deployment calibrations for fresh online CoSplit-UCB.

The existing captured graph is reused. Calibration performs split backward but
no optimizer step; state, gradients and random streams are restored. Samples
initialize the existing discounted learners, never replace their predictions.
"""

from __future__ import annotations

import copy
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
    """Three structural anchors, independent of previously measured timings."""
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("Calibration requires an admitted training catalog")
    anchors = (min(candidates, key=lambda c: (c.graph_position_ratio, c.boundary)),
               min(candidates, key=lambda c: (abs(c.graph_position_ratio - .25), c.boundary)),
               max(candidates, key=lambda c: (c.graph_position_ratio, c.boundary)))
    return tuple(dict.fromkeys(candidate.boundary for candidate in anchors))


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def calibrate_split(model, inputs, targets, *, boundaries, make_handle, loss_fn,
                    device, source):
    """Measure F/B on this device, with no optimization or persistent RNG change."""
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
    try:
        model.train()
        for cut in boundaries:
            # Buffers such as BatchNorm statistics must start from the same
            # state at every anchor.
            model.load_state_dict(state)
            handle = make_handle(cut)
            if handle.plan.boundary != cut:
                raise ValueError("Calibration must use an exact admitted canonical boundary")
            model.zero_grad(set_to_none=True)
            _synchronize(device)
            begin = time.perf_counter()
            boundary = handle.backend.run_prefix(*inputs.args, input_kwargs=dict(inputs.kwargs), training=True)
            _synchronize(device)
            forward_ms = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            loss, returned = handle.backend.train_suffix(boundary, targets, loss_fn=loss_fn, optimizer=None)
            _synchronize(device)
            local_service_ms = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            handle.backend.backward_prefix(boundary, boundary_grads=returned, optimizer=None)
            _synchronize(device)
            backward_ms = (time.perf_counter() - begin) * 1000
            if not bool(torch.isfinite(loss).all()):
                raise ValueError("Calibration loss is non-finite")
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Calibration gradients are non-finite")
            records.append({"boundary": cut, "graph_signature": handle.plan.graph_signature,
                "feature_abi_id": handle.plan.feature_abi_id,
                "client_forward_ms": forward_ms, "client_backward_ms": backward_ms,
                "local_tail_service_ms": local_service_ms, "loss": float(loss.detach()),
                "measured_batches": 1, "optimizer_steps": 0})
            del boundary, returned, loss, handle
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
    return {"schema": "splitfleet.online-calibration.v1", "source": source,
        "device": str(device), "model_hash_before": initial_hash, "model_hash_after": final_hash,
        "state_and_torch_rng_preserved": True, "optimizer_steps": 0,
        "elapsed_sec": time.perf_counter() - started, "records": records,
        "interpretation": "Synchronized local split execution without Adam updates or wire transport; online training feedback corrects initialization bias"}


def provider_handle(provider, boundary):
    """Materialize an anchor from the existing capture, without retracing."""
    provider.get_candidates(training=True)
    backend = copy.copy(provider._backends[True])
    backend.split(provider._candidates[True][boundary])
    return backend.make_handle()


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
        measurements = []
        # Validate every receipt before mutating any sufficient statistics.
        receipts = [(cid, json.loads(self.clients[cid]["online_calibration_receipt"]))
                    for cid in calibration_clients]
        receipts.append((None, self.server_receipt))
        for cid, receipt in receipts:
            expected_source = ("client_private_sample_current_deployment" if cid is not None
                               else "server_shape_matched_current_deployment")
            if receipt.get("schema") != "splitfleet.online-calibration.v1" or receipt.get("source") != expected_source:
                raise ValueError("Calibration must come from the current deployment")
            if cid is not None and str(receipt.get("device", "")).split(":", 1)[0] != self.clients[cid]["device_type"]:
                raise ValueError("Calibration device differs from the worker execution profile")
            if not math.isfinite(float(receipt.get("elapsed_sec", -1))) or float(receipt["elapsed_sec"]) < 0:
                raise ValueError("Calibration duration must be finite and nonnegative")
            if (receipt.get("model_hash_before") != self.initial_model_hash or
                receipt.get("model_hash_after") != self.initial_model_hash or
                receipt.get("state_and_torch_rng_preserved") is not True or receipt.get("optimizer_steps") != 0):
                raise ValueError("Calibration state does not match the frozen initial model")
            records = receipt["records"]
            if len(records) != len(expected) or {r["boundary"] for r in records} != expected:
                raise ValueError("Calibration anchors differ from the canonical catalog")
            for row in records:
                candidate = catalog[row["boundary"]]
                if row["graph_signature"] != candidate.graph_signature or row["feature_abi_id"] != candidate.feature_abi_id:
                    raise ValueError("Calibration graph / boundary ABI mismatch")
                names = ("client_forward_ms", "client_backward_ms") if cid is not None else ("local_tail_service_ms",)
                if row.get("measured_batches") != 1 or row.get("optimizer_steps") != 0:
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
        for cid, row, contexts, profile in prepared:
            if cid is not None:
                policy.learners.edge.update(profile, contexts.edge,
                    forward_ms=row["client_forward_ms"], backward_ms=row["client_backward_ms"], round_id=0)
                logical.append({"logical_client_id": self.clients[cid]["logical_client_id"], **row})
            else:
                policy.learners.server.update(contexts.server, row["local_tail_service_ms"], round_id=0)
        self.receipt = {"source": "current_deployment_no_update_split_calibration",
            "initial_model_hash": self.initial_model_hash, "client_samples": logical,
            "server_calibration": self.server_receipt,
            "client_calibration_elapsed_sec": {self.clients[cid]["logical_client_id"]: receipt["elapsed_sec"] for cid, receipt in receipts if cid is not None},
            "network_initialized": False, "prediction_override": False,
            "discounted_online_updates_retained": True,
            "mean_exploration_budget_enabled": True}
        self.initialized = True
        self.parameters = None
