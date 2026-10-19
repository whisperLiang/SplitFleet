"""State-preserving deployment calibration for measured Greedy placement.

The existing captured graph is reused. Calibration performs temporary stage-owned optimizer steps; model, gradients
and RNG are restored. Samples
initialize private interpolation anchors and affine transport costs.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
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
    CLIENT_ID_CONFIG_KEY,
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY, AUTOSPLIT_MODE_CONFIG_KEY, AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
    PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY,
    PLACEMENT_CALIBRATION_HASH_HANDSHAKE_KEY, PLACEMENT_CALIBRATION_MODEL_HASH_CONFIG_KEY,
    PLACEMENT_CALIBRATION_PARAMETERS_REQUIRED_KEY,
)
from splitfleet.common.model_state import tensor_state_hash
from splitfleet.common.network_probe import PROBE_CONFIG_KEY, PROBE_RESULT_KEY, PROBE_MODE, estimate_network
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
    from splitfleet.backends.utils import adapter_for
    adapter = adapter_for(model, inputs.args if inputs.args else inputs.kwargs)
    if adapter.backend_name != 'torch':
        from .native_calibration import calibrate_native_split
        return calibrate_native_split(model, inputs, targets, boundaries=boundaries,
            make_handle=make_handle, loss_fn=loss_fn, device=device, source=source,
            optimizer_fn=optimizer_fn, adapter=adapter)
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
            # Production server replicas use a private RNG even at capacity
            # one. Calibrate the same suffix path rather than an unseeded
            # backend reference that omits its isolation overhead.
            isolate = getattr(handle.backend, "enable_replica_rng_isolation", None)
            if optimizer_stage == "suffix" and isolate is not None:
                isolate(True, seed=torch.initial_seed())
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
                    optimizer=optimizer if optimizer_stage == "suffix" else None,
                    boundary_gradients_only=optimizer_stage == "prefix")
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
    return {"schema": "splitfleet.training-calibration", "source": source,
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
    config = {"splitfleet_probe_payload": payload}
    if reply_size is not None:
        config["splitfleet_probe_reply_bytes"] = reply_size
    begin = time.perf_counter_ns()
    result = client.get_properties(GetPropertiesIns(config=config),
                                   timeout=60, group_id=round_id)
    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
    expected = payload if reply_size is None or reply_size == size else bytes(reply_size)
    if result.properties.get("splitfleet_probe_payload") != expected or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError("Current-deployment transport probe failed")
    return elapsed_ms


class ClientTelemetry:
    """Bind device metadata and actively measure the current training link."""

    def __init__(self, *, policy=None):
        self.clients: dict[str, dict[str, Any]] = {}
        self.policy = policy
        self.receipt = None
        self.round_probes = {}

    def _client_workers(self, count):
        return min(count, getattr(self.policy, 'client_concurrency', 1) or 1)

    def _map_clients(self, call, clients):
        workers = self._client_workers(len(clients))
        if workers <= 1:
            return [call(client) for client in clients]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Ordered results preserve cost initialization and receipt order.
            return list(executor.map(call, clients))

    def _bind_properties(self, clients, round_id: int, *, config=None, refresh=False) -> None:
        pending = [client for client in clients if refresh or str(client.cid) not in self.clients]
        if not pending:
            return

        def fetch(client):
            request = dict(config or {})
            payload = request.get(PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY)
            capabilities = {**self.clients.get(str(client.cid), {}), **getattr(client, 'properties', {})}
            handshake = (payload is not None and PLACEMENT_CALIBRATION_MODEL_HASH_CONFIG_KEY in request
                         and capabilities.get(PLACEMENT_CALIBRATION_HASH_HANDSHAKE_KEY) is True)
            if handshake:
                request.pop(PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY)
            result = client.get_properties(
                GetPropertiesIns(config=request), timeout=300, group_id=round_id,
            )
            props = dict(result.properties)
            fallback = props.get(PLACEMENT_CALIBRATION_PARAMETERS_REQUIRED_KEY) is True
            if fallback:
                if not handshake:
                    raise ValueError('Client requested calibration parameters after receiving the full payload')
                result = client.get_properties(GetPropertiesIns(config=dict(config)), timeout=300, group_id=round_id)
                props = dict(result.properties)
                if props.get(PLACEMENT_CALIBRATION_PARAMETERS_REQUIRED_KEY):
                    raise ValueError('Client still requires calibration parameters after the full payload')
            if not props.get("logical_client_id"):
                raise ValueError("Online placement requires a physical worker identity")
            for field in ("num_batches", "batch_size"):
                if int(props.get(field, 0)) < 1:
                    raise ValueError(f"Online placement requires positive {field}")
            return str(client.cid), props, {
                'requests': 1 + int(fallback), 'hash_matched_clients': int(handshake and not fallback),
                'fallback_clients': int(fallback),
                'parameter_payload_bytes': len(payload) if payload is not None and (not handshake or fallback) else 0,
                'avoided_parameter_payload_bytes': len(payload) if handshake and not fallback else 0,
            }

        # The virtual runtime supplies its admitted worker bound. Each client
        # still measures its own private model/data and returns its own receipt.
        # Physical deployments retain the sequential default unless bounded.
        results = self._map_clients(fetch, pending)
        self.clients.update((cid, props) for cid, props, _ in results)
        return {key: sum(transfer[key] for _, _, transfer in results) for key in (
            'requests', 'hash_matched_clients', 'fallback_clients',
            'parameter_payload_bytes', 'avoided_parameter_payload_bytes')}

    def bind_clients(self, clients, round_id):
        clients = tuple(clients)
        self._bind_properties(clients, round_id)
        self.refresh_transport(clients, round_id)

    def refresh_transport(self, clients, round_id):
        """Complete fresh split-channel probes before this round's selection."""
        if not getattr(self.policy, "uses_active_network_probes", False):
            return
        clients = tuple(clients)
        previous = self.round_probes.get(int(round_id), {})
        pending = [client for client in clients if str(client.cid) not in previous]

        def measure(client):
            cid = str(client.cid)
            began = time.perf_counter()
            result = client.get_properties(GetPropertiesIns(config={
                PROBE_CONFIG_KEY: int(round_id), CLIENT_ID_CONFIG_KEY: cid,
            }), timeout=600, group_id=round_id)
            receipt = json.loads(result.properties[PROBE_RESULT_KEY])
            context = estimate_network(receipt, round_id=int(round_id))
            return cid, {**receipt, "context": context,
                         "coordinator_elapsed_sec": time.perf_counter() - began}

        # Publish no partial cohort if any probe fails. A failure cannot
        # silently reuse a configured rate or an earlier round's observation.
        # Probing clients concurrently makes the shared RPC service's queue
        # dominate small-RPC RTT and can produce negative directional deltas.
        # Calibration and training retain their independent worker bound.
        measured = [measure(client) for client in pending]
        self.round_probes[int(round_id)] = {**previous, **dict(measured)}
        observer = getattr(self.policy, "transport_probe_observer", None)
        if observer is not None:
            for cid, receipt in measured:
                observer({"round_id": int(round_id), "client_id": cid, **receipt})

    def network_context(self, client_id, round_id):
        receipt = self.round_probes.get(int(round_id), {}).get(str(client_id))
        return estimate_network(receipt, round_id=int(round_id)) if receipt is not None else None

    def bind_evaluation_clients(self, clients, round_id):
        self._bind_properties(clients, round_id)

    def client_context(self, client_id: str) -> dict[str, Any]:
        props = self.clients.get(str(client_id))
        if props is None:
            raise ValueError(f"Online placement has no bound worker {client_id!r}")
        return {key: props[key] for key in (
            "num_batches", "batch_size", "device_type", "accelerator", "precision",
            "transport_intercept_ms", "transport_up_ms_per_byte", "transport_down_ms_per_byte",
            "transport_probe_round", "transport_changed_round",
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
    """Validate current-device anchors and bootstrap communication measurements once."""

    def __init__(self, *, initial_model_hash, server_receipt, parameters=None, policy=None):
        super().__init__(policy=policy)
        self.initial_model_hash = initial_model_hash
        # ``initial_model_hash`` identifies the frozen experiment receipt.
        # The handshake hash can move with the global model so a client that
        # joins after bootstrap can be calibrated against the model it will
        # actually execute.
        self.calibration_model_hash = initial_model_hash
        self.server_receipt = server_receipt
        self.parameters = parameters
        self.initialized = False

    def bind_clients(self, clients, round_id):
        if self.policy is None or self.server_receipt is None:
            raise ValueError("Calibration must be ready before client binding")
        policy = self.policy
        catalog = {candidate.boundary: candidate for candidate in policy._catalog()}
        if self.initialized or policy._state_restored:
            clients = tuple(clients)
            missing = tuple(
                client for client in clients
                if "online_calibration_receipt" not in self.clients.get(str(client.cid), {})
            )
            if missing:
                self._bind_late_clients(missing, round_id, catalog)
            super().bind_clients(clients, round_id)
            return
        if (any(key[1] for key in policy._round_assignments) or
                policy._observed_actions or
                policy._last_boundary or policy.cost_tables):
            raise ValueError("Bootstrap only a fresh, unplanned cost table")
        clients = tuple(clients)
        if not clients:
            raise ValueError("Calibration requires bound physical clients")
        provider = policy.candidate_provider
        config = {"splitfleet_calibration_boundaries": json.dumps(calibration_boundaries(catalog.values()))}
        if getattr(provider, "dynamic_batch", None) is not None:
            config[AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY] = json.dumps(provider.dynamic_batch)
        config[AUTOSPLIT_MODE_CONFIG_KEY] = getattr(provider, "mode", "generated_eager")
        if getattr(provider, "trace_batch_mode", None):
            config[AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY] = provider.trace_batch_mode
        if self.parameters is not None:
            config[PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY] = self.parameters
            config[PLACEMENT_CALIBRATION_MODEL_HASH_CONFIG_KEY] = self.calibration_model_hash
        # Evaluation may have bound these workers without calibration. Request
        # receipts only for the first training cohort, using its global model.
        binding_started = time.perf_counter()
        transfer = self._bind_properties(clients, round_id, config=config, refresh=True)
        binding_elapsed = time.perf_counter() - binding_started
        validation_started = time.perf_counter()
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
            expected_schema = "splitfleet.training-calibration"
            if receipt.get("schema") != expected_schema or receipt.get("source") != expected_source:
                raise ValueError("Calibration must come from the current deployment")
            if cid is not None and str(receipt.get("device", "")).split(":", 1)[0] != self.clients[cid]["device_type"]:
                raise ValueError("Calibration device differs from the worker execution profile")
            if not math.isfinite(float(receipt.get("elapsed_sec", -1))) or float(receipt["elapsed_sec"]) < 0:
                raise ValueError("Calibration duration must be finite and nonnegative")
            if (receipt.get("model_hash_before") != self.calibration_model_hash or
                receipt.get("model_hash_after") != self.calibration_model_hash or
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
        logical = []
        transport_samples, transport_contexts, transport_warmup = [], [], []
        exchange_samples, exchange_contexts = [], []
        # Application-level echo is a bootstrap prior measured here, not
        # a one-way timing or an assertion that Flower and split RPCs
        # have identical serialization. Actual training RTT replaces it.
        template = next(iter(catalog.values()))
        validation_elapsed = time.perf_counter() - validation_started
        probe_started = time.perf_counter()

        def probe_client(client):
            # A client has one outstanding request at a time, as on a physical
            # connection. Independent clients use the admitted worker pool.
            samples, contexts, warmup = [], [], []
            exchanges, exchange_features = [], []
            cid = str(client.cid)
            for repeat in range(2):
                elapsed_ms = measure_transport_echo(client, size=65536, round_id=round_id)
                warmup.append({"logical_client_id": self.clients[cid]["logical_client_id"],
                    "payload_bytes": 65536, "repeat": repeat, "roundtrip_ms": elapsed_ms, "echo_verified": True})
            for size in (65536, 2097152):
                batch_size = int(self.client_context(cid)["batch_size"])
                candidate = replace(template, boundary_forward_bytes=size,
                                    boundary_gradient_bytes=None, boundary_tensor_count=1,
                                    metadata={**template.metadata, "payload_batch_size": batch_size,
                                              "boundary_forward_bytes_by_batch_size": {batch_size: size}})
                context = None
                for repeat in range(2):
                    elapsed_ms = measure_transport_echo(client, size=size, round_id=round_id)
                    samples.append({"logical_client_id": self.clients[cid]["logical_client_id"],
                        "payload_bytes": size, "repeat": repeat, "roundtrip_ms": elapsed_ms,
                        "echo_verified": True})
                    contexts.append((cid, context, elapsed_ms))
            for request_size, reply_size in ((65536, 65536), (2097152, 65536), (65536, 2097152)):
                context = None
                for repeat in range(2):
                    elapsed_ms = measure_transport_echo(client, size=request_size, reply_size=reply_size, round_id=round_id)
                    exchanges.append({"logical_client_id": self.clients[cid]["logical_client_id"],
                        "request_bytes": request_size, "reply_bytes": reply_size, "repeat": repeat,
                        "roundtrip_ms": elapsed_ms, "reply_verified": True})
                    exchange_features.append((cid, context, elapsed_ms))
            return samples, contexts, warmup, exchanges, exchange_features

        # Finish and validate every RPC before initializing any cost table. Physical
        # deployments keep their sequential default when no pool is declared.
        for samples, contexts, warmup, exchanges, exchange_features in self._map_clients(probe_client, clients):
            transport_samples.extend(samples)
            transport_contexts.extend(contexts)
            transport_warmup.extend(warmup)
            exchange_samples.extend(exchanges)
            exchange_contexts.extend(exchange_features)
        probe_elapsed = time.perf_counter() - probe_started
        update_started = time.perf_counter()
        for cid, candidate, row in measurements:
            if cid is not None:
                logical.append({"logical_client_id": self.clients[cid]["logical_client_id"], **row})
        self.receipt = {"source": "current_deployment_restored_training_cost_calibration",
            "parameter_handshake": transfer,
            "timing": {"client_binding_sec": binding_elapsed, "receipt_validation_sec": validation_elapsed,
                       "transport_probe_sec": probe_elapsed,
                       "cost_initialization_sec": time.perf_counter() - update_started,
                       "client_concurrency_limit": self._client_workers(len(clients))},
            "initial_model_hash": self.initial_model_hash, "client_samples": logical,
            "server_calibration": self.server_receipt,
            "client_calibration_elapsed_sec": {self.clients[cid]["logical_client_id"]: receipt["elapsed_sec"] for cid, receipt in receipts if cid is not None},
            "network_initialized": True, "prediction_override": False,
            "online_feedback_updates_retained": True,
            "exploration_enabled": False}
        self.receipt["temporary_optimizer_updates"] = True
        self.receipt["transport_learning_mode"] = (
            "affine_bootstrap_with_actual_round_feedback")
        self.receipt["network_context_mode"] = (PROBE_MODE if policy.uses_active_network_probes
                                                 else "actual_fit_feedback")
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
        self.refresh_transport(clients, round_id)

    def _calibration_config(self, catalog):
        """Build the client calibration request for the current model."""
        provider = self.policy.candidate_provider
        config = {
            "splitfleet_calibration_boundaries": json.dumps(
                calibration_boundaries(catalog.values())
            ),
            PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY: self.parameters,
            PLACEMENT_CALIBRATION_MODEL_HASH_CONFIG_KEY: self.calibration_model_hash,
            AUTOSPLIT_MODE_CONFIG_KEY: getattr(provider, "mode", "generated_eager"),
        }
        if getattr(provider, "dynamic_batch", None) is not None:
            config[AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY] = json.dumps(provider.dynamic_batch)
        if getattr(provider, "trace_batch_mode", None):
            config[AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY] = provider.trace_batch_mode
        return config

    def _validate_late_client_receipt(self, cid, receipt, catalog):
        """Validate one late client's prefix calibration against the current model."""
        expected = set(calibration_boundaries(catalog.values()))
        if receipt.get("schema") != "splitfleet.training-calibration":
            raise ValueError("Calibration must come from the current deployment")
        if receipt.get("source") != "client_private_sample_current_deployment":
            raise ValueError("Calibration must come from the current deployment")
        if str(receipt.get("device", "")).split(":", 1)[0] != self.clients[cid]["device_type"]:
            raise ValueError("Calibration device differs from the worker execution profile")
        if (
            receipt.get("model_hash_before") != self.calibration_model_hash
            or receipt.get("model_hash_after") != self.calibration_model_hash
            or receipt.get("state_and_torch_rng_preserved") is not True
            or receipt.get("persistent_optimizer_steps") != 0
        ):
            raise ValueError("Calibration state does not match the current global model")
        if not math.isfinite(float(receipt.get("elapsed_sec", -1))) or float(receipt["elapsed_sec"]) < 0:
            raise ValueError("Calibration duration must be finite and nonnegative")
        records = receipt.get("records")
        if not isinstance(records, list) or len(records) != len(expected) or {
            row.get("boundary") for row in records
        } != expected:
            raise ValueError("Calibration anchors differ from the canonical catalog")
        expected_steps = {
            cut: 0
            if catalog[cut].metadata.get("optimizer_prefix_parameter_bytes") == 0
            else 2
            for cut in expected
        }
        for row in records:
            boundary = row.get("boundary")
            candidate = catalog[boundary]
            if (
                row.get("graph_signature") != candidate.graph_signature
                or row.get("feature_abi_id") != candidate.feature_abi_id
                or row.get("measured_batches") != 1
                or row.get("optimizer_steps") != expected_steps[boundary]
                or row.get("optimizer_stage") != "prefix"
            ):
                raise ValueError("Calibration sample scope mismatch")
            for name in ("client_forward_ms", "client_backward_ms"):
                value = row.get(name)
                if value is None or not math.isfinite(float(value)) or float(value) < 0:
                    raise ValueError("Calibration costs must be finite and nonnegative")
        return records

    def _bind_late_clients(self, clients, round_id, catalog):
        """Calibrate clients that were not part of the initial cohort.

        A partially participating run can keep a client unbound for several
        rounds.  Such a client must receive current-model anchors and its own
        transport bootstrap before Greedy scores it; borrowing another client's
        table would silently mix hardware and link measurements.
        """
        if self.parameters is None:
            raise ValueError(
                "Greedy needs the current global parameters to calibrate a new client"
            )
        clients = tuple(clients)
        previous = {str(client.cid): self.clients.get(str(client.cid)) for client in clients}
        config = self._calibration_config(catalog)
        started = time.perf_counter()
        completed = False
        try:
            transfer = self._bind_properties(clients, round_id, config=config, refresh=True) or {}
            logical_samples = []
            elapsed = {}
            for client in clients:
                cid = str(client.cid)
                raw = self.clients[cid].get("online_calibration_receipt")
                if raw is None:
                    raise ValueError(f"Client {cid!r} did not return calibration anchors")
                receipt = json.loads(raw) if isinstance(raw, str) else raw
                records = self._validate_late_client_receipt(cid, receipt, catalog)
                logical = self.clients[cid]["logical_client_id"]
                logical_samples.extend(
                    {"logical_client_id": logical, **row} for row in records
                )
                elapsed[logical] = receipt["elapsed_sec"]

            def probe_client(client):
                cid = str(client.cid)
                logical = self.clients[cid]["logical_client_id"]
                transport, warmup, exchange = [], [], []
                for repeat in range(2):
                    warmup.append({
                        "logical_client_id": logical,
                        "payload_bytes": 65536,
                        "repeat": repeat,
                        "roundtrip_ms": measure_transport_echo(
                            client, size=65536, round_id=round_id
                        ),
                        "echo_verified": True,
                    })
                for size in (65536, 2097152):
                    for repeat in range(2):
                        transport.append({
                            "logical_client_id": logical,
                            "payload_bytes": size,
                            "repeat": repeat,
                            "roundtrip_ms": measure_transport_echo(
                                client, size=size, round_id=round_id
                            ),
                            "echo_verified": True,
                        })
                for request_size, reply_size in (
                    (65536, 65536), (2097152, 65536), (65536, 2097152)
                ):
                    for repeat in range(2):
                        exchange.append({
                            "logical_client_id": logical,
                            "request_bytes": request_size,
                            "reply_bytes": reply_size,
                            "repeat": repeat,
                            "roundtrip_ms": measure_transport_echo(
                                client,
                                size=request_size,
                                reply_size=reply_size,
                                round_id=round_id,
                            ),
                            "reply_verified": True,
                        })
                return transport, warmup, exchange

            transport, warmup, exchange = [], [], []
            for rows, warmup_rows, exchange_rows in self._map_clients(probe_client, clients):
                transport.extend(rows)
                warmup.extend(warmup_rows)
                exchange.extend(exchange_rows)

            receipt = self.receipt
            receipt.setdefault("client_samples", []).extend(logical_samples)
            receipt.setdefault("client_calibration_elapsed_sec", {}).update(elapsed)
            receipt.setdefault("transport_bootstrap", {}).setdefault("samples", []).extend(transport)
            receipt.setdefault("transport_bootstrap", {}).setdefault("warmup_samples", []).extend(warmup)
            receipt.setdefault("state_exchange_bootstrap", {}).setdefault("samples", []).extend(exchange)
            handshake = receipt.setdefault("parameter_handshake", {})
            for key in (
                "requests", "hash_matched_clients", "fallback_clients",
                "parameter_payload_bytes", "avoided_parameter_payload_bytes",
            ):
                handshake[key] = int(handshake.get(key, 0)) + int(transfer.get(key, 0))
            receipt["latest_calibration_model_hash"] = self.calibration_model_hash
            receipt.setdefault("timing", {})["late_client_calibration_sec"] = (
                time.perf_counter() - started
            )
            completed = True
        except BaseException:
            # Do not leave a client with properties from a failed calibration;
            # it must be retried atomically when it next becomes eligible.
            for cid, value in previous.items():
                if value is None:
                    self.clients.pop(cid, None)
                else:
                    self.clients[cid] = value
            raise
        finally:
            # The payload is a one-round handshake value; prepare_calibration
            # refreshes it before the next round if another client joins. Keep
            # it on failure so a retry in the same round remains possible.
            if completed:
                self.parameters = None
