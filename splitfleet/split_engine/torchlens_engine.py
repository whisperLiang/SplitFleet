"""TorchLens 2.31 adapter using public APIs only."""

from __future__ import annotations

import importlib.metadata
import uuid
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import torch
from torchlens.split import ReplayBoundary, prepare

from splitfleet.backends import BACKEND_ADAPTERS
from splitfleet.runtime import PrefixContextStore
from splitfleet.runtime.torch_suffix_training import train_torch_suffix
from splitfleet.split_engine.base import PrefixContextToken, SplitRuntimeHandle, SuffixResult
from splitfleet.split_engine.contracts import GraphContract, contract_hash
from splitfleet.transport import (
    BoundaryEnvelope,
    GradientEnvelope,
    encode_bundle,
)


def _value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        # dataclasses.asdict() deep-copies every non-container leaf. JAX graph
        # records can contain jaxlib Traceback objects, which intentionally
        # cannot be pickled/deep-copied. Walk fields directly for the
        # JSON-compatible contract projection.
        return {
            field.name: _value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(v) for v in value]
    if hasattr(value, "to_dict"):
        return _value(value.to_dict())
    if hasattr(value, "__dict__"):
        public = {k: _value(v) for k, v in vars(value).items() if not k.startswith("_")}
        return public if public else str(value)
    return str(value)


def _attr(runtime: Any, *names: str, default: Any = "") -> Any:
    for owner in (runtime, getattr(runtime, "plan", None), getattr(runtime, "graph", None), getattr(runtime, "split_graph", None)):
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if value not in (None, ""):
                return value
    return default


class TorchLensSplitEngine:
    engine_name = "torchlens"

    def __init__(self, *, context_store: PrefixContextStore | None = None) -> None:
        self.context_store = context_store or PrefixContextStore()

    def prepare(self, model: Any, sample_inputs: Any, request: Any, *, sample_kwargs: dict[str, Any] | None = None) -> SplitRuntimeHandle:
        runtime = prepare(model, sample_inputs, request, input_kwargs=sample_kwargs)
        backend = str(getattr(request, "backend", None) or _attr(runtime, "backend", "backend_name"))
        return SplitRuntimeHandle(
            runtime=runtime,
            backend=backend,
            engine=self.engine_name,
            metadata={"model": model, "sample_inputs": sample_inputs, "request": request},
        )

    def export_contract(self, handle: SplitRuntimeHandle) -> GraphContract:
        runtime = handle.runtime
        exported = getattr(runtime, "export_contract", None)
        if callable(exported):
            raw = _value(exported())
            if isinstance(raw, dict):
                known = {name: raw[name] for name in GraphContract.__dataclass_fields__ if name in raw}
                known.setdefault("backend", handle.backend)
                return GraphContract(**known)
        boundary = _value(_attr(runtime, "boundary_schema", default={}))
        request = _value(_attr(runtime, "request", "split_request", default={}))
        capabilities = _value(_attr(runtime, "capabilities", "capability_report", default={}))
        try:
            version = importlib.metadata.version("torchlens")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        model = handle.metadata.get("model")
        try:
            state_schema_hash = BACKEND_ADAPTERS.create(handle.backend).state_manifest(model).schema_hash
        except (KeyError, TypeError, AttributeError):
            state_schema_hash = ""
        input_schema = _tensor_schema(handle.metadata.get("sample_inputs"))
        graph_hash = str(
            _attr(runtime, "canonical_graph_hash", default="")
            or getattr(getattr(runtime, "graph_ir", None), "graph_hash", "")
            or ""
        )
        if not graph_hash:
            graph = _value(
                _attr(runtime, "graph_ir", "split_graph", "graph", "ir", default={})
            )
            graph_hash = contract_hash(graph)
        profile = getattr(runtime, "model_profile", None)
        model_profile_id = str(getattr(profile, "id", "") or "")
        model_revision = str(
            handle.metadata.get("model_revision")
            or getattr(model, "model_revision", "")
            or (f"{model.__class__.__module__}.{model.__class__.__qualname__}" if model is not None else "")
        )
        return GraphContract(
            torchlens_version=version,
            backend=handle.backend,
            model_profile_id=model_profile_id or str(_attr(runtime, "model_profile_id", "profile_id")),
            model_revision=model_revision,
            model_state_schema_hash=state_schema_hash,
            input_schema_hash=contract_hash(input_schema),
            canonical_graph_hash=graph_hash,
            split_id=str(_attr(runtime, "split_id")),
            boundary_schema_hash=str(_attr(runtime, "boundary_schema_hash", default="")) or contract_hash(boundary),
            split_request_hash=contract_hash(request),
            capability_hash=contract_hash(capabilities),
            metadata={"source": "derived"},
        )

    def run_prefix(
        self,
        handle: SplitRuntimeHandle,
        inputs: Any,
        *,
        training: bool,
    ) -> tuple[BoundaryEnvelope, PrefixContextToken | None]:
        runtime = handle.runtime
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        native = runtime.run_training_prefix(*args) if training else runtime.run_prefix(*args)
        contract = self.export_contract(handle)
        round_id = int(handle.metadata.get("round_id", 0))
        client_id = str(handle.metadata.get("client_id", ""))
        step_id = str(handle.metadata.get("step_id") or uuid.uuid4().hex)
        token = None
        if training:
            self.context_store.put(round_id, client_id, step_id, native)
            token = PrefixContextToken(round_id, client_id, step_id)
        adapter = BACKEND_ADAPTERS.create(handle.backend)
        tensors = tuple(
            adapter.encode_tensor(name, tensor) for name, tensor in native.tensors.items()
        )
        batch_size = next(
            (
                int(tensor.shape[0])
                for tensor in native.tensors.values()
                if tuple(getattr(tensor, "shape", ()) or ())
            ),
            0,
        )
        envelope = BoundaryEnvelope(
            tensors=tensors,
            engine=self.engine_name,
            backend=handle.backend,
            round_id=round_id,
            client_id=client_id,
            step_id=step_id,
            plan_id=str(handle.metadata.get("plan_id", "")),
            split_id=str(runtime.split_id),
            canonical_graph_hash=contract.canonical_graph_hash,
            boundary_schema_hash=contract.boundary_schema_hash,
            model_version=int(handle.metadata.get("model_version", 0)),
            batch_size=batch_size,
            metadata={"contract_digest": contract.digest},
        )
        return envelope, token

    def _native_boundary(self, handle: SplitRuntimeHandle, boundary: BoundaryEnvelope) -> ReplayBoundary:
        runtime = handle.runtime
        if boundary.backend != handle.backend:
            raise ValueError(f"Boundary backend mismatch: {boundary.backend!r} != {handle.backend!r}")
        contract = self.export_contract(handle)
        if boundary.split_id != runtime.split_id:
            raise ValueError("Boundary split id mismatch")
        if boundary.canonical_graph_hash != contract.canonical_graph_hash:
            raise ValueError("Boundary canonical graph hash mismatch")
        if boundary.boundary_schema_hash != contract.boundary_schema_hash:
            raise ValueError("Boundary schema hash mismatch")
        adapter = BACKEND_ADAPTERS.create(handle.backend)
        device = _runtime_device(handle)
        tensors = {
            item.tensor_id: adapter.decode_tensor(item, device)
            for item in boundary.tensors
        }
        return ReplayBoundary(
            backend=handle.backend,
            tensors=tensors,
            spec=runtime.boundary_spec,
            metadata={
                "split_id": runtime.split_id,
                "graph_shape_hash": contract.canonical_graph_hash,
                "batch_size": boundary.batch_size,
            },
        )

    def run_suffix(
        self,
        handle: SplitRuntimeHandle,
        boundary: BoundaryEnvelope,
        targets: Any = None,
        optimizer: Any = None,
    ) -> SuffixResult:
        native = self._native_boundary(handle, boundary)
        if targets is None:
            outputs = handle.runtime.run_suffix(native)
            return SuffixResult(
                outputs=encode_bundle(outputs, backend=handle.backend),
                num_examples=boundary.batch_size,
            )
        if handle.backend == "torch":
            loss, gradients = train_torch_suffix(
                handle.runtime, native, targets, optimizer=optimizer
            )
        else:
            loss, gradients = handle.runtime.train_suffix(
                native, targets, optimizer=optimizer
            )
        adapter = BACKEND_ADAPTERS.create(handle.backend)
        gradient_envelope = GradientEnvelope(
            tensors=tuple(
                adapter.encode_tensor(name, value) for name, value in gradients.items()
            ),
            backend=handle.backend,
            round_id=boundary.round_id,
            client_id=boundary.client_id,
            step_id=boundary.step_id,
            plan_id=boundary.plan_id,
            split_id=boundary.split_id,
            model_version=boundary.model_version,
        )
        return SuffixResult(
            outputs=None,
            gradients=gradient_envelope,
            loss=BACKEND_ADAPTERS.create(handle.backend).scalar_value(loss),
            num_examples=boundary.batch_size,
        )

    def backward_prefix(
        self,
        handle: SplitRuntimeHandle,
        context_token: PrefixContextToken,
        gradients: GradientEnvelope,
        optimizer: Any = None,
    ) -> Any:
        if (gradients.round_id, gradients.client_id, gradients.step_id) != (
            context_token.round_id, context_token.client_id, context_token.step_id
        ):
            raise ValueError("Gradient context identity mismatch")
        native = self.context_store.pop(
            context_token.round_id, context_token.client_id, context_token.step_id
        )
        adapter = BACKEND_ADAPTERS.create(handle.backend)
        device = _runtime_device(handle)
        decoded = {
            item.tensor_id: adapter.decode_tensor(item, device)
            for item in gradients.tensors
        }
        return handle.runtime.backward_prefix(native, decoded, optimizer=optimizer)


def _iter_tensor_values(value: Any):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensor_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensor_values(item)


def _runtime_device(handle: SplitRuntimeHandle) -> Any:
    """Infer the native device used to reconstruct wire tensors."""
    if handle.backend == "torch":
        return next(iter(handle.runtime.model.parameters()), torch.empty(0)).device
    for tensor in _iter_tensor_values(handle.metadata.get("sample_inputs")):
        device = getattr(tensor, "device", None)
        if callable(device):
            device = device()
        if device is not None:
            return device
        place = getattr(tensor, "place", None)
        if place is not None:
            return place
    return None


def _tensor_schema(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        shape = ["B", *[int(dim) for dim in value.shape[1:]]] if value.ndim else []
        return {"shape": shape, "dtype": str(value.dtype).removeprefix("torch.")}
    if isinstance(value, dict):
        return {str(key): _tensor_schema(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_schema(item) for item in value]
    return type(value).__name__


def graph_contract_for_runtime_handle(handle: Any) -> GraphContract:
    """Export a contract from SplitFleet's prepared TorchLens runtime handle."""
    engine_handle = SplitRuntimeHandle(
        runtime=handle.runtime,
        backend=str(getattr(getattr(handle.runtime, "adapter", None), "name", "")),
        metadata={
            "model": handle.model,
            "sample_inputs": handle.plan.metadata.get("_example_inputs"),
            "request": getattr(handle.runtime, "request", None),
        },
    )
    if not engine_handle.backend:
        raise RuntimeError("TorchLens runtime handle does not declare a backend")
    return TorchLensSplitEngine().export_contract(engine_handle)
