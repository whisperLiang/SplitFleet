"""TorchLens 2.34.1 adapter using public APIs only."""

from __future__ import annotations

import importlib.metadata
import uuid
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from torchlens.split import ReplayBoundary

from splitfleet.backends import BACKEND_ADAPTERS
from splitfleet.tasks import ModelInputs
from splitfleet.runtime import PrefixContextStore
from splitfleet.runtime.torch_suffix_training import train_torch_suffix
from splitfleet.split_engine.base import PrefixContextToken, SplitRuntimeHandle, SuffixResult
from splitfleet.split_engine.contracts import GraphContract, contract_hash
from splitfleet.transport import (
    BoundaryEnvelope,
    GradientEnvelope,
    encode_bundle,
)
from splitfleet.transport.split_wire import replay_metadata


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



class TorchLensSplitEngine:
    engine_name = "torchlens"

    def __init__(self, *, context_store: PrefixContextStore | None = None) -> None:
        self.context_store = context_store or PrefixContextStore()

    def prepare(self, model: Any, sample_inputs: Any, request: Any, *, sample_kwargs: dict[str, Any] | None = None) -> SplitRuntimeHandle:
        from splitfleet.autosplit.torchlens_runtime import prepare_split_runtime, normalize_model_call

        call = ModelInputs.from_value(sample_inputs)
        kwargs = dict(call.kwargs) if sample_kwargs is None else sample_kwargs
        runtime = prepare_split_runtime(model, call.args, request, input_kwargs=kwargs)
        args, kwargs, _ = normalize_model_call(model, call.args, kwargs)
        backend = runtime.adapter.name
        adapter = BACKEND_ADAPTERS.create(backend)
        if backend == "jax" and not hasattr(model, "params"):
            if not args:
                raise ValueError("Functional JAX models require parameters as the first argument.")
            adapter.bind_external_params(args[0])
        return SplitRuntimeHandle(
            runtime=runtime,
            backend=backend,
            engine=self.engine_name,
            metadata={"model": model, "sample_inputs": args, "sample_kwargs": kwargs, "request": runtime.request},
        )

    def export_contract(self, handle: SplitRuntimeHandle) -> GraphContract:
        runtime = handle.runtime
        if runtime.graph_ir is None:
            raise RuntimeError("TorchLens runtime has no captured graph IR.")
        model = handle.metadata["model"]
        state_adapter = BACKEND_ADAPTERS.create(handle.backend)
        if handle.backend == "jax" and not hasattr(model, "params"):
            state_adapter.bind_external_params(handle.metadata["sample_inputs"][0])
        state_schema_hash = state_adapter.state_manifest(model).schema_hash
        input_schema = _tensor_schema(
            {"args": handle.metadata["sample_inputs"], "kwargs": handle.metadata["sample_kwargs"]},
            batch_axes=runtime.batch_spec.axes,
        )
        profile = runtime.model_profile
        return GraphContract(
            torchlens_version=importlib.metadata.version("torchlens"),
            backend=handle.backend,
            model_profile_id=profile.id if profile is not None else "",
            model_revision=handle.metadata.get("model_revision", f"{model.__class__.__module__}.{model.__class__.__qualname__}"),
            model_state_schema_hash=state_schema_hash,
            input_schema_hash=contract_hash(input_schema),
            canonical_graph_hash=runtime.graph_ir.graph_hash,
            split_id=runtime.split_id,
            boundary_schema_hash=contract_hash(_value(runtime.boundary_schema)),
            split_request_hash=contract_hash(_value(runtime.request)),
            capability_hash=contract_hash(_value(runtime.capability_report)),
            metadata={"source": "torchlens_2.34.1"},
        )

    def run_prefix(
        self,
        handle: SplitRuntimeHandle,
        inputs: Any,
        *,
        training: bool,
        input_kwargs: dict[str, Any] | None = None,
    ) -> tuple[BoundaryEnvelope, PrefixContextToken | None]:
        from splitfleet.autosplit.torchlens_runtime import normalize_model_call

        runtime = handle.runtime
        call = ModelInputs.from_value(inputs)
        kwargs = (
            dict(call.kwargs) if isinstance(inputs, ModelInputs)
            else dict(handle.metadata.get("sample_kwargs", {}))
        ) if input_kwargs is None else input_kwargs
        args, kwargs, _ = normalize_model_call(handle.metadata.get("model"), call.args, kwargs)
        native = runtime.run_training_prefix(*args, input_kwargs=kwargs) if training else runtime.run_prefix(*args, input_kwargs=kwargs)
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
        batch_size = int(native.metadata["runtime_batch_size"])
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
            metadata={**replay_metadata(native.metadata), "contract_digest": contract.digest},
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
                **replay_metadata(boundary.metadata),
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
        *,
        loss_fn: Any = None,
    ) -> SuffixResult:
        native = self._native_boundary(handle, boundary)
        if targets is None:
            outputs = handle.runtime.run_suffix(native)
            return SuffixResult(
                outputs=encode_bundle(outputs, backend=handle.backend),
                num_examples=boundary.batch_size,
            )
        if loss_fn is None:
            raise ValueError("Split training requires an explicit loss_fn.")
        if handle.backend == "torch":
            loss, gradients = train_torch_suffix(
                handle.runtime, native, targets, loss_fn=loss_fn, optimizer=optimizer
            )
        else:
            loss, gradients = handle.runtime.train_suffix(
                native, targets, loss_fn=loss_fn, optimizer=optimizer
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
        if gradients.backend != handle.backend:
            raise ValueError("Gradient backend mismatch")
        if gradients.split_id != handle.runtime.split_id:
            raise ValueError("Gradient split id mismatch")
        for field_name in ("plan_id", "model_version"):
            if field_name in handle.metadata and getattr(gradients, field_name) != handle.metadata[field_name]:
                raise ValueError(f"Gradient {field_name} mismatch")
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
        model = handle.runtime.model
        for tensor in (*model.parameters(), *model.buffers()):
            return tensor.device
        return None
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


def _tensor_schema(value: Any, *, batch_axes=None, path: str = "") -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = [int(dim) for dim in value.shape]
        axis = (batch_axes or {}).get(path)
        if axis is not None:
            shape[axis] = "B"
        return {"shape": shape, "dtype": str(value.dtype).removeprefix("torch.")}
    if isinstance(value, dict):
        return {str(key): _tensor_schema(item, batch_axes=batch_axes,
                    path=path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_schema(item, batch_axes=batch_axes, path=f"{path}/{index}")
                for index, item in enumerate(value)]
    return type(value).__name__


def graph_contract_for_runtime_handle(handle: Any) -> GraphContract:
    """Export a contract from SplitFleet's prepared TorchLens runtime handle."""
    engine_handle = SplitRuntimeHandle(
        runtime=handle.runtime,
        backend=handle.runtime.adapter.name,
        metadata={
            "model": handle.model,
            "sample_inputs": handle.plan.metadata.get("_example_inputs"),
            "sample_kwargs": handle.plan.metadata.get("_example_kwargs", {}),
            "request": handle.runtime.request,
        },
    )
    return TorchLensSplitEngine().export_contract(engine_handle)
