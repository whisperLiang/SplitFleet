"""Backend facade for TorchLens native autosplit runtimes."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import numpy as np
from torchlens.split import after

from splitfleet.autosplit.boundary import (
    BoundaryPayload,
    from_torchlens_boundary,
    to_torchlens_boundary,
)
from splitfleet.autosplit.torchlens_candidate import (
    SplitCandidate,
    build_candidate_descriptor,
    candidate_from_plan,
)
from splitfleet.autosplit.torchlens_contract import build_runtime_contract
from splitfleet.autosplit.torchlens_runtime import (
    TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION,
    first_tensor_batch_size,
    infer_trace_batch_mode,
    make_split_spec,
    normalize_example_inputs,
    prepare_split_runtime,
    repartition_split_runtime,
    torchlens_runtime_version,
    trace_signature,
)
from splitfleet.autosplit.types import SplitRuntimePlan
from splitfleet.backends import BACKEND_ADAPTERS
from splitfleet.backends.utils import inference_context
from splitfleet.runtime.torch_suffix_training import train_torch_suffix


@dataclass
class TorchLensRuntimeHandle:
    model: torch.nn.Module
    runtime: Any
    plan: SplitRuntimePlan
    backend: "TorchLensSplitBackend"

    @property
    def feature_abi_id(self) -> str:
        return self.plan.feature_abi_id

    @feature_abi_id.setter
    def feature_abi_id(self, value: str) -> None:
        self.plan.feature_abi_id = value


SplitRuntimeHandle = TorchLensRuntimeHandle


def _make_plan_id(graph_signature: str, split_id: str, boundary: str, mode: str) -> str:
    digest = hashlib.sha1(
        "|".join([graph_signature, split_id, boundary, mode]).encode("utf-8")
    ).hexdigest()
    return f"torchlens_{digest[:12]}"


def _iter_tensors(value: Any):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _runtime_device(runtime: Any) -> torch.device | None:
    model = getattr(runtime, "model", None)
    if isinstance(model, torch.nn.Module):
        for tensor in model.parameters():
            return tensor.device
        for tensor in model.buffers():
            return tensor.device
    return None


def _move_to_device(value: Any, device: torch.device | str | None) -> Any:
    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _boundary_batch_matches(payload: BoundaryPayload, batch_size: int) -> bool:
    for tensor in payload.tensors.values():
        shape = tuple(getattr(tensor, "shape", ()) or ())
        if shape and int(shape[0]) == batch_size:
            return True
    return False


def _normalise_boundary_batch_metadata(
    payload: BoundaryPayload,
    inputs: tuple[Any, ...],
) -> BoundaryPayload:
    batch_size = first_tensor_batch_size(inputs)
    if batch_size is None or not _boundary_batch_matches(payload, batch_size):
        return payload
    metadata = dict(payload.metadata)
    metadata["batch_size"] = int(batch_size)
    payload.metadata = metadata
    payload.batch_size = int(batch_size)
    return payload


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    return list(_iter_tensors(value))


def _compare_outputs(expected: Any, actual: Any, backend: str = "torch") -> tuple[bool, float, float]:
    max_abs = 0.0
    max_rel = 0.0
    success = True

    def visit(left: Any, right: Any) -> None:
        nonlocal max_abs, max_rel, success
        if hasattr(left, "shape") and hasattr(left, "dtype"):
            if not (hasattr(right, "shape") and hasattr(right, "dtype")) or tuple(left.shape) != tuple(right.shape):
                success = False
                max_abs = float("inf")
                max_rel = float("inf")
                return
            adapter = BACKEND_ADAPTERS.create(backend)
            left_array = np.asarray(adapter._to_numpy(left) if hasattr(adapter, "_to_numpy") else left.detach().cpu().numpy())
            right_array = np.asarray(adapter._to_numpy(right) if hasattr(adapter, "_to_numpy") else right.detach().cpu().numpy())
            diff = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
            abs_value = float(diff.max()) if diff.size else 0.0
            denom = np.maximum(np.abs(right_array.astype(np.float64)), 1e-12)
            rel_value = float((diff / denom).max()) if diff.size else 0.0
            max_abs = max(max_abs, abs_value)
            max_rel = max(max_rel, rel_value)
            if not np.allclose(left_array, right_array, atol=1e-5, rtol=1e-4):
                success = False
            return
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or set(left) != set(right):
                success = False
                return
            for key in left:
                visit(left[key], right[key])
            return
        if isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                success = False
                return
            for left_item, right_item in zip(left, right):
                visit(left_item, right_item)
            return
        if left != right:
            success = False

    visit(expected, actual)
    return success, max_abs, max_rel


class TorchLensSplitBackend:
    """SplitFleet facade over TorchLens native SplitRuntime."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        model_name: str | None = None,
        model_family: str | None = None,
    ) -> Any:
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.runtime: Any | None = None
        self.split_spec: Any | None = None
        self.current_candidate: SplitCandidate | None = None
        self.candidates: list[SplitCandidate] = []
        self.trace_sample_input: Any = None
        self.model_name = model_name
        self.model_family = model_family
        self.trace_batch_size: int | None = None
        self.validation: dict[str, Any] | None = None
        self.framework_backend = "torch"

    def trace(
        self,
        model: torch.nn.Module,
        sample_inputs: Any,
        *,
        split_spec: Any = None,
        boundary: str = "50%",
        mode: str = "generated_eager",
        trainable: bool = True,
        dynamic_batch: tuple[int, int] | None = None,
        trace_batch_mode: str | None = None,
        model_name: str | None = None,
        model_family: str | None = None,
        **_: Any,
    ) -> "TorchLensSplitBackend":
        self.model = model
        self.model_name = model_name or self.model_name or model.__class__.__name__
        self.model_family = model_family or self.model_family or self.model_name
        self.trace_sample_input = normalize_example_inputs(sample_inputs)
        from splitfleet.backends.utils import detect_torchlens_backend
        self.framework_backend = detect_torchlens_backend(model, self.trace_sample_input)
        self.trace_batch_size = first_tensor_batch_size(self.trace_sample_input)
        resolved_trace_batch_mode = trace_batch_mode or infer_trace_batch_mode(self.trace_sample_input)
        if dynamic_batch is None:
            dynamic_batch = (2, 64) if resolved_trace_batch_mode == "batch_gt1" else (1, 64)
        probe_boundary = "50%" if str(boundary) == "auto" else str(boundary)
        self.split_spec = split_spec or make_split_spec(
            probe_boundary,
            dynamic_batch=dynamic_batch,
            trainable=trainable,
            trace_batch_mode=resolved_trace_batch_mode,
            mode=mode,
            backend=self.framework_backend,
        )
        self.runtime = prepare_split_runtime(
            model,
            self.trace_sample_input,
            self.split_spec,
            mode=mode,
        )
        self.current_candidate = candidate_from_plan(
            self.runtime,
            self.split_spec,
            self.runtime.plan,
            graph_signature=trace_signature(self.runtime),
        )
        self.candidates = [self.current_candidate]
        return self

    def _ensure_runtime(self) -> Any:
        if self.runtime is None:
            raise RuntimeError("TorchLens split runtime has not been traced.")
        return self.runtime

    def _ensure_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("TorchLens split backend has no model.")
        return self.model

    def enumerate_candidates(
        self,
        *,
        max_boundary_count: int | None = None,
        max_payload_bytes: int | None = None,
        max_candidates: int | None = None,
    ) -> list[SplitCandidate]:
        runtime = self._ensure_runtime()
        graph = getattr(runtime, "trace_graph", None)
        if graph is None:
            raise RuntimeError("TorchLens runtime does not expose trace_graph.")
        candidates: list[SplitCandidate] = []
        for node_index, node in enumerate(graph.nodes):
            label = str(getattr(node, "label", "") or "")
            if (
                not label
                or bool(getattr(node, "is_input", False))
                or bool(getattr(node, "is_output", False))
            ):
                continue
            try:
                spec = make_split_spec(
                    after(label), dynamic_batch=self.split_spec.dynamic_batch,
                    trainable=self.split_spec.trainable,
                    backend=self.framework_backend,
                )
                candidate_runtime = repartition_split_runtime(runtime, spec)
            except Exception:
                continue
            candidate = candidate_from_plan(
                candidate_runtime,
                spec,
                candidate_runtime.plan,
                node_index=node_index,
                graph_signature=trace_signature(runtime),
            )
            if max_boundary_count is not None and candidate.boundary_count > int(max_boundary_count):
                continue
            if max_payload_bytes is not None and candidate.estimated_payload_bytes > int(max_payload_bytes):
                continue
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                int(item.estimated_payload_bytes),
                int(item.boundary_count),
                int(item.node_index if item.node_index is not None else 10**9),
                item.candidate_id,
            )
        )
        if max_candidates is not None:
            candidates = candidates[: max(0, int(max_candidates))]
        self.candidates = candidates
        return candidates

    def repartition(self, boundary: str) -> TorchLensRuntimeHandle:
        """Select another boundary while reusing the existing model capture."""
        runtime = self._ensure_runtime()
        if self.split_spec is None:
            raise RuntimeError("TorchLens split backend has not been traced.")
        spec = make_split_spec(
            boundary,
            dynamic_batch=self.split_spec.dynamic_batch,
            trainable=self.split_spec.trainable,
            backend=self.framework_backend,
        )
        self.runtime = repartition_split_runtime(runtime, spec)
        self.split_spec = spec
        self.current_candidate = candidate_from_plan(
            self.runtime,
            spec,
            self.runtime.plan,
            graph_signature=trace_signature(self.runtime),
        )
        return self.make_handle()

    def split(self, candidate: SplitCandidate | None = None) -> SplitCandidate:
        chosen = candidate or self.current_candidate
        if chosen is None:
            raise RuntimeError("No TorchLens split candidate is selected.")
        model = self._ensure_model()
        if self.trace_sample_input is None or self.split_spec is None:
            raise RuntimeError("TorchLens split backend has not been traced.")
        self.split_spec = make_split_spec(
            chosen.boundary, dynamic_batch=self.split_spec.dynamic_batch,
            trainable=self.split_spec.trainable,
            backend=self.framework_backend,
        )
        self.runtime = prepare_split_runtime(model, self.trace_sample_input, self.split_spec)
        self.current_candidate = candidate_from_plan(
            self.runtime,
            self.split_spec,
            self.runtime.plan,
            node_index=chosen.node_index,
            graph_signature=trace_signature(self.runtime),
        )
        return self.current_candidate

    def run_prefix(self, *inputs: Any, training: bool = False) -> BoundaryPayload:
        runtime = self._ensure_runtime()
        args = normalize_example_inputs(inputs)
        raw = (
            runtime.run_training_prefix(*args)
            if training and hasattr(runtime, "run_training_prefix")
            else runtime.run_prefix(*args)
        )
        return _normalise_boundary_batch_metadata(from_torchlens_boundary(raw), args)

    def run_suffix(self, boundary: BoundaryPayload) -> Any:
        runtime = self._ensure_runtime()
        native = to_torchlens_boundary(boundary)
        device = _runtime_device(runtime)
        if device is not None:
            native = native.to(device)
        return runtime.run_suffix(native)

    def train_suffix(
        self,
        boundary: BoundaryPayload,
        targets: Any,
        *,
        loss_fn=None,
        optimizer=None,
        measurements: dict[str, float] | None = None,
    ):
        """Run the suffix training step, optionally filling ``measurements``.

        Every backend reports ``server_total_ms``. Backends whose suffix step is
        executed phase by phase also report ``server_forward_ms`` and
        ``server_backward_ms``; the others cannot separate the phases and do not
        report a fabricated split.
        """

        runtime = self._ensure_runtime()
        native = to_torchlens_boundary(boundary)
        device = _runtime_device(runtime)
        if device is not None:
            native = native.to(device)
            targets = _move_to_device(targets, device)
        if self.framework_backend == "torch":
            return train_torch_suffix(
                runtime,
                native,
                targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
                measurements=measurements,
            )
        if measurements is not None:
            started = time.perf_counter_ns()
            result = runtime.train_suffix(native, targets, loss_fn=loss_fn, optimizer=optimizer)
            measurements["server_total_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
            return result
        return runtime.train_suffix(native, targets, loss_fn=loss_fn, optimizer=optimizer)

    def backward_prefix(
        self,
        boundary: BoundaryPayload,
        boundary_grads: Any,
        *,
        optimizer=None,
    ) -> Any:
        runtime = self._ensure_runtime()
        backward = getattr(runtime, "backward_prefix", None)
        if not callable(backward):
            raise RuntimeError("TorchLens SplitRuntime does not support backward_prefix.")
        native = to_torchlens_boundary(boundary)
        device = _runtime_device(runtime)
        if device is not None:
            native = native.to(device)
            boundary_grads = _move_to_device(boundary_grads, device)
        return backward(native, boundary_grads=boundary_grads, optimizer=optimizer)

    def validate_candidate(self, candidate: SplitCandidate | None = None) -> dict[str, Any]:
        chosen = candidate or self.current_candidate
        if chosen is None:
            return {
                "success": False,
                "candidate_id": None,
                "runtime": "torchlens_native",
                "error": "no candidate selected",
            }
        if candidate is not None and (
            self.current_candidate is None
            or candidate.candidate_id != self.current_candidate.candidate_id
        ):
            try:
                chosen = self.split(candidate)
            except Exception as exc:
                return {
                    "success": False,
                    "candidate_id": candidate.candidate_id,
                    "runtime": "torchlens_native",
                    "error": str(exc),
                }
        runtime = self._ensure_runtime()
        model = self._ensure_model()
        inputs = normalize_example_inputs(self.trace_sample_input)
        try:
            with inference_context(self.framework_backend):
                boundary = self.run_prefix(*inputs)
                replay_output = self.run_suffix(boundary)
                expected_output = model(*inputs)
            success, max_abs, max_rel = _compare_outputs(expected_output, replay_output, self.framework_backend)
        except Exception as exc:
            return {
                "success": False,
                "candidate_id": chosen.candidate_id,
                "runtime": "torchlens_native",
                "error": str(exc),
            }
        return {
            "success": bool(success),
            "max_abs_diff": float(max_abs),
            "max_rel_diff": float(max_rel),
            "candidate_id": chosen.candidate_id,
            "runtime": "torchlens_native",
            "error": None if success else "split replay output mismatch",
            "tail_trainability": bool(chosen.is_trainable_tail),
            "split_id": chosen.boundary,
        }

    def make_handle(self) -> TorchLensRuntimeHandle:
        runtime = self._ensure_runtime()
        candidate = self.current_candidate
        if candidate is None:
            raise RuntimeError("No TorchLens split candidate is selected.")
        plan = getattr(runtime, "plan", None)
        split_spec = getattr(runtime, "request", self.split_spec)
        boundary_schema = dict(candidate.descriptor.get("boundary_schema") or {})
        feature_layout = dict(candidate.descriptor.get("feature_layout") or {})
        contract = build_runtime_contract(
            model_family=str(self.model_family or self.model_name or ""),
            canonical_split_key=candidate.boundary,
            graph_signature=trace_signature(runtime),
            boundary_tensor_labels=list(candidate.boundary_tensor_labels),
            boundary_schema=boundary_schema,
            feature_layout=feature_layout,
            runtime_backend="torchlens_native",
            adapter_version=TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION,
            runtime_version=torchlens_runtime_version(),
            trace_batch_mode=str(getattr(split_spec, "trace_batch_mode", "")),
            dynamic_batch=getattr(split_spec, "dynamic_batch", None),
            trace_batch_size=self.trace_batch_size,
        )
        actual_split_id = str(candidate.boundary)
        runtime_plan = SplitRuntimePlan(
            plan_id=_make_plan_id(
                trace_signature(runtime),
                actual_split_id,
                candidate.boundary,
                str(getattr(split_spec, "mode", "generated_eager")),
            ),
            split_id=actual_split_id,
            graph_signature=trace_signature(runtime),
            boundary=candidate.boundary,
            mode="generated_eager",
            trainable=bool(getattr(split_spec, "trainable", True)),
            dynamic_batch=getattr(split_spec, "dynamic_batch", None),
            trace_batch_mode=str(getattr(split_spec, "trace_batch_mode", "batch_gt1")),
            trace_batch_size=self.trace_batch_size,
            boundary_bytes=int(candidate.estimated_payload_bytes),
            prefix_node_count=len(getattr(plan, "prefix_node_ids", ()) or ()),
            suffix_node_count=len(getattr(plan, "suffix_node_ids", ()) or ()),
            trainable_suffix=bool(candidate.is_trainable_tail),
            candidate_id=candidate.candidate_id,
            split_label=candidate.split_label,
            boundary_tensor_labels=list(candidate.boundary_tensor_labels),
            torchlens_version=torchlens_runtime_version(),
            feature_layout_id=str(contract.get("feature_layout_id", "")),
            feature_abi_id=str(contract.get("feature_abi_id", "")),
            runtime_contract=contract,
            metadata={
                "boundary_nodes": tuple(candidate.boundary_tensor_labels),
                "requested_boundary": getattr(split_spec, "boundary", candidate.boundary),
                "candidate_descriptor": build_candidate_descriptor(candidate),
                "feature_layout_id": contract.get("feature_layout_id", ""),
                "feature_abi_id": contract.get("feature_abi_id", ""),
                "_example_inputs": self.trace_sample_input,
            },
        )
        return TorchLensRuntimeHandle(
            model=self._ensure_model(),
            runtime=runtime,
            plan=runtime_plan,
            backend=self,
        )


def prepare_torchlens_runtime(
    model: torch.nn.Module,
    sample_inputs: Any,
    *,
    boundary: str = "50%",
    mode: str = "generated_eager",
    trainable: bool = True,
    dynamic_batch: tuple[int, int] | None = None,
    trace_batch_mode: str | None = None,
    model_name: str | None = None,
    model_family: str | None = None,
    candidate: SplitCandidate | None = None,
) -> TorchLensRuntimeHandle:
    backend = TorchLensSplitBackend(model_name=model_name, model_family=model_family)
    backend.trace(
        model,
        sample_inputs,
        boundary=boundary,
        mode=mode,
        trainable=trainable,
        dynamic_batch=dynamic_batch,
        trace_batch_mode=trace_batch_mode,
        model_name=model_name,
        model_family=model_family,
    )
    if candidate is not None:
        backend.split(candidate)
    return backend.make_handle()


def run_prefix(handle: TorchLensRuntimeHandle, *inputs: Any) -> BoundaryPayload:
    return handle.backend.run_prefix(*inputs)


def run_training_prefix(handle: TorchLensRuntimeHandle, *inputs: Any) -> BoundaryPayload:
    return handle.backend.run_prefix(*inputs, training=True)


def run_suffix(handle: TorchLensRuntimeHandle, boundary: BoundaryPayload) -> Any:
    return handle.backend.run_suffix(boundary)


def train_suffix(
    handle: TorchLensRuntimeHandle,
    boundary: BoundaryPayload,
    targets: Any,
    *,
    loss_fn=None,
    optimizer=None,
    measurements: dict[str, float] | None = None,
):
    return handle.backend.train_suffix(
        boundary,
        targets,
        loss_fn=loss_fn,
        optimizer=optimizer,
        measurements=measurements,
    )


def backward_prefix(
    handle: TorchLensRuntimeHandle,
    boundary: BoundaryPayload,
    boundary_grads: Any,
    *,
    optimizer=None,
) -> Any:
    return handle.backend.backward_prefix(
        boundary,
        boundary_grads,
        optimizer=optimizer,
    )


__all__ = [
    "SplitRuntimeHandle",
    "TorchLensRuntimeHandle",
    "TorchLensSplitBackend",
    "backward_prefix",
    "prepare_torchlens_runtime",
    "run_prefix",
    "run_suffix",
    "run_training_prefix",
    "train_suffix",
]
