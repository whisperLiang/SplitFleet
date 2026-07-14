"""Compatibility helpers for the TorchLens 2.31 public split API."""

from __future__ import annotations

import importlib.metadata
from contextvars import ContextVar
from dataclasses import fields
from functools import lru_cache
from hashlib import sha256
from threading import RLock
from typing import Any

import torch
from torchlens.split import ReplayBoundary, SplitFeatures, SplitRequest, SplitRuntime, after, before, percent, prepare
from splitfleet.backends.utils import detect_torchlens_backend

DEFAULT_SPLIT_MODE = "generated_eager"
TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION = "splitfleet-torchlens-public-v2"
SplitSpec = SplitRequest
_REPLAY_COMPAT_INSTALLED = False
_TINYGRAD_SIGNATURE_CACHE = None
_TINYGRAD_DEVICE_REWRITE_CACHE = None
_TINYGRAD_CAPTURE_LOCK = RLock()


def torchlens_runtime_version() -> str:
    try:
        return importlib.metadata.version("torchlens")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def require_torchlens_231() -> None:
    version = torchlens_runtime_version()
    if version != "2.31.0":
        raise RuntimeError(
            f"SplitFleet requires torchlens==2.31.0, but the active installation is {version!r}."
        )


def normalize_example_inputs(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def _iter_tensors(value: Any):
    module = type(value).__module__
    if isinstance(value, torch.Tensor) or (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and module.startswith(("tensorflow", "jax", "jaxlib", "paddle", "tinygrad"))
    ):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def first_tensor_batch_size(example_inputs: Any) -> int | None:
    for tensor in _iter_tensors(example_inputs):
        if tensor.ndim:
            return int(tensor.shape[0])
    return None


def infer_trace_batch_mode(example_inputs: Any) -> str:
    size = first_tensor_batch_size(example_inputs)
    if size is None:
        raise ValueError("TorchLens tracing requires at least one batched tensor in sample inputs.")
    return "batch_gt1" if size > 1 else "batch_1"


def _reset_torch_capture_wrappers(model: torch.nn.Module) -> None:
    """Work around TorchLens 2.31 repeated-root capture wrapper state."""
    try:
        from torchlens import _state

        _state._prepared_models.discard(model)
    except Exception:
        pass
    for module in model.modules():
        forward = vars(module).get("forward")
        wrapped = getattr(forward, "__wrapped__", None)
        if wrapped is not None:
            try:
                delattr(module, "forward")
            except AttributeError:
                module.forward = wrapped


def _install_torchlens_replay_compat() -> None:
    """Install narrow Torch replay compatibility for portable captured values."""
    global _REPLAY_COMPAT_INSTALLED
    if _REPLAY_COMPAT_INSTALLED:
        return
    from torchlens.intervention.types import Unsupported
    from torchlens.split.adapters.torch import GeneratedSuffix, GeneratedPrefix

    for cls in (GeneratedPrefix, GeneratedSuffix):
        original = cls._resolve_component
        original_param_handles = cls._param_handles_for_node
        original_source_value = cls._source_value

        def resolve(self, component, node, overlay, *, param_cursor, _original=original):
            if isinstance(component, Unsupported) and component.value_type == "ellipsis":
                return Ellipsis
            return _original(
                self,
                component,
                node,
                overlay,
                param_cursor=param_cursor,
            )

        cls._resolve_component = resolve

        def param_handles(self, node, _original=original_param_handles):
            try:
                return _original(self, node)
            except KeyError:
                # TorchLens 2.31 can record a direct Parameter (for example
                # RF-DETR's refpoint_embed) with a module address that is not
                # present in Trace.modules. The live handle is still valid;
                # only optional owning-module buffer discovery must be skipped.
                handles = []
                seen = set()
                for param_ref in node.param_refs:
                    handle = getattr(param_ref, "handle", None)
                    if handle is not None and id(handle) not in seen:
                        handles.append(handle)
                        seen.add(id(handle))
                return handles

        cls._param_handles_for_node = param_handles

        def source_value(self, node, _original=original_source_value):
            value = _original(self, node)
            if value is None:
                # A captured Buffer may retain a BufferRef whose live handle
                # was released after tracing. TorchLens currently returns that
                # None instead of falling back to the captured tensor payload.
                captured = getattr(getattr(node, "op", None), "out", None)
                if captured is not None:
                    return captured
            return value

        cls._source_value = source_value
    _REPLAY_COMPAT_INSTALLED = True


def _tinygrad_signature_cache():
    """Memoize recursive UOp signatures for one capture at a time."""
    global _TINYGRAD_SIGNATURE_CACHE
    if _TINYGRAD_SIGNATURE_CACHE is None:
        from torchlens.backends.tinygrad import backend as tinygrad_backend

        current = tinygrad_backend._uop_signature
        if hasattr(current, "cache_clear"):
            _TINYGRAD_SIGNATURE_CACHE = current
        else:
            @lru_cache(maxsize=None)
            def cached(uop):
                children = ",".join(cached(child) for child in (getattr(uop, "src", ()) or ()))
                payload = (
                    f"{tinygrad_backend._uop_name(uop)}:"
                    f"{getattr(uop, 'dtype', None)}:{getattr(uop, 'arg', None)}"
                    f"[{children}]"
                ).encode("utf-8", errors="backslashreplace")
                return sha256(payload).hexdigest()

            tinygrad_backend._uop_signature = cached
            _TINYGRAD_SIGNATURE_CACHE = cached
    return _TINYGRAD_SIGNATURE_CACHE


def _install_tinygrad_replay_compat() -> None:
    """Memoize recursive device rewrites over tinygrad's shared UOp DAG."""
    global _TINYGRAD_DEVICE_REWRITE_CACHE
    if _TINYGRAD_DEVICE_REWRITE_CACHE is not None:
        return
    from torchlens.split.adapters import tinygrad as tinygrad_adapter

    current = tinygrad_adapter._rewrite_tinygrad_uop_device
    if hasattr(current, "cache_clear"):
        _TINYGRAD_DEVICE_REWRITE_CACHE = current
        return
    cached = lru_cache(maxsize=None)(current)
    tinygrad_adapter._rewrite_tinygrad_uop_device = cached
    _TINYGRAD_DEVICE_REWRITE_CACHE = cached

    segment_type = tinygrad_adapter._TinygradGeneratedSegmentBase
    original_parameter_rewrite = segment_type._rewrite_parameter_expand_tree
    original_shape_rewrite = segment_type._rewrite_shape_uop_tree
    original_constant_lineage = segment_type._is_constant_lineage
    suffix_type = tinygrad_adapter.TinygradGeneratedSuffix
    original_suffix_call = suffix_type.__call__
    original_literal_check = tinygrad_adapter._is_tinygrad_literal_uop
    remote_training = ContextVar("splitfleet_tinygrad_remote_training", default=False)
    remote_root_ids = ContextVar("splitfleet_tinygrad_remote_root_ids", default=frozenset())

    def suffix_call(self, boundary):
        self._splitfleet_parameter_rewrite_cache = {}
        self._splitfleet_shape_rewrite_cache = {}
        is_remote_training = bool(
            getattr(boundary, "metadata", {}).get("suffix_training_roots")
        )
        self._splitfleet_remote_training = is_remote_training
        token = remote_training.set(is_remote_training)
        root_token = remote_root_ids.set(
            frozenset(
                id(getattr(value, "uop", None))
                for value in getattr(boundary, "tensors", {}).values()
            )
            if is_remote_training
            else frozenset()
        )
        try:
            return original_suffix_call(self, boundary)
        finally:
            remote_root_ids.reset(root_token)
            remote_training.reset(token)
            self._splitfleet_remote_training = False
            self._splitfleet_parameter_rewrite_cache.clear()
            self._splitfleet_shape_rewrite_cache.clear()

    def parameter_rewrite(self, node, uop):
        cache = getattr(self, "_splitfleet_parameter_rewrite_cache", None)
        if cache is None:
            cache = self._splitfleet_parameter_rewrite_cache = {}
        key = (node.canonical_id, id(uop))
        if key not in cache:
            cache[key] = original_parameter_rewrite(self, node, uop)
        return cache[key]

    def shape_rewrite(self, node, uop):
        cache = getattr(self, "_splitfleet_shape_rewrite_cache", None)
        if cache is None:
            cache = self._splitfleet_shape_rewrite_cache = {}
        key = (node.canonical_id, id(uop))
        if key not in cache:
            cache[key] = original_shape_rewrite(self, node, uop)
        return cache[key]

    def constant_lineage(self, node, seen=None):
        if getattr(self, "_splitfleet_remote_training", False):
            return False
        return original_constant_lineage(self, node, seen)

    def literal_check(uop):
        if remote_training.get():
            return False
        return original_literal_check(uop)

    @lru_cache(maxsize=None)
    def already_on_device(uop, target_device):
        if target_device is None:
            return True
        op = getattr(uop, "op", None)
        if op is tinygrad_adapter._tinygrad_ops().DEVICE:
            return getattr(uop, "arg", None) == target_device
        return all(
            already_on_device(child, target_device)
            for child in (getattr(uop, "src", ()) or ())
        )

    def device_rewrite(uop, target_device):
        if remote_training.get() and id(uop) in remote_root_ids.get():
            return uop
        if remote_training.get() and already_on_device(uop, target_device):
            return uop
        return cached(uop, target_device)

    suffix_type.__call__ = suffix_call
    segment_type._rewrite_parameter_expand_tree = parameter_rewrite
    segment_type._rewrite_shape_uop_tree = shape_rewrite
    segment_type._is_constant_lineage = constant_lineage
    tinygrad_adapter._is_tinygrad_literal_uop = literal_check
    tinygrad_adapter._rewrite_tinygrad_uop_device = device_rewrite


def _point(boundary: Any):
    if hasattr(boundary, "kind") and hasattr(boundary, "target"):
        return boundary
    text = str(boundary or "50%").strip()
    if text.startswith("after:"):
        return after(text.split(":", 1)[1])
    if text.startswith("before:"):
        return before(text.split(":", 1)[1])
    if text.startswith("percent:"):
        return percent(float(text.split(":", 1)[1]))
    if text.endswith("%"):
        return percent(float(text[:-1]))
    raise ValueError(f"Unsupported TorchLens boundary {boundary!r}")


def make_split_spec(
    boundary: Any,
    *,
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = (2, 64),
    trainable: bool = True,
    trace_batch_mode: str = "batch_gt1",
    mode: str = DEFAULT_SPLIT_MODE,
    backend: str = "torch",
) -> SplitRequest:
    del trace_batch_mode, mode
    if isinstance(boundary, SplitRequest):
        return boundary
    return SplitRequest(
        point=_point(getattr(boundary, "boundary", boundary)),
        backend=str(backend),
        features=SplitFeatures(replay=True, dynamic_batch=dynamic_batch, training=bool(trainable)),
        validation="strict",
        batch_symbol=batch_symbol,
    )


def prepare_split_runtime(model: Any, example_inputs: Any, split_spec_or_boundary: Any, mode: str | None = None) -> SplitRuntime:
    del mode
    backend = detect_torchlens_backend(model, example_inputs)
    request = make_split_spec(split_spec_or_boundary, backend=backend) if not isinstance(split_spec_or_boundary, SplitRequest) else split_spec_or_boundary
    if backend == "torch":
        _install_torchlens_replay_compat()
        _reset_torch_capture_wrappers(model)
    if backend == "tinygrad":
        _install_tinygrad_replay_compat()
        signature = _tinygrad_signature_cache()
        with _TINYGRAD_CAPTURE_LOCK:
            signature.cache_clear()
            try:
                return prepare(model, normalize_example_inputs(example_inputs), request)
            finally:
                signature.cache_clear()
    return prepare(model, normalize_example_inputs(example_inputs), request)


def repartition_split_runtime(runtime: SplitRuntime, request: SplitRequest) -> SplitRuntime:
    """Build a new split from an existing capture without retracing the model."""
    from torchlens.split.pipeline import (
        analyze_split_capabilities,
        execute_split_runtime,
        lower_split_program,
        plan_split,
    )
    from torchlens.split.program import ensure_capability_report_supported

    graph = runtime.trace_graph
    adapter = runtime.adapter
    plan = plan_split(graph, request)
    prefix_program = lower_split_program(
        graph, plan, request, segment="prefix", adapter=adapter,
    )
    suffix_program = lower_split_program(
        graph, plan, request, segment="suffix", adapter=adapter,
    )
    feature_values = {
        field.name: getattr(request.features, field.name)
        for field in fields(request.features)
    }
    capability_report = analyze_split_capabilities(
        adapter,
        graph,
        plan,
        request,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
        graph_ir=getattr(runtime, "graph_ir", None),
        model_profile=getattr(runtime, "model_profile", None),
        features=feature_values,
    )
    if request.validation == "strict":
        ensure_capability_report_supported(capability_report, request)
    segments = execute_split_runtime(adapter, graph, plan, request)
    return SplitRuntime(
        model=runtime.model,
        trace=runtime.trace,
        trace_graph=graph,
        request=request,
        plan=plan,
        adapter=adapter,
        segments=segments,
        capability_report=capability_report,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
        graph_ir=getattr(runtime, "graph_ir", None),
        model_profile=getattr(runtime, "model_profile", None),
        prepared_input_kwargs=getattr(runtime, "prepared_input_kwargs", None),
    )


def prepare_split_replay_runtime(model: torch.nn.Module, example_inputs: Any, split_spec_or_boundary: Any, mode: str | None = None) -> SplitRuntime:
    request = make_split_spec(
        split_spec_or_boundary,
        backend=detect_torchlens_backend(model, example_inputs),
    ) if not isinstance(split_spec_or_boundary, SplitRequest) else split_spec_or_boundary
    request = SplitRequest(point=request.point, backend=request.backend, model_profile=request.model_profile,
        features=SplitFeatures(replay=True, dynamic_batch=request.dynamic_batch, training=False),
        validation=request.validation, device_policy=request.device_policy, batch_symbol=request.batch_symbol)
    return prepare_split_runtime(model, example_inputs, request, mode)


def build_split_runtime(model: Any, example_batch: Any, config: Any) -> SplitRuntime:
    return prepare_split_runtime(model, example_batch, make_split_spec(config))


def trace_signature(runtime: Any) -> str:
    graph_ir = getattr(runtime, "graph_ir", None)
    return str(getattr(graph_ir, "graph_hash", "") or getattr(getattr(runtime, "trace_graph", None), "graph_shape_hash", "") or "")


def get_split_runtime_metadata(runtime: Any) -> dict[str, Any]:
    plan, request = runtime.plan, runtime.request
    return {"split_id": runtime.split_id, "graph_signature": trace_signature(runtime),
        "runtime_backend": "torchlens_native", "boundary": request.boundary,
        "split_label": getattr(next((n for n in runtime.trace_graph.nodes if n.canonical_id == plan.target_node_id), None), "label", ""),
        "boundary_tensor_labels": list(plan.boundary_node_ids), "prefix_nodes": list(plan.prefix_node_ids),
        "suffix_nodes": list(plan.suffix_node_ids), "trainable_suffix": request.trainable}


__all__ = ["DEFAULT_SPLIT_MODE", "ReplayBoundary", "SplitRuntime", "SplitRequest", "SplitSpec",
    "TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION", "build_split_runtime", "first_tensor_batch_size",
    "get_split_runtime_metadata", "infer_trace_batch_mode", "make_split_spec", "normalize_example_inputs",
    "prepare_split_replay_runtime", "prepare_split_runtime", "repartition_split_runtime",
    "torchlens_runtime_version", "trace_signature"]
