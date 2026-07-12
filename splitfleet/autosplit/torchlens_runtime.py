"""Compatibility helpers for the TorchLens 2.31 public split API."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import torch
from torchlens.split import ReplayBoundary, SplitFeatures, SplitRequest, SplitRuntime, after, before, percent, prepare

DEFAULT_SPLIT_MODE = "generated_eager"
TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION = "splitfleet-torchlens-public-v2"
SplitSpec = SplitRequest
_REPLAY_COMPAT_INSTALLED = False


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
    if isinstance(value, torch.Tensor):
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
    """Install narrow compatibility for portable Python ellipsis indices."""
    global _REPLAY_COMPAT_INSTALLED
    if _REPLAY_COMPAT_INSTALLED:
        return
    from torchlens.intervention.types import Unsupported
    from torchlens.split.adapters.torch import GeneratedSuffix, GeneratedPrefix

    for cls in (GeneratedPrefix, GeneratedSuffix):
        original = cls._resolve_component

        def resolve(self, component, node, overlay, *, param_cursor, runtime_batch_size, _original=original):
            if isinstance(component, Unsupported) and component.value_type == "ellipsis":
                return Ellipsis
            return _original(
                self,
                component,
                node,
                overlay,
                param_cursor=param_cursor,
                runtime_batch_size=runtime_batch_size,
            )

        cls._resolve_component = resolve
    _REPLAY_COMPAT_INSTALLED = True


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
) -> SplitRequest:
    del trace_batch_mode, mode
    if isinstance(boundary, SplitRequest):
        return boundary
    return SplitRequest(
        point=_point(getattr(boundary, "boundary", boundary)),
        backend="torch",
        features=SplitFeatures(replay=True, dynamic_batch=dynamic_batch, training=bool(trainable)),
        validation="strict",
        batch_symbol=batch_symbol,
    )


def prepare_split_runtime(model: torch.nn.Module, example_inputs: Any, split_spec_or_boundary: Any, mode: str | None = None) -> SplitRuntime:
    del mode
    request = make_split_spec(split_spec_or_boundary) if not isinstance(split_spec_or_boundary, SplitRequest) else split_spec_or_boundary
    _install_torchlens_replay_compat()
    _reset_torch_capture_wrappers(model)
    return prepare(model, normalize_example_inputs(example_inputs), request)


def prepare_split_replay_runtime(model: torch.nn.Module, example_inputs: Any, split_spec_or_boundary: Any, mode: str | None = None) -> SplitRuntime:
    request = make_split_spec(split_spec_or_boundary) if not isinstance(split_spec_or_boundary, SplitRequest) else split_spec_or_boundary
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
    "prepare_split_replay_runtime", "prepare_split_runtime", "torchlens_runtime_version", "trace_signature"]
