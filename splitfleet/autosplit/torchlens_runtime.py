"""TorchLens native split runtime preparation for SplitFleet."""

from __future__ import annotations

import importlib.metadata
from dataclasses import replace
from typing import Any, Literal

import torch
from torchlens.options import CaptureOptions, VisualizationOptions
from torchlens.split import ReplayBoundary, SplitRuntime, SplitSpec
from torchlens.split.codegen import build_segments
from torchlens.split.planner import plan_split
from torchlens.split.shape import infer_traced_batch_size
from torchlens.split.trace_graph import trace_graph_from_model_log
from torchlens.user_funcs import log_forward_pass


DEFAULT_SPLIT_MODE = "generated_eager"
TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION = "splitfleet-torchlens-native-runtime-v1"


def torchlens_runtime_version() -> str:
    try:
        return importlib.metadata.version("torchlens")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def normalize_example_inputs(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def first_tensor_batch_size(example_inputs: Any) -> int | None:
    for tensor in _iter_tensors(example_inputs):
        if tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def infer_trace_batch_mode(example_inputs: Any) -> str:
    batch_size = first_tensor_batch_size(example_inputs)
    if batch_size is None:
        raise ValueError("TorchLens tracing requires at least one batched tensor in sample inputs.")
    return "batch_gt1" if batch_size > 1 else "batch_1"


def _normalize_mode(mode: str | None) -> Literal["generated_eager", "compiled"]:
    normalized = str(mode or DEFAULT_SPLIT_MODE).strip()
    if normalized not in {"generated_eager", "compiled"}:
        raise ValueError(f"Unsupported TorchLens split mode: {mode!r}.")
    return normalized  # type: ignore[return-value]


def make_split_spec(
    boundary: Any,
    *,
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = (2, 64),
    trainable: bool = True,
    trace_batch_mode: str = "batch_gt1",
    mode: str = DEFAULT_SPLIT_MODE,
) -> SplitSpec:
    if isinstance(boundary, SplitSpec):
        if mode is None or getattr(boundary, "mode", None) == mode:
            return boundary
        return replace(boundary, mode=_normalize_mode(mode))
    if hasattr(boundary, "boundary") and not isinstance(boundary, str):
        config = boundary
        return SplitSpec(
            boundary=str(config.boundary),
            batch_symbol=batch_symbol,
            dynamic_batch=getattr(config, "dynamic_batch", dynamic_batch),
            trainable=bool(getattr(config, "trainable", trainable)),
            trace_batch_mode=str(getattr(config, "trace_batch_mode", trace_batch_mode)),
            device_policy="runtime",
            mode=_normalize_mode(getattr(config, "mode", mode)),
        )
    return SplitSpec(
        boundary=str(boundary),
        batch_symbol=batch_symbol,
        dynamic_batch=dynamic_batch,
        trainable=bool(trainable),
        trace_batch_mode=str(trace_batch_mode or "batch_gt1"),
        device_policy="runtime",
        mode=_normalize_mode(mode),
    )


def _spec_with_mode(split_spec_or_boundary: SplitSpec | str, mode: str | None) -> SplitSpec:
    spec = (
        make_split_spec(split_spec_or_boundary, mode=mode or DEFAULT_SPLIT_MODE)
        if isinstance(split_spec_or_boundary, str)
        else split_spec_or_boundary
    )
    if mode is None:
        return spec
    normalized_mode = _normalize_mode(mode)
    if getattr(spec, "mode", None) == normalized_mode:
        return spec
    return replace(spec, mode=normalized_mode)


def _validate_trace_batch(inputs: tuple[Any, ...], spec: SplitSpec) -> None:
    traced_batch_size = infer_traced_batch_size(inputs)
    if getattr(spec, "trace_batch_mode", None) == "batch_gt1" and (
        traced_batch_size is None or int(traced_batch_size) <= 1
    ):
        raise ValueError("TorchLens batch_gt1 tracing requires trace sample batch size > 1.")


def _clear_torchlens_module_state(model: torch.nn.Module) -> None:
    try:
        from torchlens.decoration.model_prep import _state

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
                setattr(module, "forward", wrapped)


def _prepare_split(model: torch.nn.Module, example_inputs: Any, spec: SplitSpec) -> SplitRuntime:
    inputs = normalize_example_inputs(example_inputs)
    _validate_trace_batch(inputs, spec)
    _clear_torchlens_module_state(model)
    model_log = log_forward_pass(
        model,
        inputs,
        {},
        capture=CaptureOptions(
            layers_to_save="all",
            keep_unsaved_layers=True,
            detach_saved_tensors=False,
            save_function_args=True,
            intervention_ready=True,
        ),
        visualization=VisualizationOptions(view="none"),
    )
    traced_batch_size = infer_traced_batch_size(inputs)
    graph = trace_graph_from_model_log(
        model_log,
        traced_batch_size=traced_batch_size,
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
    )
    plan = plan_split(graph, spec)
    segments = build_segments(graph, plan, mode=spec.mode)
    return SplitRuntime(
        model=model,
        trace_graph=graph,
        split_spec=spec,
        plan=plan,
        segments=segments,
    )


def prepare_split_runtime(
    model: torch.nn.Module,
    example_inputs: Any,
    split_spec_or_boundary: SplitSpec | str,
    mode: str | None = None,
) -> SplitRuntime:
    spec = _spec_with_mode(split_spec_or_boundary, mode)
    return _prepare_split(model, example_inputs, spec)


def prepare_split_replay_runtime(
    model: torch.nn.Module,
    example_inputs: Any,
    split_spec_or_boundary: SplitSpec | str,
    mode: str | None = None,
) -> SplitRuntime:
    spec = _spec_with_mode(split_spec_or_boundary, mode)
    replay_spec = SplitSpec(
        boundary=spec.boundary,
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
        trainable=False,
        trace_batch_mode=spec.trace_batch_mode,
        device_policy=spec.device_policy,
        mode=spec.mode,
    )
    return _prepare_split(model, example_inputs, replay_spec)


def build_split_runtime(model: Any, example_batch: Any, config: Any) -> SplitRuntime:
    inputs = normalize_example_inputs(example_batch)
    batch_size = first_tensor_batch_size(inputs)
    if batch_size is None or batch_size <= 1:
        raise ValueError("TorchLens batch_gt1 tracing requires example_batch batch size > 1.")
    spec = make_split_spec(config)
    return prepare_split_runtime(model, inputs, spec, mode=spec.mode)


def trace_signature(runtime: Any) -> str:
    graph = getattr(runtime, "trace_graph", None)
    return str(getattr(graph, "graph_shape_hash", "") or "")


def get_split_runtime_metadata(runtime: Any) -> dict[str, Any]:
    plan = getattr(runtime, "plan", None)
    split_spec = getattr(runtime, "split_spec", None)
    return {
        "split_id": getattr(runtime, "split_id", None),
        "graph_signature": trace_signature(runtime),
        "runtime_backend": "torchlens_native",
        "torchlens_mode": getattr(split_spec, "mode", None),
        "boundary": getattr(split_spec, "boundary", None),
        "split_label": getattr(plan, "split_label", None),
        "boundary_tensor_labels": list(getattr(plan, "boundary_nodes", ()) or ()),
        "prefix_nodes": list(getattr(plan, "prefix_nodes", ()) or ()),
        "suffix_nodes": list(getattr(plan, "suffix_nodes", ()) or ()),
        "trainable_suffix": bool(getattr(split_spec, "trainable", True)),
    }


__all__ = [
    "DEFAULT_SPLIT_MODE",
    "ReplayBoundary",
    "SplitRuntime",
    "SplitSpec",
    "TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION",
    "build_split_runtime",
    "first_tensor_batch_size",
    "get_split_runtime_metadata",
    "infer_trace_batch_mode",
    "make_split_spec",
    "normalize_example_inputs",
    "prepare_split_replay_runtime",
    "prepare_split_runtime",
    "torchlens_runtime_version",
    "trace_signature",
]
