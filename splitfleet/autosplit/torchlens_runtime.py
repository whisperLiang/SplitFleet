"""Thin TorchLens 2.18 native split runtime wrappers for SplitFleet."""

from __future__ import annotations

import importlib.metadata
from dataclasses import fields, replace
from typing import Any, Literal

import torch
from torchlens.split import (
    ReplayBoundary,
    SplitRuntime,
    SplitSpec,
    prepare_split,
    prepare_split_replay,
)
from torchlens.split.shape import infer_traced_batch_size


DEFAULT_SPLIT_MODE = "generated_eager"
TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION = "splitfleet-torchlens-native-runtime-v2"
TORCHLENS_MIN_NATIVE_RUNTIME_VERSION = "2.18.0"


def torchlens_runtime_version() -> str:
    try:
        import torchlens as tl

        version = getattr(tl, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        return importlib.metadata.version("torchlens")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def require_torchlens_218() -> None:
    version = torchlens_runtime_version()
    if _version_tuple(version) < _version_tuple(TORCHLENS_MIN_NATIVE_RUNTIME_VERSION):
        raise RuntimeError(
            "SplitFleet TorchLens autosplit requires torchlens>=2.18.0, "
            f"but imported torchlens version is {version!r}."
        )


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in str(value).split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if digits:
            parts.append(int(digits))
    return tuple(parts or [0])


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


def _split_spec_field_names() -> set[str]:
    return {field.name for field in fields(SplitSpec)}


def make_split_spec(
    boundary: Any,
    *,
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = None,
    trainable: bool = True,
    trace_batch_mode: str = "batch_gt1",
    device_policy: str = "runtime",
    mode: str = DEFAULT_SPLIT_MODE,
    use_live_param_sources: bool | None = True,
) -> SplitSpec:
    if isinstance(boundary, SplitSpec):
        updates: dict[str, Any] = {}
        if mode is not None and getattr(boundary, "mode", None) != mode:
            updates["mode"] = _normalize_mode(mode)
        if "use_live_param_sources" in _split_spec_field_names() and (
            use_live_param_sources is not None
            and getattr(boundary, "use_live_param_sources", None) != use_live_param_sources
        ):
            updates["use_live_param_sources"] = use_live_param_sources
        return replace(boundary, **updates) if updates else boundary

    if hasattr(boundary, "boundary") and not isinstance(boundary, str):
        config = boundary
        boundary_value = str(config.boundary)
        dynamic_batch_value = getattr(config, "dynamic_batch", dynamic_batch)
        trainable_value = bool(getattr(config, "trainable", trainable))
        trace_batch_mode_value = str(getattr(config, "trace_batch_mode", trace_batch_mode))
        mode_value = _normalize_mode(getattr(config, "mode", mode))
        use_live_value = getattr(config, "use_live_param_sources", use_live_param_sources)
    else:
        boundary_value = str(boundary)
        dynamic_batch_value = dynamic_batch
        trainable_value = bool(trainable)
        trace_batch_mode_value = str(trace_batch_mode or "batch_gt1")
        mode_value = _normalize_mode(mode)
        use_live_value = use_live_param_sources

    kwargs: dict[str, Any] = {
        "boundary": boundary_value,
        "batch_symbol": batch_symbol,
        "dynamic_batch": dynamic_batch_value,
        "trainable": trainable_value,
        "trace_batch_mode": trace_batch_mode_value,
        "device_policy": device_policy,
        "mode": mode_value,
    }
    if "use_live_param_sources" in _split_spec_field_names():
        kwargs["use_live_param_sources"] = use_live_value
    return SplitSpec(**kwargs)


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


def prepare_split_runtime(
    model: torch.nn.Module,
    example_inputs: Any,
    split_spec_or_boundary: SplitSpec | str,
    mode: str | None = None,
) -> SplitRuntime:
    require_torchlens_218()
    inputs = normalize_example_inputs(example_inputs)
    spec = _spec_with_mode(split_spec_or_boundary, mode)
    _validate_trace_batch(inputs, spec)
    _clear_torchlens_module_state(model)
    return prepare_split(model, inputs, spec)


def prepare_split_replay_runtime(
    model: torch.nn.Module,
    example_inputs: Any,
    split_spec_or_boundary: SplitSpec | str,
    mode: str | None = None,
) -> SplitRuntime:
    require_torchlens_218()
    inputs = normalize_example_inputs(example_inputs)
    spec = _spec_with_mode(split_spec_or_boundary, mode)
    _validate_trace_batch(inputs, spec)
    _clear_torchlens_module_state(model)
    return prepare_split_replay(model, inputs, spec)


def prepare_torchlens_runtime(
    model: torch.nn.Module,
    example_inputs: Any,
    split_spec_or_boundary: SplitSpec | str,
    mode: str | None = None,
) -> SplitRuntime:
    return prepare_split_runtime(model, example_inputs, split_spec_or_boundary, mode)


def prepare_torchlens_replay_runtime(
    model: torch.nn.Module,
    example_inputs: Any,
    split_spec_or_boundary: SplitSpec | str,
    mode: str | None = None,
) -> SplitRuntime:
    return prepare_split_replay_runtime(model, example_inputs, split_spec_or_boundary, mode)


def build_split_runtime(model: Any, example_batch: Any, config: Any) -> SplitRuntime:
    inputs = normalize_example_inputs(example_batch)
    batch_size = first_tensor_batch_size(inputs)
    if batch_size is None or batch_size <= 1:
        raise ValueError("TorchLens batch_gt1 tracing requires example_batch batch size > 1.")
    spec = make_split_spec(config)
    return prepare_split_runtime(model, inputs, spec, mode=spec.mode)


def build_torchlens_runtime(model: Any, example_batch: Any, config: Any) -> SplitRuntime:
    return build_split_runtime(model, example_batch, config)


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
        "torchlens_version": torchlens_runtime_version(),
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
    "TORCHLENS_MIN_NATIVE_RUNTIME_VERSION",
    "TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION",
    "build_split_runtime",
    "build_torchlens_runtime",
    "first_tensor_batch_size",
    "get_split_runtime_metadata",
    "infer_trace_batch_mode",
    "make_split_spec",
    "normalize_example_inputs",
    "prepare_split_replay_runtime",
    "prepare_split_runtime",
    "prepare_torchlens_replay_runtime",
    "prepare_torchlens_runtime",
    "require_torchlens_218",
    "torchlens_runtime_version",
    "trace_signature",
]
