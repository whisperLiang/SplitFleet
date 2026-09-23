"""Preparation and batch contracts for the TorchLens 2.34.1 public API."""

from __future__ import annotations

import importlib.metadata
import inspect
from dataclasses import replace
from typing import Any, Mapping

import torch
from torchlens import release_model
from torchlens.split import SplitFeatures, SplitPoint, SplitRequest, SplitRuntime, after, before, percent, prepare
from splitfleet.backends.utils import detect_torchlens_backend

TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION = "splitfleet-torchlens-public-v4-build1"
REQUIRED_TORCHLENS_VERSION = "2.34.1"
REQUIRED_TORCHLENS_BUILD = "SplitFleet local build 1 of TorchLens 2.34.1."

def torchlens_runtime_version() -> str:
    return importlib.metadata.version("torchlens")


def require_torchlens_version() -> None:
    version = torchlens_runtime_version()
    if version != REQUIRED_TORCHLENS_VERSION:
        raise RuntimeError(
            f"SplitFleet requires torchlens=={REQUIRED_TORCHLENS_VERSION}, but the active installation is {version!r}."
        )
    provenance = importlib.metadata.distribution("torchlens").read_text("SPLITFLEET_PATCHES")
    if not provenance or provenance.splitlines()[0] != REQUIRED_TORCHLENS_BUILD:
        raise RuntimeError(
            "SplitFleet requires the patched TorchLens 2.34.1 build 1 wheel; "
            "install the repository dependency with uv sync --reinstall-package torchlens."
        )


def normalize_example_inputs(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    return (example_inputs,)


def normalize_model_call(model: Any, inputs: tuple[Any, ...], input_kwargs: Mapping[str, Any] | None = None,
                         batch_axes: Mapping[str, int] | None = None):
    """Bind Torch keyword calls to the same positional convention used by capture.

    TorchLens 2.34.1 mistakes an empty args tuple for a single tuple argument
    when ``forward(x)`` is called as ``forward(x=...)``. Python's public binding
    API removes that ambiguity and also rejects duplicate arguments. Keyword
    only inputs retain their names; explicit batch pointers follow moved args.
    """
    kwargs = dict(input_kwargs or {})
    if not isinstance(model, torch.nn.Module) or not kwargs:
        return inputs, kwargs, batch_axes
    signature = inspect.signature(model.forward)
    bound = signature.bind(*inputs, **kwargs)
    args, canonical_kwargs = bound.args, bound.kwargs
    axes = None if batch_axes is None else dict(batch_axes)
    if axes is not None:
        positional = [name for name, parameter in signature.parameters.items()
                      if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
        for index, name in enumerate(positional[:len(args)]):
            if name not in kwargs or name in canonical_kwargs:
                continue
            source = "/kwargs/" + name.replace("~", "~0").replace("/", "~1")
            for path in list(axes):
                if path == source or path.startswith(source + "/"):
                    axes[f"/args/{index}" + path[len(source):]] = axes.pop(path)
    # GeneratedPrefix flattens keyword tensors in repr-sorted key order;
    # capture must assign input IDs in that same order.
    canonical_kwargs = {key: canonical_kwargs[key] for key in sorted(canonical_kwargs, key=repr)}
    return args, canonical_kwargs, axes


def runtime_input_batch_size(runtime: Any, inputs: tuple[Any, ...], input_kwargs: Mapping[str, Any] | None = None) -> int | None:
    """Read declared batch axes without mistaking parameters or image channels for B.

    TorchLens resolves these JSON Pointers during preparation. An empty axis
    map represents fixed input shapes and intentionally has no batch dimension.
    """
    inputs, input_kwargs, _ = normalize_model_call(runtime.model, inputs, input_kwargs)
    axes = runtime.batch_spec.axes
    if not axes:
        return None
    roots = {"args": inputs, "kwargs": input_kwargs or {}}
    sizes: dict[str, int] = {}
    for path, axis in axes.items():
        value: Any = roots
        try:
            for part in path.split("/")[1:]:
                key = part.replace("~1", "/").replace("~0", "~")
                if isinstance(value, Mapping):
                    value = value[key]
                elif isinstance(value, (tuple, list)):
                    value = value[int(key)]
                else:
                    value = getattr(value, key)
            sizes[path] = int(value.shape[axis])
        except (KeyError, IndexError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid declared batch input {path!r} at axis {axis}.") from exc
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Declared batch inputs disagree: {sizes!r}.")
    batch_size = next(iter(sizes.values()))
    if batch_size < 1:
        raise ValueError("Input batch size must be positive.")
    return batch_size


def _point(boundary: Any):
    if isinstance(boundary, SplitPoint):
        return boundary
    if not isinstance(boundary, str):
        raise TypeError("A split boundary must be a string or SplitPoint.")
    text = boundary.strip()
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
    trainable: bool = True,
    backend: str = "torch",
    batch_axes: dict[str, int] | None = None,
) -> SplitRequest:
    return SplitRequest(
        point=_point(boundary),
        backend=str(backend),
        features=SplitFeatures(replay=True, training=bool(trainable), batch_axes=batch_axes),
        validation="strict",
        batch_symbol=batch_symbol,
    )


def prepare_split_runtime(
    model: Any,
    example_inputs: Any,
    split_spec_or_boundary: Any,
    *,
    input_kwargs: dict[str, Any] | None = None,
) -> SplitRuntime:
    """Prepare with the supported public API, without mutating TorchLens internals."""
    require_torchlens_version()
    backend = detect_torchlens_backend(model, (example_inputs, input_kwargs or {}))
    request = (
        split_spec_or_boundary
        if isinstance(split_spec_or_boundary, SplitRequest)
        else make_split_spec(split_spec_or_boundary, backend=backend)
    )
    inputs, input_kwargs, axes = normalize_model_call(
        model, normalize_example_inputs(example_inputs), input_kwargs, request.features.batch_axes,
    )
    if axes != request.features.batch_axes:
        request = replace(request, features=replace(request.features, batch_axes=axes))
    # Native capture leaves persistent forward closures on submodules. Those
    # closures are not rebound by deepcopy, which is needed for federated
    # replicas. Use the public cleanup API while retaining user-owned tl_ attrs
    # (release_model also removes attributes with that historical prefix).
    user_attributes = [
        (module, {name: value for name, value in vars(module).items() if name.startswith("tl_")})
        for module in model.modules()
    ] if isinstance(model, torch.nn.Module) else []
    try:
        return prepare(
            model, inputs, request,
            input_kwargs=input_kwargs,
        )
    finally:
        if isinstance(model, torch.nn.Module):
            try:
                release_model(model)
            finally:
                for module, attributes in user_attributes:
                    for name, value in attributes.items():
                        setattr(module, name, value)


def repartition_split_runtime(runtime: SplitRuntime, request: SplitRequest) -> SplitRuntime:
    """Reuse capture, batch contract, placement and live state through ``at``."""
    if replace(request, point=runtime.request.point) != runtime.request:
        raise ValueError("Repartition may change only the split point; prepare a new runtime for other request changes.")
    return runtime.at(request.point)


def trace_signature(runtime: SplitRuntime) -> str:
    if runtime.graph_ir is None:
        raise RuntimeError("TorchLens runtime has no captured graph IR.")
    return runtime.graph_ir.graph_hash


__all__ = ["SplitRuntime", "SplitRequest",
    "TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION", "make_split_spec", "normalize_example_inputs", "normalize_model_call",
    "prepare_split_runtime", "repartition_split_runtime",
    "require_torchlens_version", "runtime_input_batch_size", "torchlens_runtime_version", "trace_signature"]
