"""Backend-neutral helpers used by the Flower split-learning path."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np

from splitfleet.backends.registry import BACKEND_ADAPTERS


def _iter_tensors(value: Any):
    module = type(value).__module__
    if (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and module.startswith(("torch", "tensorflow", "jax", "jaxlib", "paddle", "tinygrad"))
    ):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def detect_torchlens_backend(model: Any, example_inputs: Any) -> str:
    """Resolve a framework backend without importing optional frameworks."""
    candidates = [model, *_iter_tensors(example_inputs)]
    for value in candidates:
        value_type = type(value)
        modules = {
            cls.__module__.lower()
            for cls in getattr(value_type, "__mro__", (value_type,))
        }
        for module in modules:
            if module.startswith("torch"):
                return "torch"
            if module.startswith(("tensorflow", "keras")):
                return "tf"
            if module.startswith(("jax", "jaxlib", "flax", "haiku")):
                return "jax"
            if module.startswith("paddle"):
                return "paddle"
            if module.startswith("tinygrad"):
                return "tinygrad"
    raise TypeError(
        "Cannot infer a TorchLens split backend; supported backends are "
        "torch, tf, jax, paddle, and tinygrad."
    )


def adapter_for(model: Any, sample_inputs: Any):
    adapter = BACKEND_ADAPTERS.create(detect_torchlens_backend(model, sample_inputs))
    if adapter.backend_name == "jax" and not hasattr(model, "params"):
        values = sample_inputs if isinstance(sample_inputs, (tuple, list)) else (sample_inputs,)
        if not values:
            raise ValueError("Functional JAX models require params as the first sample input.")
        adapter.bind_external_params(values[0])
    return adapter


def move_value(value: Any, adapter: Any, device: Any) -> Any:
    if isinstance(value, np.ndarray):
        if adapter.backend_name == "torch":
            import torch
            return torch.as_tensor(value, device=device)
        return adapter._from_numpy(value, device=device)
    if isinstance(value, dict):
        return {key: move_value(item, adapter, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_value(item, adapter, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_value(item, adapter, device) for item in value)
    to = getattr(value, "to", None)
    if callable(to) and adapter.backend_name == "torch":
        return to(device)
    return value


def bind_model_inputs(value: Any, adapter: Any) -> Any:
    """Replace a functional JAX call's params argument with synchronized state."""
    if adapter.backend_name != "jax" or not getattr(adapter, "has_external_params", False):
        return value
    if isinstance(value, tuple) and value:
        return (adapter.external_params, *value[1:])
    if isinstance(value, list) and value:
        return [adapter.external_params, *value[1:]]
    raise ValueError("Functional JAX inputs must be a tuple/list whose first item is params.")


def inference_context(backend: str):
    if backend == "torch":
        import torch
        return torch.no_grad()
    return nullcontext()


def zero_grad(model: Any, optimizer: Any = None) -> None:
    method = getattr(model, "zero_grad", None)
    if callable(method):
        try: method(set_to_none=True)
        except TypeError: method()
    method = getattr(optimizer, "zero_grad", None)
    if callable(method):
        try: method(set_to_none=True)
        except TypeError: method()


def model_training(model: Any) -> bool:
    return bool(getattr(model, "training", False))
