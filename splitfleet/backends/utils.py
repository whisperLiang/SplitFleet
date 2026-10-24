"""Backend-neutral helpers used by the Flower split-learning path."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any

import numpy as np

from splitfleet.backends.registry import BACKEND_ADAPTERS


def paddle_tensor_device(value: Any) -> str:
    """Read the actual native place without creating a CUDA tensor or context."""
    place = value.place
    if place.is_gpu_place():
        return f"cuda:{place.gpu_device_id()}"
    if place.is_cpu_place():
        return "cpu"
    raise ValueError(f"Unsupported native Paddle place {place}")


def paddle_model_device(model: Any, inputs: Any = None) -> str:
    """Follow native model state, or actual inputs for a stateless callable."""
    state_dict = getattr(model, "state_dict", None)
    variables = list(state_dict().values()) if callable(state_dict) else []
    for name in ("state_tensors", "canonical_variables"):
        declaration = getattr(model, name, None)
        if callable(declaration):
            variables.extend(declaration().values())
    if variables:
        places = {paddle_tensor_device(value) for value in variables}
        if len(places) != 1:
            raise ValueError("A Paddle split runtime requires model state on one native device")
        return places.pop()
    places = {paddle_tensor_device(value) for value in _iter_tensors(inputs)}
    if len(places) > 1:
        raise ValueError("A stateless Paddle runtime requires tensor inputs on one actual native device")
    if places:
        return places.pop()
    # Cloning/loading an empty state has no input tensors to establish a place.
    import paddle
    return paddle.get_device().replace("gpu:", "cuda:")


def _tensorflow_native_device(value: Any) -> str:
    underlying = getattr(value, "value", None)
    if underlying is not None and not callable(underlying):
        value = underlying
    device = getattr(value, "device", None)
    if not isinstance(device, str) or not device:
        raise ValueError("TensorFlow state/input has no actual native device")
    return device


def _tensorflow_device_alias(device: str) -> str:
    import tensorflow as tf
    specification = tf.DeviceSpec.from_string(device)
    if specification.device_index is None:
        raise ValueError(f"TensorFlow native device has no explicit index: {device}")
    if specification.device_type == "GPU":
        return f"cuda:{specification.device_index}"
    if specification.device_type == "CPU":
        return "cpu" if specification.device_index == 0 else f"cpu:{specification.device_index}"
    raise ValueError(f"Unsupported native TensorFlow device {device}")


def tensorflow_tensor_device(value: Any) -> str:
    """Read a real TensorFlow tensor or variable's public native device."""
    return _tensorflow_device_alias(_tensorflow_native_device(value))


def tensorflow_model_device(model: Any, inputs: Any = None) -> str:
    """Require one actual place across all TF state, or stateless tensor inputs.

    Keras exposes ResourceVariables through ``value``; tf.Module exposes them
    directly. Include canonical hooks as well as the public variables list so
    buffers and counters cannot be silently omitted from the residency proof.
    """
    variables = list(getattr(model, "variables", ()) or ())
    canonical = getattr(model, "canonical_variables", None)
    if callable(canonical):
        variables.extend(canonical().values())
    if variables:
        # Inspect every actual variable, but parse each distinct device string
        # once. The quality native SGD path calls this for every real batch.
        places = {_tensorflow_device_alias(place) for place in
                  {_tensorflow_native_device(value) for value in variables}}
        if len(places) != 1:
            raise ValueError("A TensorFlow split runtime requires all native model state on one device")
        return places.pop()
    places = {_tensorflow_device_alias(place) for place in
              {_tensorflow_native_device(value) for value in _iter_tensors(inputs)}}
    if len(places) != 1:
        raise ValueError("A stateless TensorFlow runtime requires tensor inputs on one actual native device")
    return places.pop()


def _tensorflow_scope_device(device: Any) -> str:
    """Translate logical SplitFleet aliases while retaining an explicit index."""
    requested = str(device)
    alias = requested.lower()
    if alias == "cpu":
        return "/CPU:0"
    if alias.startswith("cpu:"):
        return f"/CPU:{int(alias.split(':', 1)[1])}"
    if alias.startswith("cuda:"):
        return f"/GPU:{int(alias.split(':', 1)[1])}"
    if requested.startswith("/"):
        import tensorflow as tf
        specification = tf.DeviceSpec.from_string(requested)
        if specification.device_type in {"CPU", "GPU"} and specification.device_index is not None:
            return requested
    raise ValueError(f"Unsupported explicit TensorFlow scope device {device!r}")


@contextmanager
def native_device_scope(backend: str, device: Any):
    """Bind the native kernel context in the thread doing the work.

    Paddle's process-wide default place stays stable while same-device native
    operations overlap. Different-device operations wait for the current group.
    Never hold the scope across an RPC.
    TensorFlow's public scope is thread-local; generated raw-op replay needs
    this scope even when its captured inputs and ResourceVariables are on CPU.
    """
    if backend in {"tf", "tensorflow"}:
        import tensorflow as tf
        with tf.device(_tensorflow_scope_device(device)):
            yield
        return
    if backend != "paddle":
        yield
        return
    requested = str(device).lower().removeprefix("/").replace("cuda:", "gpu:")
    if requested == "cpu:0":
        requested = "cpu"
    from splitfleet.runtime.paddle_device_scope import paddle_device_scope
    with paddle_device_scope(requested):
        yield


def native_device_method(method=None, *, capture=False):
    """Scope a synchronous native operation with its model's declared device."""
    if method is None:
        return lambda wrapped: native_device_method(wrapped, capture=capture)
    @wraps(method)
    def scoped(self, *args, **kwargs):
        adapter = getattr(self, "backend_adapter", None)
        backend = getattr(adapter, "backend_name", getattr(self, "framework_backend", ""))
        if backend in {"tf", "tensorflow"}:
            from splitfleet.runtime.tensorflow_capture_scope import tensorflow_capture_scope, tensorflow_execution_scope
            execution = tensorflow_capture_scope() if capture else tensorflow_execution_scope()
        elif backend == 'paddle':
            from splitfleet.runtime.paddle_device_scope import paddle_execution_scope
            execution = paddle_execution_scope(capture=capture)
        else:
            execution = nullcontext()
        with execution, native_device_scope(backend, self.device):
            return method(self, *args, **kwargs)
    return scoped


def _iter_tensors(value: Any):
    module = type(value).__module__
    if (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and module.startswith(("torch", "tensorflow", "paddle"))
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
            if module.startswith("paddle"):
                return "paddle"
    raise TypeError(
        "Cannot infer a TorchLens split backend; supported backends are "
        "torch, tf, and paddle."
    )


def adapter_for(model: Any, sample_inputs: Any):
    adapter = BACKEND_ADAPTERS.create(detect_torchlens_backend(model, sample_inputs))
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


def bind_model_inputs(value: Any, adapter: Any, *, model: Any = None) -> Any:
    """Return model inputs after native framework placement has been applied."""
    _ = adapter, model
    return value


def inference_context(backend: str):
    if backend == "torch":
        import torch
        return torch.no_grad()
    return nullcontext()


def zero_grad(model: Any, optimizer: Any = None) -> None:
    method = getattr(model, "zero_grad", None)
    if callable(method):
        method()
    method = getattr(optimizer, "zero_grad", None)
    if callable(method):
        method()


def model_training(model: Any) -> bool:
    return bool(getattr(model, "training", False))
