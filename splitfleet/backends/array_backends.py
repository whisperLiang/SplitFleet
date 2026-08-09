"""Optional TorchLens framework adapters with NumPy-based wire encoding."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from typing import Any, Mapping

import numpy as np

from splitfleet.backends.base import ParameterEnvelope, StateEntry, StateManifest
from splitfleet.transport import TensorEnvelope


class ArrayBackendAdapter:
    backend_name = "array"

    def _to_numpy(self, value: Any) -> np.ndarray:
        raise NotImplementedError

    def _from_numpy(self, value: np.ndarray, *, device: Any = None) -> Any:
        raise NotImplementedError

    def _state(self, model: Any) -> Mapping[str, Any]:
        raise NotImplementedError

    def _load_arrays(self, model: Any, values: Mapping[str, np.ndarray]) -> None:
        raise NotImplementedError

    def clone_model(self, model: Any) -> Any:
        return copy.deepcopy(model)

    def move_model(self, model: Any, device: Any) -> Any:
        return model

    def set_training(self, model: Any, training: bool) -> None:
        setattr(model, "training", bool(training))

    def state_manifest(self, model: Any) -> StateManifest:
        entries = tuple(
            StateEntry(str(name), tuple(int(x) for x in self._to_numpy(value).shape), str(self._to_numpy(value).dtype))
            for name, value in self._state(model).items()
        )
        raw = json.dumps([entry.__dict__ for entry in entries], sort_keys=True).encode()
        return StateManifest(self.backend_name, entries, hashlib.sha256(raw).hexdigest())

    def tied_state_groups(self, model: Any) -> tuple[tuple[int, ...], ...]:
        positions: dict[int, list[int]] = {}
        for index, value in enumerate(self._state(model).values()):
            positions.setdefault(id(value), []).append(index)
        return tuple(tuple(group) for group in positions.values() if len(group) > 1)

    def encode_tensor(self, tensor_id: str, tensor: Any) -> TensorEnvelope:
        value = np.ascontiguousarray(self._to_numpy(tensor))
        requires_grad = bool(
            getattr(tensor, "requires_grad", False)
            or getattr(tensor, "trainable", False)
            or (hasattr(tensor, "stop_gradient") and not bool(tensor.stop_gradient))
        )
        return TensorEnvelope(
            str(tensor_id), tuple(value.shape), str(value.dtype), value.tobytes(),
            requires_grad=requires_grad,
        )

    def decode_tensor(self, envelope: TensorEnvelope, device: Any = None) -> Any:
        value = np.frombuffer(envelope.payload, dtype=np.dtype(envelope.dtype)).reshape(envelope.shape).copy()
        tensor = self._from_numpy(value, device=device)
        if envelope.requires_grad:
            method = getattr(tensor, "requires_grad_", None)
            if callable(method):
                method(True)
            elif hasattr(tensor, "stop_gradient"):
                tensor.stop_gradient = False
            else:
                tensor.requires_grad = True
        return tensor

    def export_state(self, model: Any) -> ParameterEnvelope:
        manifest = self.state_manifest(model)
        tensors = tuple(self.encode_tensor(name, value) for name, value in self._state(model).items())
        return ParameterEnvelope(self.backend_name, manifest.schema_hash, tensors, {})

    def load_state(self, model: Any, state: ParameterEnvelope) -> None:
        if state.backend != self.backend_name or state.schema_hash != self.state_manifest(model).schema_hash:
            raise ValueError("Model state backend or schema mismatch")
        self._load_arrays(model, {item.tensor_id: np.frombuffer(item.payload, dtype=np.dtype(item.dtype)).reshape(item.shape).copy() for item in state.tensors})

    def build_optimizer(self, model: Any, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError(f"{self.backend_name} optimizer construction requires a framework-specific factory")

    def scalar_value(self, value: Any) -> float:
        return float(np.asarray(self._to_numpy(value)).item())

    def export_ndarrays(self, model: Any) -> list[np.ndarray]:
        return [self._to_numpy(value).copy() for value in self._state(model).values()]

    def load_ndarrays(self, model: Any, values: list[np.ndarray]) -> None:
        names = list(self._state(model))
        if len(names) != len(values):
            raise ValueError(f"Expected {len(names)} state tensors, received {len(values)}")
        self._load_arrays(model, dict(zip(names, values)))


class TensorFlowBackendAdapter(ArrayBackendAdapter):
    backend_name = "tf"
    def _to_numpy(self, value): return np.asarray(value.numpy())
    def _from_numpy(self, value, *, device=None):
        import tensorflow as tf
        with tf.device(device or "/CPU:0"): return tf.convert_to_tensor(value)
    @staticmethod
    def _variable_key(index, variable):
        name = getattr(variable, "path", None) or getattr(variable, "name", None) or "weight"
        return f"{index}:{name}"
    def _state(self, model):
        return {
            self._variable_key(index, variable): variable
            for index, variable in enumerate(model.weights)
        }
    def _load_arrays(self, model, values):
        for index, variable in enumerate(model.weights):
            variable.assign(values[self._variable_key(index, variable)])
    def set_training(self, model, training):
        training = bool(training)
        setattr(model, "training", training)
        setattr(model, "_splitfleet_training", training)
        if getattr(model, "_splitfleet_training_call_installed", False):
            return

        original_call = model.call
        try:
            parameters = inspect.signature(original_call).parameters.values()
        except (TypeError, ValueError):
            return
        supports_training = any(
            parameter.name == "training" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if not supports_training:
            return

        def splitfleet_call(*args, **kwargs):
            kwargs.setdefault("training", bool(getattr(model, "_splitfleet_training", False)))
            return original_call(*args, **kwargs)

        model.call = splitfleet_call
        setattr(model, "_splitfleet_training_call_installed", True)
    def build_optimizer(self, model, config):
        import tensorflow as tf
        name = str(config.get("name", "sgd")).lower()
        lr = float(config.get("lr", config.get("learning_rate", 0.01)))
        choices = {"sgd": tf.keras.optimizers.SGD, "adam": tf.keras.optimizers.Adam}
        if name not in choices: raise ValueError(f"Unsupported TensorFlow optimizer {name!r}")
        return choices[name](learning_rate=lr)


class PaddleBackendAdapter(ArrayBackendAdapter):
    backend_name = "paddle"
    def _to_numpy(self, value): return np.asarray(value.numpy())
    def _from_numpy(self, value, *, device=None):
        import paddle
        return paddle.to_tensor(value, place=device)
    def _state(self, model): return model.state_dict()
    def _load_arrays(self, model, values):
        model.set_state_dict({name: self._from_numpy(value) for name, value in values.items()})
    def set_training(self, model, training): model.train() if training else model.eval()
    def build_optimizer(self, model, config):
        import paddle
        name = str(config.get("name", "sgd")).lower()
        lr = float(config.get("lr", config.get("learning_rate", 0.01)))
        choices = {"sgd": paddle.optimizer.SGD, "adam": paddle.optimizer.Adam}
        if name not in choices: raise ValueError(f"Unsupported Paddle optimizer {name!r}")
        return choices[name](learning_rate=lr, parameters=model.parameters())


class JaxBackendAdapter(ArrayBackendAdapter):
    backend_name = "jax"
    def __init__(self):
        self._external_params = None
    @property
    def has_external_params(self):
        return self._external_params is not None
    @property
    def external_params(self):
        if self._external_params is None:
            raise AttributeError("No external JAX parameter tree has been bound")
        return self._external_params
    def bind_external_params(self, params):
        current = self._external_params
        if isinstance(current, dict) and isinstance(params, Mapping):
            current.clear()
            current.update(params)
        elif isinstance(current, list) and isinstance(params, (list, tuple)):
            current[:] = params
        else:
            self._external_params = params
    def _params(self, model):
        if hasattr(model, "params"):
            return model.params
        return self.external_params
    def _to_numpy(self, value): return np.asarray(value)
    def _from_numpy(self, value, *, device=None):
        import jax
        if isinstance(device, str):
            normalized = device.strip().lower().replace("/", "")
            platform, _, raw_index = normalized.partition(":")
            if platform == "cuda":
                platform = "gpu"
            devices = jax.devices(platform)
            device = devices[int(raw_index or 0)]
        return jax.device_put(value, device) if device is not None else jax.numpy.asarray(value)
    def _state(self, model):
        import jax
        pairs, _ = jax.tree_util.tree_flatten_with_path(self._params(model))
        return {jax.tree_util.keystr(path): value for path, value in pairs}
    def _load_arrays(self, model, values):
        import jax
        _, treedef = jax.tree_util.tree_flatten(self._params(model))
        names = list(self._state(model))
        params = jax.tree_util.tree_unflatten(treedef, [self._from_numpy(values[name]) for name in names])
        setter = getattr(model, "set_params", None)
        if callable(setter): setter(params)
        elif hasattr(model, "params"): model.params = params
        else: self.bind_external_params(params)


class TinygradBackendAdapter(ArrayBackendAdapter):
    backend_name = "tinygrad"
    def _to_numpy(self, value): return np.asarray(value.numpy())
    def _from_numpy(self, value, *, device=None):
        from tinygrad import Tensor
        return Tensor(value, device=device)
    def _state(self, model):
        from tinygrad.nn.state import get_state_dict
        return get_state_dict(model)
    def _load_arrays(self, model, values):
        for name, tensor in self._state(model).items(): tensor.assign(self._from_numpy(values[name], device=tensor.device))
    def build_optimizer(self, model, config):
        from tinygrad import Tensor
        from tinygrad.nn.optim import Adam, SGD
        name = str(config.get("name", "sgd")).lower()
        lr = float(config.get("lr", 0.01))
        params = [value for value in self._state(model).values() if getattr(value, "requires_grad", False)]
        if name == "sgd":
            optimizer = SGD(params, lr=lr)
        elif name == "adam":
            optimizer = Adam(params, lr=lr)
        else:
            raise ValueError(f"Unsupported tinygrad optimizer {name!r}")

        class PartialGradientOptimizer:
            """Allow a full-model tinygrad optimizer to step one split segment."""

            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.params = wrapped.params

            def zero_grad(self):
                return self.wrapped.zero_grad()

            def step(self):
                active = [parameter for parameter in self.params if parameter.grad is not None]
                if not active:
                    return None
                missing = [parameter for parameter in self.params if parameter.grad is None]
                for parameter in missing:
                    parameter.grad = Tensor.zeros_like(parameter)
                try:
                    return self.wrapped.step()
                finally:
                    for parameter in missing:
                        parameter.grad = None

            def __getattr__(self, attribute):
                return getattr(self.wrapped, attribute)

        return PartialGradientOptimizer(optimizer)
