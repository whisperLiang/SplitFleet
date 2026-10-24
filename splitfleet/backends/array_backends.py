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


def _declared_native_state(model: Any) -> Mapping[str, Any] | None:
    """Opt in to model-declared names while retaining native tensor layouts."""
    for name in ("state_tensors", "canonical_variables"):
        getter = getattr(model, name, None)
        if callable(getter):
            state = getter()
            if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
                raise TypeError(f"{name}() must return a mapping with string state names")
            return state
    return None


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
        entries = []
        for name, value in self._state(model).items():
            array = self._to_numpy(value)
            entries.append(StateEntry(str(name), tuple(int(x) for x in array.shape), str(array.dtype)))
        entries = tuple(entries)
        raw = json.dumps([entry.__dict__ for entry in entries], sort_keys=True).encode()
        return StateManifest(self.backend_name, entries, hashlib.sha256(raw).hexdigest())

    def tied_state_groups(self, model: Any) -> tuple[tuple[int, ...], ...]:
        positions: dict[int, list[int]] = {}
        for index, value in enumerate(self._state(model).values()):
            positions.setdefault(id(value), []).append(index)
        return tuple(tuple(group) for group in positions.values() if len(group) > 1)

    def encode_tensor(self, tensor_id: str, tensor: Any) -> TensorEnvelope:
        # ascontiguousarray promotes a scalar to shape (1,), which changes the
        # replay schema for scalar losses, counters and reduction boundaries.
        value = np.asarray(self._to_numpy(tensor))
        requires_grad = bool(
            getattr(tensor, "requires_grad", False)
            or getattr(tensor, "trainable", False)
            or (hasattr(tensor, "stop_gradient") and not bool(tensor.stop_gradient))
        )
        return TensorEnvelope(
            str(tensor_id), tuple(value.shape), str(value.dtype), value.tobytes(order="C"),
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
        if len({item.tensor_id for item in state.tensors}) != len(state.tensors):
            raise ValueError("Model state has duplicate tensor names")
        arrays = {
            item.tensor_id: np.frombuffer(item.payload, dtype=np.dtype(item.dtype)).reshape(item.shape).copy()
            for item in state.tensors
        }
        self._validate_arrays(model, arrays)
        self._load_arrays(model, arrays)

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
        arrays = dict(zip(names, values, strict=True))
        self._validate_arrays(model, arrays)
        self._load_arrays(model, arrays)

    def _validate_arrays(self, model: Any, values: Mapping[str, np.ndarray]) -> None:
        """Validate the complete native schema before modifying any variable."""
        current = self._state(model)
        if set(values) != set(current):
            raise ValueError("Model state tensor names do not match manifest")
        for name, reference in current.items():
            expected = self._to_numpy(reference)
            value = np.asarray(values[name])
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Model state shape or dtype mismatch for {name}")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite model state for {name}")


class TensorFlowBackendAdapter(ArrayBackendAdapter):
    backend_name = "tf"
    def _to_numpy(self, value): return np.asarray(value.numpy())
    def _from_numpy(self, value, *, device=None):
        import tensorflow as tf
        requested = device or "/CPU:0"
        if isinstance(requested, str):
            alias = requested.strip().lower()
            if alias == "cpu":
                requested = "/CPU:0"
            elif alias.startswith(("cuda:", "gpu:", "cpu:")):
                platform, raw_index = alias.split(":", 1)
                index = int(raw_index)
                if index < 0:
                    raise ValueError(f"Invalid TensorFlow device {device!r}")
                requested = f"/{'CPU' if platform == 'cpu' else 'GPU'}:{index}"
        specification = requested if isinstance(requested, tf.DeviceSpec) else tf.DeviceSpec.from_string(str(requested))
        if specification.device_type in {"GPU", "CPU"}:
            index = specification.device_index or 0
            available = tf.config.list_logical_devices(specification.device_type)
            if index < 0 or index >= len(available):
                raise RuntimeError(f"Requested TensorFlow device {requested!r} is unavailable")
        with tf.device(requested):
            tensor = tf.convert_to_tensor(value)
        actual = tf.DeviceSpec.from_string(tensor.device)
        if specification.device_type is not None and (
            actual.device_type != specification.device_type
            or actual.device_index != (specification.device_index or 0)
        ):
            raise RuntimeError(f"TensorFlow tensor could not be placed on requested device {requested!r}")
        return tensor
    @staticmethod
    def _variable_key(index, variable, model=None):
        name = getattr(variable, "path", None) or getattr(variable, "name", None) or "weight"
        model_name = getattr(model, "name", None)
        if model_name and name.startswith(f"{model_name}/"):
            name = name[len(model_name) + 1:]
        return f"{index}:{name}"
    def _state(self, model):
        declared = _declared_native_state(model)
        if declared is not None:
            return declared
        return {
            self._variable_key(index, variable, model): variable
            for index, variable in enumerate(model.weights)
        }
    def _load_arrays(self, model, values):
        declared = _declared_native_state(model)
        if declared is not None:
            for name, variable in declared.items():
                variable.assign(values[name])
            return
        for index, variable in enumerate(model.weights):
            variable.assign(values[self._variable_key(index, variable, model)])
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
        if name not in choices:
            raise ValueError(f"Unsupported TensorFlow optimizer {name!r}")
        from splitfleet.runtime.tensorflow_updates import native_optimizer
        return native_optimizer(choices[name](learning_rate=lr), model)


class PaddleBackendAdapter(ArrayBackendAdapter):
    backend_name = "paddle"
    def _to_numpy(self, value):
        from splitfleet.backends.utils import native_device_scope, paddle_tensor_device
        with native_device_scope("paddle", paddle_tensor_device(value)):
            return np.asarray(value.numpy())
    def clone_model(self, model):
        from splitfleet.backends.utils import native_device_scope, paddle_model_device
        with native_device_scope("paddle", paddle_model_device(model)):
            return copy.deepcopy(model)
    def _from_numpy(self, value, *, device=None):
        import paddle
        from splitfleet.backends.utils import native_device_scope
        place = device
        requested = paddle.get_device()
        if isinstance(device, str):
            alias = device.strip().lower().removeprefix("/")
            if alias in {"cpu", "cpu:0"}:
                place = paddle.CPUPlace()
                requested = "cpu"
            elif alias.startswith(("cuda:", "gpu:")):
                index = int(alias.split(":", 1)[1])
                if index < 0:
                    raise ValueError(f"Invalid Paddle device {device!r}")
                if not paddle.is_compiled_with_cuda() or index >= paddle.device.cuda.device_count():
                    raise RuntimeError(f"Requested Paddle device {device!r} is unavailable")
                place = paddle.CUDAPlace(index)
                requested = f"cuda:{index}"
            else:
                raise ValueError(f"Unsupported Paddle device {device!r}")
        elif place is not None:
            requested = "cpu" if str(place) == "Place(cpu)" else f"cuda:{place.get_device_id()}"
        with native_device_scope("paddle", requested):
            tensor = paddle.to_tensor(value, place=place)
        if place is not None and str(tensor.place) != str(place):
            raise RuntimeError(f"Paddle tensor could not be placed on requested device {device!r}")
        return tensor
    def _state(self, model):
        declared = _declared_native_state(model)
        return model.state_dict() if declared is None else declared
    def _load_arrays(self, model, values):
        from splitfleet.backends.utils import native_device_scope, paddle_model_device
        with native_device_scope("paddle", paddle_model_device(model)):
            self._load_arrays_on_device(model, values)
    def _load_arrays_on_device(self, model, values):
        declared = _declared_native_state(model)
        if declared is not None:
            loader = getattr(model, "load_state", None)
            if callable(loader):
                loader(values)
            else:
                for name, variable in declared.items():
                    variable.set_value(values[name])
            return
        model.set_state_dict({name: self._from_numpy(value) for name, value in values.items()})
    def set_training(self, model, training): model.train() if training else model.eval()
    def build_optimizer(self, model, config):
        import paddle
        name = str(config.get("name", "sgd")).lower()
        lr = float(config.get("lr", config.get("learning_rate", 0.01)))
        choices = {"sgd": paddle.optimizer.SGD, "adam": paddle.optimizer.Adam}
        if name not in choices:
            raise ValueError(f"Unsupported Paddle optimizer {name!r}")
        return choices[name](learning_rate=lr, parameters=model.parameters())
