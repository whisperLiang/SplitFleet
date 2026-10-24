"""Native framework operations for the shared Flower virtual runtime.

Wire state uses each adapter's native schema.  Experiment-specific canonical
layouts and initialization hashes are handled by the model factory.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import os
import pickle
import random
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from splitfleet.backends import BACKEND_ADAPTERS
from splitfleet.backends.utils import adapter_for, bind_model_inputs, move_value


def isolate_unused_tensorflow():
    """Keep Flower's unused optional TensorFlow out of a native worker.

    Flower's Ray simulation backend imports TensorFlow for its optional
    GPU-growth helper.  Its
    exported LLVM symbols conflict with Paddle 3.3's CINN library at load time.
    A process dedicated to another backend does not use that helper; Python's explicit
    absent-module sentinel lets Flower take its documented optional-import
    branch without changing either installed framework.
    """
    if sys.modules.get("tensorflow") is not None:
        raise RuntimeError("Native backend isolation requires a fresh process without TensorFlow loaded")
    sys.modules["tensorflow"] = None


class FrameworkOps:
    def __init__(self, backend="torch"):
        backend = {"pytorch": "torch"}.get(backend, backend)
        self.adapter = BACKEND_ADAPTERS.create(backend)
        self.backend = self.adapter.backend_name

    @classmethod
    def for_spec(cls, spec):
        return cls(getattr(spec, "framework_backend", "torch"))

    @classmethod
    def for_model(cls, model, inputs=()):
        if inputs is None or isinstance(inputs, (tuple, list, dict)) and not inputs:
            inputs = (model.params,) if hasattr(model, "params") else ()
            state = getattr(model, "state_tensors", None)
            if not inputs and callable(state):
                inputs = (state(),)
        adapter = adapter_for(model, inputs)
        result = cls(adapter.backend_name)
        result.adapter = adapter
        return result

    def build_model(self, factory, device):
        parameters = inspect.signature(factory).parameters
        model = factory(device=device) if "device" in parameters else factory()
        model = self.adapter.move_model(model, device)
        self.assert_model_device(model, device)
        return model

    def prepare_process(self, *, isolate_unused_frameworks=False):
        if self.backend == "paddle" or (isolate_unused_frameworks and self.backend != "tf"):
            isolate_unused_tensorflow()

    def names(self, model):
        return list(self.native_state(model))

    def native_state(self, model):
        return model.state_dict() if self.backend == "torch" else self.adapter._state(model)

    def export_arrays(self, model):
        return self.adapter.export_ndarrays(model)

    def load_arrays(self, model, arrays):
        self.adapter.load_ndarrays(model, arrays)

    def hash_arrays(self, model, arrays):
        names = self.names(model)
        if len(names) != len(arrays):
            raise ValueError("Native state hash tensor count differs from the model")
        state = dict(zip(names, arrays, strict=True))
        if self.backend == "torch":
            import torch

            from splitfleet.common.model_state import tensor_state_hash
            return tensor_state_hash({name: torch.from_numpy(value) for name, value in state.items()})
        from splitfleet.common.model_state import native_array_state_hash
        return native_array_state_hash(self.backend, names, arrays)

    def state_hash(self, model):
        return self.hash_arrays(model, self.export_arrays(model))

    def state_bytes(self, model):
        return sum(value.nbytes for value in self.export_arrays(model))

    def environment_receipt(self, model, device):
        self.assert_model_device(model, device)
        module = importlib.import_module({"tf": "tensorflow"}.get(self.backend, self.backend))
        version = getattr(module, "__version__", None)
        if version is None:
            version = importlib.metadata.version(self.backend)
        return {"framework_backend": self.backend, "module_version": str(version),
                "device": str(device), "state_device_verified": True,
                "native_state_tensors": len(self.names(model)),
                "state_layout": "native_adapter_schema",
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "optional_framework_isolation": ["tensorflow"] if (
                    "tensorflow" in sys.modules and sys.modules["tensorflow"] is None) else [],
                "rng_scope": self.rng_scope_description}

    def trainable_names(self, model):
        if self.backend == "torch":
            return {name for name, value in model.named_parameters() if value.requires_grad}
        names = set()
        for name, value in self.native_state(model).items():
            active = (value.trainable if self.backend == "tf" else not value.stop_gradient
                      if self.backend == "paddle" else getattr(value, "requires_grad", False))
            if active:
                names.add(name)
        return names

    def set_training(self, model, training):
        self.adapter.set_training(model, training)
        method = getattr(model, "train", None)
        if self.backend == "torch" and callable(method):
            method(bool(training))

    def optimizer(self, model, name, learning_rate):
        if self.backend == "torch":
            import torch
            factory = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}.get(name)
            if factory is None:
                raise ValueError(f"Unsupported Torch optimizer {name!r}")
            return factory(model.parameters(), lr=learning_rate)
        if self.backend == "tf":
            from splitfleet.backends.utils import (
                native_device_scope,
                tensorflow_model_device,
            )
            with native_device_scope("tf", tensorflow_model_device(model)):
                optimizer = self.adapter.build_optimizer(model, {"name": name, "lr": learning_rate})
                optimizer.build(model.trainable_variables)
            return optimizer
        return self.adapter.build_optimizer(model, {"name": name, "lr": learning_rate})

    def prepare_batch(self, model, batch, device):
        call = batch.inputs.map_values(lambda value: move_value(value, self.adapter, device))
        call = replace(call, args=bind_model_inputs(call.args, self.adapter, model=model))
        return replace(batch, inputs=call, targets=move_value(batch.targets, self.adapter, device))

    @staticmethod
    def _device(device):
        platform, _, index = str(device).lower().removeprefix("/").partition(":")
        if platform in {"cuda", "gpu"}:
            platform = "cuda"
        elif platform != "cpu":
            raise ValueError(f"Unsupported virtual device {device!r}")
        index = int(index or 0)
        if index < 0 or (platform == "cpu" and index != 0):
            raise ValueError(f"Unsupported virtual device {device!r}")
        return platform, index

    def gpu_count(self):
        if self.backend == "torch":
            import torch
            return torch.cuda.device_count()
        if self.backend == "tf":
            import tensorflow as tf
            return len(tf.config.list_physical_devices("GPU"))
        if self.backend == "paddle":
            import paddle
            return paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0
        raise ValueError(f'Unsupported native GPU backend {self.backend}')

    def admit_device(self, device):
        platform, index = self._device(device)
        if platform == "cuda" and index >= self.gpu_count():
            raise RuntimeError(f"No native {self.backend} GPU is available for {device}")
        if self.backend == "torch" and platform == "cuda":
            import torch
            torch.cuda.set_device(index)
        elif self.backend == "paddle":
            import paddle
            paddle.set_device(f"gpu:{index}" if platform == "cuda" else "cpu")

    def assert_model_device(self, model, device):
        platform, index = self._device(device)
        self.admit_device(device)
        for name, value in self.native_state(model).items():
            if self.backend == "torch":
                actual = (value.device.type, value.device.index or 0)
            elif self.backend == "tf":
                import tensorflow as tf
                native = getattr(value, "value", value)
                native = value if callable(native) else native
                specification = tf.DeviceSpec.from_string(native.device)
                actual = ("cuda" if specification.device_type == "GPU" else "cpu", specification.device_index or 0)
            elif self.backend == "paddle":
                requested = f"Place(gpu:{index})" if platform == "cuda" else "Place(cpu)"
                if str(value.place) != requested:
                    raise RuntimeError(f"Native {self.backend} state {name!r} is on {value.place}, expected {requested}")
                continue
            else:
                native = str(value.device).upper()
                actual = ("cuda", int(native.partition(":")[2] or 0)) if native.startswith("CUDA") else ("cpu", 0)
            if actual != (platform, index):
                raise RuntimeError(f"Native {self.backend} state {name!r} is on {actual}, expected {(platform, index)}")

    def free_gpu_memory(self, device):
        if self.backend == "torch" and self._device(device)[0] == "cuda":
            import torch
            return torch.cuda.mem_get_info(device)[0]
        return None

    def reset_peak_memory(self, device):
        platform, index = self._device(device)
        if platform != "cuda":
            return
        if self.backend == "torch":
            import torch
            torch.cuda.reset_peak_memory_stats(device)
        elif self.backend == "tf":
            import tensorflow as tf
            tf.config.experimental.reset_memory_stats(f"GPU:{index}")
        elif self.backend == "paddle":
            import paddle
            paddle.device.cuda.reset_max_memory_allocated(index)

    def peak_gpu_memory_mb(self, device):
        platform, index = self._device(device)
        if platform != "cuda":
            return None
        if self.backend == "torch":
            import torch
            value = torch.cuda.max_memory_allocated(device)
        elif self.backend == "tf":
            import tensorflow as tf
            value = tf.config.experimental.get_memory_info(f"GPU:{index}")["peak"]
        elif self.backend == "paddle":
            import paddle
            value = paddle.device.cuda.max_memory_allocated(index)
        else:
            return None
        return value / 2**20

    def native_memory_receipt(self, device):
        """Return optional backend allocator details for a worker receipt.

        The virtual worker records this hook for every backend.  The current
        backends do not expose a comparable allocator receipt, so they
        deliberately return ``None`` while preserving the worker metrics
        contract.
        """
        _ = device
        return None

    @contextmanager
    def worker_resources(self, profile, *, mode):
        from .resources import worker_resources
        if self.backend == "torch":
            with worker_resources(profile, mode=mode) as receipt:
                yield receipt
            return
        self.admit_device(profile.device)
        with worker_resources(replace(profile, device="cpu"), mode=mode) as receipt:
            yield {**receipt, "framework_backend": self.backend,
                   "backend_thread_environment": {name: os.environ.get(name) for name in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}}

    def capture_rng(self, device="cpu"):
        if self.backend == "torch":
            from .state import capture_rng
            return capture_rng(device)
        return pickle.dumps(("splitfleet.host-rng.v1", random.getstate(), np.random.get_state()), protocol=5)

    @property
    def rng_scope_description(self):
        return "Torch/Python/NumPy RNG" if self.backend == "torch" else "Python/NumPy host RNG; native weights and batches are explicitly keyed"

    @contextmanager
    def rng_scope(self, *, seed, saved=None, device="cpu"):
        if self.backend == "torch":
            from .state import rng_scope
            with rng_scope(seed=seed, saved=saved, device=device):
                yield
            return
        previous = self.capture_rng()

        def restore(value):
            marker, python_rng, numpy_rng = pickle.loads(value)
            if marker != "splitfleet.host-rng.v1":
                raise ValueError("Host RNG state has an incompatible backend contract")
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

        try:
            if saved is None:
                random.seed(seed)
                np.random.seed(seed)
            else:
                restore(saved)
            yield
        finally:
            restore(previous)

    def save_checkpoint(self, model, arrays, path):
        path = Path(path)
        if len(arrays) != len(self.names(model)) or any(not np.isfinite(value).all() for value in arrays):
            raise FloatingPointError("Checkpoint state is incomplete or nonfinite")
        if self.backend == "torch":
            import torch
            state = {name: torch.from_numpy(value) for name, value in zip(self.names(model), arrays, strict=True)}
            torch.save(state, path.with_suffix(".pt"))
            return path.with_suffix(".pt")
        path = path.with_suffix(".npz")
        np.savez(path, **dict(zip(self.names(model), arrays, strict=True)))
        metadata = {"backend": self.backend, "layout": "native_adapter_schema",
                    "schema_hash": self.adapter.state_manifest(model).schema_hash,
                    "native_state_hash": self.hash_arrays(model, arrays)}
        path.with_suffix(".json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
        return path

    def load_checkpoint(self, model, path):
        path = Path(path)
        if self.backend == "torch":
            import torch
            state = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            return
        path = path.with_suffix(".npz")
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["backend"] != self.backend or metadata["schema_hash"] != self.adapter.state_manifest(model).schema_hash:
            raise ValueError("Native checkpoint backend or schema differs from the model")
        with np.load(path, allow_pickle=False) as record:
            if set(record.files) != set(self.names(model)):
                raise ValueError("Native checkpoint state names differ from the model")
            arrays = [record[name].copy() for name in self.names(model)]
        if self.hash_arrays(model, arrays) != metadata["native_state_hash"]:
            raise ValueError("Native checkpoint hash mismatch")
        self.load_arrays(model, arrays)
