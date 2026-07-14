from splitfleet.backends.base import BackendAdapter, ParameterEnvelope, StateEntry, StateManifest
from splitfleet.backends.torch_backend import TorchBackendAdapter
from splitfleet.backends.registry import BACKEND_ADAPTERS, BackendAdapterRegistry
from splitfleet.backends.array_backends import (
    JaxBackendAdapter, PaddleBackendAdapter, TensorFlowBackendAdapter, TinygradBackendAdapter,
)

BACKEND_ADAPTERS.register("torch", TorchBackendAdapter)
BACKEND_ADAPTERS.register("tf", TensorFlowBackendAdapter)
BACKEND_ADAPTERS.register("tensorflow", TensorFlowBackendAdapter)
BACKEND_ADAPTERS.register("jax", JaxBackendAdapter)
BACKEND_ADAPTERS.register("paddle", PaddleBackendAdapter)
BACKEND_ADAPTERS.register("tinygrad", TinygradBackendAdapter)

__all__ = [
    "BACKEND_ADAPTERS", "BackendAdapter", "BackendAdapterRegistry", "ParameterEnvelope",
    "StateEntry", "StateManifest", "TorchBackendAdapter", "TensorFlowBackendAdapter",
    "JaxBackendAdapter", "PaddleBackendAdapter", "TinygradBackendAdapter",
]
