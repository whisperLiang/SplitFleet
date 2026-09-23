from splitfleet.backends.base import BackendAdapter, ParameterEnvelope, StateEntry, StateManifest
from splitfleet.backends.torch_backend import TorchBackendAdapter
from splitfleet.backends.registry import BACKEND_ADAPTERS, BackendAdapterRegistry, BackendAvailability
from splitfleet.backends.array_backends import (
    JaxBackendAdapter, PaddleBackendAdapter, TensorFlowBackendAdapter, TinygradBackendAdapter,
)

BACKEND_ADAPTERS.register("torch", TorchBackendAdapter, required_modules=("torch",))
BACKEND_ADAPTERS.register("tf", TensorFlowBackendAdapter, required_modules=("tensorflow",), install_extra="tensorflow")
BACKEND_ADAPTERS.register("tensorflow", TensorFlowBackendAdapter, required_modules=("tensorflow",), install_extra="tensorflow")
BACKEND_ADAPTERS.register("jax", JaxBackendAdapter, required_modules=("jax",), install_extra="jax")
BACKEND_ADAPTERS.register("paddle", PaddleBackendAdapter, required_modules=("paddle",), install_extra="paddle")
BACKEND_ADAPTERS.register("tinygrad", TinygradBackendAdapter, required_modules=("tinygrad",), install_extra="tinygrad")

__all__ = [
    "BACKEND_ADAPTERS", "BackendAdapter", "BackendAdapterRegistry", "BackendAvailability", "ParameterEnvelope",
    "StateEntry", "StateManifest", "TorchBackendAdapter", "TensorFlowBackendAdapter",
    "JaxBackendAdapter", "PaddleBackendAdapter", "TinygradBackendAdapter",
]
