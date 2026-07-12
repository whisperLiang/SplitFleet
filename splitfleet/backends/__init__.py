from splitfleet.backends.base import BackendAdapter, ParameterEnvelope, StateEntry, StateManifest
from splitfleet.backends.torch_backend import TorchBackendAdapter
from splitfleet.backends.registry import BACKEND_ADAPTERS, BackendAdapterRegistry

BACKEND_ADAPTERS.register("torch", TorchBackendAdapter)

__all__ = [
    "BACKEND_ADAPTERS", "BackendAdapter", "BackendAdapterRegistry", "ParameterEnvelope",
    "StateEntry", "StateManifest", "TorchBackendAdapter",
]
