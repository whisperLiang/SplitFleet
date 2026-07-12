"""Model state and tensor operations owned by a framework backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from splitfleet.transport import TensorEnvelope


@dataclass(frozen=True)
class StateEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class StateManifest:
    backend: str
    entries: tuple[StateEntry, ...]
    schema_hash: str


@dataclass(frozen=True)
class ParameterEnvelope:
    backend: str
    schema_hash: str
    tensors: tuple[TensorEnvelope, ...]
    metadata: Mapping[str, Any]


@runtime_checkable
class BackendAdapter(Protocol):
    backend_name: str

    def clone_model(self, model: Any) -> Any: ...
    def move_model(self, model: Any, device: Any) -> Any: ...
    def set_training(self, model: Any, training: bool) -> None: ...
    def state_manifest(self, model: Any) -> StateManifest: ...
    def export_state(self, model: Any) -> ParameterEnvelope: ...
    def load_state(self, model: Any, state: ParameterEnvelope) -> None: ...
    def encode_tensor(self, tensor_id: str, tensor: Any) -> TensorEnvelope: ...
    def decode_tensor(self, envelope: TensorEnvelope, device: Any) -> Any: ...
    def build_optimizer(self, model: Any, config: Mapping[str, Any]) -> Any: ...
    def scalar_value(self, value: Any) -> float: ...
