"""Framework-neutral split engine protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from splitfleet.split_engine.contracts import GraphContract
from splitfleet.transport import BoundaryEnvelope, GradientEnvelope


@dataclass
class SplitRuntimeHandle:
    runtime: Any
    backend: str
    engine: str = "torchlens"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefixContextToken:
    round_id: int
    client_id: str
    step_id: str


@dataclass
class SuffixResult:
    outputs: Any
    gradients: GradientEnvelope | None = None
    loss: float | None = None
    num_examples: int = 0


@runtime_checkable
class SplitEngine(Protocol):
    engine_name: str

    def prepare(self, model: Any, sample_inputs: Any, request: Any, *, sample_kwargs: dict[str, Any] | None = None) -> SplitRuntimeHandle: ...
    def export_contract(self, handle: SplitRuntimeHandle) -> GraphContract: ...
    def run_prefix(self, handle: SplitRuntimeHandle, inputs: Any, *, training: bool) -> tuple[BoundaryEnvelope, PrefixContextToken | None]: ...
    def run_suffix(self, handle: SplitRuntimeHandle, boundary: BoundaryEnvelope, targets: Any = None, optimizer: Any = None) -> SuffixResult: ...
    def backward_prefix(self, handle: SplitRuntimeHandle, context_token: Any, gradients: GradientEnvelope, optimizer: Any = None) -> Any: ...
