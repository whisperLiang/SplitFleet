"""Backend-neutral contracts used by CoSplit-UCB."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Protocol, Sequence


ALGORITHM_VERSION = "cosplit_ucb_v1"
FEATURE_SCHEMA_VERSION = "cosplit_context_v1"


@dataclass(frozen=True, order=True)
class ExecutionProfileKey:
    """Stable cooperative-sharing key for an edge execution environment."""

    framework_backend: str
    runtime_backend: str
    device_type: str
    accelerator: str
    precision: str

    def __post_init__(self) -> None:
        for field_name in (
            "framework_backend",
            "runtime_backend",
            "device_type",
            "accelerator",
            "precision",
        ):
            value = str(getattr(self, field_name)).strip().lower()
            if not value:
                raise ValueError(f"ExecutionProfileKey.{field_name} must not be empty")
            object.__setattr__(self, field_name, value)

    @property
    def stable_id(self) -> str:
        """Return the portable, hostname-free profile identifier."""

        return "|".join(
            (
                self.framework_backend,
                self.runtime_backend,
                self.device_type,
                self.accelerator,
                self.precision,
            )
        )

    @classmethod
    def from_value(cls, value: "ExecutionProfileKey | Mapping[str, Any] | str") -> "ExecutionProfileKey":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            parts = value.split("|")
            if len(parts) != 5:
                raise ValueError("Execution profile strings must contain five pipe-separated fields")
            return cls(*parts)
        return cls(
            framework_backend=str(value.get("framework_backend", "unknown")),
            runtime_backend=str(value.get("runtime_backend", "unknown")),
            device_type=str(value.get("device_type", "cpu")),
            accelerator=str(value.get("accelerator", "generic_cpu")),
            precision=str(value.get("precision", "fp32")),
        )


@dataclass(frozen=True)
class SplitCandidateDescriptor:
    """Canonical description of one valid operation-level split boundary."""

    boundary: str
    split_id: str
    graph_position_ratio: float
    prefix_node_count: int
    suffix_node_count: int
    total_node_count: int
    boundary_forward_bytes: int | None
    boundary_gradient_bytes: int | None
    boundary_tensor_count: int
    prefix_parameter_bytes: int | None
    suffix_parameter_bytes: int | None
    client_memory_bytes: int | None
    server_memory_bytes: int | None
    trainable: bool
    feature_abi_id: str
    graph_signature: str
    framework_backend: str = "unknown"
    runtime_backend: str = "torchlens_native"
    valid: bool = True
    validation_error: str | None = None
    runtime_contract: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.boundary or not self.split_id:
            raise ValueError("candidate boundary and split_id must not be empty")
        if self.total_node_count < 1:
            raise ValueError("total_node_count must be positive")
        if self.prefix_node_count < 0 or self.suffix_node_count < 0:
            raise ValueError("candidate node counts must be non-negative")
        if not isfinite(float(self.graph_position_ratio)):
            raise ValueError("candidate graph position must be finite")
        object.__setattr__(self, "graph_position_ratio", min(max(float(self.graph_position_ratio), 0.0), 1.0))


@dataclass(frozen=True)
class CandidateEstimate:
    """Learned component costs and confidence radii for one client/cut pair."""

    client_id: str
    boundary: str
    client_forward_mean_ms: float
    client_backward_mean_ms: float
    network_upload_mean_ms: float
    network_download_mean_ms: float
    server_service_mean_ms: float
    switch_mean_ms: float
    client_forward_uncertainty_ms: float
    client_backward_uncertainty_ms: float
    network_upload_uncertainty_ms: float
    network_download_uncertainty_ms: float
    server_service_uncertainty_ms: float
    switch_uncertainty_ms: float
    feasible: bool = True
    infeasible_reason: str | None = None

    def __post_init__(self) -> None:
        numeric_fields = (
            "client_forward_mean_ms",
            "client_backward_mean_ms",
            "network_upload_mean_ms",
            "network_download_mean_ms",
            "server_service_mean_ms",
            "switch_mean_ms",
            "client_forward_uncertainty_ms",
            "client_backward_uncertainty_ms",
            "network_upload_uncertainty_ms",
            "network_download_uncertainty_ms",
            "server_service_uncertainty_ms",
            "switch_uncertainty_ms",
        )
        for name in numeric_fields:
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, max(value, 0.0))

    @property
    def mean_total_without_queue_ms(self) -> float:
        return sum(
            (
                self.client_forward_mean_ms,
                self.client_backward_mean_ms,
                self.network_upload_mean_ms,
                self.network_download_mean_ms,
                self.server_service_mean_ms,
                self.switch_mean_ms,
            )
        )

    @property
    def uncertainty_total_ms(self) -> float:
        return sum(
            (
                self.client_forward_uncertainty_ms,
                self.client_backward_uncertainty_ms,
                self.network_upload_uncertainty_ms,
                self.network_download_uncertainty_ms,
                self.server_service_uncertainty_ms,
                self.switch_uncertainty_ms,
            )
        )

    @property
    def lcb_total_without_queue_ms(self) -> float:
        return max(self.mean_total_without_queue_ms - self.uncertainty_total_ms, 0.0)

    @property
    def ucb_total_without_queue_ms(self) -> float:
        return self.mean_total_without_queue_ms + self.uncertainty_total_ms


@dataclass
class PlacementFeedback:
    """One round-level observation for a client and selected boundary."""

    round_id: int
    client_id: str
    boundary: str
    client_forward_ms: float | None = None
    client_backward_ms: float | None = None
    network_upload_ms: float | None = None
    network_download_ms: float | None = None
    server_service_ms: float | None = None
    switch_ms: float | None = None
    completion_ms: float | None = None
    num_examples: int = 0
    num_batches: int = 0
    client_peak_memory_mb: float | None = None
    server_peak_memory_mb: float | None = None
    execution_profile: ExecutionProfileKey | Mapping[str, Any] | str | None = None
    success: bool = True


@dataclass(frozen=True)
class PlacementFailure:
    """A classified placement failure that must not become a latency sample."""

    round_id: int
    client_id: str
    boundary: str | None
    kind: str = "runtime"
    reason: str | None = None


class RoundPlacementPolicy(Protocol):
    """Joint round placement interface consumed by ``AutoSplitStrategy``."""

    def plan_round(
        self,
        *,
        round_id: int,
        client_ids: Sequence[str],
        training: bool,
    ) -> Mapping[str, str]:
        """Plan one boundary for every selected client in a single joint call."""

    def observe_round(
        self,
        *,
        round_id: int,
        feedback: Sequence[PlacementFeedback],
    ) -> None:
        """Consume at most one aggregate observation per client/cut/round."""

    def observe_failure(
        self,
        *,
        round_id: int,
        client_id: str,
        boundary: str | None = None,
        kind: str = "runtime",
        reason: Any = None,
    ) -> None:
        """Record failure feasibility state without a fabricated cost target."""


class RuntimeTelemetryProvider(Protocol):
    """Supply raw/recent context without converting it into a latency cost."""

    def client_context(self, client_id: str) -> Mapping[str, Any]:
        """Return client compute, memory, and recent link telemetry."""

    def server_context(self) -> Mapping[str, Any]:
        """Return recent server utilization, queue, jobs, and memory context."""

    def execution_profile(self, client_id: str) -> ExecutionProfileKey | Mapping[str, Any] | str:
        """Return the stable, hostname-free execution profile for a client."""


__all__ = [
    "ALGORITHM_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "CandidateEstimate",
    "ExecutionProfileKey",
    "PlacementFailure",
    "PlacementFeedback",
    "RoundPlacementPolicy",
    "RuntimeTelemetryProvider",
    "SplitCandidateDescriptor",
]
