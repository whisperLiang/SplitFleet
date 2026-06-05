"""Autosplit types shared across TorchLens planning and strategy code."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from splitfleet.autosplit.boundary import BoundaryPayload


class ReplicaScope(str, Enum):
    """Supported state-sharing semantics for server-side suffix replicas."""

    SHARED = "shared"
    PER_CLIENT = "per_client"
    PER_GROUP = "per_group"


@dataclass(frozen=True)
class WorkerSpec:
    """Describe a worker candidate that can host a split side."""

    worker_id: str
    address: Optional[str] = None
    device: str = "cpu"
    bandwidth_mbps: float = 1000.0
    memory_bytes: Optional[int] = None
    tags: tuple[str, ...] = ()
    online: bool = True


@dataclass(frozen=True)
class PlacementConstraint:
    """Hard constraints applied before scoring prefix/suffix placements."""

    max_stages: int = 2
    max_frontier_size: int = 1
    max_candidates: int = 32
    max_payload_bytes: int = 32 * 1024 * 1024
    max_stage_memory_bytes: Optional[int] = None
    privacy_metric_lower_bound: float = 0.0
    max_privacy_leakage: Optional[float] = None
    max_layer_freezing_ratio: Optional[float] = None
    require_trainable_tail: bool = True


@dataclass(frozen=True)
class PlacementObjective:
    """Weighted objective for candidate ranking."""

    latency_weight: float = 1.0
    bandwidth_weight: float = 0.5
    memory_weight: float = 0.25
    privacy_weight: float = 0.25


@dataclass
class SplitRuntimePlan:
    """Concrete TorchLens runtime summary retained with a prepared handle."""

    plan_id: str
    split_id: str
    graph_signature: str
    boundary: str
    mode: str
    trainable: bool
    dynamic_batch: tuple[int, int] | None
    trace_batch_mode: str
    trace_batch_size: int | None
    boundary_bytes: int
    prefix_node_count: int
    suffix_node_count: int
    trainable_suffix: bool
    candidate_id: str
    split_label: str
    boundary_tensor_labels: list[str]
    runtime_contract: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitPlan:
    """Concrete two-stage SplitFleet placement backed by TorchLens native split."""

    plan_id: str
    split_id: str
    graph_signature: str
    boundary: str
    mode: str
    prefix_worker_id: str
    suffix_worker_id: str
    score: float
    backend: str = "torchlens"
    runtime_backend: str = "torchlens_native"
    candidate_id: str = ""
    split_label: str = ""
    boundary_tensor_labels: list[str] = field(default_factory=list)
    payload_bytes: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    candidate_descriptor: dict[str, Any] = field(default_factory=dict)
    trace_signature: str = ""
    trace_batch_mode: str = "batch_gt1"
    dynamic_batch: tuple[int, int] | None = None
    trace_batch_size: int | None = None
    canonical_split_key: str = ""
    feature_layout_id: str = ""
    feature_abi_id: str = ""
    runtime_contract: dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    worker_specs: Dict[str, WorkerSpec] = field(default_factory=dict)
    objective: PlacementObjective = field(default_factory=PlacementObjective)
    constraints: PlacementConstraint = field(default_factory=PlacementConstraint)

    @property
    def split_config_id(self) -> str:
        return self.plan_id

    @property
    def stage_count(self) -> int:
        return 2

    @property
    def stage_to_worker(self) -> dict[str, str]:
        return {
            "prefix": self.prefix_worker_id,
            "suffix": self.suffix_worker_id,
        }


PlacementPlan = SplitPlan
