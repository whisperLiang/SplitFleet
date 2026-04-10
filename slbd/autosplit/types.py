"""Autosplit IR shared across planning, runtime, and strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from torchlens.replay_plan import BoundaryPayload, ExecNode, ExecutionPlan


class ReplicaScope(str, Enum):
    """Supported state-sharing semantics for server-side stages."""

    SHARED = "shared"
    PER_CLIENT = "per_client"
    PER_GROUP = "per_group"


@dataclass(frozen=True)
class WorkerSpec:
    """Describe a worker candidate that can host one or more stages."""

    worker_id: str
    address: Optional[str] = None
    device: str = "cpu"
    bandwidth_mbps: float = 1000.0
    memory_bytes: Optional[int] = None
    tags: tuple[str, ...] = ()
    online: bool = True


@dataclass(frozen=True)
class PlacementConstraint:
    """Hard constraints applied before scoring candidate placements."""

    max_stages: int = 3
    max_frontier_size: int = 4
    max_candidates: int = 24
    max_payload_bytes: int = 32 * 1024 * 1024
    max_stage_memory_bytes: Optional[int] = None
    privacy_metric_lower_bound: float = 0.0
    require_trainable_tail: bool = True


@dataclass(frozen=True)
class PlacementObjective:
    """Weighted objective for candidate ranking."""

    latency_weight: float = 1.0
    bandwidth_weight: float = 0.5
    memory_weight: float = 0.25
    privacy_weight: float = 0.25


@dataclass
class PartitionStage:
    """One ordered stage inside a partition plan."""

    stage_id: str
    node_indices: List[int]
    input_indices: List[int]
    output_indices: List[int]
    input_labels: List[str]
    output_labels: List[str]
    passthrough_input_indices: List[int] = field(default_factory=list)
    passthrough_input_labels: List[str] = field(default_factory=list)
    estimated_compute_cost: float = 0.0
    estimated_activation_bytes: int = 0
    estimated_parameter_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PartitionPlan:
    """Ordered multi-stage partition over one concrete execution plan."""

    plan_id: str
    model_name: str
    execution_plan: ExecutionPlan
    stages: List[PartitionStage]
    graph_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def stage_count(self) -> int:
        return len(self.stages)


@dataclass
class PlacementPlan:
    """Concrete worker assignment for a partition plan."""

    partition_plan: PartitionPlan
    stage_to_worker: Dict[str, str]
    worker_specs: Dict[str, WorkerSpec]
    score: float
    stage_scores: Dict[str, float] = field(default_factory=dict)
    objective: PlacementObjective = field(default_factory=PlacementObjective)
    constraints: PlacementConstraint = field(default_factory=PlacementConstraint)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def plan_id(self) -> str:
        return self.partition_plan.plan_id
