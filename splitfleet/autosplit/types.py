"""Autosplit types shared across Ariadne planning and strategy code."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ariadne import BoundaryPayload


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
    """Hard constraints applied before scoring Ariadne prefix/suffix placements."""

    max_stages: int = 2
    max_frontier_size: int = 1
    max_candidates: int = 1
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
class AriadnePlacementPlan:
    """Concrete two-stage SplitFleet placement backed by Ariadne."""

    plan_id: str
    split_id: str
    graph_signature: str
    boundary: str
    mode: str
    prefix_worker_id: str
    suffix_worker_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    worker_specs: Dict[str, WorkerSpec] = field(default_factory=dict)
    objective: PlacementObjective = field(default_factory=PlacementObjective)
    constraints: PlacementConstraint = field(default_factory=PlacementConstraint)

    @property
    def stage_count(self) -> int:
        return 2

    @property
    def stage_to_worker(self) -> dict[str, str]:
        return {
            "prefix": self.prefix_worker_id,
            "suffix": self.suffix_worker_id,
        }


PlacementPlan = AriadnePlacementPlan
