"""Public autosplit facade for SplitFleet."""

from splitfleet.autosplit.cache import PlanCacheEntry, PlanCacheStore
from splitfleet.autosplit.policies import (
    AggregationPolicy,
    ExecutionSchedulePolicy,
    PartitionSelectionPolicy,
    ReplicaScopePolicy,
)
from splitfleet.autosplit.planner import AutoSplitPlanner
from splitfleet.autosplit.runtime import (
    AutoSplitSession,
    PreparedExecutionContext,
    StageBackwardResult,
    StageForwardResult,
)
from splitfleet.autosplit.serde import (
    deserialize_plan_descriptor,
    dump_model_state,
    dumps_torch_object,
    load_model_state,
    loads_torch_object,
    serialize_plan_descriptor,
)
from splitfleet.autosplit.tracer import ModelTracer
from splitfleet.autosplit.types import (
    BoundaryPayload,
    ExecNode,
    ExecutionPlan,
    PartitionPlan,
    PartitionStage,
    PlacementConstraint,
    PlacementObjective,
    PlacementPlan,
    ReplicaScope,
    WorkerSpec,
)

__all__ = [
    "AggregationPolicy",
    "AutoSplitPlanner",
    "AutoSplitSession",
    "BoundaryPayload",
    "deserialize_plan_descriptor",
    "dump_model_state",
    "dumps_torch_object",
    "ExecNode",
    "ExecutionPlan",
    "ExecutionSchedulePolicy",
    "load_model_state",
    "loads_torch_object",
    "ModelTracer",
    "PartitionPlan",
    "PartitionSelectionPolicy",
    "PartitionStage",
    "PlacementConstraint",
    "PlacementObjective",
    "PlacementPlan",
    "PlanCacheEntry",
    "PlanCacheStore",
    "PreparedExecutionContext",
    "ReplicaScope",
    "ReplicaScopePolicy",
    "serialize_plan_descriptor",
    "StageBackwardResult",
    "StageForwardResult",
    "WorkerSpec",
]
