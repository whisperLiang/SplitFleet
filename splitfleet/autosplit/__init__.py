"""Public autosplit facade for SplitFleet's Ariadne backend."""

from ariadne import BoundaryPayload

from splitfleet.autosplit.ariadne_adapter import (
    AriadneRuntimeHandle,
    AriadneSplitPlan,
    backward_prefix,
    infer_trace_batch_mode,
    normalize_example_inputs,
    prepare_ariadne_runtime,
    run_prefix,
    run_suffix,
    run_training_prefix,
    train_suffix,
)
from splitfleet.autosplit.cache import PlanCacheEntry, PlanCacheStore
from splitfleet.autosplit.policies import (
    AggregationPolicy,
    ExecutionSchedulePolicy,
    PartitionSelectionPolicy,
    ReplicaScopePolicy,
)
from splitfleet.autosplit.planner import AutoSplitPlanner
from splitfleet.autosplit.runtime import AutoSplitSession, compute_loss, normalize_inputs
from splitfleet.autosplit.serde import (
    deserialize_plan_descriptor,
    dump_model_state,
    dumps_torch_object,
    load_model_state,
    loads_torch_object,
    serialize_plan_descriptor,
)
from splitfleet.autosplit.types import (
    AriadnePlacementPlan,
    PlacementConstraint,
    PlacementObjective,
    PlacementPlan,
    ReplicaScope,
    WorkerSpec,
)

__all__ = [
    "AggregationPolicy",
    "AriadnePlacementPlan",
    "AriadneRuntimeHandle",
    "AriadneSplitPlan",
    "AutoSplitPlanner",
    "AutoSplitSession",
    "BoundaryPayload",
    "ExecutionSchedulePolicy",
    "PartitionSelectionPolicy",
    "PlacementConstraint",
    "PlacementObjective",
    "PlacementPlan",
    "PlanCacheEntry",
    "PlanCacheStore",
    "ReplicaScope",
    "ReplicaScopePolicy",
    "WorkerSpec",
    "backward_prefix",
    "compute_loss",
    "deserialize_plan_descriptor",
    "dump_model_state",
    "dumps_torch_object",
    "infer_trace_batch_mode",
    "load_model_state",
    "loads_torch_object",
    "normalize_example_inputs",
    "normalize_inputs",
    "prepare_ariadne_runtime",
    "run_prefix",
    "run_suffix",
    "run_training_prefix",
    "serialize_plan_descriptor",
    "train_suffix",
]
