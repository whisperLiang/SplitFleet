"""Public autosplit facade for SplitFleet's TorchLens backend."""

from splitfleet.autosplit.boundary import (
    BoundaryPayload,
    BoundarySpec,
    from_torchlens_boundary,
    to_torchlens_boundary,
)
from splitfleet.runtime import PrefixContextKey, PrefixContextStore
from splitfleet.autosplit.batch_window import (
    BatchWindowError,
    batch_window_contains,
    describe_batch_window,
    normalize_batch_window,
    require_batch_in_window,
)
from splitfleet.autosplit.cache import PlanCacheEntry, PlanCacheStore
from splitfleet.autosplit.policies import (
    AggregationPolicy,
    ExecutionSchedulePolicy,
    PartitionSelectionPolicy,
    ReplicaScopePolicy,
)
from splitfleet.autosplit.planner import AutoSplitPlanner, validate_stage_counts
from splitfleet.autosplit.runtime import AutoSplitSession, compute_loss
from splitfleet.autosplit.serde import (
    deserialize_plan_descriptor,
    serialize_plan_descriptor,
)
from splitfleet.autosplit.torchlens_backend import (
    TorchLensRuntimeHandle,
    TorchLensSplitBackend,
    backward_prefix,
    prepare_torchlens_runtime,
    run_prefix,
    run_suffix,
    run_training_prefix,
    train_suffix,
)
from splitfleet.autosplit.torchlens_candidate import SplitCandidate
from splitfleet.autosplit.torchlens_contract import (
    FeatureAbiSpec,
    build_feature_abi_spec,
    build_runtime_contract,
    classify_contract_compatibility,
    feature_abi_id,
    runtime_contract_digest,
    runtime_identity_id,
    stable_json,
)
from splitfleet.autosplit.torchlens_runtime import (
    SplitRuntime,
    make_split_spec,
    normalize_example_inputs,
    prepare_split_runtime,
    torchlens_runtime_version,
)
from splitfleet.autosplit.types import (
    PlacementConstraint,
    PlacementObjective,
    ReplicaScope,
    SplitPlacementPlan,
    SplitRuntimePlan,
    WorkerSpec,
)

__all__ = [
    "AggregationPolicy",
    "AutoSplitPlanner",
    "AutoSplitSession",
    "BatchWindowError",
    "BoundaryPayload",
    "PrefixContextKey",
    "PrefixContextStore",
    "BoundarySpec",
    "ExecutionSchedulePolicy",
    "FeatureAbiSpec",
    "PartitionSelectionPolicy",
    "PlacementConstraint",
    "PlacementObjective",
    "PlanCacheEntry",
    "PlanCacheStore",
    "ReplicaScope",
    "ReplicaScopePolicy",
    "SplitCandidate",
    "SplitPlacementPlan",
    "SplitRuntime",
    "SplitRuntimePlan",
    "TorchLensRuntimeHandle",
    "TorchLensSplitBackend",
    "WorkerSpec",
    "backward_prefix",
    "batch_window_contains",
    "build_feature_abi_spec",
    "build_runtime_contract",
    "classify_contract_compatibility",
    "compute_loss",
    "describe_batch_window",
    "deserialize_plan_descriptor",
    "feature_abi_id",
    "from_torchlens_boundary",
    "make_split_spec",
    "normalize_batch_window",
    "normalize_example_inputs",
    "prepare_split_runtime",
    "prepare_torchlens_runtime",
    "require_batch_in_window",
    "run_prefix",
    "run_suffix",
    "run_training_prefix",
    "runtime_identity_id",
    "runtime_contract_digest",
    "serialize_plan_descriptor",
    "stable_json",
    "to_torchlens_boundary",
    "torchlens_runtime_version",
    "train_suffix",
    "validate_stage_counts",
]
