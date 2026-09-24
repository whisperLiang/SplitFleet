"""SplitFleet placement APIs.

CoSplit-UCB is the sole built-in dynamic split-point policy. Fixed/manual
boundaries remain available directly through ``AutoSplitStrategy``.
"""

from splitfleet.server.placement.base import PlacementFeedback, RoundPlacementPolicy
from splitfleet.server.placement.cosplit_ucb import (
    BanditStateStore,
    CandidateEstimate,
    CoSplitUCBConfig,
    CoSplitUCBPlacementPolicy,
    ContextEncoder,
    CooperativeLearners,
    DiscountedLinUCB,
    ExecutionProfileKey,
    FeasibilityFilter,
    GlobalPlacementSolver,
    SafeExplorationController,
    RuntimeTelemetryProvider,
    SplitCandidateDescriptor,
    StaticCandidateProvider,
    TorchLensCandidateProvider,
)

__all__ = [
    "BanditStateStore",
    "CandidateEstimate",
    "CoSplitUCBConfig",
    "CoSplitUCBPlacementPolicy",
    "ContextEncoder",
    "CooperativeLearners",
    "DiscountedLinUCB",
    "ExecutionProfileKey",
    "FeasibilityFilter",
    "GlobalPlacementSolver",
    "PlacementFeedback",
    "RoundPlacementPolicy",
    "RuntimeTelemetryProvider",
    "SafeExplorationController",
    "SplitCandidateDescriptor",
    "StaticCandidateProvider",
    "TorchLensCandidateProvider",
]
