"""Policy primitives used by AutoSplitStrategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from splitfleet.autosplit.types import PlacementPlan, ReplicaScope


@dataclass(frozen=True)
class PartitionSelectionPolicy:
    """Select the placement plan to use for the current round."""

    preferred_stage_count: int | None = None

    def choose(self, placements: Sequence[PlacementPlan]) -> PlacementPlan:
        if not placements:
            raise ValueError("No placement plans were provided.")
        if self.preferred_stage_count is None:
            return min(placements, key=lambda plan: (plan.score, plan.partition_plan.stage_count))
        matching = [
            plan for plan in placements
            if plan.partition_plan.stage_count == self.preferred_stage_count
        ]
        pool = matching or list(placements)
        return min(pool, key=lambda plan: (plan.score, plan.partition_plan.stage_count))


@dataclass(frozen=True)
class ReplicaScopePolicy:
    """Resolve the replica scope for server-side execution."""

    replica_scope: ReplicaScope = ReplicaScope.SHARED

    def resolve(self) -> ReplicaScope:
        return self.replica_scope

    def requires_independent_replicas(self) -> bool:
        return self.resolve() != ReplicaScope.SHARED


@dataclass(frozen=True)
class ExecutionSchedulePolicy:
    """Decide whether stages should be processed as a synchronized batch."""

    process_clients_as_batch: bool = False

    def should_batch(self) -> bool:
        return self.process_clients_as_batch


@dataclass(frozen=True)
class AggregationPolicy:
    """Describe how server-side state should be aggregated."""

    name: str = "weighted_average"

    def normalized_name(self) -> str:
        normalized = self.name.strip().lower().replace("-", "_")
        aliases = {
            "fedavg": "weighted_average",
            "split_fed": "splitfed",
            "splitfed_average": "splitfed",
            "splitfed_avg": "splitfed",
        }
        return aliases.get(normalized, normalized)

    def requires_per_client_replica_scope(self) -> bool:
        return self.normalized_name() == "splitfed"

    def reduce_weights(self, num_examples: Iterable[int]) -> list[int]:
        return [max(1, int(count)) for count in num_examples]
