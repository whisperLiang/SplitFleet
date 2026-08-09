"""Common interfaces for pluggable per-client split placement policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass
class ClientPlacementState:
    """Mutable per-client placement statistics."""

    cid: str
    ladder_index: int
    round_duration_sec: float = 0.0
    last_num_examples: int = 0
    rounds_observed: int = 0
    failures: int = 0
    last_switch_round: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementDecision:
    """One placement change applied at the start of a round."""

    round_id: int
    cid: str
    previous_boundary: str
    boundary: str
    reason: str


class ClientPlacementPolicy(Protocol):
    """Protocol implemented by ``AutoSplitStrategy.client_placement_fn`` policies.

    A policy is called once per client and round and returns the boundary that
    client should use.  ``AutoSplitStrategy`` additionally feeds fit results and
    failures back into the policy when it implements the observer methods.
    """

    def __call__(self, server_round: int, cid: str, training: bool) -> Any:
        """Return the boundary (or placement plan) for one client and round."""

    def observe_fit_metrics(
        self,
        *,
        round_id: int,
        cid: str,
        num_examples: int,
        metrics: Mapping[str, Any],
    ) -> None:
        """Record one successful client fit result."""

    def observe_failure(self, *, round_id: int, cid: str, reason: Any = None) -> None:
        """Record one failed client."""


__all__ = ["ClientPlacementPolicy", "ClientPlacementState", "PlacementDecision"]
