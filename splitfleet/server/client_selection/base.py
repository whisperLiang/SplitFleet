"""Common interfaces for pluggable server-side client selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass
class ClientState:
    """Mutable per-client statistics consumed by selectors."""

    cid: str
    reward: float = 0.0
    duration: float = 1.0
    num_examples: int = 0
    last_selected_round: int = 0
    selected_count: int = 0
    available: bool = True
    last_loss: float | None = None
    last_accuracy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """Result returned by a client selector for one fit round."""

    selected_cids: list[str]
    scores: dict[str, float]
    explore_cids: list[str]
    exploit_cids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class ClientSelector(Protocol):
    """Protocol implemented by fit-stage client selectors."""

    def register_client(
        self,
        cid: str,
        *,
        num_examples: int = 0,
        duration: float = 1.0,
        reward: float = 0.0,
    ) -> None:
        """Register a client if it is not already known."""

    def select(
        self,
        *,
        round_id: int,
        candidate_cids: Sequence[str],
        num_clients: int,
    ) -> SelectionResult:
        """Select client ids from the currently available candidates."""

    def update_after_fit(
        self,
        *,
        round_id: int,
        cid: str,
        num_examples: int,
        metrics: Mapping[str, Any],
    ) -> None:
        """Update selector state after a successful client fit result."""

    def update_after_failure(
        self,
        *,
        round_id: int,
        cid: str,
        reason: Any = None,
    ) -> None:
        """Update selector state after a selected client failed."""
