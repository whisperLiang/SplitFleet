"""Capability-aware per-client split placement for heterogeneous devices."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence

from splitfleet.server.client_selection.metrics import extract_duration
from splitfleet.server.placement.base import ClientPlacementState, PlacementDecision

DECISION_LOG_SIZE = 1024


@dataclass(frozen=True)
class CapabilityPlacementConfig:
    """Tuning knobs for :class:`CapabilityAwarePlacementPolicy`.

    ``tolerance`` is a relative band around the target round duration.  Only
    clients outside the band move, which keeps a fleet with similar devices on
    a stable cut.
    """

    tolerance: float = 0.20
    min_rounds_between_switches: int = 2
    warmup_rounds: int = 1
    max_steps_per_round: int = 1
    duration_smoothing: float = 0.5
    target_round_seconds: float | None = None
    failure_backoff_steps: int = 1


class CapabilityAwarePlacementPolicy:
    """Move each client along a boundary ladder to level synchronous rounds.

    ``boundary_ladder`` is ordered from the **lightest** client prefix (an early
    cut: little device compute, more server compute and boundary traffic) to the
    **heaviest** client prefix (a late cut: more device compute, less server
    work).  Devices slower than the round target move toward the lighter end and
    devices faster than the target move toward the heavier end, so a synchronous
    round is bounded by fewer stragglers without changing what is aggregated.

    The instance is usable directly as ``AutoSplitStrategy.client_placement_fn``;
    the strategy feeds fit results back through :meth:`observe_fit_metrics`.
    Decisions are taken once per round, so the fit and evaluate calls of the same
    round always agree on a client's boundary.
    """

    def __init__(
        self,
        *,
        boundary_ladder: Sequence[str],
        start_index: int | None = None,
        config: CapabilityPlacementConfig | None = None,
    ) -> None:
        ladder = [str(boundary) for boundary in boundary_ladder]
        if len(ladder) < 2:
            raise ValueError(
                "A capability-aware boundary ladder needs at least two boundaries, "
                "ordered from the lightest to the heaviest client prefix."
            )
        self.boundary_ladder = ladder
        self.config = config or CapabilityPlacementConfig()
        if self.config.tolerance < 0:
            raise ValueError("CapabilityPlacementConfig.tolerance must not be negative.")
        if not 0.0 < self.config.duration_smoothing <= 1.0:
            raise ValueError(
                "CapabilityPlacementConfig.duration_smoothing must be in (0, 1]."
            )
        self.start_index = (
            len(ladder) // 2 if start_index is None else self._clamp_index(start_index)
        )
        self.clients: dict[str, ClientPlacementState] = {}
        # Kept for observability only, so the log is bounded on long runs.
        self.decisions: deque[PlacementDecision] = deque(maxlen=DECISION_LOG_SIZE)
        self._decided_round: int | None = None

    def __call__(self, server_round: int, cid: str, training: bool = True) -> str:
        _ = training
        self._apply_round_decisions(int(server_round))
        return self.boundary_for(str(cid))

    def boundary_for(self, cid: str) -> str:
        return self.boundary_ladder[self._state(str(cid)).ladder_index]

    def placement_state(self) -> dict[str, str]:
        """Return the current boundary of every known client."""

        return {cid: self.boundary_ladder[state.ladder_index] for cid, state in self.clients.items()}

    def observe_fit_metrics(
        self,
        *,
        round_id: int,
        cid: str,
        num_examples: int,
        metrics: Mapping[str, Any],
    ) -> None:
        state = self._state(str(cid))
        duration = extract_duration(metrics, default=state.round_duration_sec or 1.0)
        alpha = self.config.duration_smoothing
        state.round_duration_sec = (
            duration
            if state.rounds_observed == 0
            else alpha * duration + (1.0 - alpha) * state.round_duration_sec
        )
        state.last_num_examples = max(int(num_examples), 0)
        state.rounds_observed += 1
        state.metadata["last_round"] = int(round_id)
        for key in ("tail_wait_sec", "prefix_compute_sec", "upload_bytes", "max_batch_size"):
            if key in metrics:
                state.metadata[key] = metrics[key]

    def observe_failure(self, *, round_id: int, cid: str, reason: Any = None) -> None:
        state = self._state(str(cid))
        state.failures += 1
        state.metadata["last_failure_round"] = int(round_id)
        if reason is not None:
            state.metadata["last_failure_reason"] = str(reason)[:200]
        steps = int(self.config.failure_backoff_steps)
        if steps <= 0:
            return
        # A device that dropped out is the one most likely to be overloaded, so
        # relieve it immediately instead of waiting for the hysteresis window.
        self._move(state, -steps, round_id=int(round_id), reason="failure_backoff")

    def _state(self, cid: str) -> ClientPlacementState:
        state = self.clients.get(cid)
        if state is None:
            state = ClientPlacementState(cid=cid, ladder_index=self.start_index)
            self.clients[cid] = state
        return state

    def _clamp_index(self, index: int) -> int:
        return max(0, min(len(self.boundary_ladder) - 1, int(index)))

    def _move(self, state: ClientPlacementState, steps: int, *, round_id: int, reason: str) -> bool:
        target = self._clamp_index(state.ladder_index + int(steps))
        if target == state.ladder_index:
            return False
        previous = self.boundary_ladder[state.ladder_index]
        state.ladder_index = target
        state.last_switch_round = int(round_id)
        self.decisions.append(
            PlacementDecision(
                round_id=int(round_id),
                cid=state.cid,
                previous_boundary=previous,
                boundary=self.boundary_ladder[target],
                reason=reason,
            )
        )
        return True

    def _apply_round_decisions(self, round_id: int) -> None:
        if self._decided_round == round_id:
            return
        self._decided_round = round_id
        observed = [state for state in self.clients.values() if state.rounds_observed > 0]
        if not observed or round_id <= int(self.config.warmup_rounds):
            return
        target = self._target_duration(observed)
        if target is None:
            return
        tolerance = float(self.config.tolerance)
        slow_threshold = target * (1.0 + tolerance)
        fast_threshold = target * (1.0 - tolerance)
        for state in observed:
            # A client that has never switched is never held back by hysteresis.
            if (
                state.last_switch_round is not None
                and round_id - state.last_switch_round < int(self.config.min_rounds_between_switches)
            ):
                continue
            duration = state.round_duration_sec
            if duration > slow_threshold:
                self._move(
                    state,
                    -self._step_size(duration, target),
                    round_id=round_id,
                    reason="slower_than_round_target",
                )
            elif duration < fast_threshold:
                self._move(
                    state,
                    self._step_size(target, duration),
                    round_id=round_id,
                    reason="faster_than_round_target",
                )

    def _step_size(self, slower: float, faster: float) -> int:
        """Scale the ladder step with how far a client is from the target."""

        max_steps = max(1, int(self.config.max_steps_per_round))
        if faster <= 0:
            return max_steps
        return max(1, min(max_steps, int(slower / faster)))

    def _target_duration(self, observed: Sequence[ClientPlacementState]) -> float | None:
        if self.config.target_round_seconds is not None:
            return max(float(self.config.target_round_seconds), 1e-9)
        durations = [state.round_duration_sec for state in observed if state.round_duration_sec > 0]
        if not durations:
            return None
        return median(durations)


__all__ = ["CapabilityAwarePlacementPolicy", "CapabilityPlacementConfig"]
