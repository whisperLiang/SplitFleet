"""Slack-aware safe exploration for synchronous split federated rounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .solver import GlobalPlacementSolver
from .types import CandidateEstimate


@dataclass(frozen=True)
class ExplorationDecision:
    client_id: str
    baseline_boundary: str
    boundary: str
    reason: str
    uncertainty_ms: float
    safe_budget_ms: float
    predicted_upper_makespan_ms: float


class SafeExplorationController:
    """Explore only assignments inside a conservative global slowdown budget."""

    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        max_explorations_per_round: int = 1,
        forced_probe_interval: int = 10,
        seed: int = 233,
    ) -> None:
        if epsilon < 0 or max_explorations_per_round < 0 or forced_probe_interval < 1:
            raise ValueError("invalid safe exploration parameters")
        self.epsilon = float(epsilon)
        self.max_explorations_per_round = int(max_explorations_per_round)
        self.forced_probe_interval = int(forced_probe_interval)
        self._rng = np.random.default_rng(int(seed))
        self.last_probe_round: dict[str, int] = {}
        self.candidate_last_explored_round: dict[tuple[str, str], int] = {}

    def apply(
        self,
        *,
        round_id: int,
        baseline: Mapping[str, CandidateEstimate],
        estimates: Mapping[str, Sequence[CandidateEstimate]],
        solver: GlobalPlacementSolver,
        batch_counts: Mapping[str, int] | None = None,
        residence_locked: set[str] | None = None,
        component_last_observation: Mapping[tuple[str, str], int] | None = None,
    ) -> tuple[dict[str, CandidateEstimate], list[ExplorationDecision]]:
        final = dict(baseline)
        decisions: list[ExplorationDecision] = []
        if not baseline or self.max_explorations_per_round == 0:
            return final, decisions
        locked = residence_locked or set()
        observations = component_last_observation or {}
        baseline_upper = solver.simulate(
            baseline, use_upper=True, batch_counts=batch_counts
        ).max_client_completion_ms
        safe_budget = (1.0 + self.epsilon) * baseline_upper
        alternatives: list[tuple[tuple[float, ...], str, CandidateEstimate, float, bool]] = []
        for client_id in sorted(baseline):
            if client_id in locked:
                continue
            for option in estimates.get(client_id, ()):
                if not option.feasible or option.boundary == baseline[client_id].boundary:
                    continue
                forced = (
                    int(round_id)
                    - self.last_probe_round.get(client_id, 0)
                    >= self.forced_probe_interval
                    or int(round_id)
                    - self.candidate_last_explored_round.get(
                        (client_id, option.boundary), 0
                    )
                    >= self.forced_probe_interval
                    or int(round_id)
                    - observations.get((client_id, "network"), 0)
                    >= self.forced_probe_interval
                    or int(round_id)
                    - observations.get(("__global__", "server"), 0)
                    >= self.forced_probe_interval
                )
                trial = {**baseline, client_id: option}
                upper = solver.simulate(
                    trial, use_upper=True, batch_counts=batch_counts
                ).max_client_completion_ms
                if upper > safe_budget + 1e-9:
                    continue
                # Forced-safe probes take precedence. Information gain is
                # represented by the confidence radius; optimistic cost only
                # breaks ties among equally informative actions.
                priority = (
                    0.0 if forced else 1.0,
                    -option.uncertainty_total_ms,
                    option.lcb_total_without_queue_ms,
                    float(self._rng.random()),
                )
                alternatives.append((priority, client_id, option, upper, forced))
        alternatives.sort(key=lambda row: (row[0], row[1], row[2].boundary))

        used_clients: set[str] = set()
        for _, client_id, option, _, forced in alternatives:
            if len(decisions) >= self.max_explorations_per_round:
                break
            if client_id in used_clients:
                continue
            trial = {**final, client_id: option}
            final_upper = solver.simulate(
                trial, use_upper=True, batch_counts=batch_counts
            ).max_client_completion_ms
            if final_upper > safe_budget + 1e-9:
                continue
            baseline_boundary = final[client_id].boundary
            final[client_id] = option
            used_clients.add(client_id)
            decisions.append(
                ExplorationDecision(
                    client_id=client_id,
                    baseline_boundary=baseline_boundary,
                    boundary=option.boundary,
                    reason="forced_safe_probe" if forced else "slack_uncertainty",
                    uncertainty_ms=option.uncertainty_total_ms,
                    safe_budget_ms=safe_budget,
                    predicted_upper_makespan_ms=final_upper,
                )
            )
        return final, decisions

    def record_observation(
        self,
        *,
        round_id: int,
        client_id: str,
        boundary: str,
        was_probe: bool,
    ) -> None:
        """Advance probe history only after a successful action observation."""

        key = (str(client_id), str(boundary))
        self.candidate_last_explored_round[key] = int(round_id)
        if was_probe:
            self.last_probe_round[str(client_id)] = int(round_id)

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_probe_round": dict(sorted(self.last_probe_round.items())),
            "candidate_last_explored_round": {
                client_id: {
                    boundary: round_id
                    for (stored_client_id, boundary), round_id in sorted(
                        self.candidate_last_explored_round.items()
                    )
                    if stored_client_id == client_id
                }
                for client_id in sorted(
                    {key[0] for key in self.candidate_last_explored_round}
                )
            },
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.last_probe_round = {
            str(key): int(value) for key, value in state.get("last_probe_round", {}).items()
        }
        self.candidate_last_explored_round = {
            (str(client_id), str(boundary)): int(round_id)
            for client_id, boundaries in state.get(
                "candidate_last_explored_round", {}
            ).items()
            for boundary, round_id in boundaries.items()
        }
        if "rng_state" in state:
            self._rng.bit_generator.state = dict(state["rng_state"])


__all__ = ["ExplorationDecision", "SafeExplorationController"]
