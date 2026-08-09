"""Server-global constrained scheduling for Resource-Adaptive SplitFed."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence

from .split_cost_model import SplitCostPrediction


@dataclass
class SchedulerDecision:
    client_id: str
    old_split_key: str
    new_split_key: str
    predicted_improvement_ratio: float
    switch_reason: str


class ResourceAdaptiveSplitScheduler:
    """Greedy list scheduling followed by coordinate improvement.

    Choices are evaluated jointly using bounded server lanes, so the objective
    is the predicted synchronous makespan rather than each client's isolated
    latency.  The real server pool enforces the same concurrency limit.
    """

    def __init__(
        self,
        *,
        min_relative_improvement_to_switch: float = 0.10,
        min_rounds_between_switches: int = 3,
        max_switches_per_round: int | None = None,
        use_memory_constraint: bool = True,
        use_hysteresis: bool = True,
        use_server_state: bool = True,
    ) -> None:
        self.min_relative_improvement_to_switch = float(min_relative_improvement_to_switch)
        self.min_rounds_between_switches = int(min_rounds_between_switches)
        self.max_switches_per_round = max_switches_per_round
        self.use_memory_constraint = bool(use_memory_constraint)
        self.use_hysteresis = bool(use_hysteresis)
        self.use_server_state = bool(use_server_state)
        self.current_splits: dict[str, str] = {}
        self.last_switch_round: dict[str, int] = {}
        self.last_decisions: list[SchedulerDecision] = []

    def select_splits(
        self,
        clients: Sequence[Any],
        client_resource_states: Mapping[str, Any],
        server_resource_state: Any,
        candidate_predictions: Mapping[str, Any],
        *,
        round_id: int = 1,
    ) -> dict[str, str]:
        client_ids = [str(_get(client, "client_id", _get(client, "cid", client))) for client in clients]
        server = _mapping(server_resource_state)
        concurrency = max(1, int(server.get("max_server_concurrency", 1) or 1))
        options = {
            client_id: self._feasible_options(candidate_predictions[client_id])
            for client_id in client_ids
        }
        for client_id, predictions in options.items():
            if not predictions:
                raise RuntimeError(f"No feasible split candidate for client {client_id!r}.")

        # Most constrained/slow clients are placed first.
        ordered = sorted(
            client_ids,
            key=lambda cid: min(pred.predicted_round_ms for pred in options[cid]),
            reverse=True,
        )
        assignment: dict[str, SplitCostPrediction] = {}
        for client_id in ordered:
            best = min(
                options[client_id],
                key=lambda prediction: self._objective(
                    {**assignment, client_id: prediction}, concurrency
                ),
            )
            assignment[client_id] = best

        # Bounded coordinate search repairs greedy choices when a late choice
        # changes server queue waves for clients assigned earlier.
        for _ in range(3):
            changed = False
            for client_id in ordered:
                current = assignment[client_id]
                best = min(
                    options[client_id],
                    key=lambda prediction: self._objective(
                        {**assignment, client_id: prediction}, concurrency
                    ),
                )
                proposed_objective = self._objective(
                    {**assignment, client_id: best}, concurrency
                )
                current_objective = self._objective(assignment, concurrency)
                if proposed_objective[0] + 1e-9 < current_objective[0] or (
                    abs(proposed_objective[0] - current_objective[0]) <= 1e-9
                    and proposed_objective[1:] < current_objective[1:]
                ):
                    assignment[client_id] = best
                    changed = changed or best.split_key != current.split_key
            if not changed:
                break

        selected: dict[str, str] = {}
        decisions: list[SchedulerDecision] = []
        switch_candidates: list[tuple[float, str, str, str, str]] = []
        for client_id in client_ids:
            proposed = assignment[client_id]
            old = self.current_splits.get(client_id, proposed.split_key)
            current_prediction = next(
                (item for item in options[client_id] if item.split_key == old), proposed
            )
            improvement = max(
                0.0,
                (current_prediction.predicted_round_ms - proposed.predicted_round_ms)
                / max(current_prediction.predicted_round_ms, 1e-9),
            )
            reason = "minimum_predicted_global_makespan"
            chosen = proposed.split_key
            if old != chosen and self.use_hysteresis:
                held = round_id - self.last_switch_round.get(client_id, 0)
                if held < self.min_rounds_between_switches:
                    chosen, reason = old, "minimum_hold_rounds"
                elif improvement < self.min_relative_improvement_to_switch:
                    chosen, reason = old, "improvement_below_threshold"
            if old != chosen:
                switch_candidates.append((improvement, client_id, old, chosen, reason))
            selected[client_id] = chosen
            decisions.append(SchedulerDecision(client_id, old, chosen, improvement, reason))

        if self.max_switches_per_round is not None and len(switch_candidates) > self.max_switches_per_round:
            allowed = {
                item[1]
                for item in sorted(switch_candidates, reverse=True)[: self.max_switches_per_round]
            }
            for decision in decisions:
                if decision.old_split_key != decision.new_split_key and decision.client_id not in allowed:
                    selected[decision.client_id] = decision.old_split_key
                    decision.new_split_key = decision.old_split_key
                    decision.switch_reason = "max_switches_per_round"

        for decision in decisions:
            if decision.old_split_key != decision.new_split_key:
                self.last_switch_round[decision.client_id] = round_id
            self.current_splits[decision.client_id] = decision.new_split_key
        self.last_decisions = decisions
        return selected

    def _feasible_options(self, predictions: Any) -> list[SplitCostPrediction]:
        values = predictions.values() if isinstance(predictions, Mapping) else predictions
        result = []
        for value in values:
            prediction = value if isinstance(value, SplitCostPrediction) else SplitCostPrediction(**_mapping(value))
            if prediction.feasible or not self.use_memory_constraint:
                result.append(prediction)
        return result

    def _objective(
        self, assignment: Mapping[str, SplitCostPrediction], concurrency: int
    ) -> tuple[float, float, int]:
        if not assignment:
            return (0.0, 0.0, 0)
        if not self.use_server_state:
            makespan = max(item.predicted_round_ms for item in assignment.values())
            return makespan, sum(item.predicted_round_ms for item in assignment.values()), 0
        lanes = [0.0 for _ in range(concurrency)]
        completions: list[float] = []
        early_count = 0
        jobs = sorted(
            assignment.values(),
            key=lambda item: item.predicted_client_compute_ms + item.predicted_network_ms,
        )
        for prediction in jobs:
            arrival = prediction.predicted_client_compute_ms + prediction.predicted_network_ms
            if prediction.predicted_server_gpu_time_ms <= 0:
                completions.append(prediction.predicted_round_ms)
                continue
            lane = min(range(concurrency), key=lanes.__getitem__)
            start = max(arrival, lanes[lane])
            queue = start - arrival
            lanes[lane] = start + prediction.predicted_server_gpu_time_ms
            completion = prediction.predicted_round_ms + queue
            completions.append(completion)
            early_count += int(prediction.predicted_server_gpu_time_ms > prediction.predicted_client_compute_ms)
        return max(completions), sum(completions), early_count


def select_edge_local(predictions: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for client_id, values in predictions.items():
        candidates = values.values() if isinstance(values, Mapping) else values
        feasible = [item for item in candidates if item.feasible]
        if not feasible:
            raise RuntimeError(f"No feasible split candidate for client {client_id!r}.")
        result[str(client_id)] = min(feasible, key=lambda item: item.predicted_round_ms).split_key
    return result


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return dict(vars(value))


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
