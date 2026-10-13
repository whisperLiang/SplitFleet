"""Common wall-clock endpoint for native FL and synchronous split training."""

from __future__ import annotations

import copy
import math
import time
from types import MethodType

import torch
from flwr.common import parameters_to_ndarrays
from flwr.server.history import History


def with_time_budget(server, budget_sec: float, evaluation_interval_sec: float):
    if (not math.isfinite(budget_sec) or budget_sec <= 0
            or not math.isfinite(evaluation_interval_sec) or evaluation_interval_sec <= 0):
        raise ValueError("Training and evaluation time budgets must be finite and positive")
    server.time_budget_sec = budget_sec
    server.evaluation_interval_sec = evaluation_interval_sec
    server.fit = MethodType(_fit, server)
    return server


def _fit(self, num_rounds, timeout):
    # A common time budget supersedes the round cap. Otherwise fast methods
    # could exhaust ten rounds and stop long before the common time endpoint.
    strategy = self.strategy
    started = strategy.measurement_started_ns / 1e9
    elapsed = lambda: time.monotonic() - started
    split = hasattr(self, "server_model_manager")
    attr = "client_parameters" if split else "parameters"
    setattr(self, attr, self._get_initial_parameters(server_round=0, timeout=timeout))
    if split:
        self.server_parameters = self._get_initial_server_parameters()
    history = History()
    evaluation_index = 0
    accepted_rounds = 0
    attempted_rounds = 0

    def evaluate(final=False):
        nonlocal evaluation_index
        evaluation_index += 1
        snapshot = elapsed()
        if split:
            _, metrics = strategy.evaluate(evaluation_index, client_parameters=self.client_parameters,
                                           server_parameters=self.server_parameters)
        else:
            _, metrics = strategy.evaluate(evaluation_index, self.parameters)
        strategy.evaluation_records[-1].update(
            model_version=accepted_rounds, snapshot_elapsed_sec=snapshot,
            within_time_budget=strategy.evaluation_records[-1]["elapsed_sec"] <= self.time_budget_sec,
            evaluation_kind="final_frozen_model" if final else "checkpoint")
        if final:
            strategy.evaluation_records[-1]["model_committed_elapsed_sec"] = last_commit_time
        history.add_metrics_centralized(evaluation_index, metrics)

    evaluate()
    next_evaluation = elapsed() + self.evaluation_interval_sec
    last_commit_time = 0.0
    while elapsed() < self.time_budget_sec:
        attempted_rounds += 1
        before = {name: value.detach().clone() for name, value in strategy.model.state_dict().items()}
        parameters_before = copy.deepcopy(getattr(self, attr))
        tails_before = copy.deepcopy(self.server_parameters) if split else None
        result = self.fit_round(attempted_rounds, timeout)
        if result is None or strategy.fit_failures:
            raise RuntimeError("Timed training did not return a complete failure-free round")
        now = elapsed()
        accepted = now < self.time_budget_sec
        fits = [row for row in strategy.fit_records if row["round_id"] == attempted_rounds]
        for row in fits:
            row.update(update_status="aggregated" if accepted else "discarded_at_stop",
                       aggregation_elapsed_sec=now,
                       aggregated_model_version=accepted_rounds + 1 if accepted else None)
        if not accepted:
            setattr(self, attr, parameters_before)
            if split:
                self.server_parameters = tails_before
            strategy.model.load_state_dict(before)
            break
        if split:
            parameters, tails, fit_metrics, _ = result
            if tails:
                self.server_parameters = tails
        else:
            parameters, fit_metrics, _ = result
        if parameters:
            setattr(self, attr, parameters)
        arrays = parameters_to_ndarrays(getattr(self, attr))
        if split:
            strategy.backend_adapter.load_ndarrays(strategy.model, arrays)
        else:
            strategy.model.load_state_dict({name: torch.from_numpy(value.copy())
                for name, value in zip(strategy.model.state_dict(), arrays, strict=True)})
        now = elapsed()
        if now >= self.time_budget_sec:
            setattr(self, attr, parameters_before)
            if split:
                self.server_parameters = tails_before
            strategy.model.load_state_dict(before)
            for row in fits:
                row.update(update_status="discarded_at_stop", aggregation_elapsed_sec=now,
                           aggregated_model_version=None)
            break
        accepted_rounds += 1
        for row in fits:
            row["aggregation_elapsed_sec"] = now
        last_commit_time = now
        history.add_metrics_distributed_fit(attempted_rounds, fit_metrics)
        if now >= next_evaluation:
            evaluate()
            next_evaluation = elapsed() + self.evaluation_interval_sec
    stopped = elapsed()
    evaluate(final=True)
    self.budget_metadata = {
        "aggregation_protocol": "synchronous", "time_budget_sec": self.time_budget_sec,
        "evaluation_interval_sec": self.evaluation_interval_sec,
        "evaluation_policy": "initial, wall-clock cadence and final frozen model",
        "rounds": attempted_rounds, "global_updates": accepted_rounds,
        "last_committed_model_elapsed_sec": last_commit_time,
        "training_stopped_elapsed_sec": min(stopped, self.time_budget_sec),
        "drain_duration_sec": max(0.0, stopped - self.time_budget_sec),
        "budget_interpretation": "wall_clock", "stop_reason": "time_budget",
    }
    return history, elapsed()
