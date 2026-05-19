"""Metric extraction helpers for Oort-style client selection."""

from __future__ import annotations

import math
from typing import Any, Mapping

from splitfleet.server.client_selection.base import ClientState


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        converted = float(value)
        if math.isfinite(converted):
            return converted
    return None


def extract_duration(metrics: Mapping[str, Any], default: float = 1.0) -> float:
    """Extract a positive fit duration from Flower metrics."""

    for key in ("fit_duration_sec", "train_duration_sec", "duration"):
        value = _as_float(metrics.get(key))
        if value is not None and value > 0:
            return value
    return max(float(default), 1e-12)


def _extract_num_examples(num_examples: int, metrics: Mapping[str, Any]) -> int:
    metric_examples = metrics.get("num_examples")
    if isinstance(metric_examples, bool):
        return max(int(num_examples), 0)
    if isinstance(metric_examples, (int, float)):
        return max(int(metric_examples), 0)
    return max(int(num_examples), 0)


def _extract_loss(
    metrics: Mapping[str, Any],
    previous_state: ClientState | None,
) -> float:
    loss = _as_float(metrics.get("loss"))
    if loss is None and previous_state is not None:
        loss = previous_state.last_loss
    if loss is None:
        loss = 0.0
    return max(float(loss), 0.0)


def compute_oort_reward(
    num_examples: int,
    metrics: Mapping[str, Any],
    previous_state: ClientState | None,
    config,
) -> float:
    """Compute statistical utility used by the Oort selector.

    The reward intentionally does not subtract training duration. Oort applies
    duration as a separate penalty during selection so the same signal is not
    counted twice.
    """

    _ = config
    examples = _extract_num_examples(num_examples, metrics)
    loss = _extract_loss(metrics, previous_state)
    return math.log1p(examples) * max(loss, 1e-12)
