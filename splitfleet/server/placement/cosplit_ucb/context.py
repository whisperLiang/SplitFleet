"""Stable backend-neutral feature encoding for CoSplit-UCB."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .types import FEATURE_SCHEMA_VERSION, SplitCandidateDescriptor


def _number(values: Mapping[str, Any], key: str) -> tuple[float, float]:
    raw = values.get(key)
    if raw is None:
        return 0.0, 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, 1.0
    if not np.isfinite(value):
        return 0.0, 1.0
    return value, 0.0


def _ratio(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _utilization(value: float) -> float:
    # Providers may expose either [0, 1] ratios or percentages.
    return _ratio(value / 100.0 if value > 1.0 else value)


class ContextEncoder:
    """Encode fixed-schema numerical contexts without online renormalization."""

    feature_schema_version = FEATURE_SCHEMA_VERSION
    edge_dimension = 12
    network_dimension = 13
    server_dimension = 12
    switch_dimension = 9

    def edge_context(
        self,
        candidate: SplitCandidateDescriptor,
        telemetry: Mapping[str, Any] | None = None,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        values = telemetry or {}
        cpu, cpu_missing = _number(values, "cpu_utilization")
        gpu, gpu_missing = _number(values, "gpu_utilization")
        memory, memory_missing = _number(values, "memory_free_ratio")
        prefix_ratio = candidate.prefix_node_count / max(candidate.total_node_count, 1)
        return np.asarray(
            [
                1.0,
                candidate.graph_position_ratio,
                _ratio(prefix_ratio),
                np.log1p(max(candidate.prefix_parameter_bytes or 0, 0)),
                np.log1p(max(candidate.boundary_tensor_count, 0)),
                np.log1p(max(int(batch_size or values.get("batch_size") or 0), 0)),
                _utilization(cpu),
                cpu_missing,
                _utilization(gpu),
                gpu_missing,
                _ratio(memory),
                memory_missing,
            ],
            dtype=np.float64,
        )

    def network_context(
        self,
        candidate: SplitCandidateDescriptor,
        telemetry: Mapping[str, Any] | None = None,
        *,
        direction: str,
    ) -> np.ndarray:
        if direction not in {"upload", "download"}:
            raise ValueError("network direction must be 'upload' or 'download'")
        values = telemetry or {}
        forward_bytes = candidate.boundary_forward_bytes
        gradient_bytes = candidate.boundary_gradient_bytes
        uplink, uplink_missing = _number(values, "uplink_mbps")
        downlink, downlink_missing = _number(values, "downlink_mbps")
        rtt, rtt_missing = _number(values, "rtt_ms")
        return np.asarray(
            [
                1.0,
                np.log1p(max(forward_bytes or 0, 0)),
                1.0 if forward_bytes is None else 0.0,
                np.log1p(max(gradient_bytes or 0, 0)),
                1.0 if gradient_bytes is None else 0.0,
                np.log1p(max(candidate.boundary_tensor_count, 0)),
                np.log1p(max(uplink, 0.0)),
                uplink_missing,
                np.log1p(max(downlink, 0.0)),
                downlink_missing,
                np.log1p(max(rtt, 0.0)),
                rtt_missing,
                1.0 if direction == "download" else 0.0,
            ],
            dtype=np.float64,
        )

    def server_context(
        self,
        candidate: SplitCandidateDescriptor,
        telemetry: Mapping[str, Any] | None = None,
        *,
        max_concurrency: int,
    ) -> np.ndarray:
        values = telemetry or {}
        gpu, gpu_missing = _number(values, "gpu_utilization")
        active, active_missing = _number(values, "active_jobs")
        queue, queue_missing = _number(values, "queue_length")
        memory, memory_missing = _number(values, "memory_free_ratio")
        suffix_ratio = candidate.suffix_node_count / max(candidate.total_node_count, 1)
        return np.asarray(
            [
                1.0,
                _ratio(suffix_ratio),
                np.log1p(max(candidate.suffix_parameter_bytes or 0, 0)),
                _utilization(gpu),
                gpu_missing,
                np.log1p(max(active, 0.0)),
                active_missing,
                np.log1p(max(queue, 0.0)),
                queue_missing,
                np.log1p(max(int(max_concurrency), 1)),
                _ratio(memory),
                memory_missing,
            ],
            dtype=np.float64,
        )

    def switch_context(
        self,
        candidate: SplitCandidateDescriptor,
        *,
        previous: SplitCandidateDescriptor | None,
        telemetry: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        values = telemetry or {}
        cache_hit, cache_missing = _number(values, "placement_cache_hit")
        prepared, prepared_missing = _number(values, "placement_prepared")
        state_bytes, state_missing = _number(values, "state_bytes")
        changed = previous is not None and previous.boundary != candidate.boundary
        distance = (
            abs(previous.graph_position_ratio - candidate.graph_position_ratio)
            if previous is not None
            else 0.0
        )
        return np.asarray(
            [
                1.0,
                1.0 if changed else 0.0,
                _ratio(distance),
                _ratio(cache_hit),
                cache_missing,
                _ratio(prepared),
                prepared_missing,
                np.log1p(max(state_bytes, 0.0)),
                state_missing,
            ],
            dtype=np.float64,
        )


__all__ = ["ContextEncoder"]
