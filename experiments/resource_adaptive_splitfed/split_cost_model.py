"""Profile-backed, online-updated split cost model."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class SplitCostPrediction:
    split_key: str
    predicted_round_ms: float
    predicted_client_compute_ms: float
    predicted_network_ms: float
    predicted_server_compute_ms: float
    predicted_server_queue_ms: float
    predicted_client_peak_memory_mb: float
    predicted_server_gpu_time_ms: float
    predicted_switch_ms: float
    feasible: bool
    infeasible_reason: str | None
    predicted_server_peak_memory_mb: float | None = None


class SplitCostModel:
    """Lookup/linear-scaling model fitted only from successfully executed batches."""

    def __init__(self, *, ema_alpha: float = 0.25) -> None:
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1].")
        self.ema_alpha = float(ema_alpha)
        self._profiles: dict[tuple[str, str, int], dict[str, float | None]] = {}
        self._online: dict[tuple[str, str], dict[str, float]] = {}

    def fit(self, profile_records: Iterable[Any]) -> "SplitCostModel":
        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for raw in profile_records:
            record = _mapping(raw)
            if not bool(record.get("success")):
                continue
            key = (
                str(record.get("device_profile", "default")),
                str(record["split_key"]),
                int(record["batch_size"]),
            )
            grouped[key].append(record)
        if not grouped:
            raise ValueError("SplitCostModel.fit requires at least one successful real profile record.")
        metrics = (
            "client_forward_ms",
            "client_backward_ms",
            "server_forward_ms",
            "server_backward_ms",
            "network_upload_ms",
            "network_download_ms",
            "server_queue_ms",
            "client_peak_memory_mb",
            "server_peak_memory_mb",
            "boundary_forward_bytes",
            "boundary_gradient_bytes",
            "runtime_prepare_ms",
            "client_compute_score",
        )
        for key, records in grouped.items():
            summary: dict[str, float | None] = {}
            for metric in metrics:
                values = [float(item[metric]) for item in records if item.get(metric) is not None]
                summary[metric] = statistics.fmean(values) if values else None
            self._profiles[key] = summary
        return self

    def _profile_group(self, device_profile: str, split_key: str) -> dict[int, dict[str, float | None]]:
        group = {
            profile_batch: values
            for (profile_device, profile_split, profile_batch), values in self._profiles.items()
            if profile_device == device_profile and profile_split == split_key
        }
        if group:
            return group
        for (_, profile_split, profile_batch), values in self._profiles.items():
            if profile_split == split_key:
                group.setdefault(profile_batch, values)
        return group

    def _lookup(self, device_profile: str, split_key: str, batch_size: int) -> dict[str, float | None]:
        group = self._profile_group(device_profile, split_key)
        if not group:
            raise KeyError(f"No successful profile exists for split {split_key!r}.")
        source_batch = (
            batch_size if batch_size in group else min(group, key=lambda item: abs(item - batch_size))
        )
        scaled = dict(group[source_batch])
        if source_batch != batch_size:
            scale = batch_size / max(source_batch, 1)
            for metric in (
                "client_forward_ms",
                "client_backward_ms",
                "server_forward_ms",
                "server_backward_ms",
                "network_upload_ms",
                "network_download_ms",
                "boundary_forward_bytes",
                "boundary_gradient_bytes",
            ):
                if scaled.get(metric) is not None:
                    scaled[metric] = float(scaled[metric]) * scale
        # Peak memory is a fixed model/optimizer footprint plus a per-sample
        # activation footprint, so it is neither batch-invariant nor linear.
        # Leaving it at the profiled batch's value under-reports the memory a
        # larger batch really needs, which is the one direction the feasibility
        # test must never fail in.
        for metric in ("client_peak_memory_mb", "server_peak_memory_mb"):
            scaled[metric] = _memory_at_batch(group, metric, batch_size)
        return scaled

    def predict(
        self,
        client_state: Any,
        server_state: Any,
        split_candidate: Any,
    ) -> SplitCostPrediction:
        client = _mapping(client_state)
        server = _mapping(server_state)
        candidate = _mapping(split_candidate)
        split_key = str(candidate["split_key"])
        device_profile = str(client.get("device_profile", "default"))
        batch_size = int(client.get("batch_size", candidate.get("batch_size", 1)))
        try:
            profile = self._lookup(device_profile, split_key, batch_size)
        except KeyError as exc:
            return SplitCostPrediction(
                split_key, math.inf, math.inf, math.inf, math.inf, math.inf,
                math.inf, math.inf, 0.0, False, str(exc), None,
            )

        client_compute = _sum(profile, "client_forward_ms", "client_backward_ms")
        server_compute = _sum(profile, "server_forward_ms", "server_backward_ms")
        profiled_compute_score = _positive(profile.get("client_compute_score"))
        current_compute_score = _positive(client.get("client_compute_score"))
        if profiled_compute_score is not None and current_compute_score is not None:
            compute_scale = min(10.0, max(0.1, profiled_compute_score / current_compute_score))
            client_compute *= compute_scale
        server_gpu_utilization = server.get("server_gpu_utilization")
        if server_gpu_utilization is not None:
            remaining_capacity = max(0.25, 1.0 - float(server_gpu_utilization) / 100.0)
            server_compute /= remaining_capacity
        forward_bytes = _first_number(
            candidate.get("boundary_forward_bytes"), profile.get("boundary_forward_bytes")
        )
        gradient_bytes = _first_number(
            candidate.get("boundary_gradient_bytes"), profile.get("boundary_gradient_bytes")
        )
        uplink = _positive(client.get("uplink_mbps"))
        downlink = _positive(client.get("downlink_mbps"))
        rtt = max(0.0, float(client.get("rtt_ms") or 0.0))
        if split_key == "full_local":
            network = 0.0
            server_compute = 0.0
        elif uplink is not None and downlink is not None:
            network = (
                forward_bytes * 8.0 / (uplink * 1000.0)
                + gradient_bytes * 8.0 / (downlink * 1000.0)
                + rtt
            )
        else:
            network = _sum(profile, "network_upload_ms", "network_download_ms")

        active = int(server.get("server_active_jobs", 0) or 0)
        queued = int(server.get("server_queue_length", 0) or 0)
        concurrency = max(1, int(server.get("max_server_concurrency", 1) or 1))
        measured_queue = float(profile.get("server_queue_ms") or 0.0)
        queue_waves = max(0, active + queued + 1 - concurrency) / concurrency
        server_queue = measured_queue + queue_waves * server_compute
        switch_ms = 0.0
        if str(client.get("current_split_key", split_key)) != split_key:
            # An explicit 0.0 is a real prediction (the `no_switch_cost` ablation
            # sets exactly that) and must not fall through to the profiled cost.
            switch_ms = _first_number(
                candidate.get("predicted_switch_ms"), profile.get("runtime_prepare_ms")
            )

        client_memory = float(profile.get("client_peak_memory_mb") or 0.0)
        server_memory_value = profile.get("server_peak_memory_mb")
        server_memory = None if server_memory_value is None else float(server_memory_value)
        feasible = True
        reason = None
        available_client = _positive(client.get("client_available_memory_mb"))
        if available_client is not None and client_memory > available_client:
            feasible, reason = False, "client_memory"
        available_server = _positive(
            server.get("server_gpu_memory_free_mb")
            if server.get("server_gpu_memory_free_mb") is not None
            else server.get("server_available_memory_mb")
        )
        if (
            feasible
            and split_key != "full_local"
            and available_server is not None
            and server_memory is not None
            and server_memory > available_server
        ):
            feasible, reason = False, "server_memory"

        # Every profiled compute/network metric is per batch, while a prediction
        # is compared against a whole round. Scaling here keeps the profile path
        # and the observed `completion_ms` in the same unit.
        batches = max(1, int(client.get("batches_per_round", 1) or 1))
        client_compute *= batches
        network *= batches
        server_compute *= batches
        server_queue *= batches

        online = self._online.get((str(client.get("client_id", "")), split_key))
        if online is not None:
            # Preserve the current resource-sensitive terms.  Online observations
            # calibrate systematic model error instead of freezing a completion
            # time measured under an old CPU/network/server state.  Scaling the
            # components as well as their sum keeps ablations and the global
            # server-lane scheduler internally consistent.
            scale = float(online["scale"])
            client_compute *= scale
            network *= scale
            server_compute *= scale
            server_queue *= scale
            switch_ms *= scale
        total = client_compute + network + server_compute + server_queue + switch_ms
        return SplitCostPrediction(
            split_key=split_key,
            predicted_round_ms=total,
            predicted_client_compute_ms=client_compute,
            predicted_network_ms=network,
            predicted_server_compute_ms=server_compute,
            predicted_server_queue_ms=server_queue,
            predicted_client_peak_memory_mb=client_memory,
            predicted_server_gpu_time_ms=server_compute,
            predicted_switch_ms=switch_ms,
            feasible=feasible,
            infeasible_reason=reason,
            predicted_server_peak_memory_mb=server_memory,
        )

    def update_online(self, observation: Any) -> None:
        """Blend one measured round into a resource-sensitive correction scale.

        ``predicted_completion_ms`` is the prediction used for the measured
        round.  It already contains the previous scale, so the new target scale
        removes that scale before applying the observed/predicted correction.
        Callers predating this field retain a neutral scale rather than turning
        the observation into a resource-insensitive absolute prediction.
        """

        record = _mapping(observation)
        if not bool(record.get("success", True)) or record.get("completion_ms") is None:
            return
        key = (str(record["client_id"]), str(record["split_key"]))
        observed = float(record["completion_ms"])
        predicted = float(record.get("predicted_completion_ms", observed))
        if not math.isfinite(observed) or not math.isfinite(predicted) or predicted <= 0:
            return
        previous = self._online.get(key)
        previous_scale = float(previous["scale"]) if previous is not None else 1.0
        target_scale = previous_scale * observed / predicted
        scale = target_scale if previous is None else (
            self.ema_alpha * target_scale + (1.0 - self.ema_alpha) * previous_scale
        )
        self._online[key] = {"scale": max(scale, 1e-9)}


def _memory_at_batch(
    group: Mapping[int, Mapping[str, Any]],
    metric: str,
    batch_size: int,
) -> float | None:
    """Predict peak memory at ``batch_size`` from the profiled batch sizes."""

    points = sorted(
        (int(profile_batch), float(values[metric]))
        for profile_batch, values in group.items()
        if values.get(metric) is not None
    )
    if not points:
        return None
    for profile_batch, value in points:
        if profile_batch == batch_size:
            return value
    if len(points) >= 2:
        slope, intercept = _least_squares(points)
        fitted = intercept + slope * batch_size
        # Peak memory is monotonic in batch size, so a fit must never fall below
        # a measurement taken at a smaller batch.
        floor = max(
            (value for profile_batch, value in points if profile_batch <= batch_size),
            default=0.0,
        )
        return max(fitted, floor, 0.0)
    # A single profiled batch cannot separate the fixed and per-sample parts.
    # Scaling the whole measurement over-estimates the fixed part, which keeps
    # the feasibility test conservative instead of optimistic.
    profile_batch, value = points[0]
    return value * (batch_size / max(profile_batch, 1))


def _least_squares(points: Sequence[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return slope, mean_y - slope * mean_x


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return dict(vars(value))


def _sum(record: Mapping[str, Any], *keys: str) -> float:
    return sum(float(record.get(key) or 0.0) for key in keys)


def _positive(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if number > 0 else None


def _first_number(*values: Any) -> float:
    """Return the first value that was actually provided, keeping an explicit 0."""

    for value in values:
        if value is not None:
            return float(value)
    return 0.0
