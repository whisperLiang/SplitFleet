"""Time to a frozen quality threshold at recorded evaluation checkpoints.

Times include setup and evaluation from the server's monotonic start clock.
No interpolation or extrapolation is performed between observations.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
from collections import defaultdict


def time_to_quality(
    evaluations: Sequence[dict[str, Any]], *, metric: str, threshold: float,
) -> dict[str, Any]:
    if not math.isfinite(threshold):
        raise ValueError("Quality threshold must be finite")
    base = {"metric": metric, "threshold": threshold, "time_sec": None,
            "round_id": None, "censor_time_sec": None}
    if not evaluations:
        return {**base, "status": "unavailable", "reason": "No evaluation checkpoints"}
    if any("elapsed_sec" not in row for row in evaluations):
        return {**base, "status": "unavailable", "reason": "Monotonic checkpoint times were not recorded"}
    previous_round, previous_time = -1, -1.0
    crossing = None
    for row in evaluations:
        round_id, elapsed = int(row["round_id"]), float(row["elapsed_sec"])
        quality = float(row["metrics"][metric])
        if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(quality):
            raise ValueError("Evaluation quality and elapsed time must be finite; time must be nonnegative")
        if round_id <= previous_round or elapsed < previous_time:
            raise ValueError("Evaluation rounds and monotonic times must be ordered")
        if crossing is None and quality >= threshold:
            crossing = {**base, "status": "reached", "time_sec": elapsed,
                        "round_id": round_id, "quality_at_crossing": quality}
        previous_round, previous_time = round_id, elapsed
    return crossing or {**base, "status": "not_reached", "censor_time_sec": previous_time}


def summarize_paired_ttq(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize only valid complete pairs, without a survivor-only mean.

    If any eligible seed is censored or unavailable, the population's mean TTQ
    remains unknown. Observed crossings stay available in the per-seed rows.
    """
    groups = defaultdict(list)
    for row in rows:
        key = (row["stage"], row.get("task", ""), row["scheme"], row["metric"], row["threshold"])
        groups[key].append(row)
    summaries = []
    for (stage, task, scheme, metric, threshold), values in sorted(groups.items()):
        eligible = [row for row in values if row.get("valid") and row.get("complete_paired_seed")]
        if len({row["seed"] for row in eligible}) != len(eligible):
            raise ValueError("TTQ summary has duplicate paired seeds")
        if any(row["status"] not in ("reached", "not_reached", "unavailable") for row in eligible):
            raise ValueError("Unknown TTQ status")
        reached = [row for row in eligible if row["status"] == "reached"]
        if any(row["time_sec"] is None or not math.isfinite(row["time_sec"]) or row["time_sec"] < 0
               for row in reached):
            raise ValueError("Reached thresholds require finite nonnegative times")
        summaries.append({"stage": stage, "task": task, "scheme": scheme, "metric": metric, "threshold": threshold,
            "complete_paired_n": len(eligible), "reached_n": len(reached),
            "censored_n": sum(row["status"] == "not_reached" for row in eligible),
            "unavailable_n": sum(row["status"] == "unavailable" for row in eligible),
            "mean_ttq_sec": sum(row["time_sec"] for row in reached) / len(reached)
                if reached and len(reached) == len(eligible) else None})
    return summaries
