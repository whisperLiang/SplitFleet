"""Paired wall-time reporting for complete, equally resourced training runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import t

from .queue import read_json
from .protocol import SEEDS

TRAINING_WALL_SCOPE = (
    "Measured immediately before runtime invocation through its return after cleanup; "
    "includes model construction, Ray startup, graph capture, placement/calibration, "
    "worker and suffix waits, state synchronization, RPC, all SGD updates, aggregation "
    "and all scheduled validation passes, final checkpoint and runtime shutdown. "
    "Excludes the two final test evaluations, interpreter startup, experiment imports "
    "and frozen input verification before runtime invocation."
)


def measured_training_time(directory):
    """Read a persisted whole-run clock; never synthesize it from task durations."""
    receipt = read_json(Path(directory) / "result.json")
    seconds = receipt.get("wall_sec")
    if (receipt.get("status") != "passed" or receipt.get("wall_scope") != TRAINING_WALL_SCOPE
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not np.isfinite(seconds) or seconds <= 0):
        raise ValueError("Complete training requires a positive measured wall_sec receipt")
    return {"wall_seconds": float(seconds), "source": "result.json:wall_sec",
            "scope": TRAINING_WALL_SCOPE}


def validate_timing_pair(left, right):
    """Different hardware or runtime versions cannot form an efficiency pair."""
    for key in ("source_identity", "device", "CUDA_VISIBLE_DEVICES", "native_environment"):
        if key not in left["manifest"] or key not in right["manifest"]:
            raise ValueError(f"Timing comparison is missing its {key} identity")
        if left["manifest"][key] != right["manifest"][key]:
            raise ValueError(f"Timing pair differs in {key}")
    for key in ("resources", "profiles", "client_profiles", "server_device", "networks",
                "network_by_round", "batch_size", "local_epochs", "rounds", "optimizer",
                "learning_rate", "virtual_clients", "transport"):
        if key not in left["config"] or key not in right["config"]:
            raise ValueError(f"Timing comparison is missing its {key} configuration")
        if left["config"][key] != right["config"][key]:
            raise ValueError(f"Timing pair differs in {key}")


def paired_efficiency_statistics(fedavg_seconds, splitfleet_seconds):
    """Use paired log ratios so a speedup interval remains positive."""
    fedavg = np.asarray(fedavg_seconds, dtype=np.float64)
    splitfleet = np.asarray(splitfleet_seconds, dtype=np.float64)
    if fedavg.shape != (len(SEEDS),) or splitfleet.shape != (len(SEEDS),):
        raise ValueError("The timing comparison needs all two paired seeds")
    if (not np.isfinite(fedavg).all() or not np.isfinite(splitfleet).all()
            or (fedavg <= 0).any() or (splitfleet <= 0).any()):
        raise ValueError("Measured training times must be finite and positive")
    log_ratios = np.log(fedavg) - np.log(splitfleet)
    mean = float(log_ratios.mean())
    half_width = float(t.ppf(.975, len(SEEDS) - 1) * log_ratios.std(ddof=1) / np.sqrt(len(SEEDS)))
    interval = np.exp([mean - half_width, mean + half_width]).tolist()
    speedup = float(np.exp(mean))
    return {
        "fedavg_wall_seconds": fedavg.tolist(),
        "splitfleet_wall_seconds": splitfleet.tolist(),
        "fedavg_wall_mean_seconds": float(fedavg.mean()),
        "fedavg_wall_sample_sd_seconds": float(fedavg.std(ddof=1)),
        "splitfleet_wall_mean_seconds": float(splitfleet.mean()),
        "splitfleet_wall_sample_sd_seconds": float(splitfleet.std(ddof=1)),
        "paired_speedups": np.exp(log_ratios).tolist(),
        "paired_time_reduction_pct": (100 * (1 - splitfleet / fedavg)).tolist(),
        "geometric_mean_speedup": speedup,
        "geometric_time_reduction_pct": 100 * (1 - 1 / speedup),
        "paired_95pct_t_interval_speedup": interval,
        "speedup_interval_above_one": interval[0] > 1,
        "degrees_of_freedom": len(SEEDS) - 1,
        "scope": TRAINING_WALL_SCOPE,
        "assumption": (
            "Paired log time ratios are independent and approximately normally distributed; "
            "two seeds give limited precision. Speedup is FedAvg time / SplitFleet time."
        ),
    }
