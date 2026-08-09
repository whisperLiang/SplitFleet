"""Aggregate valid measured runs across seeds with no missing-data imputation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Any, Mapping

import yaml

from .metrics import descriptive_stats
from .validate_results import validate_run


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate(results_root: str | Path, output: str | Path) -> Path:
    root = Path(results_root)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    round_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    run_configs: dict[str, Any] = {}
    skipped = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and path != destination):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("method") == "resource_profile":
            continue
        report = validate_run(run_dir)
        if not report["valid"]:
            skipped.append({"run_id": run_dir.name, "errors": report["errors"]})
            warnings.warn(f"Skipping invalid run {run_dir.name}: {report['errors']}", RuntimeWarning)
            continue
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        run_configs[run_dir.name] = config
        for collection, name in (
            (round_rows, "round_metrics.jsonl"),
            (client_rows, "client_metrics.jsonl"),
            (decision_rows, "split_decisions.jsonl"),
            (resource_rows, "resource_metrics.jsonl"),
        ):
            for row in _jsonl(run_dir / name):
                row["source_run_id"] = run_dir.name
                row["experiment"] = config.get("experiment")
                row["num_clients"] = config.get("num_clients")
                row["server_concurrency"] = config.get("server_concurrency")
                collection.append(row)
    if not round_rows:
        raise RuntimeError("No valid measured training runs were available for aggregation.")
    for name, rows in (
        ("round_metrics.jsonl", round_rows),
        ("client_metrics.jsonl", client_rows),
        ("split_decisions.jsonl", decision_rows),
        ("resource_metrics.jsonl", resource_rows),
    ):
        with (destination / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    (destination / "manifest.json").write_text(
        json.dumps({"runs": sorted(run_configs), "configs": run_configs, "skipped": skipped}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _aggregate_rounds(round_rows, destination / "aggregated_round_metrics.csv")
    _aggregate_seed_summaries(round_rows, destination / "summary.csv")
    _paired_comparisons(round_rows, destination / "paired_comparisons.csv")
    return destination


def _aggregate_rounds(rows: list[dict[str, Any]], path: Path) -> None:
    numeric = sorted(
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"seed", "round_id"}
    )
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), int(row["round_id"])), []).append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "round_id", "metric", "mean", "std", "median", "p95", "confidence_interval_95", "number_of_valid_runs"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (method, round_id), values in sorted(groups.items()):
            for metric in numeric:
                stats = descriptive_stats(row[metric] for row in values if row.get(metric) is not None)
                stats["confidence_interval_95"] = json.dumps(stats["confidence_interval_95"])
                writer.writerow({"method": method, "round_id": round_id, "metric": metric, **stats})


def _aggregate_seed_summaries(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = ["round_time_ms", "test_accuracy", "macro_f1", "server_queue_ms", "client_idle_wait_ms", "oom_count", "timeout_count"]
    per_run: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        for metric in metrics:
            if row.get(metric) is not None:
                per_run.setdefault((str(row["method"]), int(row["seed"]), metric), []).append(float(row[metric]))
    groups: dict[tuple[str, str], list[float]] = {}
    for (method, _, metric), values in per_run.items():
        groups.setdefault((method, metric), []).append(sum(values) / len(values))
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "metric", "mean", "std", "median", "p95", "confidence_interval_95", "number_of_valid_runs"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (method, metric), values in sorted(groups.items()):
            stats = descriptive_stats(values)
            stats["confidence_interval_95"] = json.dumps(stats["confidence_interval_95"])
            writer.writerow({"method": method, "metric": metric, **stats})


def _paired_comparisons(rows: list[dict[str, Any]], path: Path) -> None:
    per_seed: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        per_seed.setdefault((str(row["method"]), int(row["seed"])), []).append(float(row["round_time_ms"]))
    means = {key: sum(values) / len(values) for key, values in per_seed.items()}
    methods = sorted({key[0] for key in means})
    fields = ["method_a", "method_b", "paired_seeds", "mean_difference_ms", "normal_approx_p_value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, left in enumerate(methods):
            for right in methods[index + 1 :]:
                seeds = sorted({seed for method, seed in means if method == left} & {seed for method, seed in means if method == right})
                differences = [means[(left, seed)] - means[(right, seed)] for seed in seeds]
                p_value = None
                if len(differences) >= 3:
                    mean = sum(differences) / len(differences)
                    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
                    if variance > 0:
                        statistic = mean / math.sqrt(variance / len(differences))
                        p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
                writer.writerow(
                    {
                        "method_a": left,
                        "method_b": right,
                        "paired_seeds": json.dumps(seeds),
                        "mean_difference_ms": sum(differences) / len(differences) if differences else None,
                        "normal_approx_p_value": p_value,
                    }
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(aggregate(args.results_root, args.output))


if __name__ == "__main__":
    main()
