"""Aggregate valid measured runs across seeds with no missing-data imputation."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

import yaml

from .config_utils import stable_hash
from .metrics import descriptive_stats
from .statistics import holm_adjust, paired_effect
from .validate_results import validate_run


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _protocol_hash(config: dict[str, Any], metadata: dict[str, Any]) -> str:
    # Run selection and destination are administrative, not experimental factors.
    factors = {key: value for key, value in config.items()
               if key not in {"methods", "seeds", "results_root"}}
    environment = {
        key: metadata.get(key)
        for key in ("hostname", "torch_version", "cuda_version", "device_names")
    }
    return stable_hash({"config": factors, "environment": environment})


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
        report = validate_run(run_dir)
        if not report["valid"]:
            skipped.append({"run_id": run_dir.name, "errors": report["errors"]})
            warnings.warn(f"Skipping invalid run {run_dir.name}: {report['errors']}", RuntimeWarning)
            continue
        config_path = run_dir / "resolved_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {} if config_path.exists() else {}
        run_configs[run_dir.name] = config
        protocol_hash = _protocol_hash(config, metadata)
        for collection, name in (
            (round_rows, "round_metrics.jsonl"),
            (client_rows, "client_metrics.jsonl"),
            (decision_rows, "split_decisions.jsonl"),
            (resource_rows, "resource_metrics.jsonl"),
        ):
            for row in _jsonl(run_dir / name):
                row.setdefault("method", metadata.get("method"))
                row.setdefault("seed", metadata.get("seed"))
                if "round_time_ms" not in row and "selected_makespan_ms" in row:
                    row["round_time_ms"] = row["selected_makespan_ms"]
                row["source_run_id"] = run_dir.name
                row["experiment"] = config.get("experiment")
                row["protocol_hash"] = protocol_hash
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
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("experiment") or "unknown"), str(row["protocol_hash"]),
            str(row["method"]), int(row["round_id"]),
        )
        groups.setdefault(key, []).append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment", "protocol_hash", "method", "round_id", "metric", "mean", "std", "median", "p95", "confidence_interval_95", "number_of_valid_runs"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (experiment, protocol_hash, method, round_id), values in sorted(groups.items()):
            for metric in numeric:
                stats = descriptive_stats(row[metric] for row in values if row.get(metric) is not None)
                stats["confidence_interval_95"] = json.dumps(stats["confidence_interval_95"])
                writer.writerow({"experiment": experiment, "protocol_hash": protocol_hash, "method": method, "round_id": round_id, "metric": metric, **stats})


def _aggregate_seed_summaries(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = ["round_time_ms", "test_accuracy", "macro_f1", "server_queue_ms", "client_idle_wait_ms", "oom_count", "timeout_count"]
    per_run: dict[tuple[str, str, str, int, str], list[float]] = {}
    for row in rows:
        for metric in metrics:
            if row.get(metric) is not None:
                key = (
                    str(row.get("experiment") or "unknown"), str(row["protocol_hash"]),
                    str(row["method"]), int(row["seed"]), metric,
                )
                per_run.setdefault(key, []).append(float(row[metric]))
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for (experiment, protocol_hash, method, _, metric), values in per_run.items():
        groups.setdefault((experiment, protocol_hash, method, metric), []).append(sum(values) / len(values))
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment", "protocol_hash", "method", "metric", "mean", "std", "median", "p95", "confidence_interval_95", "number_of_valid_runs"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (experiment, protocol_hash, method, metric), values in sorted(groups.items()):
            stats = descriptive_stats(values)
            stats["confidence_interval_95"] = json.dumps(stats["confidence_interval_95"])
            writer.writerow({"experiment": experiment, "protocol_hash": protocol_hash, "method": method, "metric": metric, **stats})


def _paired_comparisons(rows: list[dict[str, Any]], path: Path) -> None:
    protocols = sorted({
        (str(row.get("experiment") or "unknown"), str(row["protocol_hash"]))
        for row in rows
    })
    records: list[dict[str, Any]] = []
    for experiment, protocol_hash in protocols:
        selected = [
            row for row in rows
            if (str(row.get("experiment") or "unknown"), str(row["protocol_hash"]))
            == (experiment, protocol_hash)
        ]
        methods = sorted({str(row["method"]) for row in selected})
        summaries: dict[tuple[str, int, str], float] = {}
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in selected:
            grouped.setdefault((str(row["method"]), int(row["seed"])), []).append(row)
        for (method, seed), values in grouped.items():
            ordered = sorted(values, key=lambda row: int(row["round_id"]))
            summaries[(method, seed, "mean_round_time_ms")] = sum(
                float(row["round_time_ms"]) for row in ordered
            ) / len(ordered)
            for metric in ("test_accuracy", "macro_f1", "worst_client_accuracy"):
                available = [row for row in ordered if row.get(metric) is not None]
                if available:
                    summaries[(method, seed, f"final_{metric}")] = float(available[-1][metric])
        metrics = sorted({key[2] for key in summaries})
        for metric in metrics:
            for index, first in enumerate(methods):
                for second in methods[index + 1 :]:
                    left, right = (
                        (second, first) if second == "cosplit_ucb"
                        else (first, second)
                    )
                    seeds = sorted(
                        {seed for method, seed, name in summaries if method == left and name == metric}
                        & {seed for method, seed, name in summaries if method == right and name == metric}
                    )
                    if len(seeds) < 2:
                        continue
                    effect = paired_effect(
                        [summaries[(left, seed, metric)] for seed in seeds],
                        [summaries[(right, seed, metric)] for seed in seeds],
                        seed=2026 + sum(seeds),
                    )
                    lower, upper = effect.confidence_interval_95
                    margin = 0.005 if left == "cosplit_ucb" and metric == "final_test_accuracy" else None
                    records.append(
                        {
                            "experiment": experiment,
                            "protocol_hash": protocol_hash,
                            "metric": metric,
                            "method_a": left,
                            "method_b": right,
                            "paired_seeds": json.dumps(seeds),
                            "paired_n": effect.paired_n,
                            "mean_difference": effect.mean_difference,
                            "median_difference": effect.median_difference,
                            "relative_difference": effect.relative_difference,
                            "confidence_interval_95": json.dumps([lower, upper]),
                            "cohens_dz": effect.cohens_dz,
                            "sign_flip_p_value": effect.sign_flip_p_value,
                            "holm_adjusted_p_value": None,
                            "noninferiority_margin": margin,
                            "method_a_noninferior": lower >= -margin if margin is not None and effect.paired_n >= 8 else None,
                        }
                    )
    by_family: dict[tuple[str, str, str], list[int]] = {}
    for index, record in enumerate(records):
        by_family.setdefault((record["experiment"], record["protocol_hash"], record["metric"]), []).append(index)
    for indices in by_family.values():
        adjusted = holm_adjust(records[index]["sign_flip_p_value"] for index in indices)
        for index, value in zip(indices, adjusted, strict=True):
            records[index]["holm_adjusted_p_value"] = value
    fields = [
        "experiment", "protocol_hash", "metric", "method_a", "method_b", "paired_seeds", "paired_n",
        "mean_difference", "median_difference", "relative_difference",
        "confidence_interval_95", "cohens_dz", "sign_flip_p_value",
        "holm_adjusted_p_value", "noninferiority_margin", "method_a_noninferior",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(aggregate(args.results_root, args.output))


if __name__ == "__main__":
    main()
