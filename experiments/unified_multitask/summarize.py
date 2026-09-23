"""Summarize validated multi-task runs without inventing paired observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.resource_adaptive_splitfed.statistics import holm_adjust, paired_effect

from .validate import validate_run


def summarize(
    results_root: str | Path,
    output: str | Path,
    *,
    include_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = Path(results_root)
    runs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if include_prefixes and not any(path.name.startswith(prefix) for prefix in include_prefixes):
            continue
        if not path.is_dir() or not (path / "metadata.json").exists():
            continue
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema") != "splitfleet.unified-benchmark.v1":
            continue
        report = validate_run(path)
        if not report["valid"]:
            excluded.append({"run_id": path.name, "errors": report["errors"]})
            continue
        if not metadata.get("device") or not metadata.get("hostname"):
            excluded.append({
                "run_id": path.name,
                "errors": ["Run lacks device/hostname provenance required for paired timing."],
            })
            continue
        rows = [json.loads(line) for line in (path / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        primary_metric = str(rows[-1]["primary_metric"])
        method_config = {}
        if metadata["method"] == "splitfed_fixed":
            method_config["fixed_boundary"] = metadata.get("fixed_boundary")
        elif metadata["method"] == "fedprox":
            method_config["proximal_mu"] = metadata.get("proximal_mu")
        runs.append({
            "run_id": path.name,
            "task": metadata["task"],
            "source": metadata["source"],
            "method": metadata["method"],
            "method_config": method_config,
            "seed": int(metadata["seed"]),
            "rounds": int(metadata["rounds"]),
            "protocol": {
                "num_clients": int(metadata["num_clients"]),
                "batch_size": int(metadata["batch_size"]),
                "max_batches_per_client": metadata.get("max_batches_per_client"),
                "train_examples": int(metadata["train_examples"]),
                "test_examples": int(metadata["test_examples"]),
                "learning_rate": float(metadata["learning_rate"]),
                "dirichlet_alpha": metadata.get("dirichlet_alpha"),
                "candidate_cuts": metadata["candidate_cuts"],
                "sample_selection": metadata.get("sample_selection"),
                "hostname": metadata["hostname"],
                "device": metadata["device"],
                "device_name": metadata.get("device_name"),
                "torch_version": metadata.get("torch_version"),
            },
            "primary_metric": primary_metric,
            "final_task_metric": float(rows[-1]["metrics"][primary_metric]),
            "total_wall_sec": sum(float(row["round_time_sec"]) for row in rows),
            "total_boundary_bytes": sum(
                int(row["boundary_upload_bytes"]) + int(row["boundary_download_bytes"])
                for row in rows
            ),
            "initial_model_hash": metadata["initial_model_hash"],
            "partition_hash": metadata["partition_hash"],
            "data_content_hash": metadata["data_content_hash"],
        })
    groups: dict[tuple[str, str, int, str], dict[tuple[str, str], dict[int, dict[str, Any]]]] = {}
    for run in runs:
        protocol_key = json.dumps(run["protocol"], sort_keys=True)
        key = (run["task"], run["source"], run["rounds"], protocol_key)
        variant = (run["method"], json.dumps(run["method_config"], sort_keys=True))
        by_seed = groups.setdefault(key, {}).setdefault(variant, {})
        if run["seed"] in by_seed:
            raise ValueError(f"Duplicate task/source/method/seed run: {key}, {variant}, {run['seed']}")
        by_seed[run["seed"]] = run
    comparisons: list[dict[str, Any]] = []
    for (task, source, rounds, protocol_key), methods in sorted(groups.items()):
        proposed_key = ("splitfleet", "{}")
        if proposed_key not in methods:
            continue
        for baseline_key in sorted(name for name in methods if name != proposed_key):
            baseline, baseline_config_json = baseline_key
            seeds = sorted(set(methods[proposed_key]) & set(methods[baseline_key]))
            if not seeds:
                continue
            for seed in seeds:
                proposed = methods[proposed_key][seed]
                control = methods[baseline_key][seed]
                if (
                    proposed["initial_model_hash"] != control["initial_model_hash"]
                    or proposed["partition_hash"] != control["partition_hash"]
                    or proposed["data_content_hash"] != control["data_content_hash"]
                ):
                    raise ValueError(f"Pair identity mismatch for {task}/{baseline}/seed={seed}")
            row: dict[str, Any] = {
                "task": task,
                "source": source,
                "rounds": rounds,
                "protocol": json.loads(protocol_key),
                "proposed": "splitfleet",
                "baseline": baseline,
                "baseline_config": json.loads(baseline_config_json),
                "paired_seeds": seeds,
                "paired_n": len(seeds),
                "evidence_status": "fixture_smoke" if source == "fixture" else (
                    "real_data_pilot" if len(seeds) < 5 or rounds < 100 else "exploratory_multi_seed"
                ),
            }
            for metric in ("final_task_metric", "total_wall_sec", "total_boundary_bytes"):
                left = [methods[proposed_key][seed][metric] for seed in seeds]
                right = [methods[baseline_key][seed][metric] for seed in seeds]
                if len(seeds) >= 2:
                    row[metric] = paired_effect(left, right, seed=2026 + sum(seeds)).to_dict()
                else:
                    row[metric] = {
                        "mean_difference": left[0] - right[0],
                        "paired_n": 1,
                        "confidence_interval_95": None,
                        "sign_flip_p_value": None,
                    }
            comparisons.append(row)
    for task, source, rounds, protocol_key in sorted(groups):
        family = [row for row in comparisons if (
            row["task"], row["source"], row["rounds"], json.dumps(row["protocol"], sort_keys=True)
        ) == (task, source, rounds, protocol_key)]
        for metric in ("final_task_metric", "total_wall_sec", "total_boundary_bytes"):
            eligible = [row for row in family if row[metric]["sign_flip_p_value"] is not None]
            if not eligible:
                continue
            adjusted = holm_adjust(row[metric]["sign_flip_p_value"] for row in eligible)
            for row, value in zip(eligible, adjusted, strict=True):
                row[metric]["holm_adjusted_p_value"] = value
    report = {
        "schema": "splitfleet.unified-benchmark-summary.v1",
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "validation_v1",
        },
        "valid_runs": len(runs),
        "excluded_runs": excluded,
        "comparisons": comparisons,
        "interpretation": "Pilot and fixture observations cannot establish multi-task superiority.",
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-prefix", action="append", default=[])
    args = parser.parse_args(argv)
    report = summarize(
        args.results_root,
        args.output,
        include_prefixes=tuple(args.include_prefix),
    )
    print(json.dumps({"valid_runs": report["valid_runs"], "excluded_runs": len(report["excluded_runs"]), "comparisons": len(report["comparisons"])}, sort_keys=True))


if __name__ == "__main__":
    main()
