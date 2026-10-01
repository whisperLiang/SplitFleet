"""Audit and summarize a paired physical four-task comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.physical_multitask import FIXED_BOUNDARIES, METHODS, TASKS, scheme_name


PRIMARY = {
    "image_classification": "accuracy",
    "text_classification": "macro_f1",
    "object_detection": "map50",
    "semantic_segmentation": "miou",
}
MODELS = {
    "image_classification": "ImageClassifier",
    "text_classification": "TextClassifier",
    "object_detection": "GridDetector",
    "semantic_segmentation": "Segmenter",
}
DATASETS = {
    "image_classification": "CIFAR-10",
    "text_classification": "AG News",
    "object_detection": "Pascal VOC 2007",
    "semantic_segmentation": "Oxford-IIIT Pet",
}
RUNTIME_SOURCES = (
    "experiments/physical_multitask.py",
    "experiments/orchestrate_physical_multitask.py",
    "splitfleet/autosplit/runtime.py",
    "splitfleet/autosplit/torchlens_backend.py",
    "splitfleet/autosplit/torchlens_runtime.py",
    "splitfleet/client/autosplit_split_client.py",
    "experiments/common/identity.py",
    "experiments/common/partition.py",
    "experiments/common/training.py",
    "experiments/unified_multitask/data.py",
    "experiments/unified_multitask/models.py",
    "experiments/unified_multitask/run.py",
    "splitfleet/server/placement/cosplit_ucb/candidate_provider.py",
    "splitfleet/server/placement/cosplit_ucb/config.py",
    "splitfleet/server/placement/cosplit_ucb/policy.py",
    "splitfleet/server/strategy/autosplit_strategy.py",
)


def _runtime_code_hash(folder: Path) -> str:
    source = json.loads((folder.parent / "source_hashes.json").read_text(encoding="utf-8"))
    if any(path not in source for path in RUNTIME_SOURCES):
        raise ValueError(f"runtime source hash is missing for {folder}")
    frozen = {path: source[path] for path in RUNTIME_SOURCES}
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if result.get("image_model") == "rfdetr_nano":
        path = "experiments/rfdetr_nano_physical.py"
        if path not in source:
            raise ValueError(f"RF-DETR runtime source hash is missing for {folder}")
        frozen[path] = source[path]
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def summarize(root: Path) -> dict[str, Any]:
    rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    results = {
        row["run_dir"]: json.loads((Path(row["run_dir"]) / "result.json").read_text(encoding="utf-8"))
        for row in rows
    }
    by_key = {
        (row["task"], scheme_name(row["method"], results[row["run_dir"]]["fixed_boundary"])): row
        for row in rows
    }
    if not rows or len(by_key) != len(rows):
        raise ValueError("comparison has no runs or contains duplicate task/scheme pairs")
    if any(row["task"] not in TASKS for row in rows):
        raise ValueError("comparison contains an unknown task")
    tasks = [task for task in TASKS if any(row["task"] == task for row in rows)]
    methods = [method for method in METHODS if any(row["method"] == method for row in rows)]
    scheme_order = [
        scheme_name(method, boundary)
        for method in METHODS
        for boundary in (FIXED_BOUNDARIES if method == "splitfed_fixed" else (None,))
    ]
    schemes = [scheme for scheme in scheme_order if any(key[1] == scheme for key in by_key)]
    if set(by_key) != {(task, scheme) for task in tasks for scheme in schemes}:
        raise ValueError("comparison must use the same scheme set for every task")
    output_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_rows = [by_key[task, scheme] for scheme in schemes]
        runtime_hashes = {_runtime_code_hash(Path(row["run_dir"])) for row in task_rows}
        if len(runtime_hashes) != 1:
            raise ValueError(f"{task} methods used different runtime source snapshots")
        runtime_code_hash = next(iter(runtime_hashes))
        for key in ("seed", "rounds", "data_content_hash", "partition_hash", "initial_model_hash",
                    "batch_size"):
            values = []
            for row in task_rows:
                result = results[row["run_dir"]]
                values.append(result[key])
            if len(set(values)) != 1:
                raise ValueError(f"{task} methods differ in {key}")
        for key in ("local_epochs", "optimizer", "learning_rate", "dirichlet_alpha",
                    "train_size", "test_size", "image_model", "pretrain_checkpoint_sha256"):
            values = [results[row["run_dir"]][key] for row in task_rows]
            if len(set(values)) != 1:
                raise ValueError(f"{task} methods differ in {key}")
        fedavg_time = by_key[task, "fedavg"]["elapsed_sec"] if "fedavg" in methods else None
        fixed_time = (by_key[task, "splitfed_fixed50"]["elapsed_sec"]
                      if "splitfed_fixed50" in schemes else None)
        fedavg_metric = by_key[task, "fedavg"]["final_metrics"][PRIMARY[task]] if "fedavg" in methods else None
        for scheme in schemes:
            row = by_key[task, scheme]
            method = row["method"]
            folder = Path(row["run_dir"])
            report = json.loads((folder / "validation_report.json").read_text(encoding="utf-8"))
            result = results[row["run_dir"]]
            processes = json.loads((folder / "process_manifest.json").read_text(encoding="utf-8"))
            if not report["valid"] or len(result["fit_records"]) != result["rounds"] * 6 or result["fit_failures"]:
                raise ValueError(f"{task}/{method} failed completeness checks")
            if processes["server_exit_code"] != 0 or set(processes["worker_exit_codes"].values()) != {0}:
                raise ValueError(f"{task}/{method} has a nonzero process exit")
            primary = float(row["final_metrics"][PRIMARY[task]])
            elapsed = float(row["elapsed_sec"])
            if not math.isfinite(primary) or not math.isfinite(elapsed):
                raise ValueError(f"{task}/{method} has a nonfinite comparison value")
            for round_id in range(1, result["rounds"] + 1):
                fits = [fit for fit in result["fit_records"] if fit["round_id"] == round_id]
                evaluations = [value for value in result["evaluation_records"]
                               if value["round_id"] == round_id]
                if len(fits) != 6 or len(evaluations) != 1:
                    raise ValueError(f"{task}/{method} round {round_id} is incomplete")
                examples = sum(fit["num_examples"] for fit in fits)
                training_loss = sum(
                    float(fit["metrics"]["task_loss" if method in ("fedavg", "fedprox") else "loss"])
                    * fit["num_examples"] for fit in fits
                ) / examples
                learning_rows.append({
                    "task": task, "method": method, "scheme": scheme,
                    "fixed_boundary": result["fixed_boundary"], "round": round_id,
                    "train_examples": examples, "train_loss": training_loss,
                    "primary_metric": PRIMARY[task],
                    "test_primary": evaluations[0]["metrics"][PRIMARY[task]],
                    "slowest_client_fit_sec": max(
                        float(fit["metrics"]["fit_duration_sec"]) for fit in fits
                    ),
                    "round_fit_span_sec": (
                        max(fit["metrics"]["fit_finished_unix_ns"] for fit in fits)
                        - min(fit["metrics"]["fit_started_unix_ns"] for fit in fits)
                    ) / 1e9,
                })
            output_rows.append({
                "task": task, "dataset": DATASETS[task],
                "model": ("ResNet-50 GroupNorm" if result["image_model"] == "resnet50"
                          else "RF-DETR Nano" if result["image_model"] == "rfdetr_nano"
                          else MODELS[task]),
                "method": method, "scheme": scheme, "fixed_boundary": result["fixed_boundary"],
                "primary_metric": PRIMARY[task],
                "final_primary": primary,
                "delta_vs_fedavg": primary - fedavg_metric if fedavg_metric is not None else None,
                "elapsed_sec": elapsed,
                "time_ratio_vs_fedavg": elapsed / fedavg_time if fedavg_time else None,
                "time_ratio_vs_fixed50": elapsed / fixed_time if fixed_time else None,
                "rounds": result["rounds"],
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "local_epochs": result["local_epochs"],
                "optimizer": result["optimizer"],
                "learning_rate": result["learning_rate"],
                "dirichlet_alpha": result["dirichlet_alpha"],
                "final_model_hash": result["final_model_hash"],
                "runtime_code_hash": runtime_code_hash,
                "min_six_worker_overlap_sec": min(report["fit_interval_overlap_sec"].values()),
                "run_dir": str(folder),
            })
    summary = {
        "schema": "splitfleet.physical-multitask-comparison.v3",
        "scope": "single-seed, real-data physical training comparison",
        "hosts": 3, "workers": 6, "task_count": len(tasks), "method_count": len(methods),
        "scheme_count": len(schemes),
        "runs": output_rows, "learning_curves": learning_rows,
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with (root / "learning_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(learning_rows[0]))
        writer.writeheader()
        writer.writerows(learning_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run_root)
    print(json.dumps({"runs": len(summary["runs"]),
                      "output": str(args.run_root / "comparison.json")}, indent=2))


if __name__ == "__main__":
    main()
