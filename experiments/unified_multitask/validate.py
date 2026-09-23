"""Check completeness and identity of one unified multi-task benchmark run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from splitfleet.tasks import TASK_SPECS

from .run import METHODS


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_run(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assignments = json.loads((root / "assignments.json").read_text(encoding="utf-8"))
    clients = _jsonl(root / "client_metrics.jsonl")
    rounds = _jsonl(root / "round_metrics.jsonl")
    errors: list[str] = []
    if metadata.get("schema") != "splitfleet.unified-benchmark.v1":
        errors.append("Unexpected benchmark schema.")
    if not metadata.get("completed"):
        errors.append("Run has no completion marker.")
    if metadata.get("method") not in METHODS:
        errors.append("Unknown method.")
    try:
        spec = TASK_SPECS.get(metadata["task"])
    except (KeyError, ValueError):
        spec = None
        errors.append("Unknown task.")
    expected_rounds = int(metadata["rounds"])
    expected_clients = int(metadata["num_clients"])
    if [int(row["round_id"]) for row in rounds] != list(range(1, expected_rounds + 1)):
        errors.append("Round records are missing, duplicated or unordered.")
    if len(clients) != expected_rounds * expected_clients:
        errors.append("Client metric count differs from rounds × clients.")
    if len(assignments) != expected_clients:
        errors.append("Assignment client count differs from metadata.")
    assigned_indices = [int(index) for values in assignments.values() for index in values]
    if sorted(assigned_indices) != list(range(int(metadata["train_examples"]))):
        errors.append("Assignments do not cover the training set exactly once.")
    if metadata.get("partition_hash") != _stable_hash(assignments):
        errors.append("Partition hash differs from assignments.json.")
    if not isinstance(metadata.get("data_content_hash"), str) or len(metadata["data_content_hash"]) != 64:
        errors.append("Dataset content hash is missing.")
    seen = Counter((int(row["round_id"]), str(row["client_id"])) for row in clients)
    expected_keys = {(round_id, str(client_id)) for round_id in range(1, expected_rounds + 1)
                     for client_id in range(expected_clients)}
    if set(seen) != expected_keys or any(count != 1 for count in seen.values()):
        errors.append("Client round identities are missing or duplicated.")
    for row in clients:
        if row.get("method") != metadata.get("method"):
            errors.append("Client method differs from metadata.")
        if int(row.get("num_examples", 0)) <= 0 or int(row.get("num_batches", 0)) <= 0:
            errors.append("Client recorded no training work.")
        if not _finite_between(row.get("task_loss"), 0, None):
            errors.append("Client loss is non-finite or negative.")
        for field in ("fit_duration_sec", "runtime_prepare_sec", "boundary_upload_bytes", "boundary_download_bytes"):
            if not _finite_between(row.get(field), 0, None):
                errors.append(f"Client {field} is non-finite or negative.")
        split = metadata.get("method") in {"splitfed_fixed", "splitfleet"}
        if split and (row.get("boundary") == "full_local" or row.get("boundary_upload_bytes", 0) <= 0
                      or row.get("boundary_download_bytes", 0) <= 0):
            errors.append("Split method has no valid boundary traffic.")
        if not split and (row.get("boundary") != "full_local" or row.get("boundary_upload_bytes") != 0
                          or row.get("boundary_download_bytes") != 0):
            errors.append("Full-local method recorded split traffic.")
    for row in rounds:
        round_id = int(row["round_id"])
        matching = [item for item in clients if int(item["round_id"]) == round_id]
        if row.get("initial_model_hash") != metadata.get("initial_model_hash"):
            errors.append(f"Round {round_id} changed its initial model identity.")
        if row.get("partition_hash") != metadata.get("partition_hash"):
            errors.append(f"Round {round_id} changed its partition identity.")
        if row.get("data_content_hash") != metadata.get("data_content_hash"):
            errors.append(f"Round {round_id} changed its dataset identity.")
        if int(row.get("successful_clients", -1)) != len(matching):
            errors.append(f"Round {round_id} successful-client count differs from raw records.")
        for field in ("num_examples", "boundary_upload_bytes", "boundary_download_bytes"):
            if int(row.get(field, -1)) != sum(int(item[field]) for item in matching):
                errors.append(f"Round {round_id} {field} differs from client records.")
        if not _finite_between(row.get("round_time_sec"), 0, None):
            errors.append(f"Round {round_id} time is non-finite or negative.")
        metrics = row.get("metrics") or {}
        if spec is not None:
            if row.get("primary_metric") != spec.primary_metric or spec.primary_metric not in metrics:
                errors.append(f"Round {round_id} lacks the task's primary metric.")
            for name, value in metrics.items():
                if not _finite_between(value, 0, 1):
                    errors.append(f"Round {round_id} {name} is outside [0, 1].")
    _check_sibling_identities(root, metadata, errors)
    return {
        "schema": "splitfleet.unified-benchmark-validation.v1",
        "valid": not errors,
        "errors": sorted(set(errors)),
        "task": metadata.get("task"),
        "method": metadata.get("method"),
        "source": metadata.get("source"),
        "seed": metadata.get("seed"),
        "rounds": len(rounds),
        "client_records": len(clients),
        "initial_model_hash": metadata.get("initial_model_hash"),
        "partition_hash": metadata.get("partition_hash"),
    }


def _stable_hash(value: Any) -> str:
    from experiments.resource_adaptive_splitfed.config_utils import stable_hash

    return stable_hash(value)


def _finite_between(value: Any, low: float, high: float | None) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= low and (high is None or numeric <= high)


def _check_sibling_identities(root: Path, metadata: dict[str, Any], errors: list[str]) -> None:
    for sibling in root.parent.iterdir():
        if sibling == root or not sibling.is_dir() or not (sibling / "metadata.json").exists():
            continue
        other = json.loads((sibling / "metadata.json").read_text(encoding="utf-8"))
        if not other.get("completed") or not other.get("data_content_hash"):
            continue
        if all(other.get(key) == metadata.get(key) for key in (
            "task", "source", "sample_selection", "seed", "train_examples", "test_examples",
            "num_clients", "batch_size", "rounds"
        )):
            for key in ("initial_model_hash", "partition_hash", "data_content_hash"):
                if other.get(key) != metadata.get(key):
                    errors.append(f"Sibling {sibling.name} has a different {key}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    report = validate_run(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
