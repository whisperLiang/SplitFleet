"""Validate RA-SplitFed raw results without repairing or interpolating data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .split_candidates import SEMANTIC_SPLIT_ORDER


REQUIRED_FILES = {
    "metadata.json",
    "resolved_config.yaml",
    "git_commit.txt",
    "environment.json",
    "client_assignments.json",
    "split_candidates.json",
    "round_metrics.jsonl",
    "client_metrics.jsonl",
    "split_decisions.jsonl",
    "resource_metrics.jsonl",
    "failures.jsonl",
    "summary.csv",
}
PLACEHOLDER_TOKENS = {"placeholder", "dummy_result", "synthetic_result", "interpolated_result"}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path.name}:{line_number}: record is not an object")
                continue
            _validate_values(row, f"{path.name}:{line_number}", errors)
            rows.append(row)
    return rows


def _validate_values(value: Any, location: str, errors: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{location}: contains NaN or Inf")
    elif isinstance(value, str) and value.lower() in PLACEHOLDER_TOKENS:
        errors.append(f"{location}: contains placeholder token {value!r}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _validate_values(child, f"{location}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_values(child, f"{location}[{index}]", errors)


def validate_run(results_dir: str | Path, *, write_report: bool = True) -> dict[str, Any]:
    root = Path(results_dir)
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).exists())
    if missing:
        errors.append(f"Missing required files: {missing}")
    if errors:
        report = {"valid": False, "errors": errors, "warnings": warnings}
        if write_report:
            (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    metadata = _read_json(root / "metadata.json")
    config = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    assignments = _read_json(root / "client_assignments.json")
    candidates = _read_json(root / "split_candidates.json")
    candidate_keys = {str(item["split_key"]) for item in candidates}
    if not candidate_keys.issubset(set(SEMANTIC_SPLIT_ORDER)) or "full_local" not in candidate_keys:
        errors.append("split_candidates.json contains illegal or incomplete canonical split keys")
    streams = {
        name: _read_jsonl(root / name, errors)
        for name in (
            "round_metrics.jsonl", "client_metrics.jsonl", "split_decisions.jsonl",
            "resource_metrics.jsonl", "failures.jsonl",
        )
    }
    if metadata.get("method") == "resource_profile":
        _validate_profile_run(root, config, candidate_keys, streams, errors)
        report = {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "counts": {
                **{name: len(rows) for name, rows in streams.items()},
                "profile_records.jsonl": _count_jsonl(
                    root / "profiles" / "profile_records.jsonl"
                ),
            },
            "checked_rounds": 0,
            "partition_hash": None,
            "initial_model_hash": assignments.get("initial_model_hash"),
            "client_sampling_hash": None,
        }
        if write_report:
            (root / "validation_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return report
    rounds = streams["round_metrics.jsonl"]
    expected_rounds = int(config.get("rounds", len(rounds)))
    observed = [int(row.get("round_id", -1)) for row in rounds]
    duplicates = sorted({value for value in observed if observed.count(value) > 1})
    if duplicates:
        errors.append(f"Duplicate round records: {duplicates}")
    missing_rounds = sorted(set(range(1, expected_rounds + 1)) - set(observed))
    if missing_rounds:
        errors.append(f"Missing round records: {missing_rounds}")
    for stream_name, rows in streams.items():
        seen_client_keys = set()
        for row in rows:
            for key, value in row.items():
                lowered = key.lower()
                if value is not None and (
                    lowered.endswith("_ms") or lowered.endswith("_bytes") or lowered.endswith("_mb")
                ) and isinstance(value, (int, float)) and value < 0:
                    errors.append(f"{stream_name}: negative {key}={value}")
            split_key = row.get("split_key")
            if split_key is not None and split_key not in candidate_keys:
                errors.append(f"{stream_name}: illegal split key {split_key!r}")
            if stream_name == "client_metrics.jsonl":
                identity = (row.get("method"), row.get("seed"), row.get("round_id"), row.get("client_id"))
                if identity in seen_client_keys:
                    errors.append(f"Duplicate client-round record: {identity}")
                seen_client_keys.add(identity)
                if not row.get("success", True):
                    failures = streams["failures.jsonl"]
                    if not any(
                        failure.get("round_id") == row.get("round_id")
                        and failure.get("client_id") == row.get("client_id")
                        for failure in failures
                    ):
                        errors.append(f"Unrecorded client failure: {identity}")
    stable_fields = ("initial_model_hash", "partition_hash", "client_sampling_hash")
    for field in stable_fields:
        values = {row.get(field) for row in rounds}
        if len(values) > 1:
            errors.append(f"Configuration/data identity changed mid-run: {field}")
    if rounds and any(row.get("final_accuracy_source") != "cifar10_test_set" for row in rounds):
        errors.append("At least one accuracy value is not traced to the real CIFAR-10 test set")
    if assignments.get("initial_model_hash") != metadata.get("initial_model_hash", assignments.get("initial_model_hash")):
        warnings.append("metadata does not duplicate initial_model_hash; client_assignments remains authoritative")
    _validate_summary(root / "summary.csv", rounds, errors)
    _compare_sibling_identities(root, metadata, config, assignments, errors)
    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {name: len(rows) for name, rows in streams.items()},
        "checked_rounds": expected_rounds,
        "partition_hash": assignments.get("partition_hash"),
        "initial_model_hash": assignments.get("initial_model_hash"),
        "client_sampling_hash": assignments.get("client_sampling_hash"),
    }
    if write_report:
        (root / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _validate_profile_run(
    root: Path,
    config: Mapping[str, Any],
    candidate_keys: set[str],
    streams: Mapping[str, list[dict[str, Any]]],
    errors: list[str],
) -> None:
    record_path = root / "profiles" / "profile_records.jsonl"
    summary_path = root / "profiles" / "profile_summary.csv"
    if not record_path.exists() or not summary_path.exists():
        errors.append("Profile run is missing profile_records.jsonl or profile_summary.csv")
        return
    records = _read_jsonl(record_path, errors)
    seen = set()
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        identity = (
            str(record.get("device_profile")),
            str(record.get("network_profile")),
            str(record.get("split_key")),
            int(record.get("batch_size", -1)),
            int(record.get("repetition", -1)),
        )
        if identity in seen:
            errors.append(f"Duplicate profile record: {identity}")
        seen.add(identity)
        if identity[2] not in candidate_keys:
            errors.append(f"Profile record has illegal split key {identity[2]!r}")
        group_key = identity[:4]
        grouped.setdefault(group_key, []).append(record)
        if not bool(record.get("success")) and not any(
            failure.get("device_profile") == identity[0]
            and failure.get("network_profile") == identity[1]
            and failure.get("split_key") == identity[2]
            and int(failure.get("batch_size", -1)) == identity[3]
            and int(failure.get("repetition", -1)) == identity[4]
            for failure in streams["failures.jsonl"]
        ):
            errors.append(f"Unrecorded profile failure: {identity}")

    selected_splits = [str(value) for value in config.get("split_keys", candidate_keys)]
    expected = {
        (str(device), str(network), split, int(batch), repetition)
        for device in (config.get("device_profiles") or {"measured_host": {}})
        for network in (config.get("network_profiles") or {"measured_link": {}})
        for batch in config.get("batch_sizes", [1, 2, 4, 8])
        for split in selected_splits
        for repetition in range(int(config.get("repetitions", 5)))
    }
    missing = sorted(expected - seen)
    if missing:
        errors.append(f"Missing profile records: {missing[:10]}")

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary_by_key = {
        (
            str(row["device_profile"]),
            str(row["network_profile"]),
            str(row["split_key"]),
            int(row["batch_size"]),
        ): row
        for row in summary_rows
    }
    for key, values in grouped.items():
        successful = [value for value in values if bool(value.get("success"))]
        if not successful:
            continue
        row = summary_by_key.get(key)
        if row is None:
            errors.append(f"profile_summary.csv is missing group {key}")
            continue
        if int(row["valid_repetitions"]) != len(successful):
            errors.append(f"profile_summary.csv valid_repetitions mismatch for {key}")
        for metric, recorded in row.items():
            if metric in {
                "device_profile", "network_profile", "split_key", "batch_size",
                "valid_repetitions",
            } or recorded in (None, ""):
                continue
            measured = [
                float(value[metric])
                for value in successful
                if value.get(metric) is not None
            ]
            if not measured:
                errors.append(f"profile_summary.csv has unmeasured value for {key}/{metric}")
                continue
            recomputed = sum(measured) / len(measured)
            if not math.isclose(recomputed, float(recorded), rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"profile_summary.csv cannot be recomputed for {key}/{metric}")


def _validate_summary(path: Path, rounds: list[dict[str, Any]], errors: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        summary = {row["metric"]: row for row in csv.DictReader(handle)}
    for metric, row in summary.items():
        values = [float(item[metric]) for item in rounds if item.get(metric) is not None]
        if not values:
            continue
        recomputed = sum(values) / len(values)
        recorded = float(row["mean"])
        if not math.isclose(recomputed, recorded, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"summary.csv metric {metric!r} cannot be recomputed from round_metrics")


def _compare_sibling_identities(
    root: Path,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    assignments: Mapping[str, Any],
    errors: list[str],
) -> None:
    for sibling in root.parent.iterdir():
        if sibling == root or not sibling.is_dir():
            continue
        try:
            other_metadata = _read_json(sibling / "metadata.json")
            other = _read_json(sibling / "client_assignments.json")
            other_config = yaml.safe_load(
                (sibling / "resolved_config.yaml").read_text(encoding="utf-8")
            ) or {}
        except (OSError, json.JSONDecodeError):
            continue
        if other_metadata.get("method") == "resource_profile":
            continue
        if (
            other_metadata.get("dataset") == metadata.get("dataset")
            and other_metadata.get("model") == metadata.get("model")
            and other_metadata.get("seed") == metadata.get("seed")
            and other_config.get("experiment") == config.get("experiment")
            and other_config.get("num_clients") == config.get("num_clients")
            and other_config.get("max_train_samples") == config.get("max_train_samples")
            and other_config.get("normalization") == config.get("normalization")
            and other_config.get("client_holdout_fraction")
            == config.get("client_holdout_fraction")
        ):
            for key in ("partition_hash", "initial_model_hash", "client_sampling_hash"):
                if other.get(key) != assignments.get(key):
                    errors.append(f"Cross-method identity mismatch with {sibling.name}: {key}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args(argv)
    report = validate_run(args.results_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
