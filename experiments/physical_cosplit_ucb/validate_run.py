"""Validate a completed physical CoSplit-UCB run before using its results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def validate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if result.get("schema") != "splitfleet.physical-cosplit-ucb.v1":
        errors.append("unexpected result schema")
    if result.get("placement_policy") != "cosplit_ucb":
        errors.append("run did not use CoSplit-UCB")
    rounds = int(result.get("rounds", 0))
    expected = int(result.get("expected_clients", 0))
    if rounds < 1 or expected < 2:
        errors.append("rounds and expected_clients must be positive")
    if int(result.get("candidate_count", 0)) < 1:
        errors.append("candidate catalog is empty")
    if not result.get("partition_hash"):
        errors.append("partition hash is missing")
    manifest = result.get("partition_manifest", {})
    if manifest.get("partition_hash") != result.get("partition_hash"):
        errors.append("partition manifest hash does not match the run")
    partitions = manifest.get("clients", {})
    if len(partitions) != expected:
        errors.append("partition manifest does not cover all clients")
    if result.get("history") in (None, "None", ""):
        errors.append("Flower history is missing")
    if result.get("fit_failures"):
        errors.append(f"{len(result['fit_failures'])} client fit failure(s) occurred")

    fits = result.get("fit_records", [])
    server_fits = result.get("server_fit_records", [])
    evaluations = result.get("evaluation_records", [])
    diagnostics = result.get("round_diagnostics", {})
    local_epochs = int(result.get("local_epochs", 0))
    if local_epochs < 1:
        errors.append("local_epochs must be positive")
    baseline_ids: set[str] | None = None
    baseline_partitions: dict[str, str] | None = None
    baseline_cids: dict[str, str] | None = None
    for round_id in range(1, max(rounds, 0) + 1):
        label = f"round {round_id}"
        round_fits = [row for row in fits if int(row.get("round_id", -1)) == round_id]
        round_server = [row for row in server_fits if int(row.get("round_id", -1)) == round_id]
        round_evaluations = [row for row in evaluations if int(row.get("round_id", -1)) == round_id]
        cids = [str(row.get("flower_cid", "")) for row in round_fits]
        logical_ids = [str(row.get("logical_client_id", "")) for row in round_fits]
        if len(round_fits) != expected or len(set(cids)) != expected or "" in cids:
            errors.append(f"{label}: expected {expected} distinct client fit records")
        if len(set(logical_ids)) != expected or "" in logical_ids:
            errors.append(f"{label}: physical client identities are missing or duplicated")
        if baseline_ids is None:
            baseline_ids = set(logical_ids)
        elif set(logical_ids) != baseline_ids:
            errors.append(f"{label}: physical client identities changed")
        partition_ids = {
            str(row.get("logical_client_id", "")): str(row.get("metrics", {}).get("client_index", ""))
            for row in round_fits
        }
        if len(set(partition_ids.values())) != expected:
            errors.append(f"{label}: physical clients reused a partition")
        if baseline_partitions is None:
            baseline_partitions = partition_ids
        elif partition_ids != baseline_partitions:
            errors.append(f"{label}: physical client partition assignment changed")
        logical_cids = {
            str(row.get("logical_client_id", "")): str(row.get("flower_cid", ""))
            for row in round_fits
        }
        if baseline_cids is None:
            baseline_cids = logical_cids
        elif logical_cids != baseline_cids:
            errors.append(f"{label}: Flower client identity changed across rounds")
        suffix_counts = Counter(str(row.get("flower_cid", "")) for row in round_server)
        if suffix_counts != Counter(cids):
            errors.append(f"{label}: suffix result is missing or duplicated for a client")
        if len(round_evaluations) != 1 or int(round_evaluations[0].get("num_examples", 0)) != 10000:
            errors.append(f"{label}: complete CIFAR-10 test evaluation is missing")
        round_diagnostics = diagnostics.get(str(round_id), diagnostics.get(round_id, {}))
        assignments = round_diagnostics.get("assignment", {})
        if set(assignments) != set(cids):
            errors.append(f"{label}: placement diagnostics do not cover all clients")
        for row in round_fits:
            cid = str(row.get("flower_cid", ""))
            logical_id = str(row.get("logical_client_id", ""))
            partition = partitions.get(str(row.get("metrics", {}).get("client_index", "")))
            if partition is None:
                errors.append(f"{label}: client {logical_id} has no partition manifest entry")
            elif int(row.get("num_examples", 0)) != int(partition.get("num_examples", 0)) * local_epochs:
                errors.append(f"{label}: client {logical_id} did not process its complete local epochs")
            if row.get("metrics", {}).get("boundary") != assignments.get(cid):
                errors.append(f"{label}: client {cid} executed a different boundary")
        for row in round_server:
            cid = str(row.get("flower_cid", ""))
            matching = next((fit for fit in round_fits if str(fit.get("flower_cid", "")) == cid), None)
            if matching is None or int(row.get("num_examples", 0)) != int(matching.get("num_examples", 0)):
                errors.append(f"{label}: suffix for {cid} processed a different number of samples")
    return {
        "schema": "splitfleet.physical-cosplit-ucb-validation.v1",
        "valid": not errors,
        "errors": errors,
        "rounds": rounds,
        "expected_clients": expected,
    }


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    folder = Path(run_dir)
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    report = validate_result(result)
    (folder / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    report = validate_run(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
