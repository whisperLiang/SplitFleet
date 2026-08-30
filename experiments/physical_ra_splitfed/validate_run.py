"""Strict validation for one physical confirmatory or pilot training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(value)
    return rows


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    is_fedavg = metadata.get("method") == "fedavg_full_local"
    is_native_prefix = metadata.get("split_runtime") == "native_prefix"
    clients = _read_jsonl(root / "client_metrics.jsonl")
    servers = _read_jsonl(root / "server_metrics.jsonl", required=not is_fedavg)
    rounds = _read_jsonl(root / "round_metrics.jsonl")
    evaluations = _read_jsonl(root / "evaluation_metrics.jsonl")
    decisions = _read_jsonl(root / "split_decisions.jsonl", required=False)
    failures = _read_jsonl(root / "failures.jsonl", required=False)
    errors: list[str] = []
    if metadata.get("data_pipeline") != "flower-cifar10-v1":
        errors.append("run does not use the Flower CIFAR-10 data pipeline")
    if int(metadata.get("train_samples", 0)) != 50000:
        errors.append("run does not use the complete 50,000-sample CIFAR-10 train split")
    if int(metadata.get("test_samples", 0)) != 10000:
        errors.append("run does not use the complete CIFAR-10 test split")
    if "max_train_samples" in metadata or "max_batches_per_round" in metadata:
        errors.append("deprecated truncated-data or truncated-batch metadata is present")
    # Runs produced before A8 are explicitly ResNet-18 and did not yet carry
    # a model field. Preserve their historical validity.
    model = str(metadata.get("model") or "resnet18")
    if metadata.get("protocol_path", "").endswith("protocol_large_model.yaml"):
        if model == "resnet18":
            errors.append("large-model protocol cannot use resnet18")
    expected_rounds = int(metadata["rounds"])
    expected_round_ids = list(range(1, expected_rounds + 1))
    if [int(row["round_id"]) for row in rounds] != expected_round_ids:
        errors.append("round metrics are incomplete, duplicated, or unordered")
    if [int(row["round_id"]) for row in evaluations] != expected_round_ids:
        errors.append("central evaluations are incomplete, duplicated, or unordered")
    if failures:
        errors.append(f"run contains {len(failures)} recorded failure(s)")
    client_keys = [
        (int(row["round_id"]), str(row["logical_client_id"])) for row in clients
    ]
    server_keys = [
        (int(row["round_id"]), str(row["logical_client_id"])) for row in servers
    ]
    expected_records = expected_rounds * int(metadata.get("partition_manifest", {}).get("clients", {}).__len__())
    if expected_records != expected_rounds * 4:
        errors.append("metadata does not describe exactly four logical clients")
    if len(client_keys) != expected_records or len(client_keys) != len(set(client_keys)):
        errors.append("client record count or uniqueness check failed")
    if is_fedavg:
        if servers:
            errors.append("FedAvg must not create suffix-server records")
    else:
        if len(server_keys) != expected_records or len(server_keys) != len(set(server_keys)):
            errors.append("suffix record count or uniqueness check failed")
        if sorted(client_keys) != sorted(server_keys):
            errors.append("client and suffix record identities differ")
    if not is_fedavg and metadata.get("environment", {}).get("cuda_available"):
        required_server_memory = {
            "server_cuda_allocated_mb",
            "server_cuda_reserved_mb",
            "server_cuda_peak_allocated_mb",
        }
        for row in servers:
            config = row.get("config") or {}
            missing = sorted(required_server_memory - set(config))
            if missing:
                errors.append(
                    f"server CUDA memory metrics missing at round {row.get('round_id')} "
                    f"client {row.get('logical_client_id')}: {missing}"
                )
    if is_native_prefix:
        for row in servers:
            config = row.get("config") or {}
            if config.get("persistent_server_model") is not True:
                errors.append("native suffix replica is not marked persistent")
            if int(row.get("round_id", 0)) > 1 and config.get("persistent_reused") is not True:
                errors.append(
                    f"native suffix replica was not reused at round {row.get('round_id')} "
                    f"client {row.get('logical_client_id')}"
                )
            if config.get("torchlens_in_timed_path") is not False:
                errors.append("native suffix did not prove TorchLens exclusion")
    partition_hash = str(metadata["partition_hash"])
    local_epochs = int(metadata.get("local_epochs", 0))
    batch_size = int(metadata.get("batch_size", 0))
    logical_partitions = metadata.get("logical_client_partitions") or {}
    if local_epochs < 1 or batch_size < 1:
        errors.append("invalid Flower local-epoch or batch-size configuration")
    logical_ids = set()
    runner_hashes = set()
    source_manifest_hashes = set()
    for row in clients:
        metrics = row.get("metrics") or {}
        logical_ids.add(str(row.get("logical_client_id", "")))
        runner_hashes.add(str(metrics.get("runner_sha256", "")))
        source_manifest_hashes.add(str(metrics.get("source_manifest_hash", "")))
        if metrics.get("partition_hash") != partition_hash:
            errors.append(
                f"partition mismatch at round {row.get('round_id')} "
                f"client {row.get('logical_client_id')}"
            )
        if not row.get("success") or int(row.get("num_examples", 0)) <= 0:
            errors.append(
                f"unsuccessful or empty update at round {row.get('round_id')} "
                f"client {row.get('logical_client_id')}"
            )
        logical_id = str(row.get("logical_client_id", ""))
        partition_id = str(logical_partitions.get(logical_id, ""))
        partition_record = (metadata.get("partition_manifest", {}).get("clients", {})).get(
            partition_id, {}
        )
        partition_examples = int(partition_record.get("num_examples", 0))
        expected_examples = partition_examples * local_epochs
        expected_batches = (
            (partition_examples + batch_size - 1) // batch_size
        ) * local_epochs if batch_size else 0
        if int(row.get("num_examples", 0)) != expected_examples:
            errors.append(
                f"client {logical_id} did not consume its complete local partition "
                f"at round {row.get('round_id')}: {row.get('num_examples')}/{expected_examples}"
            )
        if int(metrics.get("num_batches", 0)) != expected_batches:
            errors.append(
                f"client {logical_id} did not execute complete local epochs at round "
                f"{row.get('round_id')}: {metrics.get('num_batches')}/{expected_batches} batches"
            )
        if is_fedavg:
            if row.get("split_key") != "full_local":
                errors.append(
                    f"FedAvg client {row.get('logical_client_id')} did not use full_local"
                )
            for field in (
                "full_model_compute_sec",
                "model_download_bytes",
                "model_upload_bytes",
            ):
                if float(metrics.get(field, 0.0)) <= 0.0:
                    errors.append(
                        f"FedAvg metric {field} is missing or empty at round "
                        f"{row.get('round_id')} client {row.get('logical_client_id')}"
                    )
            if int(metrics.get("boundary_upload_bytes", -1)) != 0 or int(
                metrics.get("boundary_download_bytes", -1)
            ) != 0:
                errors.append(
                    f"FedAvg unexpectedly recorded split-boundary traffic at round "
                    f"{row.get('round_id')} client {row.get('logical_client_id')}"
                )
        if is_native_prefix:
            prefix_down = int(metrics.get("prefix_parameter_download_bytes", 0))
            prefix_up = int(metrics.get("prefix_parameter_upload_bytes", 0))
            full_bytes = int(metrics.get("full_model_parameter_bytes", 0))
            if not 0 < prefix_down < full_bytes or not 0 < prefix_up < full_bytes:
                errors.append(
                    f"native prefix synchronization is not a strict subset at round "
                    f"{row.get('round_id')} client {row.get('logical_client_id')}"
                )
            if int(metrics.get("suffix_parameter_download_bytes", -1)) != 0 or int(
                metrics.get("suffix_parameter_upload_bytes", -1)
            ) != 0:
                errors.append("native client transferred suffix parameters")
            if metrics.get("parameter_sync_mode") != "prefix_parameters":
                errors.append("native client used the wrong parameter sync mode")
            if metrics.get("torchlens_in_timed_path") is not False:
                errors.append("native client did not prove TorchLens exclusion")
    if logical_ids != {"win136", "orin140", "orin118", "orin238"}:
        errors.append(f"unexpected logical clients: {sorted(logical_ids)}")
    expected_runner_hash = str(metadata.get("environment", {}).get("runner_sha256", ""))
    if runner_hashes != {expected_runner_hash} or not expected_runner_hash:
        errors.append(
            f"runner source hashes differ: clients={sorted(runner_hashes)}, "
            f"server={expected_runner_hash}"
        )
    expected_source_manifest_hash = str(metadata.get("source_manifest_hash", ""))
    if (
        source_manifest_hashes != {expected_source_manifest_hash}
        or not expected_source_manifest_hash
    ):
        errors.append(
            "source manifest hashes differ: "
            f"clients={sorted(source_manifest_hashes)}, "
            f"server={expected_source_manifest_hash}"
        )
    if not metadata.get("protocol_sha256"):
        errors.append("protocol_sha256 is missing")
    for row in evaluations:
        accuracy = float(row["test_accuracy"])
        if not 0.0 <= accuracy <= 1.0 or int(row["num_examples"]) != 10000:
            errors.append(f"invalid CIFAR-10 evaluation in round {row['round_id']}")
    calibration = (
        {1: "full_local", 2: "full_local", 3: "full_local"}
        if is_fedavg
        else {1: "stem", 2: "layer2", 3: "layer4"}
    )
    for row in rounds:
        round_id = int(row["round_id"])
        if round_id in calibration:
            if row.get("split_key") != calibration[round_id]:
                errors.append(
                    f"calibration round {round_id} used {row.get('split_key')}, "
                    f"expected {calibration[round_id]}"
                )
            if set((row.get("placements") or {}).values()) != {calibration[round_id]}:
                errors.append(f"calibration placements differ in round {round_id}")
    if metadata.get("fit_failure_count") not in (0, None):
        errors.append("metadata reports fit failures")
    if not is_fedavg and metadata.get("profile_report_hash") is None:
        errors.append("profile-guided run is missing profile_report_hash")
    expected_decisions = 0 if is_fedavg else max(0, expected_rounds - 3) * 4
    if len(decisions) != expected_decisions:
        errors.append(
            f"expected {expected_decisions} decisions, found {len(decisions)}"
        )
    return {
        "schema": "splitfleet.physical-run-validation.v1",
        "valid": not errors,
        "errors": errors,
        "run_id": metadata["run_id"],
        "model": model,
        "protocol_path": metadata.get("protocol_path"),
        "method": metadata["method"],
        "seed": metadata["seed"],
        "rounds": len(rounds),
        "client_records": len(clients),
        "server_records": len(servers),
        "evaluation_records": len(evaluations),
        "decision_records": len(decisions),
        "failure_records": len(failures),
        "initial_model_hash": metadata["initial_model_hash"],
        "partition_hash": partition_hash,
        "profile_report_hash": metadata.get("profile_report_hash"),
        "source_manifest_hash": expected_source_manifest_hash,
        "protocol_sha256": metadata.get("protocol_sha256"),
        "max_server_cuda_allocated_mb": max(
            (
                float((row.get("config") or {}).get("server_cuda_allocated_mb", 0.0))
                for row in servers
            ),
            default=0.0,
        ),
        "max_server_cuda_reserved_mb": max(
            (
                float((row.get("config") or {}).get("server_cuda_reserved_mb", 0.0))
                for row in servers
            ),
            default=0.0,
        ),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = validate_run(args.run_dir)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
