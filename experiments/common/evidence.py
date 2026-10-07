"""Fresh, source-frozen output directories for download-free evidence runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def begin_run(output: Path, config: dict) -> dict:
    from experiments.www2027_study import freeze_runtime, save_json

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _, manifest = freeze_runtime(Path(__file__).resolve().parents[2], output)
    frozen = {**config, "config_sha256": hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    save_json(output / "config.json", frozen)
    save_json(output / "runtime_manifest.json", manifest)
    save_json(output / "result.json", {"status": "running"})
    save_json(output / "validation_report.json", {"status": "not_completed"})
    (output / "stdout.log").touch()
    (output / "metrics.jsonl").touch()
    return manifest


def claim_frozen_run(output: Path) -> dict:
    """Admit one child execution against frozen source/config/dependency bytes.

    A crashed child keeps its attempt marker. Reusing a hidden --frozen entry
    point must never overwrite completed, failed or partially measured runs.
    """
    from experiments.analysis.partition_coverage import _torchlens_source_identity

    output = output.resolve()
    snapshot = output / "runtime_snapshot"
    if Path(__file__).resolve().parents[2] != snapshot:
        raise ValueError("Execute from the recorded frozen source")
    config = json.loads((output / "config.json").read_text())
    contents = {key: value for key, value in config.items() if key != "config_sha256"}
    digest = hashlib.sha256(json.dumps(contents, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if config.get("config_sha256") != digest:
        raise ValueError("Frozen configuration changed before execution")
    manifest = json.loads((output / "runtime_manifest.json").read_text())
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in ("splitfleet", "experiments")
              for path in sorted((snapshot / folder).rglob("*.py"))}
    if hashes != manifest["source_hashes"]:
        raise ValueError("Frozen runtime source changed before execution")
    if _torchlens_source_identity() != manifest["torchlens_source_sha256"]:
        raise ValueError("TorchLens source changed before execution")
    if json.loads((output / "result.json").read_text()).get("status") != "running":
        raise FileExistsError("Run already completed or failed; preserve it and use a new directory")
    with (output / "execution_attempt.json").open("x") as stream:
        json.dump({"status": "claimed", "config_sha256": digest,
                   "source_identity": manifest["source_identity"]}, stream, sort_keys=True)
    return config


def finish_run(output: Path, result: dict, *, validation: dict, metrics=()) -> None:
    from experiments.www2027_study import save_json

    save_json(output / "result.json", result)
    save_json(output / "validation_report.json", validation)
    with (output / "metrics.jsonl").open("w") as stream:
        for row in metrics:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows, fields=None) -> None:
    import csv

    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
