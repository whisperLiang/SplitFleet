"""Deleting unused attempts must never replace a published measured endpoint."""

import hashlib
import json

import pytest

from experiments.analysis import fedcompass_study as audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def published(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "__file__", str(tmp_path / "experiments/analysis/fedcompass_study.py"))
    methods = ["fedavg", "splitfed_fixed25", "splitfed_fixed50", "splitfed_fixed75", "splitfleet"]
    models = ["resnet50_pretrained", "bert_base", "rfdetr_nano", "deeplabv3_resnet50"]
    observations = []
    for model in models:
        for seed in range(10501, 10506):
            for method in methods:
                path = tmp_path / "results" / model / str(seed) / method / "result.json"
                path.parent.mkdir(parents=True)
                raw = {"model_id": model, "task": "fixture", "seed": seed,
                       "method": "splitfed_fixed" if method.startswith("splitfed_fixed") else method,
                       "fixed_boundary": method.removeprefix("splitfed_fixed") + "%" if method.startswith("splitfed_fixed") else None,
                       "fit_failures": [], "evaluation_records": [{"round_id": 10, "metrics": {"accuracy": .5}}],
                       "server_duration_sec": 12, "rounds": 10, "initial_model_hash": "initial",
                       "final_model_hash": "final", "partition_hash": "partition", "data_content_hash": "data",
                       "worker_ids": ["worker"], "train_size": 240, "test_size": 200}
                path.write_text(json.dumps(raw))
                observations.append({"model": model, "task": "fixture", "seed": seed, "method": method,
                                     "metric": "accuracy", "quality": .5, "quality_pct": 50,
                                     "run_dir": str(path.parent), "result_sha256": sha(path),
                                     **{key: raw[key] for key in ("server_duration_sec", "initial_model_hash", "final_model_hash",
                                                                "partition_hash", "data_content_hash", "worker_ids", "train_size", "test_size")}})
    discarded = tmp_path / "unused_failure.json"
    discarded.write_text('{"status":"failed"}')
    discarded_hash = sha(discarded)
    discarded.unlink()
    value = {"protocol": "fixed_rounds_sync", "methods": methods, "tasks": [{"model": model} for model in models],
             "observations": observations, "planned_jobs": 100, "completed_jobs": 100,
             "complete_planned_population": True, "cohorts_pooled": False,
             "analysis_source_sha256": {}, "source_artifacts_sha256": {str(discarded): discarded_hash}}
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(value))
    receipt = tmp_path / "paper/evidence/manuscript_cleanup_20261013/receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"published_evidence_sha256": {"evidence.json": sha(evidence)},
                                  "removed_files": [{"path": "unused_failure.json", "sha256": discarded_hash}]}))
    return evidence, receipt, discarded


def test_pruned_unused_attempt_keeps_all_published_successes_auditable(published):
    checked = audit.audit_published_endpoints(published[0])
    assert checked["published_jobs"] == 100
    assert checked["deleted_sources_verified_against_cleanup_receipt"] == 1
    assert checked["deleted_attempt_population_reaudited"] is False


def test_receipt_cannot_substitute_for_a_deleted_published_success(published):
    evidence, receipt, _ = published
    row = json.loads(evidence.read_text())["observations"][0]
    path = audit.Path(row["run_dir"]) / "result.json"
    value = json.loads(receipt.read_text())
    value["removed_files"].append({"path": str(path.relative_to(evidence.parent)), "sha256": sha(path)})
    receipt.write_text(json.dumps(value))
    path.unlink()
    with pytest.raises(ValueError, match="Missing or changed published endpoint"):
        audit.audit_published_endpoints(evidence)


@pytest.mark.parametrize("change", ["missing_without_receipt", "wrong_removed_hash", "restored_removed_file"])
def test_cleanup_receipt_must_match_each_pruned_source(published, change):
    evidence, receipt, discarded = published
    value = json.loads(receipt.read_text())
    if change == "missing_without_receipt":
        value["removed_files"] = []
    elif change == "wrong_removed_hash":
        value["removed_files"][0]["sha256"] = "wrong"
    else:
        discarded.write_text('{"status":"failed"}')
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unrecorded missing|Deleted source does not match"):
        audit.audit_published_endpoints(evidence)
