"""Distinct fixed cuts stay paired without accepting incomplete protocols."""

import json

import pytest

from experiments.physical_multitask import scheme_name
from experiments.summarize_physical_multitask import RUNTIME_SOURCES, summarize


@pytest.fixture
def comparison_root(tmp_path):
    (tmp_path / "source_hashes.json").write_text(json.dumps({
        path: "frozen-source" for path in RUNTIME_SOURCES
    }))
    rows = []
    for method, boundary in (
        ("fedavg", None), ("fedprox", None), ("splitfed_fixed", "25%"),
        ("splitfed_fixed", "50%"), ("splitfed_fixed", "75%"), ("splitfleet", None),
    ):
        scheme = scheme_name(method, boundary)
        folder = tmp_path / scheme
        folder.mkdir()
        metrics = {"accuracy": 0.5}
        result = {
            "task": "image_classification", "method": method, "fixed_boundary": boundary,
            "seed": 19, "rounds": 1, "data_content_hash": "samples",
            "partition_hash": "partitions", "initial_model_hash": "initial",
            "final_model_hash": scheme, "batch_size": 2, "local_epochs": 1,
            "optimizer": "sgd", "learning_rate": 0.01, "dirichlet_alpha": 0.5,
            "train_size": 24, "test_size": 12, "image_model": "small",
            "pretrain_checkpoint_sha256": None, "fit_failures": [],
            "fit_records": [
                {"round_id": 1, "num_examples": 4, "metrics": {
                    "task_loss" if method in ("fedavg", "fedprox") else "loss": 0.8,
                    "fit_duration_sec": 1.0, "fit_started_unix_ns": 1_000_000_000,
                    "fit_finished_unix_ns": 2_000_000_000,
                }} for _ in range(6)
            ],
            "evaluation_records": [{"round_id": 1, "metrics": metrics}],
        }
        (folder / "result.json").write_text(json.dumps(result))
        (folder / "validation_report.json").write_text(json.dumps({
            "valid": True, "fit_interval_overlap_sec": {"1": 1.0},
        }))
        (folder / "process_manifest.json").write_text(json.dumps({
            "server_exit_code": 0,
            "worker_exit_codes": {str(index): 0 for index in range(6)},
        }))
        rows.append({"task": result["task"], "method": method,
                     "elapsed_sec": 10.0, "final_metrics": metrics, "run_dir": str(folder)})
    (tmp_path / "summary.json").write_text(json.dumps(rows))
    return tmp_path


def test_summary_keeps_three_fixed_cuts_and_all_six_training_curves(comparison_root):
    report = summarize(comparison_root)
    assert report["method_count"] == 4
    assert report["scheme_count"] == 6
    assert len(report["runs"]) == len(report["learning_curves"]) == 6
    assert {row["fixed_boundary"] for row in report["runs"]
            if row["method"] == "splitfed_fixed"} == {"25%", "50%", "75%"}
    assert all(row["train_examples"] == 24 for row in report["learning_curves"])


@pytest.mark.parametrize("mutation", ["missing_optimizer", "learning_rate", "duplicate_cut"])
def test_summary_rejects_missing_protocol_mixed_protocol_and_duplicate_cut(comparison_root, mutation):
    path = comparison_root / "splitfed_fixed25" / "result.json"
    result = json.loads(path.read_text())
    if mutation == "missing_optimizer":
        del result["optimizer"]
    elif mutation == "learning_rate":
        result["learning_rate"] = 0.5
    else:
        result["fixed_boundary"] = "50%"
    path.write_text(json.dumps(result))
    with pytest.raises((ValueError, KeyError)):
        summarize(comparison_root)
