"""Strict comparisons and real isolated GPU/gRPC coverage for the phase-A gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.resnet50_quality.gate_torch import ATOL, RTOL, compare_tensor
from experiments.resnet50_quality.model_contract import NAMES, TRAINABLE_NAMES, TRAINABLE_PARAMETERS

ROOT = Path(__file__).resolve().parents[2]


def _environment(*, gpu=True):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0" if gpu else ""
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    return environment


@pytest.mark.parametrize("actual,expected,passed,reason", [
    (np.asarray([1e-6], dtype="float32"), np.asarray([0], dtype="float32"), True, None),
    (np.asarray([1e-3], dtype="float32"), np.asarray([0], dtype="float32"), False, None),
    (np.asarray(2, dtype="int64"), np.asarray(1, dtype="int64"), False, None),
    (np.asarray([np.nan], dtype="float32"), np.asarray([1], dtype="float32"), False, "nonfinite"),
    (np.asarray([1], dtype="float64"), np.asarray([1], dtype="float32"), False, "schema_mismatch"),
    (np.asarray([1], dtype="float32"), np.asarray([[1]], dtype="float32"), False, "schema_mismatch"),
])
def test_each_tensor_comparison_enforces_fixed_tolerances_and_exact_integer_state(actual, expected, passed, reason):
    result = compare_tensor(actual, expected, name="test")
    assert result["passed"] is passed
    assert result.get("reason") == reason
    assert ATOL == 1e-5 and RTOL == 1e-4


def test_unavailable_requested_gpu_retains_failure_and_never_falls_back(tmp_path):
    output = tmp_path / "unavailable-gpu"
    result = subprocess.run(
        [sys.executable, "-m", "experiments.resnet50_quality.gate_torch", "--output", str(output)],
        cwd=ROOT, env=_environment(gpu=False), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["device"] == "cuda:0"
    assert "will not fall back" in summary["error"]
    assert summary["cuts"] == []
    assert "gate_finished" in (output / "stdout.log").read_text()
    assert "Traceback" in (output / "stdout.log").read_text()
    # A failed receipt is immutable: a rerun must use a new result directory.
    repeated = subprocess.run(
        [sys.executable, "-m", "experiments.resnet50_quality.gate_torch", "--output", str(output)],
        cwd=ROOT, env=_environment(gpu=False), capture_output=True, text=True, timeout=30,
    )
    assert repeated.returncode != 0
    assert json.loads((output / "summary.json").read_text()) == summary


@pytest.mark.integration
def test_full_resnet50_gpu_native_and_three_real_grpc_cuts_match(tmp_path):
    availability = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=30,
    )
    assert availability.returncode == 0, availability.stdout + availability.stderr
    if availability.stdout.strip() != "True":
        pytest.skip("This phase-A integration check requires an available GPU")
    if not (ROOT / "data/cifar-10-batches-py/data_batch_1").exists():
        pytest.skip("The real CIFAR-10 dataset is unavailable")
    output = tmp_path / "torch-phase-A"
    result = subprocess.run(
        [sys.executable, "-m", "experiments.resnet50_quality.gate_torch", "--output", str(output)],
        cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "passed"
    assert summary["trainable_parameters"] == TRAINABLE_PARAMETERS
    assert summary["batch_size"] == 2 and summary["data"]["input_shape"] == [2, 3, 64, 64]
    assert len(set(summary["data"]["sample_ids"])) == 2
    assert summary["precision"]["dtype"] == "float32"
    assert summary["precision"]["amp"] is False
    assert summary["precision"]["cuda_matmul_tf32"] is False
    assert summary["precision"]["cudnn_tf32"] is False
    assert summary["precision"]["deterministic_algorithms"] is True
    assert summary["atol"] == ATOL and summary["rtol"] == RTOL
    assert {row["stage"] for row in summary["cuts"]} == {"stage1", "stage2", "stage3"}
    for name, digest in summary["source_hashes"].items():
        assert hashlib.sha256((ROOT / "experiments/resnet50_quality" / name).read_bytes()).hexdigest() == digest
    for cut in summary["cuts"]:
        assert cut["status"] == "passed"
        assert cut["budget_ok"] and cut["updates_ok"] and cut["counters_ok"]
        assert cut["gradient_tensor_count"] == len(TRAINABLE_NAMES) == 161
        assert cut["state_tensor_count"] == len(NAMES) == 320
        assert cut["dynamic_batch"] == [1, 16]
        assert cut["batch_validation"]["status"] == "passed"
        assert cut["batch_validation"]["traced_batch_size"] == 1
        assert cut["batch_validation"]["probe_batch_size"] == 2
        assert len(cut["bn_counters"]) == 53 and set(cut["bn_counters"].values()) == {1}
        assert cut["rpc_address"].startswith("127.0.0.1:")
        assert cut["rpc_processed_steps"] == 1
        assert cut["client_metrics"]["activation_upload_bytes"] > 0
        assert cut["client_metrics"]["gradient_download_bytes"] > 0
        assert cut["initial_hash"] == summary["initial_hash"]
        assert len(cut["updated_hash"]) == 64 and cut["updated_hash"] != cut["initial_hash"]
        assert cut["failed_tensors"] == []
        comparisons = cut["comparisons"]
        assert comparisons["logits"]["passed"] and comparisons["ce"]["passed"]
        for category in ("gradients", "sgd_updates"):
            assert set(comparisons[category]) == set(TRAINABLE_NAMES)
            assert all(row["passed"] for row in comparisons[category].values())
        assert set(comparisons["state"]) == set(NAMES)
        assert all(row["passed"] for row in comparisons["state"].values())
        for owner in ("prefix", "suffix"):
            assert cut["updates"][owner]["updated_tensors"] > 0
    assert result.stdout.count('"event": "rpc_step_finished"') == 3
