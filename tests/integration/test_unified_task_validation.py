"""Task losses and both wire directions must preserve complete SGD steps."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from splitfleet.validation import TASKS


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", params=["torch", "jax"])
def matrix_report(request, tmp_path_factory):
    backend = request.param
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"Optional backend {backend} is not installed")
    output = tmp_path_factory.mktemp(f"task-matrix-{backend}") / "report.json"
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[variable] = "1"
    result = subprocess.run(
        [
            sys.executable, "-m", "splitfleet.validation", "--backends", backend,
            "--all-nodes", "--output", str(output),
        ],
        cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True,
        text=True, timeout=660,
    )
    assert output.exists(), result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0, json.dumps(report, indent=2)
    assert report["backend_isolation"] is True
    assert report["wire_roundtrips"] == ["activations", "boundary_gradients"]
    assert report["status"] == "passed"
    return report


@pytest.mark.parametrize("task", TASKS)
def test_every_task_boundary_matches_native_training(matrix_report, task):
    case = next(result for result in matrix_report["results"] if result["task"] == task)
    assert case["status"] == "passed", case
    assert case["selection"] == "all"
    assert case["available_boundaries"] == len(case["nodes"]) > 0
    assert case["counts"] == {"passed": len(case["nodes"]), "failed": 0, "unsupported": 0}
    assert len({node["boundary"] for node in case["nodes"]}) == len(case["nodes"])
    assert any(node["boundary"].startswith("before:") for node in case["nodes"])
    assert any(node["boundary"].startswith("after:") for node in case["nodes"])
    for node in case["nodes"]:
        assert node["status"] == "passed", node
        for key in (
            "output_max_abs_error", "loss_max_abs_error",
            "gradient_max_abs_error", "parameter_max_abs_error",
        ):
            assert key in node, (node["boundary"], key)
