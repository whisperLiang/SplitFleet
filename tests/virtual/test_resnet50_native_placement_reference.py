"""Real native controls for sequential SGD and independent aggregation."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.resnet50_quality.model_contract import NAMES, canonical_state
from experiments.resnet50_quality.staged_reference import aggregate_native


def test_independent_native_aggregation_keeps_float_weights_and_integer_counter_maximum():
    first, second = canonical_state(233), canonical_state(234)
    first["bn1.num_batches_tracked"] = np.asarray(2, np.int64)
    second["bn1.num_batches_tracked"] = np.asarray(5, np.int64)
    result = aggregate_native([(first, 2), (second, 1)])
    assert tuple(result) == NAMES
    assert int(result["bn1.num_batches_tracked"]) == 5
    np.testing.assert_array_equal(result["conv1.weight"], (first["conv1.weight"] * 2 + second["conv1.weight"]) / 3)
    assert all(result[name].dtype == first[name].dtype for name in NAMES)


def test_native_stage_change_preserves_two_step_gradients_sgd_and_bn_state():
    from experiments.resnet50_quality.gate_torch import compare_tensor
    from experiments.resnet50_quality.native import (
        build_model,
        configure_backend,
        export_state,
        native_batch,
        optimizer,
        train_step,
    )
    from experiments.resnet50_quality.staged_reference import train_staged

    configure_backend("pytorch")
    initial = canonical_state(233)
    native = build_model("pytorch", initial, "cpu")
    opt = optimizer("pytorch", native, .05)
    rng = np.random.default_rng(233)
    images, labels = native_batch("pytorch", rng.normal(size=(2, 3, 64, 64)).astype(np.float32),
                                 np.asarray([2, 3], np.int64), "cpu")
    state = initial
    for step, stage in enumerate((1, 3), start=1):
        expected = train_step("pytorch", native, images, labels, native_optimizer=opt, learning_rate=.05, capture=True)
        actual = train_staged("pytorch", state, images, labels, stage=stage,
                             client_device="cpu", server_device="cpu", learning_rate=.05)
        assert all(compare_tensor(actual["gradients"][name], value, name=name)["passed"]
                   for name, value in expected["gradients"].items())
        expected_state = export_state("pytorch", native)
        assert all(compare_tensor(actual["state"][name], expected_state[name], name=name)["passed"] for name in NAMES)
        assert all(int(value) == step for name, value in actual["state"].items() if name.endswith("num_batches_tracked"))
        state = actual["state"]


def test_tensorflow_native_vjp_stage_change_and_optimizer_device():
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is not installed")
    # Isolation prevents TF from initializing CUDA in the pytest controller or
    # acquiring either GPU used by independently running native experiments.
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", NVIDIA_TF32_OVERRIDE="0", OMP_NUM_THREADS="1",
                       TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                       PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    subprocess.run([sys.executable, str(Path(__file__).with_name("tensorflow_native_reference_control.py"))],
                   env=environment, check=True, timeout=120, capture_output=True, text=True)
