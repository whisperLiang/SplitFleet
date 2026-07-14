"""Real optional-framework smoke tests for TorchLens split replay and training."""

from __future__ import annotations

import numpy as np
import os
import pytest
import subprocess
import sys


def _run_isolated(request, *, extra_env=None) -> bool:
    if os.environ.get("SPLITFLEET_OPTIONAL_BACKEND_CHILD") == "1":
        return False
    env = os.environ.copy()
    env["SPLITFLEET_OPTIONAL_BACKEND_CHILD"] = "1"
    env.update(extra_env or {})
    result = subprocess.run(
        [sys.executable, "-m", "pytest", request.node.nodeid, "-q"],
        cwd=os.getcwd(), env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if " skipped" in result.stdout:
        pytest.skip(result.stdout.strip().splitlines()[-1])
    return True


def _exercise(model, inputs, targets, loss_fn, backend: str) -> None:
    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.backends import BACKEND_ADAPTERS
    from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    adapter = BACKEND_ADAPTERS.create(backend)
    batch_size = int(targets.shape[0])
    runtime = prepare_torchlens_runtime(
        model, inputs, boundary="50%", trainable=True,
        dynamic_batch=(batch_size, batch_size),
    )
    direct = model(*args)
    boundary = runtime.backend.run_prefix(*args)
    split = runtime.backend.run_suffix(boundary)
    np.testing.assert_allclose(adapter._to_numpy(direct), adapter._to_numpy(split), rtol=1e-4, atol=1e-5)
    contract_boundary = boundary_to_envelope(
        boundary, round_id=1, client_id="optional", step_id="1", plan_id=runtime.plan.plan_id,
        split_id=runtime.plan.split_id, canonical_graph_hash=runtime.plan.graph_signature,
        boundary_schema_hash=runtime.plan.feature_abi_id, model_version=1,
    )
    restored = envelope_to_boundary(contract_boundary, runtime.runtime, None)
    assert restored.metadata["backend"] == backend
    training_boundary = runtime.backend.run_prefix(*args, training=True)
    loss, gradients = runtime.backend.train_suffix(training_boundary, targets, loss_fn=loss_fn)
    assert np.isfinite(adapter.scalar_value(loss))
    assert gradients
    runtime.backend.backward_prefix(training_boundary, gradients)


def _exercise_engine_protocol(model, inputs, targets, backend: str) -> None:
    from splitfleet.autosplit.runtime import compute_loss
    from splitfleet.autosplit.torchlens_runtime import make_split_spec
    from splitfleet.backends import BACKEND_ADAPTERS
    from splitfleet.split_engine.torchlens_engine import TorchLensSplitEngine
    from splitfleet.transport import decode_bundle

    args = inputs if isinstance(inputs, tuple) else (inputs,)
    batch_size = int(targets.shape[0])
    engine = TorchLensSplitEngine()
    handle = engine.prepare(
        model,
        args,
        make_split_spec(
            "50%", backend=backend, dynamic_batch=(batch_size, batch_size),
        ),
    )
    boundary, token = engine.run_prefix(handle, args, training=True)
    result = engine.run_suffix(handle, boundary, targets)
    assert result.gradients is not None
    assert np.isfinite(result.loss)
    assert token is not None
    engine.backward_prefix(handle, token, result.gradients)

    eval_boundary, _ = engine.run_prefix(handle, args, training=False)
    eval_result = engine.run_suffix(handle, eval_boundary)
    outputs = decode_bundle(eval_result.outputs, device=None, backend=backend)
    loss = compute_loss(outputs, targets)
    assert np.isfinite(BACKEND_ADAPTERS.create(backend).scalar_value(loss))


def test_tensorflow_split_training_optional(request) -> None:
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": ""}): return
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential([tf.keras.layers.Dense(8, activation="relu"), tf.keras.layers.Dense(2)])
    inputs, targets = tf.random.normal((3, 4)), tf.random.normal((3, 2))
    _exercise(model, inputs, targets, tf.keras.losses.MeanSquaredError(), "tf")


def test_tensorflow_split_engine_protocol_optional(request) -> None:
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": ""}): return
    tf = pytest.importorskip("tensorflow")

    model = tf.keras.Sequential([
        tf.keras.layers.Input((4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(2),
    ])
    inputs = tf.ones((3, 4))
    targets = tf.zeros((3, 2))
    _exercise_engine_protocol(model, inputs, targets, "tf")


def test_jax_split_training_optional(request) -> None:
    if _run_isolated(request): return
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    params = {"weight": jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 10}
    def model(params, x): return jnp.tanh(x) @ params["weight"]
    inputs, targets = (params, jnp.ones((3, 4))), jnp.zeros((3, 2))
    _exercise(model, inputs, targets, lambda output, target: jnp.mean((output - target) ** 2), "jax")
    _exercise_engine_protocol(model, inputs, targets, "jax")


def test_paddle_split_training_optional(request) -> None:
    if _run_isolated(request): return
    paddle = pytest.importorskip("paddle")
    model = paddle.nn.Sequential(paddle.nn.Linear(4, 8), paddle.nn.ReLU(), paddle.nn.Linear(8, 2))
    inputs, targets = paddle.randn((3, 4)), paddle.randn((3, 2))
    _exercise(model, inputs, targets, paddle.nn.MSELoss(), "paddle")
    _exercise_engine_protocol(model, inputs, targets, "paddle")


def test_tinygrad_split_training_optional(request) -> None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed_version = version("tinygrad")
    except PackageNotFoundError:
        pytest.skip("tinygrad is not installed; TorchLens 2.31 requires tinygrad==0.13.0 on Python >=3.11")
    if installed_version != "0.13.0":
        pytest.skip("TorchLens 2.31 requires tinygrad==0.13.0 (Python >=3.11)")
    if _run_isolated(request, extra_env={"DEVICE": "CPU"}): return
    tinygrad = pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    class Model:
        def __call__(self, x): return (x * 2.0).relu() + 1.0
    inputs, targets = Tensor.randn(2, 4), Tensor.randn(2, 4)
    _exercise(Model(), inputs, targets, lambda output, target: ((output - target) ** 2).mean(), "tinygrad")
    _exercise_engine_protocol(Model(), inputs, targets, "tinygrad")
