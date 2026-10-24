"""Real optional-framework smoke tests for TorchLens split replay and training."""

from __future__ import annotations

import numpy as np
import os
import pytest
import subprocess
import sys


def _run_isolated(request, *, extra_env=None, python=None) -> bool:
    if os.environ.get("SPLITFLEET_OPTIONAL_BACKEND_CHILD") == "1":
        return False
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        env[variable] = "1"
    env["SPLITFLEET_OPTIONAL_BACKEND_CHILD"] = "1"
    env.update(extra_env or {})
    result = subprocess.run(
        [python or sys.executable, "-m", "pytest", request.node.nodeid,
         "-o", "addopts=", "-q", "-rs", "--capture=sys"],
        cwd=request.config.rootpath, env=env, text=True, capture_output=True,
        timeout=300,
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


def _exercise_engine_protocol(model, inputs, targets, loss_fn, backend: str) -> None:
    from splitfleet.autosplit.runtime import compute_loss
    from splitfleet.autosplit.torchlens_runtime import make_split_spec
    from splitfleet.backends import BACKEND_ADAPTERS
    from splitfleet.split_engine.torchlens_engine import TorchLensSplitEngine
    from splitfleet.transport import decode_bundle

    args = inputs if isinstance(inputs, tuple) else (inputs,)
    engine = TorchLensSplitEngine()
    handle = engine.prepare(
        model,
        args,
        make_split_spec(
            "50%", backend=backend,
        ),
    )
    boundary, token = engine.run_prefix(handle, args, training=True)
    result = engine.run_suffix(handle, boundary, targets, loss_fn=loss_fn)
    assert result.gradients is not None
    assert np.isfinite(result.loss)
    assert token is not None
    engine.backward_prefix(handle, token, result.gradients)

    eval_boundary, _ = engine.run_prefix(handle, args, training=False)
    eval_result = engine.run_suffix(handle, eval_boundary)
    outputs = decode_bundle(eval_result.outputs, device=None, backend=backend)
    loss = compute_loss(outputs, targets, loss_fn)
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
    _exercise_engine_protocol(model, inputs, targets, tf.keras.losses.MeanSquaredError(), "tf")


def test_paddle_split_training_optional(request) -> None:
    if _run_isolated(request): return
    paddle = pytest.importorskip("paddle")
    model = paddle.nn.Sequential(paddle.nn.Linear(4, 8), paddle.nn.ReLU(), paddle.nn.Linear(8, 2))
    inputs, targets = paddle.randn((3, 4)), paddle.randn((3, 2))
    _exercise(model, inputs, targets, paddle.nn.MSELoss(), "paddle")
    _exercise_engine_protocol(model, inputs, targets, paddle.nn.MSELoss(), "paddle")


@pytest.mark.parametrize("backend", ["tf", "paddle"])
def test_cosplit_payload_context_matches_optional_backend_capture(request, backend):
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": "", "DEV": "CPU", "DEBUG": "0"}):
        return
    from splitfleet.server.placement.greedy import TorchLensCandidateProvider
    from splitfleet.server.placement.greedy.calibration import provider_handle

    if backend == "tf":
        tf = pytest.importorskip("tensorflow")
        model = tf.keras.Sequential([
            tf.keras.layers.Input((4,)), tf.keras.layers.Dense(8, activation="relu"), tf.keras.layers.Dense(2)])
        inputs, batch_argument = (tf.ones((3, 4)),), 0
    elif backend == "paddle":
        paddle = pytest.importorskip("paddle")
        model = paddle.nn.Sequential(paddle.nn.Linear(4, 8), paddle.nn.ReLU(), paddle.nn.Linear(8, 2))
        inputs, batch_argument = (paddle.ones((3, 4)),), 0
    provider = TorchLensCandidateProvider(model=model, sample_inputs=inputs,
        batch_axes={f"/args/{batch_argument}": 0}, dynamic_batch=(1, 4))
    catalog = provider.get_candidates(training=True)
    selected = {catalog[index].boundary: catalog[index] for index in (0, len(catalog) // 2, -1)}
    for candidate in selected.values():
        batch = candidate.metadata["payload_batch_size"]
        call = list(inputs)
        call[batch_argument] = call[batch_argument][:batch]
        payload = provider_handle(provider, candidate.boundary).backend.run_prefix(*call)
        actual = sum(np.asarray(value.numpy() if hasattr(value, "numpy") else value).nbytes
                     for value in payload.tensors.values())
        assert actual == candidate.boundary_forward_bytes
        assert candidate.metadata["boundary_forward_bytes_by_batch_size"][batch] == actual
