"""Canonical native layouts and genuine isolated GPU/RPC phase-A coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.integration.resnet50_reference.array_helpers import (
    RecordingFunctionalSGD, canonical_array, stage_boundaries,
)
from experiments.resnet50_quality.model_contract import NAMES, TRAINABLE_NAMES, TRAINABLE_PARAMETERS

ROOT = Path(__file__).resolve().parents[2]


def test_tf_gradient_layout_preserves_every_axis_and_scalar_counter():
    stem = np.arange(64 * 3 * 7 * 7, dtype=np.float32).reshape(64, 3, 7, 7)
    np.testing.assert_array_equal(canonical_array("tensorflow", "conv1.weight", stem.transpose(2, 3, 1, 0)), stem)
    linear = np.arange(10 * 2048, dtype=np.float32).reshape(10, 2048)
    np.testing.assert_array_equal(canonical_array("tensorflow", "fc.weight", linear.T), linear)
    count = canonical_array("tensorflow", "bn1.num_batches_tracked", np.asarray(1, np.int64))
    assert count.shape == () and count.dtype == np.int64
    with pytest.raises(RuntimeError, match="schema"):
        canonical_array("tensorflow", "fc.weight", linear.T.astype(np.float64))
    with pytest.raises(RuntimeError, match="schema"):
        canonical_array("paddle", "bn1.num_batches_tracked", np.asarray([1], np.int64))


@pytest.mark.parametrize("backend", ["tensorflow", "paddle", "jax"])
def test_stage_selection_uses_final_bottleneck_relu_metadata(backend):
    nodes = []
    for stage, blocks, channels, spatial in ((1, 3, 256, 16), (2, 4, 512, 8), (3, 6, 1024, 4)):
        shape = (1, spatial, spatial, channels) if backend == "tensorflow" else (1, channels, spatial, spatial)
        for block in range(blocks):
            nodes.append(SimpleNamespace(module_path=f"layer{stage}.{block}" + ("" if backend == "jax" else ".relu3"),
                op_type="select_n" if backend == "jax" else "Relu",
                output_shape=shape, canonical_id=f"relu_stage{stage}_block{block}", label="native_relu"))
        nodes.append(SimpleNamespace(module_path=f"layer{stage}.{blocks - 1}.bn3", op_type="Mean",
            output_shape=(channels,), canonical_id="statistics", label="statistics"))
    handle = SimpleNamespace(runtime=SimpleNamespace(trace_graph=SimpleNamespace(nodes=nodes)))
    endpoints = stage_boundaries(handle, backend)
    assert list(endpoints) == ["stage1", "stage2", "stage3"]
    assert [row["canonical_node"] for row in endpoints.values()] == ["relu_stage1_block2", "relu_stage2_block3", "relu_stage3_block5"]


def _interpreter(backend):
    selected = ROOT / {
        "paddle": ".venv-paddle-gpu/bin/python", "jax": ".venv-jax-gpu/bin/python",
        "tensorflow": ".venv/bin/python",
    }[backend]
    return str(selected) if selected.exists() else sys.executable


def _environment(backend=None):
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="0", NVIDIA_TF32_OVERRIDE="0")
    if backend == "jax":
        environment.update(CUDA_VISIBLE_DEVICES="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                           JAX_PLATFORMS="cuda,cpu")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        environment[name] = "1"
    return environment


def test_recording_functional_sgd_delegates_native_update_and_excludes_buffers():
    from splitfleet.runtime.functional_sgd import OwnedFunctionalSGD

    class Model:
        params = {"bn1.bias": np.ones(64, np.float32),
                  "bn1.num_batches_tracked": np.asarray(0, np.int64)}

        def set_params(self, values):
            self.params = values

    model = Model()
    optimizer = RecordingFunctionalSGD(model)
    assert isinstance(optimizer, OwnedFunctionalSGD)
    gradients = {"bn1.bias": np.arange(64, dtype=np.float32),
                 "bn1.num_batches_tracked": np.asarray(1, np.int64)}
    optimizer._step(gradients)
    assert optimizer.recorded_steps == 1 and set(optimizer.gradients) == {"bn1.bias"}
    np.testing.assert_array_equal(optimizer.gradients["bn1.bias"], gradients["bn1.bias"])
    np.testing.assert_array_equal(model.params["bn1.bias"], 1 - np.float32(.05) * gradients["bn1.bias"])
    assert int(model.params["bn1.num_batches_tracked"]) == 0
    with pytest.raises(RuntimeError, match="more than one"):
        optimizer._step({"bn1.bias": gradients["bn1.bias"]})


def test_jax_loss_receipt_observes_same_native_differentiated_forward(request):
    if os.environ.get("SPLITFLEET_JAX_LOSS_TEST_CHILD") != "1":
        environment = _environment("jax")
        environment.update(SPLITFLEET_JAX_LOSS_TEST_CHILD="1", CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
        result = subprocess.run([_interpreter("jax"), "-m", "pytest", request.node.nodeid,
            "-o", "addopts=", "-q"], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    import jax
    import jax.numpy as jnp
    from tests.integration.resnet50_reference.array_helpers import RecordingLoss
    from splitfleet.tasks.losses import sparse_cross_entropy

    logits = jnp.asarray([[.1, -.4, .7], [.3, .6, -.2]], dtype=jnp.float32)
    labels = jnp.asarray([2, 1], dtype=jnp.int32)
    recorder = RecordingLoss("jax")
    actual_loss, actual_gradient = jax.value_and_grad(
        lambda value: recorder({"logits": value}, labels))(logits)
    jax.effects_barrier()
    expected_loss, expected_gradient = jax.value_and_grad(
        lambda value: sparse_cross_entropy(value, labels))(logits)
    assert len(recorder.calls) == 1
    np.testing.assert_array_equal(recorder.calls[0]["logits"], np.asarray(logits))
    assert recorder.calls[0]["loss"] == float(expected_loss)
    np.testing.assert_array_equal(np.asarray(actual_loss), np.asarray(expected_loss))
    np.testing.assert_array_equal(np.asarray(actual_gradient), np.asarray(expected_gradient))


def test_tf_loss_receipt_observes_each_compiled_training_invocation(request):
    if os.environ.get("SPLITFLEET_TF_LOSS_TEST_CHILD") != "1":
        environment = _environment("tensorflow")
        environment.update(SPLITFLEET_TF_LOSS_TEST_CHILD="1", CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([_interpreter("tensorflow"), "-m", "pytest", request.node.nodeid,
            "-o", "addopts=", "-q"], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    import tensorflow as tf
    from tests.integration.resnet50_reference.array_helpers import RecordingLoss
    from splitfleet.tasks.losses import sparse_cross_entropy

    recorder = RecordingLoss("tensorflow")

    @tf.function(autograph=False)
    def differentiated(logits, labels):
        with tf.GradientTape() as tape:
            tape.watch(logits)
            loss = recorder({"logits": logits}, labels)
        return loss, tape.gradient(loss, logits)

    labels = tf.constant([2, 1])
    for index in range(2):
        logits = tf.constant([[.1 + index, -.4, .7], [.3, .6, -.2]])
        actual_loss, actual_gradient = differentiated(logits, labels)
        with tf.GradientTape() as tape:
            tape.watch(logits)
            expected_loss = sparse_cross_entropy(logits, labels)
        expected_gradient = tape.gradient(expected_loss, logits)
        assert len(recorder.calls) == index + 1
        np.testing.assert_array_equal(recorder.calls[index]["logits"], logits.numpy())
        assert recorder.calls[index]["loss"] == float(expected_loss)
        np.testing.assert_array_equal(actual_loss.numpy(), expected_loss.numpy())
        np.testing.assert_array_equal(actual_gradient.numpy(), expected_gradient.numpy())


def test_jax_factory_bn_replay_preserves_all_native_gradients(request):
    """Use the actual factory BN body between trainable upstream/head weights."""
    if os.environ.get("SPLITFLEET_JAX_BN_TEST_CHILD") != "1":
        environment = _environment("jax")
        environment.update(SPLITFLEET_JAX_BN_TEST_CHILD="1", CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
        result = subprocess.run([_interpreter("jax"), "-m", "pytest", request.node.nodeid,
            "-o", "addopts=", "-q"], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    import jax
    import jax.numpy as jnp
    from tests.integration.resnet50_reference.jax_model import ResNet50
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.tasks.losses import sparse_cross_entropy

    random = np.random.default_rng(712)

    class Model:
        def __init__(self):
            self.params = {
                "first": jnp.asarray(random.normal(size=(3, 64)).astype(np.float32) * .2),
                "bn1.weight": jnp.ones(64, dtype=jnp.float32),
                "bn1.bias": jnp.zeros(64, dtype=jnp.float32),
                "head": jnp.asarray(random.normal(size=(64, 3)).astype(np.float32) * .2),
                "head_bias": jnp.zeros(3, dtype=jnp.float32),
            }
            self.training = True

        def __call__(self, params, images):
            projected = jax.lax.dot_general(images, params["first"], (((3,), (0,)), ((), ())))
            statistics = {}
            normalized = ResNet50._batch_norm(self, projected.transpose(0, 3, 1, 2), params,
                                              "bn1", statistics, training=True)
            activated = jax.lax.select(normalized > 0, normalized, jnp.zeros_like(normalized))
            logits = activated.mean(axis=(2, 3)) @ params["head"] + params["head_bias"]
            return {"logits": logits, "batch_statistics": statistics}

        def apply_statistics(self, statistics, names=None):
            pass

    model = Model()
    images = jnp.asarray(random.normal(size=(2, 2, 2, 3)).astype(np.float32))
    labels = jnp.asarray([0, 2], dtype=jnp.int32)
    session = AutoSplitSession(device="cpu")
    base = session.prepare_runtime(model, (model.params, images), trainable=True,
                                   batch_axes={"/args/1": 0}, dynamic_batch=(1, 16))
    endpoint = next(node for node in base.runtime.trace_graph.nodes if node.op_type == "dot_general")
    handle = session.repartition_runtime(base, "after:" + endpoint.canonical_id)
    assert handle.runtime.batch_validation["status"] == "passed"
    for batch in (1, 2, 16):
        current = jnp.asarray(random.normal(size=(batch, 2, 2, 3)).astype(np.float32))
        labels = jnp.arange(batch, dtype=jnp.int32) % 3
        expected_loss, expected_gradients = jax.value_and_grad(
            lambda values: sparse_cross_entropy(model(values, current)["logits"], labels))(model.params)
        expected_statistics = model(model.params, current)["batch_statistics"]
        boundary = handle.backend.run_prefix(model.params, current, training=True)
        loss, gradients = handle.backend.train_suffix(boundary, labels,
            loss_fn=lambda output, targets: sparse_cross_entropy(output["logits"], targets))
        actual = handle.backend.backward_prefix(boundary, gradients)["inputs"][0]
        np.testing.assert_allclose(np.asarray(loss), np.asarray(expected_loss), atol=1e-5, rtol=1e-4)
        assert set(actual) == set(expected_gradients)
        for name, expected in expected_gradients.items():
            np.testing.assert_allclose(np.asarray(actual[name]), np.asarray(expected), atol=1e-5, rtol=1e-4,
                                       err_msg=f"Actual factory BN replay gradient differs at {name}, B={batch}")
        for name, expected in expected_statistics["bn1"].items():
            np.testing.assert_allclose(np.asarray(handle.backend.train_suffix_auxiliary["bn1"][name]),
                                       np.asarray(expected), atol=1e-5, rtol=1e-4)
        assert int(handle.backend.train_suffix_auxiliary["bn1"]["count"]) == 4 * batch


@pytest.mark.parametrize("norm_owner", ["prefix", "suffix"])
def test_tf_native_resource_gradients_include_upstream_weight_bn_and_head(request, norm_owner):
    """Compare every native tape gradient and real SGD state in isolation."""
    if os.environ.get("SPLITFLEET_TF_RESOURCE_TEST_CHILD") != "1":
        environment = _environment()
        environment.update(SPLITFLEET_TF_RESOURCE_TEST_CHILD="1", CUDA_VISIBLE_DEVICES="",
                           TF_ENABLE_ONEDNN_OPTS="0", TF_CPP_MIN_LOG_LEVEL="2")
        result = subprocess.run([_interpreter("tensorflow"), "-m", "pytest", request.node.nodeid,
            "-o", "addopts=", "-q"], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return

    import tensorflow as tf
    from torchlens.split import training as native_training
    from torchlens.split.adapters.tf import _tf_gradient_source
    from tests.integration.resnet50_reference.tf_model import CanonicalBatchNorm
    from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
    from splitfleet.autosplit.torchlens_candidate import ParameterCountIndex

    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__(name="resource_test")
            self.training, self.defer_statistics, self.forward_calls = True, True, 0
            self.first = tf.keras.layers.Conv2D(4, 1, use_bias=False, name="first")
            self.norm = CanonicalBatchNorm(4, name="norm")
            self.relu = tf.keras.layers.ReLU(name="relu")
            self.head = tf.keras.layers.Dense(3, name="head")

        def call(self, inputs, training=True):
            self.forward_calls += 1
            hidden, statistics = self.norm(self.first(inputs), training=True, defer_statistics=True)
            logits = self.head(tf.reduce_mean(self.relu(hidden), axis=(1, 2)))
            return {"logits": logits, "batch_statistics": {"norm": statistics}}

        def apply_statistics(self, statistics, names=None):
            if names is None or "norm" in names:
                self.norm.apply_statistics(statistics["norm"])

        def __deepcopy__(self, memo):
            clone = Model()
            clone(tf.zeros((1, 2, 2, 3)))
            clone.set_weights(self.get_weights())
            memo[id(self)] = clone
            return clone

    random = np.random.default_rng(735)
    images = tf.constant(random.normal(size=(2, 2, 2, 3)).astype(np.float32))
    targets = tf.constant(random.normal(size=(2, 3)).astype(np.float32))
    model = Model()
    model(images)
    for variable in model.trainable_variables:
        variable.assign(random.normal(size=variable.shape).astype(np.float32) * np.float32(.2))
    initial = [value.copy() for value in model.get_weights()]

    def loss(output, target):
        return tf.reduce_mean((output["logits"] - target) ** 2)

    with tf.GradientTape() as tape:
        direct = model(images)
        reference_loss = loss(direct, targets)
    reference_gradients = tape.gradient(reference_loss, model.trainable_variables)
    assert len(reference_gradients) == 5 and all(value is not None for value in reference_gradients)
    reference_gradients = [value.numpy().copy() for value in reference_gradients]
    reference_optimizer = tf.keras.optimizers.SGD(learning_rate=.05)
    reference_optimizer.apply_gradients(zip(reference_gradients, model.trainable_variables))
    model.apply_statistics(direct["batch_statistics"])
    reference_state = [value.copy() for value in model.get_weights()]
    model.set_weights(initial)

    original_native = (native_training._trainable_param_handles, native_training._train_suffix_tf,
                       native_training._backward_prefix_tf)
    backend = TorchLensSplitBackend(device="cpu").trace(
        model, images, batch_axes={"/args/0": 0}, dynamic_batch=(1, 4), trainable=True)
    endpoint = next(node for node in backend.runtime.trace_graph.nodes
                    if node.module_path == ("relu" if norm_owner == "prefix" else "first")
                    and str(node.op_type).lower() in {"relu", "conv2d"})
    backend.repartition("after:" + endpoint.canonical_id)
    graph_signature = backend.make_handle().plan.graph_signature
    index = ParameterCountIndex.from_runtime(backend.runtime)
    assert index.count(backend.runtime.trace_graph.node_by_id, trainable_only=True) == sum(
        int(np.prod(value.shape)) for value in model.trainable_variables)
    for before, after in zip(initial, model.get_weights(), strict=True):
        np.testing.assert_array_equal(after, before)

    class Recorder:
        def __init__(self):
            self.native = tf.keras.optimizers.SGD(learning_rate=.05)
            self.native.build(model.trainable_variables)
            self.identities = {id(_tf_gradient_source(value)): index
                               for index, value in enumerate(model.trainable_variables)}
            self.gradients = {}

        def apply_gradients(self, pairs):
            pairs = list(pairs)
            for gradient, variable in pairs:
                index = self.identities[id(_tf_gradient_source(variable))]
                assert index not in self.gradients, "A live variable received duplicate native SGD updates"
                self.gradients[index] = gradient.numpy().copy()
            return self.native.apply_gradients(pairs)

    optimizer = Recorder()
    training_forward_calls = model.forward_calls
    boundary = backend.run_prefix(images, training=True)
    suffix_loss_calls = []

    def recorded_loss(output, target):
        suffix_loss_calls.append(output)
        return loss(output, target)

    _, boundary_grads = backend.train_suffix(boundary, targets, loss_fn=recorded_loss, optimizer=optimizer)
    backend.backward_prefix(boundary, boundary_grads, optimizer=optimizer)
    assert len(suffix_loss_calls) == 1 and model.forward_calls == training_forward_calls
    assert int(model.norm.num_batches_tracked.numpy()) == 0
    assert set(optimizer.gradients) == set(range(5)), "Native prefix/suffix SGD omitted a declared variable"
    for index, reference in enumerate(reference_gradients):
        np.testing.assert_allclose(optimizer.gradients[index], reference, atol=1e-5, rtol=1e-4)
    statistics = backend.train_suffix_auxiliary
    for name, expected in direct["batch_statistics"]["norm"].items():
        np.testing.assert_allclose(statistics["norm"][name].numpy(), expected.numpy(), atol=1e-5, rtol=1e-4)
    model.apply_statistics(statistics)
    for actual, expected in zip(model.get_weights(), reference_state, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    assert int(model.norm.num_batches_tracked.numpy()) == 1
    assert backend.make_handle().plan.graph_signature == graph_signature
    assert original_native == (native_training._trainable_param_handles, native_training._train_suffix_tf,
                               native_training._backward_prefix_tf)
