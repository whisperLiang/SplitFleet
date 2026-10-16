"""Native TF checks; each node executes outside other backend processes."""

from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest


def _isolated(request) -> bool:
    if os.environ.get("SPLITFLEET_RESNET50_TF_MODEL_CHILD") == "1":
        return False
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is not installed")
    environment = dict(os.environ)
    environment.update({
        "SPLITFLEET_RESNET50_TF_MODEL_CHILD": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "2",
    })
    result = subprocess.run(
        [sys.executable, "-m", "pytest", request.node.nodeid, "-o", "addopts=", "-q"],
        cwd=request.config.rootpath, env=environment, text=True, capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if " skipped" in result.stdout:
        pytest.skip(result.stdout.strip().splitlines()[-1])
    return True


def test_factory_roundtrip_shapes_and_explicit_device(request):
    if _isolated(request):
        return
    import tensorflow as tf
    from experiments.resnet50_quality.model_contract import (
        BLOCKS, NAMES, SCHEMA, TRAINABLE_PARAMETERS, canonical_state, state_hash,
    )
    from tests.integration.resnet50_reference.tf_model import build_model, export_state

    state = canonical_state(233)
    for index, row in enumerate(SCHEMA):
        if row.name.endswith("num_batches_tracked"):
            state[row.name] = np.asarray(index, dtype=np.int64)
        elif row.name.endswith("running_mean"):
            state[row.name] = np.linspace(-0.1, 0.1, row.shape[0], dtype=np.float32)
        elif row.name.endswith("running_var"):
            state[row.name] = np.linspace(0.5, 1.5, row.shape[0], dtype=np.float32)
    model = build_model(state, device="cpu")
    assert isinstance(model, tf.keras.Model)
    assert sum(int(np.prod(variable.shape)) for variable in model.trainable_variables) == TRAINABLE_PARAMETERS
    assert len(model.weights) == len(SCHEMA) == 320
    assert tuple(export_state(model)) == NAMES
    assert state_hash(state) == state_hash(export_state(model))
    assert not tf.config.experimental.tensor_float_32_execution_enabled()
    assert not tf.config.get_soft_device_placement()
    for variable in model.weights:
        assert "CPU:0" in variable.value.device
        assert variable.dtype == ("int64" if variable.name == "num_batches_tracked" else "float32")
    assert model.conv1.strides == (2, 2)
    assert model.conv1.padding == "valid" and not model.conv1.use_bias
    assert model.pad1.padding == ((3, 3), (3, 3))
    value = tf.ones((2, 16, 16, 64))
    for stage, (blocks, shape) in enumerate(zip(
        BLOCKS, ((2, 16, 16, 256), (2, 8, 8, 512), (2, 4, 4, 1024), (2, 2, 2, 2048)),
        strict=True,
    ), 1):
        layer = getattr(model, f"layer{stage}")
        assert len(layer.blocks) == blocks
        for index, block in enumerate(layer.blocks):
            assert block.conv1.strides == block.conv3.strides == (1, 1)
            assert block.conv2.strides == ((2, 2) if stage > 1 and index == 0 else (1, 1))
            assert block.pad2.padding == ((1, 1), (1, 1))
            assert all(not getattr(block, f"conv{part}").use_bias for part in (1, 2, 3))
        value = layer(value, training=False)
        assert tuple(value.shape) == shape
    logits = model(tf.ones((2, 64, 64, 3)), training=False)
    assert tuple(logits.shape) == (2, 10) and logits.dtype == tf.float32
    assert bool(tf.reduce_all(tf.math.is_finite(logits)))
    assert state_hash(state) == state_hash(export_state(model))
    with pytest.raises(RuntimeError, match="unavailable"):
        build_model(state, device="cuda:0")
    with pytest.raises(ValueError, match="Unsupported"):
        build_model(state, device="automatic")


def test_batchnorm_normalization_gradients_and_running_variance(request):
    if _isolated(request):
        return
    import tensorflow as tf
    from tests.integration.resnet50_reference.tf_model import CanonicalBatchNorm

    layer = CanonicalBatchNorm(2, name="bn_test")
    layer.build((2, 3, 2, 2))
    layer.gamma.assign([1.5, 0.7])
    layer.beta.assign([0.1, -0.2])
    layer.running_mean.assign([0.8, -0.6])
    layer.running_var.assign([1.2, 1.7])
    layer.num_batches_tracked.assign(9)
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2) / 7.0
    upstream = np.random.default_rng(2).normal(size=array.shape).astype(np.float32)
    inputs = tf.convert_to_tensor(array)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        output = layer(inputs, training=True)
        loss = tf.reduce_sum(output * upstream)
    input_gradient, gamma_gradient, beta_gradient = tape.gradient(
        loss, [inputs, layer.gamma, layer.beta],
    )
    axes = (0, 1, 2)
    mean, variance = array.mean(axes), array.var(axes)
    normalized = (array - mean) / np.sqrt(variance + 1e-5)
    expected = normalized * np.asarray([1.5, 0.7], dtype=np.float32) + [0.1, -0.2]
    np.testing.assert_allclose(output.numpy(), expected, atol=5e-6, rtol=1e-5)
    expected_gradient = np.asarray([1.5, 0.7], dtype=np.float32) / np.sqrt(variance + 1e-5) * (
        upstream - upstream.mean(axes) - normalized * (upstream * normalized).mean(axes)
    )
    np.testing.assert_allclose(input_gradient.numpy(), expected_gradient, atol=5e-6, rtol=1e-5)
    np.testing.assert_allclose(gamma_gradient.numpy(), (upstream * normalized).sum(axes), atol=5e-6)
    np.testing.assert_allclose(beta_gradient.numpy(), upstream.sum(axes), atol=5e-6)
    np.testing.assert_allclose(layer.running_mean.numpy(), 0.9 * np.asarray([0.8, -0.6]) + 0.1 * mean, atol=1e-7)
    np.testing.assert_allclose(layer.running_var.numpy(), 0.9 * np.asarray([1.2, 1.7]) + 0.1 * variance * 12 / 11, atol=2e-7)
    assert int(layer.num_batches_tracked.numpy()) == 10
    buffers = [variable.numpy().copy() for variable in layer.non_trainable_variables]
    evaluation = layer(inputs, training=False)
    expected_eval = (array - layer.running_mean.numpy()) / np.sqrt(layer.running_var.numpy() + 1e-5)
    expected_eval = expected_eval * layer.gamma.numpy() + layer.beta.numpy()
    np.testing.assert_allclose(evaluation.numpy(), expected_eval, atol=1e-6)
    for expected_buffer, variable in zip(buffers, layer.non_trainable_variables, strict=True):
        np.testing.assert_array_equal(expected_buffer, variable.numpy())


def test_deferred_statistics_match_normal_full_training_and_sgd(request):
    if _isolated(request):
        return
    import tensorflow as tf
    from experiments.resnet50_quality.model_contract import canonical_state, state_hash
    from tests.integration.resnet50_reference.tf_model import build_model, export_state

    state = canonical_state(234)
    normal = build_model(state)
    deferred = build_model(state, defer_statistics=True)
    inputs = tf.convert_to_tensor(np.random.default_rng(3).normal(size=(2, 64, 64, 3)).astype(np.float32))
    targets = tf.convert_to_tensor([2, 7], dtype=tf.int64)
    with tf.GradientTape() as normal_tape:
        normal_logits = normal(inputs, training=True)
        normal_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, normal_logits, from_logits=True))
    with tf.GradientTape() as deferred_tape:
        output = deferred(inputs, training=True)
        deferred_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, output["logits"], from_logits=True))
    normal_gradients = normal_tape.gradient(normal_loss, normal.trainable_variables)
    deferred_gradients = deferred_tape.gradient(deferred_loss, deferred.trainable_variables)
    assert len(output["batch_statistics"]) == 53
    assert state_hash(export_state(deferred)) == state_hash(state)
    np.testing.assert_allclose(normal_logits.numpy(), output["logits"].numpy(), rtol=1e-5, atol=1e-6)
    for expected, actual in zip(normal_gradients, deferred_gradients, strict=True):
        assert expected is not None and actual is not None
        assert bool(tf.reduce_all(tf.math.is_finite(actual)))
        np.testing.assert_allclose(expected.numpy(), actual.numpy(), rtol=1e-5, atol=1e-6)
    deferred.apply_statistics(output["batch_statistics"])
    committed_state = export_state(deferred)
    for name, value in export_state(normal).items():
        np.testing.assert_allclose(value, committed_state[name], rtol=1e-5, atol=1e-6)
    normal_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    deferred_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    normal_optimizer.apply_gradients(zip(normal_gradients, normal.trainable_variables, strict=True))
    deferred_optimizer.apply_gradients(zip(deferred_gradients, deferred.trainable_variables, strict=True))
    actual_state = export_state(deferred)
    for name, value in export_state(normal).items():
        np.testing.assert_allclose(value, actual_state[name], rtol=1e-5, atol=1e-6)
    assert not np.array_equal(state["conv1.weight"], actual_state["conv1.weight"])
    assert not np.array_equal(state["fc.weight"], actual_state["fc.weight"])
    assert all(int(value) == 1 for name, value in actual_state.items() if name.endswith("num_batches_tracked"))


def test_deepcopy_preserves_mode_and_partial_statistics_ownership(request):
    if _isolated(request):
        return
    import tensorflow as tf
    from experiments.resnet50_quality.model_contract import canonical_state, state_hash
    from tests.integration.resnet50_reference.tf_model import build_model, export_state

    state = canonical_state(235)
    model = build_model(state, defer_statistics=True)
    model.training = True
    model._splitfleet_training = True
    clone = copy.deepcopy(model)
    assert clone is not model and clone.device == model.device
    assert clone.training and clone._splitfleet_training and clone.defer_statistics
    assert state_hash(export_state(clone)) == state_hash(state)
    assert all(left is not right for left, right in zip(model.weights, clone.weights, strict=True))
    clone.conv1.kernel.assign_add(tf.ones_like(clone.conv1.kernel) * 0.01)
    clone.bn1.running_mean.assign_add(tf.ones_like(clone.bn1.running_mean))
    assert state_hash(export_state(model)) == state_hash(state)
    inputs = tf.convert_to_tensor(np.random.default_rng(5).normal(size=(1, 64, 64, 3)).astype(np.float32))
    output = model(inputs)
    owned = {name for name in model.batch_norm_layers() if name == "bn1" or name.startswith("layer1.")}
    model.apply_statistics(output["batch_statistics"], names=owned)
    result = export_state(model)
    for name, layer in model.batch_norm_layers().items():
        assert int(layer.num_batches_tracked.numpy()) == int(name in owned)
        if name not in owned:
            np.testing.assert_array_equal(state[name + ".running_mean"], result[name + ".running_mean"])
            np.testing.assert_array_equal(state[name + ".running_var"], result[name + ".running_var"])
    before_eval = state_hash(result)
    evaluation = model(inputs, training=False)
    assert evaluation["batch_statistics"] == {}
    assert state_hash(export_state(model)) == before_eval
    with pytest.raises(ValueError, match="Unknown or missing"):
        model.apply_statistics(output["batch_statistics"], names=["layer5.bn1"])


def test_deferred_batchnorm_tf_function_is_pure(request):
    if _isolated(request):
        return
    import tensorflow as tf
    from tests.integration.resnet50_reference.tf_model import CanonicalBatchNorm

    layer = CanonicalBatchNorm(2, name="functional_bn")
    layer.build((None, 3, 2, 2))

    @tf.function(input_signature=[tf.TensorSpec((None, 3, 2, 2), tf.float32), tf.TensorSpec((), tf.bool)])
    def forward(inputs, training):
        return layer(inputs, training=training, defer_statistics=True)

    concrete = forward.get_concrete_function()
    graph = concrete.graph.as_graph_def()
    op_types = {node.op for node in graph.node}
    for function in graph.library.function:
        op_types.update(node.op for node in function.node_def)
    assert not any(op.startswith("Assign") for op in op_types)
    inputs = tf.reshape(tf.range(24, dtype=tf.float32), (2, 3, 2, 2))
    output, statistics = forward(inputs, tf.constant(True))
    assert int(statistics["count"].numpy()) == 12
    assert int(layer.num_batches_tracked.numpy()) == 0
    assert tuple(output.shape) == (2, 3, 2, 2)
    layer.apply_statistics(statistics)
    assert int(layer.num_batches_tracked.numpy()) == 1
    _, evaluation_statistics = forward(inputs, tf.constant(False))
    assert int(evaluation_statistics["count"].numpy()) == 0
    assert int(layer.num_batches_tracked.numpy()) == 1


def test_full_deferred_training_is_capturable_without_state_writes(request):
    if _isolated(request):
        return
    import tensorflow as tf
    torchlens = pytest.importorskip("torchlens")
    from experiments.resnet50_quality.model_contract import canonical_state, state_hash
    from tests.integration.resnet50_reference.tf_model import build_model, export_state

    state = canonical_state(236)
    model = build_model(state, defer_statistics=True)
    model.training = True
    inputs = tf.convert_to_tensor(np.random.default_rng(7).normal(size=(2, 64, 64, 3)).astype(np.float32))
    logged = torchlens.trace(model, (inputs,), backend="tf", layers_to_save="all")
    assert state_hash(export_state(model)) == state_hash(state)
    addresses = {module.address for module in logged.modules}
    assert {f"layer{stage}" for stage in range(1, 5)} <= addresses
    assert {f"layer1.{block}" for block in range(3)} <= addresses
    operations = [str(op.layer_type).lower() for op in logged]
    assert operations.count("conv2d") == 53
    assert not any("assign" in operation for operation in operations)
