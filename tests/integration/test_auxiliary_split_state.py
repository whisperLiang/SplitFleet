"""One-forward native split training with separately committed BN statistics."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _isolated(request, backend):
    if os.environ.get("SPLITFLEET_AUXILIARY_STATE_CHILD") == "1":
        return False
    module = "tensorflow" if backend == "tf" else backend
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} is not installed")
    environment = dict(os.environ)
    environment.update({
        "SPLITFLEET_AUXILIARY_STATE_CHILD": "1", "CUDA_VISIBLE_DEVICES": "",
        "TF_CPP_MIN_LOG_LEVEL": "2", "TF_ENABLE_ONEDNN_OPTS": "0", "DEV": "CPU",
        "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    result = subprocess.run(
        [sys.executable, "-m", "pytest", request.node.nodeid, "-o", "addopts=", "-q"],
        cwd=request.config.rootpath, env=environment, text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if " skipped" in result.stdout:
        pytest.skip(result.stdout.strip().splitlines()[-1])
    return True


def _case(backend, batch=2):
    random = np.random.default_rng(102)
    images = random.normal(size=(batch, 2, 2, 3)).astype(np.float32)
    labels = random.normal(size=(batch, 3)).astype(np.float32)
    if backend == "tf":
        import tensorflow as tf
        from tests.integration.resnet50_reference.tf_model import CanonicalBatchNorm

        class Norm(CanonicalBatchNorm):
            def _statistics(self, inputs):
                shape = tf.shape(inputs)
                mean, variance = tf.nn.moments(inputs, axes=(0, 1, 2))
                return {"mean": mean, "variance": variance, "count": shape[0] * shape[1] * shape[2]}

        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__(name="auxiliary_model")
                self.training = True
                self.defer_statistics = True
                self.first = tf.keras.layers.Conv2D(4, 1, use_bias=False, name="first")
                self.norm = Norm(4, name="norm")
                self.last = tf.keras.layers.Dense(3, name="last")

            def call(self, inputs, training=None):
                hidden, statistics = self.norm(self.first(inputs), training=True, defer_statistics=True)
                logits = self.last(tf.reduce_mean(tf.nn.relu(hidden), axis=(1, 2)))
                return {"logits": logits, "batch_statistics": {"layer1.0.bn1": statistics}}

            def apply_statistics(self, statistics, names=None):
                if names is None or "layer1.0.bn1" in names:
                    self.norm.apply_statistics(statistics["layer1.0.bn1"])

            def __deepcopy__(self, memo):
                clone = Model()
                clone(tf.zeros((1, 2, 2, 3)))
                clone.set_weights(self.get_weights())
                memo[id(self)] = clone
                return clone

        model = Model()
        inputs, targets = tf.constant(images), tf.constant(labels)
        model(inputs)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        optimizer.build(model.trainable_variables)
        return SimpleNamespace(model=model, inputs=(inputs,), targets=targets,
            loss=lambda output, target: tf.reduce_mean((output["logits"] - target) ** 2),
            optimizer=optimizer,
            numpy=lambda value: np.asarray(value.numpy()),
            first=lambda: model.first.kernel.numpy().copy(),
            count=lambda: int(model.norm.num_batches_tracked.numpy()),
            axes={"/args/0": 0}, boundary="50%")
    if backend == "paddle":
        import paddle

        class Norm(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.gamma = self.create_parameter([4], default_initializer=paddle.nn.initializer.Constant(1))
                self.beta = self.create_parameter([4], default_initializer=paddle.nn.initializer.Constant(0))
                self.register_buffer("running_mean", paddle.zeros([4]))
                self.register_buffer("running_var", paddle.ones([4]))
                self.register_buffer("counter", paddle.zeros([], dtype="int64"))

            def forward(self, inputs):
                mean = inputs.mean(axis=[0, 2, 3])
                centered = inputs - mean.reshape([1, 4, 1, 1])
                variance = (centered * centered).mean(axis=[0, 2, 3])
                count = paddle.ones_like(inputs[:, :1], dtype="int64").sum()
                output = centered * (variance + 1e-5).rsqrt().reshape([1, 4, 1, 1])
                output = output * self.gamma.reshape([1, 4, 1, 1]) + self.beta.reshape([1, 4, 1, 1])
                return output, {"mean": mean, "variance": variance, "count": count}

        class Model(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.defer_statistics = True
                self.first = paddle.nn.Conv2D(3, 4, 1, bias_attr=False)
                self.norm = Norm()
                self.last = paddle.nn.Linear(4, 3)

            def forward(self, inputs):
                hidden, statistics = self.norm(self.first(inputs))
                logits = self.last(paddle.nn.functional.relu(hidden).mean(axis=[2, 3]))
                return {"logits": logits, "batch_statistics": {"layer1.0.bn1": statistics}}

            def apply_statistics(self, statistics, names=None):
                if names is None or "layer1.0.bn1" in names:
                    row = statistics["layer1.0.bn1"]
                    count = int(row["count"].item())
                    with paddle.no_grad():
                        self.norm.running_mean.set_value(.9 * self.norm.running_mean + .1 * row["mean"])
                        self.norm.running_var.set_value(.9 * self.norm.running_var + .1 * row["variance"] * count / (count - 1))
                        self.norm.counter.set_value(self.norm.counter + 1)

        model = Model()
        model.train()
        inputs, targets = paddle.to_tensor(images.transpose(0, 3, 1, 2)), paddle.to_tensor(labels)
        return SimpleNamespace(model=model, inputs=(inputs,), targets=targets,
            loss=lambda output, target: ((output["logits"] - target) ** 2).mean(),
            optimizer=paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters()),
            numpy=lambda value: np.asarray(value.numpy()), first=lambda: model.first.weight.numpy().copy(),
            count=lambda: int(model.norm.counter.item()), axes={"/args/0": 0}, boundary="50%")
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        class Model:
            def __init__(self):
                self.training = True
                self.params = {
                    "first": jnp.asarray(random.normal(size=(3, 4)).astype(np.float32)),
                    "gamma": jnp.ones(4), "beta": jnp.zeros(4),
                    "last": jnp.asarray(random.normal(size=(4, 3)).astype(np.float32)),
                }
                self.counter = 0
                self.running_mean, self.running_var = np.zeros(4, np.float32), np.ones(4, np.float32)

            def __call__(self, parameters, inputs):
                hidden = jax.lax.dot_general(inputs, parameters["first"], (((3,), (0,)), ((), ())))
                count = jnp.ones_like(hidden[..., :1], dtype=jnp.int32).sum()
                mean = hidden.sum(axis=(0, 1, 2)) / count
                centered = hidden - mean
                variance = (centered * centered).sum(axis=(0, 1, 2)) / count
                normalized = centered * jax.lax.rsqrt(variance + np.float32(1e-5))
                normalized = jax.nn.relu(normalized * parameters["gamma"] + parameters["beta"])
                logits = normalized.mean(axis=(1, 2)) @ parameters["last"]
                statistics = {
                    "mean": jax.lax.stop_gradient(mean), "variance": jax.lax.stop_gradient(variance),
                    "count": count,
                }
                return {"logits": logits, "batch_statistics": {"layer1.0.bn1": statistics}}

            def apply_statistics(self, statistics, names=None):
                if names is None or "layer1.0.bn1" in names:
                    row = jax.device_get(statistics["layer1.0.bn1"])
                    count = int(row["count"])
                    self.running_mean = .9 * self.running_mean + .1 * row["mean"]
                    self.running_var = .9 * self.running_var + .1 * row["variance"] * count / (count - 1)
                    self.counter += 1

        model = Model()
        inputs, targets = (model.params, jnp.asarray(images)), jnp.asarray(labels)
        return SimpleNamespace(model=model, inputs=inputs, targets=targets,
            loss=lambda output, target: jnp.mean((output["logits"] - target) ** 2), optimizer=None,
            numpy=np.asarray, first=lambda: np.asarray(model.params["first"]).copy(),
            count=lambda: model.counter, axes={"/args/1": 0}, boundary="50%")
    raise ValueError(backend)


@pytest.mark.parametrize("backend", ["tf", "paddle", "jax"])
def test_native_auxiliary_forward_backward_and_owned_commit(request, backend):
    if _isolated(request, backend):
        return
    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.runtime.auxiliary_state import commit_auxiliary_state

    case = _case(backend)
    expected = case.model(*case.inputs)
    expected_statistics = {name: {key: case.numpy(value).copy() for key, value in row.items()}
                           for name, row in expected["batch_statistics"].items()}
    handle = prepare_torchlens_runtime(case.model, case.inputs, boundary=case.boundary,
        batch_axes=case.axes, dynamic_batch=(1, 16))
    if backend == "jax":
        # Select the native first projection value without building a graph or
        # assuming an operation number in its capture.
        projection = next(node for node in handle.runtime.trace_graph.nodes if node.op_type == "dot_general")
    else:
        projection = [node for node in handle.runtime.trace_graph.nodes if node.module_path == "first"
                      and not (node.is_input or node.is_output or node.is_param_source or node.is_buffer)][-1]
    handle = handle.backend.repartition("after:" + projection.canonical_id)
    assert case.count() == 0
    if backend == "paddle":
        assert all(node.output_container_spec is not None and node.output_container_paths
                   for node in handle.runtime.trace_graph.nodes if node.is_output)
    assert handle.runtime.batch_validation["status"] == "passed"
    for batch in (1, 16):
        arguments = list(case.inputs)
        index = 1 if backend == "jax" else 0
        arguments[index] = handle.runtime.adapter.resize_batch(arguments[index], 0, batch)
        expected_batch = case.model(*arguments)
        replayed_batch = handle.backend.run_suffix(handle.backend.run_prefix(*arguments))
        np.testing.assert_allclose(case.numpy(replayed_batch["logits"]),
                                   case.numpy(expected_batch["logits"]), atol=1e-5, rtol=1e-4)
        for key, value in replayed_batch["batch_statistics"]["layer1.0.bn1"].items():
            np.testing.assert_allclose(case.numpy(value),
                case.numpy(expected_batch["batch_statistics"]["layer1.0.bn1"][key]), atol=1e-5, rtol=1e-4)
        assert int(case.numpy(replayed_batch["batch_statistics"]["layer1.0.bn1"]["count"])) == 4 * batch
    assert case.count() == 0
    boundary = handle.backend.run_prefix(*case.inputs, training=True)
    calls = []
    replay = handle.runtime._run_suffix_unchecked

    def counted_suffix(*args, **kwargs):
        calls.append(1)
        return replay(*args, **kwargs)

    handle.runtime._run_suffix_unchecked = counted_suffix
    before = case.first()
    measurements = {}
    if backend == "jax":
        import jax
        from torchlens.split import training
        native_trainer, native_transform = training._train_suffix_jax, jax.value_and_grad
    loss, gradients = handle.backend.train_suffix(boundary, case.targets,
        loss_fn=case.loss, optimizer=case.optimizer, measurements=measurements)
    assert len(calls) == 1 and gradients
    assert np.isfinite(float(case.numpy(loss)))
    assert measurements["server_total_ms"] > 0
    actual = handle.backend.train_suffix_auxiliary
    assert set(actual) == set(expected_statistics)
    assert case.count() == 0
    prefix_gradients = handle.backend.backward_prefix(boundary, gradients,
        optimizer=getattr(case, "prefix_optimizer", case.optimizer))
    if backend == "jax":
        import jax
        assert training._train_suffix_jax is native_trainer and jax.value_and_grad is native_transform
        assert not any(isinstance(value, jax.core.Tracer) for row in actual.values() for value in row.values())
        direct_gradients = jax.grad(lambda parameters: case.loss(case.model(parameters, case.inputs[1]), case.targets))(case.model.params)
        for key, value in direct_gradients.items():
            np.testing.assert_allclose(case.numpy(prefix_gradients["inputs"][0][key]), case.numpy(value), atol=1e-5, rtol=1e-4)
    else:
        assert not np.array_equal(before, case.first())
    # Read only after both native optimizer updates: lazy statistics must still
    # describe the actual forward that preceded those mutations.
    for name, row in actual.items():
        for key, value in row.items():
            np.testing.assert_allclose(case.numpy(value), expected_statistics[name][key], atol=1e-5, rtol=1e-4)
    commit_auxiliary_state(case.model, actual, names=[])
    assert case.count() == 0
    commit_auxiliary_state(case.model, actual, names=["layer1.0.bn1"])
    assert case.count() == 1
    with pytest.raises(ValueError, match="no statistics"):
        commit_auxiliary_state(case.model, actual, names=["missing.bn"])
    assert case.count() == 1


def test_auxiliary_output_requires_formal_container_metadata():
    from splitfleet.runtime.auxiliary_state import restore_auxiliary_output, supports_auxiliary_state
    graph = SimpleNamespace(output_node_ids=("leaf",), node_by_id={"leaf": SimpleNamespace(
        output_container_path=(), output_container_paths=(), output_container_spec=None,
    )})
    with pytest.raises(RuntimeError, match="container metadata"):
        restore_auxiliary_output(graph, np.ones(3))
    assert not supports_auxiliary_state(SimpleNamespace())
    assert not supports_auxiliary_state(SimpleNamespace(apply_statistics=lambda *args: None, defer_statistics=False))


def test_auxiliary_flattened_order_is_resolved_by_captured_paths():
    from torchlens.ir.container import ContainerSpec, DictKey
    from splitfleet.runtime.auxiliary_state import restore_auxiliary_output

    row = ContainerSpec("dict", length=3, keys=("mean", "variance", "count"))
    statistics = ContainerSpec("dict", length=1, keys=("layer1.0.bn1",),
        child_specs=((DictKey("layer1.0.bn1"), row),))
    specification = ContainerSpec("dict", length=2, keys=("logits", "batch_statistics"),
        child_specs=((DictKey("batch_statistics"), statistics),))
    paths = (("batch_statistics", "layer1.0.bn1", "variance"), ("logits",),
             ("batch_statistics", "layer1.0.bn1", "count"), ("batch_statistics", "layer1.0.bn1", "mean"))
    values = (np.full(4, 2, np.float32), np.zeros((2, 3), np.float32), np.asarray(8, np.int64), np.ones(4, np.float32))
    nodes = {str(index): SimpleNamespace(output_container_path=(), output_container_paths=(path,),
                                        output_container_spec=specification)
             for index, path in enumerate(paths)}
    graph = SimpleNamespace(output_node_ids=tuple(nodes), node_by_id=nodes)
    restored = restore_auxiliary_output(graph, values)
    assert restored["logits"] is values[1]
    assert restored["batch_statistics"]["layer1.0.bn1"]["mean"] is values[3]
    assert restored["batch_statistics"]["layer1.0.bn1"]["variance"] is values[0]
    assert restored["batch_statistics"]["layer1.0.bn1"]["count"] is values[2]


def test_disabled_auxiliary_protocol_keeps_normal_native_training():
    import torch
    from splitfleet.autosplit import prepare_torchlens_runtime

    class Normal(torch.nn.Sequential):
        defer_statistics = False

        def apply_statistics(self, *args, **kwargs):
            raise AssertionError("A disabled auxiliary protocol must not receive commits")

    model = Normal(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    inputs, targets = torch.randn(2, 4), torch.randn(2, 3)
    handle = prepare_torchlens_runtime(model, inputs, boundary="50%", dynamic_batch=(1, 4))
    before = model[0].weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    boundary = handle.backend.run_prefix(inputs, training=True)
    handle.backend.train_suffix_auxiliary = {"stale": {}}
    _, gradients = handle.backend.train_suffix(boundary, targets,
        loss_fn=torch.nn.functional.mse_loss, optimizer=optimizer)
    handle.backend.backward_prefix(boundary, gradients, optimizer=optimizer)
    assert handle.backend.train_suffix_auxiliary is None
    assert not torch.equal(before, model[0].weight)
