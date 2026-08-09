"""Heavy ResNet-18 split-federated correctness validation for every backend node."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import pytest


RUN_ENV = "SPLITFLEET_RUN_RESNET18_ALL_NODES"
CHILD_ENV = "SPLITFLEET_RESNET18_ALL_NODES_CHILD"
THREADS_ENV = "SPLITFLEET_RESNET18_THREADS"


def _limit_child_threads(env: dict[str, str]) -> None:
    """Keep exhaustive native-backend tests from saturating the host."""
    threads = str(max(1, int(env.get(THREADS_ENV, "1"))))
    env.update({
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "TF_NUM_INTRAOP_THREADS": threads,
        "TF_NUM_INTEROP_THREADS": "1",
    })


def _run_isolated(request, backend: str) -> bool:
    if os.environ.get(CHILD_ENV) == "1":
        return False
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
    _limit_child_threads(env)
    if backend == "tf":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if backend == "tinygrad":
        env["DEVICE"] = "CPU"
        env["DEBUG"] = "0"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", request.node.nodeid, "-q", "-s"],
        cwd=str(request.config.rootpath),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, end="")
    assert result.returncode == 0, result.stdout
    return True


def _build_torch_resnet18():
    import torch
    from torchvision.models import resnet18

    torch.manual_seed(1801)
    model = resnet18(weights=None, num_classes=10)
    inputs = torch.randn(2, 3, 32, 32)
    targets = (torch.zeros(2, 10), torch.ones(2, 10))
    return model, inputs, targets, torch.nn.MSELoss()


def _build_tf_resnet18():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(1802)

    def block(x, filters: int, stride: int, name: str):
        shortcut = x
        x = tf.keras.layers.Conv2D(
            filters, 3, strides=stride, padding="same", use_bias=False,
            name=f"{name}_conv1",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv2",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
        if stride != 1 or int(shortcut.shape[-1]) != filters:
            shortcut = tf.keras.layers.Conv2D(
                filters, 1, strides=stride, use_bias=False, name=f"{name}_downsample_conv",
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(
                name=f"{name}_downsample_bn",
            )(shortcut)
        x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
        return tf.keras.layers.ReLU(name=f"{name}_relu2")(x)

    image = tf.keras.Input((32, 32, 3), name="image")
    x = tf.keras.layers.Conv2D(
        64, 7, strides=2, padding="same", use_bias=False, name="stem_conv",
    )(image)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2, padding="same", name="stem_pool")(x)
    for stage, filters in enumerate((64, 128, 256, 512), start=1):
        for block_index in range(2):
            stride = 2 if stage > 1 and block_index == 0 else 1
            x = block(x, filters, stride, f"stage{stage}_block{block_index + 1}")
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    logits = tf.keras.layers.Dense(10, name="classifier")(x)
    model = tf.keras.Model(image, logits, name="resnet18")
    inputs = tf.random.normal((2, 32, 32, 3))
    targets = (tf.zeros((2, 10)), tf.ones((2, 10)))
    return model, inputs, targets, tf.keras.losses.MeanSquaredError()


def _build_jax_resnet18():
    import jax
    import jax.numpy as jnp
    from jax.example_libraries import stax

    def conv(filters: int, kernel: int = 3, stride: int = 1):
        return stax.Conv(
            filters,
            (kernel, kernel),
            strides=(stride, stride),
            padding="SAME",
        )

    def block(filters: int, stride: int, projection: bool):
        main = stax.serial(conv(filters, stride=stride), stax.Relu, conv(filters))
        shortcut = conv(filters, kernel=1, stride=stride) if projection else stax.Identity
        return stax.serial(
            stax.FanOut(2),
            stax.parallel(main, shortcut),
            stax.FanInSum,
            stax.Relu,
        )

    def global_average_pool():
        def init_fun(_rng, input_shape):
            return (input_shape[0], input_shape[-1]), ()

        def apply_fun(_params, inputs, **_kwargs):
            return jnp.mean(inputs, axis=(1, 2))

        return init_fun, apply_fun

    layers = [
        conv(64, kernel=7, stride=2),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
    ]
    in_filters = 64
    for stage, filters in enumerate((64, 128, 256, 512)):
        for block_index in range(2):
            stride = 2 if stage > 0 and block_index == 0 else 1
            projection = stride != 1 or in_filters != filters
            layers.append(block(filters, stride, projection))
            in_filters = filters
    layers.extend((global_average_pool(), stax.Dense(10)))
    init_fun, apply_fun = stax.serial(*layers)
    _, params = init_fun(jax.random.PRNGKey(1803), (-1, 32, 32, 3))

    def model(parameters, images):
        return apply_fun(parameters, images)

    inputs = jax.random.normal(jax.random.PRNGKey(1804), (2, 32, 32, 3))
    targets = (jnp.zeros((2, 10)), jnp.ones((2, 10)))
    loss_fn = lambda output, target: jnp.mean((output - target) ** 2)
    return model, (params, inputs), targets, loss_fn


def _build_paddle_resnet18():
    import paddle

    paddle.seed(1805)
    paddle.set_device("cpu")
    model = paddle.vision.models.resnet18(pretrained=False, num_classes=10)
    inputs = paddle.randn((2, 3, 32, 32))
    targets = (paddle.zeros((2, 10)), paddle.ones((2, 10)))
    return model, inputs, targets, paddle.nn.MSELoss()


def _build_tinygrad_resnet18():
    from tinygrad import Tensor, nn

    Tensor.manual_seed(1806)

    class BasicBlock:
        def __init__(self):
            self.weight1 = Tensor([0.5]).realize()
            self.weight2 = Tensor([0.25]).realize()
            self.weight1.requires_grad = True
            self.weight2.requires_grad = True

        def __call__(self, x):
            return (x + (x * self.weight1).relu() * self.weight2).relu()

    class TinygradResNet18:
        """Eighteen trainable layers in the ResNet-18 residual topology.

        tinygrad lowers even 1x1 convolutions into hundreds of UOps. Scalar
        affine residual layers retain the 1 + 8x2 + 1 depth and skip topology
        while keeping exhaustive UOp split validation executable.
        """

        def __init__(self):
            self.stem_weight = Tensor([0.75]).realize()
            self.stem_weight.requires_grad = True
            self.blocks = [BasicBlock() for _ in range(8)]
            self.classifier_weight = Tensor([1.25]).realize()
            self.classifier_weight.requires_grad = True

        def __call__(self, inputs):
            x = (inputs * self.stem_weight).relu()
            for residual_block in self.blocks:
                x = residual_block(x)
            return x * self.classifier_weight

    inputs = Tensor([[0.5], [1.0]]).realize()
    inputs.requires_grad = True
    targets = (Tensor.zeros(2, 1).realize(), Tensor.ones(2, 1).realize())
    loss_fn = lambda output, target: ((output - target) ** 2).mean()
    return TinygradResNet18(), inputs, targets, loss_fn


BUILDERS: dict[str, Callable[[], tuple[Any, Any, tuple[Any, Any], Any]]] = {
    "torch": _build_torch_resnet18,
    "tf": _build_tf_resnet18,
    "jax": _build_jax_resnet18,
    "paddle": _build_paddle_resnet18,
    "tinygrad": _build_tinygrad_resnet18,
}


def _args(inputs: Any, adapter: Any) -> tuple[Any, ...]:
    values = inputs if isinstance(inputs, tuple) else (inputs,)
    if adapter.backend_name == "jax" and adapter.has_external_params:
        return (adapter.external_params, *values[1:])
    return values


def _trainable_arrays(backend: str, model: Any, adapter: Any) -> list[np.ndarray]:
    if backend == "torch":
        return [parameter.detach().cpu().numpy().copy() for parameter in model.parameters()]
    if backend == "tf":
        return [np.asarray(variable.numpy()).copy() for variable in model.trainable_variables]
    if backend == "paddle":
        return [np.asarray(parameter.numpy()).copy() for parameter in model.parameters()]
    if backend == "jax":
        import jax

        return [np.asarray(value).copy() for value in jax.tree_util.tree_leaves(adapter.external_params)]
    return [
        adapter._to_numpy(value).copy()
        for value in adapter._state(model).values()
        if bool(getattr(value, "requires_grad", False))
    ]


def _optimizer(backend: str, model: Any, adapter: Any):
    if backend == "jax":
        return None
    optimizer = adapter.build_optimizer(model, {"name": "sgd", "lr": 1e-4})
    if backend == "tf" and hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)
    return optimizer


def _apply_jax_update(adapter: Any, prefix_result: Any, learning_rate: float = 1e-4) -> None:
    import jax

    param_grads = prefix_result["inputs"][0]
    updated = jax.tree_util.tree_map(
        lambda parameter, grad: parameter - learning_rate * grad,
        adapter.external_params,
        param_grads,
    )
    adapter.bind_external_params(updated)


def _changed(before: list[np.ndarray], after: list[np.ndarray]) -> bool:
    return any(
        not np.allclose(left, right, rtol=1e-7, atol=1e-9)
        for left, right in zip(before, after, strict=True)
    )


def _to_numpy(adapter: Any, value: Any) -> np.ndarray:
    if hasattr(adapter, "_to_numpy"):
        return np.asarray(adapter._to_numpy(value))
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value)


def _exercise_backend(backend: str) -> None:
    # Paddle's native runtime must be loaded before TensorFlow/JAX libraries
    # that TorchLens may discover while SplitFleet is imported. Reversing this
    # order can segfault inside paddle.base.core on CPU-only Linux builds.
    model, inputs, client_targets, loss_fn = BUILDERS[backend]()

    from flwr.server.strategy.aggregate import aggregate

    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.backends.utils import adapter_for
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary

    adapter = adapter_for(model, inputs)
    adapter.set_training(model, True)
    seed = prepare_torchlens_runtime(
        model,
        _args(inputs, adapter),
        boundary="50%",
        trainable=True,
        dynamic_batch=(2, 2),
        model_name=f"resnet18-{backend}",
        model_family="resnet18",
    )
    graph_nodes = [
        node
        for node in seed.runtime.trace_graph.nodes
        if str(getattr(node, "label", "") or "")
        and not bool(getattr(node, "is_input", False))
        and not bool(getattr(node, "is_output", False))
        and not bool(getattr(node, "is_buffer", False))
        and not bool(getattr(node, "is_param_source", False))
    ]
    initial_state = adapter.export_ndarrays(model)

    built = 0
    trained = 0
    terminal = 0
    non_differentiable = 0
    for node_index, node in enumerate(graph_nodes):
        if node_index % 25 == 0:
            print(
                f"RESNET18_PROGRESS backend={backend} node={node_index}/{len(graph_nodes)} "
                f"viable={built} trained={trained}",
                flush=True,
            )
        label = str(node.label)
        split = f"after:{label}"
        try:
            if backend == "tinygrad":
                model, inputs, client_targets, loss_fn = BUILDERS[backend]()
                adapter = adapter_for(model, inputs)
                adapter.set_training(model, True)
                runtime = prepare_torchlens_runtime(
                    model,
                    inputs,
                    boundary=split,
                    trainable=True,
                    dynamic_batch=(2, 2),
                    model_name="resnet18-tinygrad",
                    model_family="resnet18",
                )
            else:
                adapter.load_ndarrays(model, initial_state)
                runtime = seed.backend.repartition(split)
        except Exception as exc:
            raise AssertionError(
                f"operational ResNet-18 node {split} could not be partitioned: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        built += 1

        # Every viable candidate must replay correctly before it is trained.
        current_args = _args(inputs, adapter)
        expected = model(*current_args)
        replay_boundary = runtime.backend.run_prefix(*current_args)
        replayed = runtime.backend.run_suffix(replay_boundary)
        np.testing.assert_allclose(
            _to_numpy(adapter, replayed),
            _to_numpy(adapter, expected),
            rtol=2e-3,
            atol=2e-4,
            err_msg=split,
        )
        if runtime.plan.suffix_node_count == 0:
            terminal += 1
            continue

        base_state = adapter.export_ndarrays(model)
        local_states: list[list[np.ndarray]] = []
        for client_index, targets in enumerate(client_targets):
            if backend == "tinygrad":
                if client_index > 0:
                    model, inputs, fresh_targets, loss_fn = BUILDERS[backend]()
                    targets = fresh_targets[client_index]
                    adapter = adapter_for(model, inputs)
                    adapter.set_training(model, True)
                    runtime = prepare_torchlens_runtime(
                        model,
                        inputs,
                        boundary=split,
                        trainable=True,
                        dynamic_batch=(2, 2),
                        model_name="resnet18-tinygrad",
                        model_family="resnet18",
                    )
            else:
                adapter.load_ndarrays(model, base_state)
            before_trainable = _trainable_arrays(backend, model, adapter)
            optimizer = _optimizer(backend, model, adapter)
            current_args = _args(inputs, adapter)
            local_boundary = runtime.backend.run_prefix(*current_args, training=True)
            contract = graph_contract_for_runtime_handle(runtime)
            envelope = boundary_to_envelope(
                local_boundary,
                round_id=1,
                client_id=f"resnet18-{backend}-client-{client_index}",
                step_id=f"{node_index}-{client_index}",
                plan_id=runtime.plan.plan_id,
                split_id=contract.split_id,
                canonical_graph_hash=contract.canonical_graph_hash,
                boundary_schema_hash=contract.boundary_schema_hash,
                model_version=1,
            )
            remote_boundary = envelope_to_boundary(envelope, runtime.runtime, None)
            loss, gradients = runtime.backend.train_suffix(
                remote_boundary,
                targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            assert np.isfinite(adapter.scalar_value(loss)), split
            if not gradients:
                non_differentiable += 1
                local_states.clear()
                break
            prefix_result = runtime.backend.backward_prefix(
                local_boundary,
                boundary_grads=gradients,
                optimizer=optimizer,
            )
            if backend == "jax":
                _apply_jax_update(adapter, prefix_result)
            after_trainable = _trainable_arrays(backend, model, adapter)
            assert _changed(before_trainable, after_trainable), (
                split,
                "training changed no trainable parameter",
            )
            local_states.append(adapter.export_ndarrays(model))

        if not local_states:
            adapter.load_ndarrays(model, base_state)
            continue
        federated_state = aggregate([(state, 1) for state in local_states])
        adapter.load_ndarrays(model, federated_state)
        loaded_state = adapter.export_ndarrays(model)
        for actual, expected_state in zip(loaded_state, federated_state, strict=True):
            np.testing.assert_allclose(actual, expected_state, rtol=1e-6, atol=1e-7)
        trained += 1

    frontier_trained = 0
    if not trained:
        adapter.load_ndarrays(model, initial_state)
        runtime = seed.backend.repartition("50%")
        base_state = adapter.export_ndarrays(model)
        local_states = []
        for client_index, targets in enumerate(client_targets):
            adapter.load_ndarrays(model, base_state)
            before_trainable = _trainable_arrays(backend, model, adapter)
            optimizer = _optimizer(backend, model, adapter)
            current_args = _args(inputs, adapter)
            local_boundary = runtime.backend.run_prefix(*current_args, training=True)
            contract = graph_contract_for_runtime_handle(runtime)
            envelope = boundary_to_envelope(
                local_boundary,
                round_id=1,
                client_id=f"resnet18-{backend}-frontier-{client_index}",
                step_id=f"frontier-{client_index}",
                plan_id=runtime.plan.plan_id,
                split_id=contract.split_id,
                canonical_graph_hash=contract.canonical_graph_hash,
                boundary_schema_hash=contract.boundary_schema_hash,
                model_version=1,
            )
            remote_boundary = envelope_to_boundary(envelope, runtime.runtime, None)
            loss, gradients = runtime.backend.train_suffix(
                remote_boundary,
                targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            assert np.isfinite(adapter.scalar_value(loss))
            assert gradients, f"{backend} has no differentiable ResNet-18 frontier"
            prefix_result = runtime.backend.backward_prefix(
                local_boundary,
                boundary_grads=gradients,
                optimizer=optimizer,
            )
            if backend == "jax":
                _apply_jax_update(adapter, prefix_result)
            after_trainable = _trainable_arrays(backend, model, adapter)
            assert _changed(before_trainable, after_trainable)
            local_states.append(adapter.export_ndarrays(model))
        federated_state = aggregate([(state, 1) for state in local_states])
        adapter.load_ndarrays(model, federated_state)
        loaded_state = adapter.export_ndarrays(model)
        for actual, expected_state in zip(loaded_state, federated_state, strict=True):
            np.testing.assert_allclose(actual, expected_state, rtol=1e-6, atol=1e-7)
        frontier_trained = 1

    assert built > 0, f"ResNet-18 exposed no viable {backend} split candidates"
    assert trained > 0 or frontier_trained, (
        f"ResNet-18 exposed no trainable {backend} split frontier"
    )
    print(
        f"RESNET18_ALL_NODES backend={backend} graph_nodes={len(graph_nodes)} "
        f"viable={built} trained={trained} terminal={terminal} "
        f"non_differentiable={non_differentiable} unsupported=0 "
        f"frontier_trained={frontier_trained}"
    )


@pytest.mark.parametrize("backend", tuple(BUILDERS))
def test_resnet18_all_split_nodes_federated_training(request, backend: str) -> None:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 to run exhaustive ResNet-18 validation")
    if _run_isolated(request, backend):
        return
    _exercise_backend(backend)
