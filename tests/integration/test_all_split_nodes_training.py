"""Train through every TorchLens-enumerated split node for each supported backend."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest


def _isolated(request, *, env_overrides=None) -> bool:
    if os.environ.get("SPLITFLEET_ALL_NODES_CHILD") == "1":
        return False
    env = os.environ.copy()
    env["SPLITFLEET_ALL_NODES_CHILD"] = "1"
    env.update(env_overrides or {})
    result = subprocess.run(
        [sys.executable, "-m", "pytest", request.node.nodeid, "-q", "-s"],
        cwd=os.getcwd(), env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def _all_nodes(model, inputs, targets, loss_fn, backend_name: str) -> None:
    from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend, prepare_torchlens_runtime
    from splitfleet.backends.utils import adapter_for
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary

    args = inputs if isinstance(inputs, tuple) else (inputs,)
    dynamic_batch = (2, 2)
    adapter = adapter_for(model, inputs)
    enumerator = TorchLensSplitBackend(model_name=f"all-nodes-{backend_name}")
    enumerator.trace(
        model, inputs, boundary="50%", trainable=True, dynamic_batch=dynamic_batch,
    )
    candidates = enumerator.enumerate_candidates()
    assert candidates, f"TorchLens enumerated no {backend_name} split candidates"

    trained: list[str] = []
    replay_only: list[str] = []
    non_differentiable: list[str] = []
    for candidate in candidates:
        report = enumerator.validate_candidate(candidate)
        assert report["success"], (candidate.boundary, report)
        if candidate.descriptor.get("suffix_node_count") == 0:
            replay_only.append(candidate.boundary)
            continue

        runtime = prepare_torchlens_runtime(
            model, inputs, boundary=candidate.boundary, trainable=True,
            dynamic_batch=dynamic_batch,
        )
        local_boundary = runtime.backend.run_prefix(*args, training=True)
        before_state = adapter.export_ndarrays(model)
        optimizer = None
        if backend_name != "jax":
            optimizer = adapter.build_optimizer(model, {"name": "sgd", "lr": 1e-4})
            if backend_name == "tf" and hasattr(optimizer, "build"):
                optimizer.build(model.trainable_variables)
        contract = graph_contract_for_runtime_handle(runtime)
        envelope = boundary_to_envelope(
            local_boundary,
            round_id=1,
            client_id=f"all-nodes-{backend_name}",
            step_id=candidate.candidate_id,
            plan_id=runtime.plan.plan_id,
            split_id=contract.split_id,
            canonical_graph_hash=contract.canonical_graph_hash,
            boundary_schema_hash=contract.boundary_schema_hash,
            model_version=1,
        )
        remote_boundary = envelope_to_boundary(envelope, runtime.runtime, None)
        loss, gradients = runtime.backend.train_suffix(
            remote_boundary, targets, loss_fn=loss_fn, optimizer=optimizer,
        )
        assert np.isfinite(adapter.scalar_value(loss)), candidate.boundary
        if not gradients:
            non_differentiable.append(candidate.boundary)
            continue
        prefix_result = runtime.backend.backward_prefix(
            local_boundary, gradients, optimizer=optimizer,
        )
        if backend_name == "jax":
            import jax

            param_grads = prefix_result["inputs"][0]
            adapter.bind_external_params(
                jax.tree_util.tree_map(
                    lambda parameter, grad: parameter - 1e-4 * grad,
                    adapter.external_params,
                    param_grads,
                )
            )
            args = (adapter.external_params, *args[1:])
        after_state = adapter.export_ndarrays(model)
        assert any(
            not np.array_equal(before, after)
            for before, after in zip(before_state, after_state, strict=True)
        ), f"{candidate.boundary} produced gradients but updated no model parameter"
        trained.append(candidate.boundary)

    assert len(trained) + len(non_differentiable) + len(replay_only) == len(candidates)
    frontier_trained = 0
    if not trained:
        # Some native IRs (currently tinygrad UOps) expose only constant/shape
        # nodes as single-node boundaries. Exercise the differentiable frontier
        # selected by TorchLens as well, while still accounting for every node
        # above rather than silently dropping non-differentiable candidates.
        runtime = prepare_torchlens_runtime(
            model, inputs, boundary="50%", trainable=True,
            dynamic_batch=dynamic_batch,
        )
        boundary = runtime.backend.run_prefix(*args, training=True)
        before_state = adapter.export_ndarrays(model)
        optimizer = None
        if backend_name != "jax":
            optimizer = adapter.build_optimizer(model, {"name": "sgd", "lr": 1e-4})
            if backend_name == "tf" and hasattr(optimizer, "build"):
                optimizer.build(model.trainable_variables)
        loss, gradients = runtime.backend.train_suffix(
            boundary, targets, loss_fn=loss_fn, optimizer=optimizer,
        )
        assert np.isfinite(adapter.scalar_value(loss))
        assert gradients, f"{backend_name} has no differentiable split-training frontier"
        prefix_result = runtime.backend.backward_prefix(
            boundary, gradients, optimizer=optimizer,
        )
        if backend_name == "jax":
            import jax

            adapter.bind_external_params(
                jax.tree_util.tree_map(
                    lambda parameter, grad: parameter - 1e-4 * grad,
                    adapter.external_params,
                    prefix_result["inputs"][0],
                )
            )
        after_state = adapter.export_ndarrays(model)
        assert any(
            not np.array_equal(before, after)
            for before, after in zip(before_state, after_state, strict=True)
        ), f"{backend_name} frontier updated no model parameter"
        frontier_trained = 1
    print(
        f"ALL_SPLIT_NODES backend={backend_name} total={len(candidates)} "
        f"trained={len(trained)} non_differentiable={len(non_differentiable)} "
        f"terminal={len(replay_only)} frontier_trained={frontier_trained}"
    )


def test_torch_yolo_detection_all_split_nodes_train(request) -> None:
    if _isolated(request): return
    import torch
    from torch import nn

    class YoloDetectionHead(nn.Module):
        """Compact YOLO-style detector producing box, objectness, and class logits."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), nn.SiLU(),
                nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.prediction = nn.Linear(8, 8)  # xywh + objectness + 3 classes

        def forward(self, images):
            return self.prediction(self.features(images))

    model = YoloDetectionHead()
    _all_nodes(model, torch.randn(2, 3, 16, 16), torch.randn(2, 8), nn.MSELoss(), "torch")


def test_tensorflow_fcn_segmentation_all_split_nodes_train(request) -> None:
    if _isolated(request, env_overrides={"CUDA_VISIBLE_DEVICES": ""}): return
    import tensorflow as tf

    # A compact fully convolutional semantic-segmentation network. Keeping the
    # output stride at two avoids ResizeBilinear, which the TorchLens TF adapter
    # explicitly reports as non-replayable.
    model = tf.keras.Sequential([
        tf.keras.layers.Input((16, 16, 3)),
        tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(12, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(4, 1),
    ], name="compact_fcn_segmenter")
    inputs = tf.random.normal((2, 16, 16, 3))
    targets = tf.random.normal((2, 8, 8, 4))
    _all_nodes(model, inputs, targets, tf.keras.losses.MeanSquaredError(), "tf")


def test_jax_retinanet_detection_all_split_nodes_train(request) -> None:
    if _isolated(request): return
    import jax.numpy as jnp
    params = {
        "stem": jnp.arange(24, dtype=jnp.float32).reshape(3, 8) / 40,
        "box": jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 50,
        "class": jnp.arange(24, dtype=jnp.float32).reshape(8, 3) / 50,
    }

    def model(params, images):
        """RetinaNet-style dense box/class heads over a shared feature map."""
        features = jnp.maximum(images @ params["stem"], 0)
        box_regression = features @ params["box"]
        class_logits = features @ params["class"]
        return jnp.concatenate((box_regression, class_logits), axis=-1)

    loss_fn = lambda output, target: jnp.mean((output - target) ** 2)
    images = jnp.ones((2, 4, 4, 3), dtype=jnp.float32)
    targets = jnp.zeros((2, 4, 4, 7), dtype=jnp.float32)
    _all_nodes(model, (params, images), targets, loss_fn, "jax")


def test_paddle_lenet_ocr_all_split_nodes_train(request) -> None:
    if _isolated(request): return
    import paddle
    model = paddle.vision.models.LeNet(num_classes=10)
    inputs = paddle.randn((2, 1, 28, 28))
    targets = paddle.randint(0, 10, shape=(2,), dtype="int64")
    _all_nodes(model, inputs, targets, paddle.nn.CrossEntropyLoss(), "paddle")


def test_tinygrad_fcn_segmentation_all_split_nodes_train(request) -> None:
    if _isolated(request, env_overrides={"DEVICE": "CPU", "DEBUG": "0"}): return
    from tinygrad import Tensor

    class FCNSegmenter:
        def __init__(self):
            self.scale = Tensor([1.5]).realize()
            self.bias = Tensor([-0.25]).realize()
            self.scale.requires_grad = True
            self.bias.requires_grad = True

        def __call__(self, images):
            # A trainable foreground-mask calibration head. This is deliberately
            # pointwise: tinygrad lowers even a 1x1 convolution into hundreds of
            # UOps, making prepare-on-every-boundary validation impractical.
            return (images * self.scale + self.bias).relu().sigmoid()

    loss_fn = lambda output, target: ((output - target) ** 2).mean()
    inputs = Tensor.randn(2, 1, 8, 8).realize()
    inputs.requires_grad = True
    targets = Tensor.randn(2, 1, 8, 8).realize()
    _all_nodes(FCNSegmenter(), inputs, targets, loss_fn, "tinygrad")
