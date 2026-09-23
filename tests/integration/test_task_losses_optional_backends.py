"""Verify standard objectives and native gradients on every supported backend."""

import numpy as np
import pytest

from splitfleet.tasks import DetectionTask, ImageClassificationTask, SemanticSegmentationTask
from tests.integration.test_torchlens_optional_frameworks import _run_isolated


pytestmark = pytest.mark.integration


def _tensor(backend, value):
    if backend == "torch":
        return pytest.importorskip("torch").tensor(value)
    if backend == "tf":
        return pytest.importorskip("tensorflow").convert_to_tensor(value)
    if backend == "jax":
        return pytest.importorskip("jax.numpy").asarray(value)
    if backend == "paddle":
        return pytest.importorskip("paddle").to_tensor(value)
    return pytest.importorskip("tinygrad").Tensor(value)


def _value_and_grad(backend, values, objective):
    values = _tensor(backend, np.asarray(values, dtype=np.float32))
    if backend == "torch":
        import torch
        values.requires_grad_(True)
        loss = objective(values)
        gradient, = torch.autograd.grad(loss, values)
        return loss.detach().numpy(), gradient.detach().numpy()
    if backend == "tf":
        import tensorflow as tf
        with tf.GradientTape() as tape:
            tape.watch(values)
            loss = objective(values)
        return loss.numpy(), tape.gradient(loss, values).numpy()
    if backend == "jax":
        import jax
        loss, gradient = jax.value_and_grad(objective)(values)
        return np.asarray(loss), np.asarray(gradient)
    if backend == "paddle":
        import paddle
        values.stop_gradient = False
        loss = objective(values)
        gradient, = paddle.grad(loss, values)
        return loss.numpy(), gradient.numpy()
    values.requires_grad = True
    loss = objective(values)
    loss.backward()
    return loss.numpy(), values.grad.numpy()


def _reference_ce(logits, labels, ignore_index=-100):
    logits = np.moveaxis(logits, 1, -1)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    valid = labels != ignore_index
    safe_labels = np.where(valid, labels, 0)
    losses = -np.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
    gradients = np.exp(log_probs)
    gradients -= np.eye(logits.shape[-1], dtype=np.float32)[safe_labels]
    gradients *= valid[..., None] / valid.sum()
    return losses[valid].mean(), np.moveaxis(gradients, -1, 1)


@pytest.mark.parametrize("backend", ["torch", "tf", "jax", "paddle", "tinygrad"])
def test_standard_task_objectives_preserve_native_gradients(request, backend):
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": "", "DEV": "CPU"}):
        return
    logits = np.asarray([[2., -1., .5], [-2., 3., 1.]], dtype=np.float32)
    labels = np.asarray([0, 2], dtype=np.int64)
    task = ImageClassificationTask()
    loss, gradient = _value_and_grad(backend, logits, lambda value: task.loss(value, _tensor(backend, labels)))
    expected_loss, expected_grad = _reference_ce(logits, labels)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gradient, expected_grad, rtol=1e-5, atol=1e-6)

    pixels = np.linspace(-2., 2., 24, dtype=np.float32).reshape(2, 3, 2, 2)
    masks = np.asarray([[[0, 1], [2, 255]], [[1, 0], [255, 2]]], dtype=np.int64)
    task = SemanticSegmentationTask(aux_weight=.25)
    loss, gradient = _value_and_grad(
        backend, pixels,
        lambda value: task.loss({"out": value, "aux": value}, _tensor(backend, masks)),
    )
    expected_loss, expected_grad = _reference_ce(pixels, masks, ignore_index=255)
    np.testing.assert_allclose(loss, expected_loss * 1.25, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gradient, expected_grad * 1.25, rtol=1e-5, atol=1e-6)

    task = DetectionTask(loss_weights={"loss_box_reg": 2.})
    loss, gradient = _value_and_grad(
        backend, [2., .5],
        lambda value: task.loss({"loss_classifier": value[0:1], "loss_box_reg": value[1:2]}),
    )
    np.testing.assert_allclose(loss, 3.)
    np.testing.assert_allclose(gradient, [1., 2.])
