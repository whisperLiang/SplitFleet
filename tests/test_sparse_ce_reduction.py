"""Spatial sparse CE keeps its loss and autograd when positions are flattened."""

import pytest
import torch
import torch.nn.functional as F

from splitfleet.tasks.losses import sparse_cross_entropy


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3, 4), (2, 4, 9), (2, 4, 5, 7), (1, 4, 3, 5, 7)])
def test_sparse_ce_matches_native_loss_and_gradient(dtype, shape):
    generator = torch.Generator().manual_seed(7703)
    logits = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
    labels = torch.randint(shape[1], (shape[0], *shape[2:]), generator=generator)
    labels.reshape(-1)[::3] = 255
    denominator = (labels != 255).sum().clamp_min(1)
    expected = F.cross_entropy(logits, labels, ignore_index=255, reduction="sum") / denominator
    actual = sparse_cross_entropy(logits, labels, ignore_index=255)
    expected_gradient, = torch.autograd.grad(expected, logits)
    actual_gradient, = torch.autograd.grad(actual, logits)
    tolerances = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 \
        else dict(rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, expected, **tolerances)
    torch.testing.assert_close(actual_gradient, expected_gradient, **tolerances)


def test_all_ignored_spatial_labels_return_zero_loss_and_gradient():
    logits = torch.randn((2, 3, 4, 5), requires_grad=True)
    labels = torch.full((2, 4, 5), 255)
    loss = sparse_cross_entropy(logits, labels, ignore_index=255)
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)
