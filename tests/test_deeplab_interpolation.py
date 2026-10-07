"""Audit deterministic bilinear math against the original native CPU kernel."""

import pytest
import torch
import torch.nn.functional as F

from experiments.deeplab_interpolation import bilinear_resize


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,size", [
    ((2, 3, 5, 7), (11, 13)),
    ((2, 3, 11, 13), (5, 7)),
    ((2, 3, 1, 1), (9, 7)),
    ((1, 2, 1, 5), (6, 3)),
    ((1, 2, 5, 1), (3, 6)),
    ((1, 2, 5, 7), (5, 7)),
])
def test_resize_output_and_gradient_match_native_bilinear(dtype, shape, size):
    generator = torch.Generator().manual_seed(6189)
    image = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
    expected = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    actual = bilinear_resize(image, size)
    weights = torch.randn(actual.shape, generator=generator, dtype=dtype)
    expected_gradient, = torch.autograd.grad(expected, image, weights)
    actual_gradient, = torch.autograd.grad(actual, image, weights)
    tolerances = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 \
        else dict(rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, expected, **tolerances)
    torch.testing.assert_close(actual_gradient, expected_gradient, **tolerances)
