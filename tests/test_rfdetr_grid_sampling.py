"""Compare deformable-attention sampling with the native reference kernel."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from experiments.rfdetr_grid_sampling import bilinear_grid_sample, integer_position_embedding


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3, 5, 7), (1, 2, 1, 4), (1, 2, 4, 1), (1, 1, 1, 1)])
def test_bilinear_sampling_matches_outputs_and_both_gradients(dtype, shape):
    generator = torch.Generator().manual_seed(1403)
    features = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
    # Repeated coordinates exercise gradient accumulation. Outside coordinates
    # and degenerate spatial axes exercise zero padding and clamped indices.
    coordinates = torch.tensor([
        [-1.4, -1.3], [-1.0, -1.0], [-0.37, 0.29], [-0.37, 0.29],
        [1.0, 1.0], [1.3, -0.83], [2.1, 2.2], [0.17, -0.63],
    ], dtype=dtype).view(1, 2, 4, 2).repeat(shape[0], 1, 1, 1).requires_grad_()
    expected = F.grid_sample(features, coordinates, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
    actual = bilinear_grid_sample(features, coordinates)
    weights = torch.randn(actual.shape, generator=generator, dtype=dtype)
    expected_gradients = torch.autograd.grad(expected, (features, coordinates), weights)
    actual_gradients = torch.autograd.grad(actual, (features, coordinates), weights)
    tolerances = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 \
        else dict(rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, expected, **tolerances)
    for result, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(result, reference, **tolerances)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("align_dim_orders", [True, False])
def test_integer_mask_counts_preserve_native_position_encoding(normalize, align_dim_orders):
    precision = torch.get_float32_matmul_precision()
    try:
        from rfdetr.models.position_encoding import PositionEmbeddingSine
        from rfdetr.utilities.tensors import NestedTensor
    finally:
        torch.set_float32_matmul_precision(precision)

    generator = torch.Generator().manual_seed(7109)
    features = torch.randn((2, 32, 24, 24), generator=generator)
    mask = torch.rand((2, 24, 24), generator=generator) < 0.3
    module = PositionEmbeddingSine(16, normalize=normalize)
    original = getattr(PositionEmbeddingSine, "_native_forward_for_math_audit", PositionEmbeddingSine.forward)
    batch = NestedTensor(features, mask)
    expected = original(module, batch, align_dim_orders=align_dim_orders)
    actual = integer_position_embedding(module, batch, align_dim_orders=align_dim_orders)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_full_rfdetr_builder_preserves_callers_math_policy():
    # Use a fresh interpreter: RF-DETR changes matmul precision at first import.
    code = """
import torch
from experiments.physical_multitask import _configure_primary_math
from experiments.rfdetr_nano_physical import RFDETRNanoDetector
_configure_primary_math({'model_id': 'rfdetr_nano'})
model = RFDETRNanoDetector(pretrain_weights=None)
assert torch.get_float32_matmul_precision() == 'highest'
assert not torch.backends.cuda.matmul.allow_tf32
assert not torch.backends.cudnn.allow_tf32
assert torch.are_deterministic_algorithms_enabled()
assert not torch.is_deterministic_algorithms_warn_only_enabled()
"""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
