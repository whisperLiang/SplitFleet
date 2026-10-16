"""Native fused and explicitly materialized FP32 operations must agree."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.parametrize('device', ['CPU', 'CUDA:0'])
def test_native_compiler_uses_separately_rounded_multiply_add(device):
    if importlib.util.find_spec('tinygrad') is None:
        pytest.skip('tinygrad is not installed')
    if device.startswith('CUDA'):
        import torch
        if not torch.cuda.is_available():
            pytest.skip('A real CUDA GPU is required')
    code = '''
import sys
import numpy as np
from splitfleet.runtime.tinygrad_precision import configure_device
from tinygrad import Tensor
device = sys.argv[1]
configure_device(device)
configure_device(device)
left = np.full((4096,), np.nextafter(np.float32(1), np.float32(2)), np.float32)
right = np.full((4096,), np.nextafter(np.float32(1), np.float32(0)), np.float32)
a, b = Tensor(left, device=device).realize(), Tensor(right, device=device).realize()
c = Tensor(np.full((4096,), -1, np.float32), device=device).realize()
whole = (a*b+c).numpy()
materialized = ((a*b).realize()+c).numpy()
np.testing.assert_array_equal(whole, materialized)
np.testing.assert_array_equal(whole, np.zeros_like(whole))
print('native FP32 rounding verified:', device)
'''
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       CPU_COUNT='1', THREADS='1', DEV=device.split(':')[0])
    result = subprocess.run([sys.executable, '-c', code, device],
        cwd=Path(__file__).resolve().parents[2], env=environment,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'native FP32 rounding verified:' in result.stdout
