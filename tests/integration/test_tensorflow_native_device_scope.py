"""Actual isolated native CPU scopes and state-derived TF runtime residency."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_tensorflow_native_scopes_state_buffers_stateless_inputs_and_threads():
    # Configure two logical CPUs before the first TensorFlow tensor. GPU use is
    # explicitly disabled in this process; native GPU proof is a retained gate.
    code = textwrap.dedent("""
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        import tensorflow as tf
        from splitfleet.backends.utils import (
            native_device_scope, tensorflow_model_device, tensorflow_tensor_device,
        )
        physical = tf.config.list_physical_devices('CPU')[0]
        tf.config.set_logical_device_configuration(physical, [
            tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration(),
        ])
        tf.config.set_soft_device_placement(False)
        with tf.device('/CPU:0'):
            weight0 = tf.Variable([[1., 2.], [3., 4.]])
            counter0 = tf.Variable(0, dtype=tf.int64, trainable=False)
            input0 = tf.ones((2, 2))
        with tf.device('/CPU:1'):
            weight1 = tf.Variable([[1., 2.], [3., 4.]])
            input1 = tf.ones((2, 2))
        # Keras value wrappers and direct ResourceVariables share one proof.
        model = SimpleNamespace(variables=[SimpleNamespace(value=weight1)],
                                device='/GPU:7', canonical_variables=lambda: {'weight':weight1})
        assert tensorflow_model_device(model) == 'cpu:1'
        assert tensorflow_tensor_device(SimpleNamespace(value=counter0)) == 'cpu'
        for model in [
                SimpleNamespace(variables=[weight1, counter0]),
                SimpleNamespace(variables=[weight1], canonical_variables=lambda: {'counter':counter0})]:
            try:
                tensorflow_model_device(model)
            except ValueError as error:
                assert 'all native model state' in str(error)
            else:
                raise AssertionError('Mixed trainable/buffer residency was silently accepted')
        assert tensorflow_model_device(lambda x:x, ((input1,), {})) == 'cpu:1'
        for inputs in [None, (input0,input1)]:
            try:
                tensorflow_model_device(lambda x:x, inputs)
            except ValueError:
                pass
            else:
                raise AssertionError('Missing/ambiguous stateless input residency was accepted')
        # A formerly valid stateless TF callable must retain its real input
        # place even though the generic facade constructor defaults to CPU0.
        from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
        stateless = lambda x: tf.nn.relu(x * 2.)
        facade = TorchLensSplitBackend(device='cpu').trace(
            stateless, (input1,), boundary='50%', trainable=True,
            batch_axes={'/args/0':0}, dynamic_batch=(1,16))
        assert str(facade.device) == 'cpu:1'
        assert facade.runtime.batch_validation['status'] == 'passed'
        boundary = facade.run_prefix(input1)
        assert facade.run_suffix(boundary).device.endswith('CPU:1')
        def operation(index):
            with native_device_scope('tf' if index==0 else 'tensorflow', f'cpu:{index}'):
                for _ in range(8):
                    out = tf.raw_ops.MatMul(a=input0, b=input0)
                    assert out.device.endswith(f'CPU:{index}')
                    assert out.numpy().tolist() == [[2.,2.],[2.,2.]]
                return out.device
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(operation,[0,1]))[1].endswith('CPU:1')
        with native_device_scope('tf','/CPU:1'):
            assert tf.identity(input0).device.endswith('CPU:1')
        try:
            with native_device_scope('tf','cuda:0'):
                tf.raw_ops.MatMul(a=input0,b=input0)
        except (tf.errors.InvalidArgumentError, RuntimeError):
            pass
        else:
            raise AssertionError('An unavailable explicit GPU silently ran on CPU')
        try:
            with native_device_scope('tf','unknown'):
                pass
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown device alias was accepted')
        print('native TF state/scopes/thread/device checks passed')
    """)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=environment,
                            capture_output=True, text=True, check=False, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native TF state/scopes/thread/device checks passed" in result.stdout
