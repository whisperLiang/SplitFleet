"""Bounded native Paddle checks, isolated from other optional framework imports."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
REAL_INPUT = """
import pickle
import tarfile
from PIL import Image
with tarfile.open('data/cifar-10-python.tar.gz') as archive:
    with archive.extractfile('cifar-10-batches-py/data_batch_1') as source:
        batch = pickle.load(source, encoding='bytes')
image = batch[b'data'][0].reshape(3, 32, 32).transpose(1, 2, 0)
image = Image.fromarray(image).resize((64, 64), resample=Image.Resampling.BILINEAR)
inputs = paddle.to_tensor(np.asarray(image).transpose(2, 0, 1)[None].astype('float32') / 255)
targets = paddle.to_tensor([batch[b'labels'][0]], dtype='int64')
"""


def run_paddle(code, *, real_input=False, timeout=90):
    if importlib.util.find_spec("paddle") is None:
        pytest.skip("Paddle is not installed")
    if real_input and not (ROOT / "data/cifar-10-python.tar.gz").is_file():
        pytest.skip("The bounded real-input test needs the existing CIFAR-10 archive")
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    prelude = """
import importlib.abc
import sys
class NoTensorFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tensorflow' or fullname.startswith('tensorflow.'):
            raise ModuleNotFoundError('TensorFlow disabled in Paddle subprocess', name=fullname)
sys.meta_path.insert(0, NoTensorFlow())
import numpy as np
import paddle
from experiments.resnet50_quality.model_contract import (
    SCHEMA, NAMES, TRAINABLE_PARAMETERS, canonical_state, state_hash,
)
from tests.integration.resnet50_reference.paddle_model import (
    CanonicalBatchNorm2D, build_model, export_state,
)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(prelude) + (REAL_INPUT if real_input else "")
         + textwrap.dedent(code) + "\nassert 'tensorflow' not in sys.modules\n"],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_canonical_state_full_architecture_fp32_and_independent_clone():
    run_paddle("""
        import copy
        paddle.set_default_dtype('float64')
        state = canonical_state(233)
        state['bn1.num_batches_tracked'] = np.asarray(11, dtype='int64')
        model = build_model(state)
        snapshot = export_state(model)
        assert tuple(snapshot) == NAMES
        assert state_hash(snapshot) == state_hash(state)
        assert sum(int(np.prod(p.shape)) for p in model.parameters() if not p.stop_gradient) == TRAINABLE_PARAMETERS
        assert tuple(len(getattr(model, f'layer{i}')) for i in range(1, 5)) == (3, 4, 6, 3)
        assert model.conv1._stride == [2, 2] and model.conv1._padding == 3
        assert model.maxpool.padding == 1 and model.maxpool.stride == 2
        for stage in range(2, 5):
            first = getattr(model, f'layer{stage}')[0]
            assert first.conv1._stride == [1, 1] and first.conv2._stride == [2, 2]
        for layer in model.sublayers():
            if isinstance(layer, paddle.nn.Conv2D):
                assert layer.bias is None
        for row in SCHEMA:
            tensor = model.state_dict()[row.name]
            assert tuple(tensor.shape) == row.shape
            assert tensor.stop_gradient == (not row.trainable)
            assert tensor.place.is_cpu_place()
        model.defer_statistics = True
        clone = copy.deepcopy(model)
        assert clone.defer_statistics
        clone.bn1.running_mean.set_value(paddle.ones([64], dtype='float32'))
        clone.bn1.weight.set_value(paddle.full([64], 2.0, dtype='float32'))
        assert np.all(model.bn1.running_mean.numpy() == 0)
        assert np.all(model.bn1.weight.numpy() == 1)
        snapshot['conv1.weight'].fill(0)
        assert np.any(model.conv1.weight.numpy() != 0)
        state['conv1.weight'].fill(0)
        assert np.any(model.conv1.weight.numpy() != 0)
        assert paddle.get_default_dtype() == 'float64'
    """)


def test_batch_norm_population_output_unbiased_running_variance_and_eval():
    run_paddle("""
        layer = CanonicalBatchNorm2D(2)
        layer.weight.set_value(paddle.to_tensor([1.5, -.5], dtype='float32'))
        layer.bias.set_value(paddle.to_tensor([.25, -.75], dtype='float32'))
        layer.running_mean.set_value(paddle.to_tensor([.4, -.8], dtype='float32'))
        layer.running_var.set_value(paddle.to_tensor([1.8, .3], dtype='float32'))
        layer.num_batches_tracked.set_value(paddle.to_tensor(np.asarray(4, dtype='int64')))
        values = np.asarray([[[[0., 2.]], [[1., 5.]]], [[[4., 6.]], [[3., 7.]]]], dtype='float32')
        inputs = paddle.to_tensor(values, stop_gradient=False)
        mean = values.mean(axis=(0, 2, 3))
        variance = values.var(axis=(0, 2, 3))
        expected = (values - mean[None, :, None, None]) / np.sqrt(variance[None, :, None, None] + 1e-5)
        expected = expected * np.asarray([1.5, -.5], dtype='float32')[None, :, None, None] + np.asarray([.25, -.75], dtype='float32')[None, :, None, None]
        output = layer(inputs)
        np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(layer.running_mean.numpy(), .9 * np.asarray([.4, -.8]) + .1 * mean, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(layer.running_var.numpy(), .9 * np.asarray([1.8, .3]) + .1 * variance * (4 / 3), rtol=1e-6, atol=1e-7)
        assert layer.num_batches_tracked.numpy().dtype == np.dtype('int64')
        assert layer.num_batches_tracked.item() == 5
        output.square().mean().backward()
        for parameter in (layer.weight, layer.bias):
            assert parameter.grad is not None and np.any(parameter.grad.numpy() != 0)
            assert np.isfinite(parameter.grad.numpy()).all()
        assert np.isfinite(inputs.grad.numpy()).all()
        assert layer.num_batches_tracked.item() == 5
        layer.eval()
        frozen = {k: v.numpy().copy() for k, v in layer.state_dict().items()}
        output = layer(inputs)
        expected = (values - frozen['running_mean'][None, :, None, None]) / np.sqrt(frozen['running_var'][None, :, None, None] + 1e-5)
        expected = expected * frozen['weight'][None, :, None, None] + frozen['bias'][None, :, None, None]
        np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-6)
        for name, value in layer.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), frozen[name])
        layer.train()
        try:
            layer(paddle.zeros([1, 2, 1, 1], dtype='float32'))
        except ValueError:
            pass
        else:
            raise AssertionError('Singleton BatchNorm training must fail')
        assert layer.num_batches_tracked.item() == 5
    """)


@pytest.mark.integration
def test_real_batch_deferred_statistics_match_native_gradients_and_sgd():
    run_paddle("""
        state = canonical_state(233)
        direct, deferred = build_model(state), build_model(state)
        deferred.defer_statistics = True
        shapes = {}
        hooks = []
        for stage in (1, 2, 3):
            hooks.append(getattr(direct, f'layer{stage}').register_forward_post_hook(
                lambda layer, arguments, output, stage=stage: shapes.__setitem__(stage, tuple(output.shape))
            ))
        optimizers = [paddle.optimizer.SGD(learning_rate=.05, parameters=m.parameters()) for m in (direct, deferred)]
        logits = direct(inputs)
        for hook in hooks:
            hook.remove()
        output = deferred(inputs)
        stats = output['batch_statistics']
        assert shapes == {1: (1, 256, 16, 16), 2: (1, 512, 8, 8), 3: (1, 1024, 4, 4)}
        assert tuple(logits.shape) == (1, 10) and len(stats) == 53
        np.testing.assert_allclose(output['logits'].numpy(), logits.numpy(), rtol=1e-6, atol=1e-6)
        assert stats['bn1']['count'].dtype == paddle.int64
        assert stats['bn1']['count'].item() == 1024
        assert stats['layer4.2.bn3']['count'].item() == 4
        assert state_hash(export_state(deferred)) == state_hash(state)
        for model, prediction in ((direct, logits), (deferred, output['logits'])):
            loss = paddle.nn.functional.cross_entropy(prediction, targets)
            assert np.isfinite(loss.item())
            loss.backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert np.isfinite(parameter.grad.numpy()).all(), name
            for parameter in (model.conv1.weight, model.bn1.weight, model.bn1.bias, model.fc.weight):
                assert np.any(parameter.grad.numpy() != 0)
        assert deferred.bn1.num_batches_tracked.item() == 0
        for name, parameter in direct.named_parameters():
            other = dict(deferred.named_parameters())[name]
            np.testing.assert_allclose(parameter.grad.numpy(), other.grad.numpy(), rtol=1e-4, atol=1e-5, err_msg=name)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.clear_grad()
        deferred.apply_statistics(stats, names=['bn1'])
        assert deferred.bn1.num_batches_tracked.item() == 1
        assert deferred.layer1[0].bn1.num_batches_tracked.item() == 0
        frozen = state_hash(export_state(deferred))
        try:
            deferred.apply_statistics(stats, names=['invalid.bn'])
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown statistics must fail')
        assert state_hash(export_state(deferred)) == frozen
        deferred.apply_statistics(stats, names=set(stats) - {'bn1'})
        direct_state, deferred_state = export_state(direct), export_state(deferred)
        for name in NAMES:
            np.testing.assert_allclose(deferred_state[name], direct_state[name], rtol=1e-4, atol=1e-5, err_msg=name)
        for name in NAMES:
            if name.endswith('num_batches_tracked'):
                assert deferred_state[name].item() == 1
        fresh = deferred(paddle.tile(inputs, [2, 1, 1, 1]))
        assert fresh['batch_statistics']['bn1']['count'].item() == 2048
        assert fresh['batch_statistics']['layer4.2.bn3']['count'].item() == 8
        assert deferred.bn1.num_batches_tracked.item() == 1
        deferred.eval()
        frozen = state_hash(export_state(deferred))
        prediction = deferred(inputs)
        assert prediction['batch_statistics'] == {}
        assert np.isfinite(prediction['logits'].numpy()).all()
        assert state_hash(export_state(deferred)) == frozen
    """, real_input=True)


def test_invalid_state_input_and_explicit_device_errors():
    run_paddle("""
        state = canonical_state(233)
        for device in ('cuda:-1', 'cuda:bad', 'gpu:0', 'invalid'):
            try:
                build_model(state, device=device)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid device must fail explicitly')
        paddle.is_compiled_with_cuda = lambda: False
        try:
            build_model(state, device='cuda:0')
        except RuntimeError as error:
            assert 'CUDA-enabled' in str(error)
        else:
            raise AssertionError('CPU Paddle must not fall back from requested CUDA')
        paddle.is_compiled_with_cuda = lambda: True
        paddle.device.cuda.device_count = lambda: 0
        try:
            build_model(state, device='cuda:0')
        except RuntimeError as error:
            assert 'no CUDA device' in str(error)
        else:
            raise AssertionError('A missing CUDA device must fail')
        paddle.device.cuda.device_count = lambda: 1
        try:
            build_model(state, device='cuda:1')
        except RuntimeError as error:
            assert 'index 1' in str(error)
        else:
            raise AssertionError('A missing second GPU must fail without falling back')
        invalid = dict(state)
        invalid['fc.weight'] = np.zeros([2048, 10], dtype='float32')
        try:
            build_model(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid canonical state shape must fail')
        model = build_model(state)
        for value in (paddle.zeros([1, 3, 32, 32], dtype='float32'), paddle.zeros([1, 3, 64, 64], dtype='float64')):
            try:
                model(value)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError('Invalid model input must fail')
    """)


@pytest.mark.integration
def test_deferred_native_torchlens_stage1_replay_exposes_batch_statistics():
    run_paddle("""
        import torchlens as tl
        from torchlens.split import SplitRequest, SplitFeatures, after, prepare
        from torchlens.split.graph import split_graph_from_trace
        state = canonical_state(233)
        model = build_model(state)
        model.defer_statistics = True
        trace = tl.trace(model, (inputs,), backend='paddle')
        graph = split_graph_from_trace(trace)
        target = [node.label for node in graph.nodes if node.module_path == 'layer1.2.relu'][-1]
        runtime = prepare(model, (inputs,), SplitRequest(
            point=after(target), backend='paddle',
            features=SplitFeatures(training=True, batch_axes={}),
        ))
        # This Paddle adapter flattens model output containers. The experiment
        # hook must restore native auxiliary leaves using their owning BN path.
        leaves = runtime.replay(inputs)
        output_nodes = [runtime.trace_graph.node_by_id[name] for name in runtime.trace_graph.output_node_ids]
        assert isinstance(leaves, tuple) and len(leaves) == 53 * 3 + 1
        statistics_leaves = {}
        for node, value in zip(output_nodes, leaves, strict=True):
            if node.module_path == 'fc':
                logits = value
            else:
                statistics_leaves.setdefault(node.module_path, []).append(value)
        statistics = {
            name: dict(zip(('mean', 'variance', 'count'), parts, strict=True))
            for name, parts in statistics_leaves.items()
        }
        expected = model(inputs)
        np.testing.assert_allclose(logits.numpy(), expected['logits'].numpy(), rtol=1e-4, atol=1e-5)
        assert statistics.keys() == expected['batch_statistics'].keys()
        for name, parts in statistics.items():
            for field, value in parts.items():
                np.testing.assert_allclose(value.numpy(), expected['batch_statistics'][name][field].numpy(), rtol=1e-4, atol=1e-5)
        assert state_hash(export_state(model)) == state_hash(state)
        model.apply_statistics(statistics)
        for name, value in export_state(model).items():
            if name.endswith('num_batches_tracked'):
                assert value.item() == 1
    """, real_input=True)
