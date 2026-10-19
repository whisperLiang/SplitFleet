"""Concurrent Paddle gradients retain private state and explicit native places."""

from concurrent.futures import ThreadPoolExecutor
import os
from threading import Barrier, Event

import numpy as np
import pytest

from splitfleet.backends.utils import native_device_scope, paddle_tensor_device
from tests.integration.test_torchlens_optional_frameworks import _run_isolated


@pytest.mark.parametrize('device', ['cpu', 'cuda:0', 'cuda:1'])
def test_same_device_training_overlaps_and_matches_serial_native_sgd(request, device):
    if _run_isolated(request, python=os.environ.get("SPLITFLEET_PADDLE_PYTHON")):
        return
    paddle = pytest.importorskip('paddle')
    if device.startswith('cuda') and (
        not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() <= int(device[-1])
    ):
        pytest.skip('Requested Paddle GPU is unavailable')
    paddle.set_device('cpu')
    initial = np.arange(12, dtype=np.float32).reshape(3, 4) / 20
    with native_device_scope('paddle', device):
        weights = [paddle.to_tensor(initial.copy(), stop_gradient=False) for _ in range(4)]
        inputs = [paddle.to_tensor(np.arange(6, dtype=np.float32).reshape(2, 3) / 10 + i)
                  for i in range(2)]
    rendezvous = Barrier(2)

    def update(index, concurrent):
        weight = weights[index if concurrent else index + 2]
        with native_device_scope('paddle', device):
            optimizer = paddle.optimizer.SGD(learning_rate=.01, parameters=[weight])
            if concurrent:
                rendezvous.wait(timeout=15)
            for _ in range(4):
                with native_device_scope('paddle', device):
                    # Factory placement checks the shared default, not only input placement.
                    assert paddle_tensor_device(paddle.ones([1])) == device
                    loss = paddle.mean(paddle.square(inputs[index] @ weight - .25))
                    loss.backward()
                    optimizer.step()
                    optimizer.clear_grad()
                if concurrent:
                    rendezvous.wait(timeout=15)
            return weight.numpy().copy()

    expected = [update(i, False) for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(update, i, True) for i in range(2)]
        actual = [f.result(timeout=40) for f in futures]
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    assert paddle.device.get_device() == 'cpu'

    # A nonconcurrent native transfer may temporarily need another place.
    with native_device_scope('paddle', device):
        with native_device_scope('paddle', 'cpu'):
            assert paddle_tensor_device(paddle.ones([1])) == 'cpu'
        assert paddle_tensor_device(paddle.ones([1])) == device
    assert paddle.device.get_device() == 'cpu'


def test_other_device_waits_for_current_operations_and_restores_default(request):
    if _run_isolated(request, python=os.environ.get("SPLITFLEET_PADDLE_PYTHON")):
        return
    paddle = pytest.importorskip('paddle')
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 2:
        pytest.skip('Two GPUs are required')
    paddle.set_device('cpu')
    entered, release, cpu_entered = Event(), Event(), Event()

    def gpu():
        with native_device_scope('paddle', 'cuda:1'):
            entered.set()
            assert release.wait(10)
            assert paddle_tensor_device(paddle.ones([1])) == 'cuda:1'

    def cpu():
        with native_device_scope('paddle', 'cpu'):
            cpu_entered.set()
            assert paddle_tensor_device(paddle.ones([1])) == 'cpu'

    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(gpu)
        assert entered.wait(10)
        waiting = pool.submit(cpu)
        try:
            assert not cpu_entered.wait(.05)
        finally:
            release.set()
        active.result(timeout=15)
        waiting.result(timeout=15)
    assert paddle.device.get_device() == 'cpu'


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('SPLITFLEET_FULL_PADDLE_CONTROL') != '1',
                    reason='Enable the dedicated full ResNet50 native control')
def test_parallel_resnet_suffixes_match_complete_native_gradients_and_state(request):
    if _run_isolated(request, python=os.environ.get("SPLITFLEET_PADDLE_PYTHON")):
        return
    paddle = pytest.importorskip('paddle')
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 2:
        pytest.skip('Two GPUs are required')
    from experiments.resnet50_quality.data import load_cifar
    from tests.integration.resnet50_reference.array_helpers import stage_boundaries
    from experiments.resnet50_quality.model_contract import canonical_state
    from tests.integration.resnet50_reference.native import (
        build_model, configure_backend, export_state, native_batch, numpy_value,
        optimizer, train_step,
    )
    from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
    from splitfleet.runtime.paddle_device_scope import paddle_execution_scope
    from splitfleet.tasks.losses import sparse_cross_entropy

    configure_backend('paddle')
    device = 'cuda:1'
    data = load_cifar('data/cifar-10-batches-py')
    populations = []
    for index, stage in enumerate(('stage1', 'stage3')):
        initial = canonical_state(233 + index)
        ids = data.train_indices[index * 20:(index + 1) * 20]
        images, labels = native_batch('paddle', data.images(ids), data.train_labels[ids], device)
        reference = build_model('paddle', initial, device)
        expected = [train_step('paddle', reference, images[a:b], labels[a:b], capture=True)
                    for a, b in ((0, 16), (16, 20))]
        expected_state = export_state('paddle', reference)
        del reference
        model = build_model('paddle', initial, device)
        model.train()
        backend = TorchLensSplitBackend(device=device).trace(
            model, images[:16], batch_axes={'/args/0': 0}, dynamic_batch=(1, 16), trainable=True)
        backend.repartition(stage_boundaries(backend.make_handle(), 'paddle')[stage]['boundary'])
        populations.append((backend, initial, images, labels, expected, expected_state))
    del data

    def check(actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    class Recorder:
        def __init__(self, model):
            self.model = model
            self.native = optimizer('paddle', model, .05)
            self.gradients = {}

        def __getattr__(self, name):
            return getattr(self.native, name)

        def step(self):
            for name, value in self.model.named_parameters():
                if value.grad is not None:
                    gradient = numpy_value(value.grad)
                    self.gradients[name] = self.gradients.get(name, 0) + gradient
            self.native.step()

    rendezvous = Barrier(2)

    def train(population, concurrent):
        backend, initial, images, labels, expected, expected_state = population
        with paddle_execution_scope(), native_device_scope('paddle', device):
            backend.model.set_state_dict({name: paddle.to_tensor(value) for name, value in initial.items()})
            opt = Recorder(backend.model)
            for step, (start, stop) in enumerate(((0, 16), (16, 20))):
                opt.gradients.clear()
                boundary = backend.run_prefix(images[start:stop], training=True)
                if concurrent:
                    # Both real private native graphs must reach suffix execution.
                    rendezvous.wait(timeout=30)
                loss, gradients = backend.train_suffix(boundary, labels[start:stop],
                    loss_fn=lambda output, target: sparse_cross_entropy(output['logits'], target),
                    optimizer=opt)
                backend.backward_prefix(boundary, gradients, optimizer=opt)
                backend.model.apply_statistics(backend.train_suffix_auxiliary)
                check(numpy_value(loss), expected[step]['loss'])
                assert set(opt.gradients) == set(expected[step]['gradients'])
                for name, value in opt.gradients.items():
                    check(value, expected[step]['gradients'][name])
            actual = export_state('paddle', backend.model)
            for name, value in actual.items():
                check(value, expected_state[name])
            assert all(int(actual[name]) == 2 for name in actual if name.endswith('num_batches_tracked'))

    for population in populations:
        train(population, False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(train, population, True) for population in populations]
        for future in futures:
            future.result(timeout=180)
