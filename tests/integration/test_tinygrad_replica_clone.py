"""Real Tinygrad gradients and updates must remain private after graph reuse."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _native_control(device):
    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.optim import SGD
    from splitfleet.runtime.tinygrad_precision import configure_device
    from splitfleet.autosplit.tinygrad_clone import clone_tinygrad_runtime_handle
    from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.backends.utils import adapter_for

    configure_device(device)

    class Projection:
        def __init__(self, rows, columns):
            self.weight = Tensor(np.arange(rows*columns, dtype=np.float32).reshape(rows, columns)/31,
                                 device=device).realize().is_param_()
            self.bias = Tensor(np.zeros(columns, np.float32), device=device).realize().is_param_()

        def __call__(self, value):
            return value.matmul(self.weight) + self.bias

    class Model:
        training, defer_statistics = True, True

        def __init__(self):
            self.device = device
            self.first, self.last = Projection(4, 5), Projection(5, 3)

        def __call__(self, value):
            hidden = self.first(value).relu().contiguous()
            observations = value[:, :1]
            count = ((observations != 0).cast(dtypes.int64)
                     + (observations == 0).cast(dtypes.int64)).sum()
            denominator = count.cast(dtypes.float32)
            mean = hidden.sum(0) / denominator
            variance = (hidden-mean).square().sum(0) / denominator
            return {'logits': self.last(hidden),
                    'batch_statistics': {'projection': {'mean': mean.detach(),
                        'variance': variance.detach(), 'count': count}}}

        def state_tensors(self):
            return {name+'.'+part: getattr(getattr(self, name), part)
                    for name in ('first', 'last') for part in ('weight', 'bias')}

        def apply_statistics(self, statistics, names=None):
            assert set(statistics) == {'projection'}

        def _splitfleet_replica_contract(self):
            return type(self), self.device, self.training

        def _splitfleet_initial_capture_boundary(self, graph):
            return [node.canonical_id for node in graph.nodes if node.module_path == 'first'
                    and node.op_type == 'add'][-1]

    source, target, reference = Model(), Model(), Model()
    for tensor in target.state_tensors().values():
        tensor.assign(tensor + .2).realize()
    for name, tensor in reference.state_tensors().items():
        tensor.assign(target.state_tensors()[name]).realize()
    original = {name: value.numpy().copy() for name, value in source.state_tensors().items()}
    values = Tensor(np.arange(64, dtype=np.float32).reshape(16, 4)/29, device=device).realize()
    labels = Tensor(np.arange(48, dtype=np.float32).reshape(16, 3)/23, device=device).realize()
    facade = TorchLensSplitBackend(device='cpu' if device == 'CPU' else 'cuda:0')
    handle = facade.trace(source, (values,), boundary='auto',
                          batch_axes={'/args/0': 0}, dynamic_batch=(1,16)).make_handle()
    assert handle.runtime.batch_validation['status'] == 'passed', handle.runtime.batch_validation
    projection = [node for node in handle.runtime.trace_graph.nodes if node.module_path == 'first'
                  and node.op_type == 'add'][-1]
    assert projection.canonical_id in str(handle.runtime.request.boundary)
    handle = handle.backend.repartition('after:' + projection.canonical_id)
    frozen = Model()
    frozen.first.weight.is_param_(False)
    assert clone_tinygrad_runtime_handle(handle, frozen) is None
    clone = clone_tinygrad_runtime_handle(handle, target)
    assert clone is not None and clone.plan.metadata['_reused_capture']
    for batch in (16, 8, 4):
        arguments = values[:batch]
        actual = clone.backend.run_suffix(clone.backend.run_prefix(arguments))
        expected = target(arguments)
        np.testing.assert_allclose(actual['logits'].numpy(), expected['logits'].numpy(), atol=1e-5, rtol=1e-4)
    schema = adapter_for(target, (values,)).state_manifest(target).schema_hash
    owned = state_ownership(clone, schema)
    owners = dict(zip(owned['names'], owned['owners'], strict=True))
    first = [value for name, value in target.state_tensors().items() if owners[name] == 'prefix']
    last = [value for name, value in target.state_tensors().items() if owners[name] == 'suffix']
    assert first and last

    def loss_fn(output, truth):
        return ((output['logits']-truth)**2).mean()

    with Tensor.train():
        native = SGD(list(reference.state_tensors().values()), lr=.01)
        prefix, suffix = SGD(first, lr=.01), SGD(last, lr=.01)
        # Return to a warmed batch after both model state and input values
        # change. Reusable shape metadata must never retain old activations.
        for step, batch in enumerate((16, 8, 16)):
            arguments, truth = values[:batch] + step / 100, labels[:batch]
            native.zero_grad()
            loss_fn(reference(arguments), truth).backward()
            Tensor.realize(*(value.grad for value in reference.state_tensors().values()))
            native.step()
            prefix.zero_grad()
            suffix.zero_grad()
            boundary = clone.backend.run_prefix(arguments, training=True)
            _, roots = clone.backend.train_suffix(boundary, truth, loss_fn=loss_fn, optimizer=suffix)
            clone.backend.backward_prefix(boundary, roots, optimizer=prefix)
            for name, value in target.state_tensors().items():
                np.testing.assert_allclose(value.numpy(), reference.state_tensors()[name].numpy(), atol=1e-5, rtol=1e-4)
                np.testing.assert_array_equal(source.state_tensors()[name].numpy(), original[name])
    print('native Tinygrad replica SGD verified:', device)


@pytest.mark.integration
@pytest.mark.parametrize('device', ['CPU', 'CUDA:0'])
def test_native_replica_gradients_and_state_are_independent(device):
    if importlib.util.find_spec('tinygrad') is None:
        pytest.skip('tinygrad is not installed')
    if device.startswith('CUDA'):
        import torch
        if not torch.cuda.is_available():
            pytest.skip('A real CUDA GPU is required')
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=str(root), DEV=device.split(':')[0],
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CPU_COUNT='1', THREADS='1')
    result = subprocess.run([sys.executable, __file__, device], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'native Tinygrad replica SGD verified:' in result.stdout


if __name__ == '__main__':
    _native_control(sys.argv[1])
