"""Native array measurements for the shared physical Greedy bootstrap."""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Mapping

import numpy as np
import torch

from splitfleet.backends.utils import bind_model_inputs, native_device_scope
from splitfleet.common.model_state import model_state_hash


class _OwnedParameters:
    """Give the existing optimizer factory only this stage's real variables."""

    def __init__(self, model, parameters):
        self.model, self.selected = model, parameters

    def __getattr__(self, name):
        return getattr(self.model, name)

    def parameters(self, *args, **kwargs):
        return list(self.selected.values())

    def state_tensors(self):
        return self.selected

    canonical_variables = state_tensors

    @property
    def variables(self):
        return list(self.selected.values())

    @property
    def trainable_variables(self):
        return self.variables


def _synchronize(backend, device, value=None):
    if backend == 'tf':
        from tensorflow.python.eager import context
        context.async_wait()
    elif backend == 'paddle' and str(device).startswith(('cuda', 'gpu')):
        import paddle
        paddle.device.cuda.synchronize()


def _finite(backend, value):
    if isinstance(value, Mapping):
        return all(_finite(backend, item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(backend, item) for item in value)
    if value is None or not hasattr(value, 'dtype'):
        return True
    if backend == 'tf':
        import tensorflow as tf
        return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    import paddle
    return bool(paddle.all(paddle.isfinite(value)).item())


def calibrate_native_split(model, inputs, targets, *, boundaries, make_handle,
                           loss_fn, device, source, optimizer_fn, adapter):
    """Use the same anchors, repetitions, owned SGD and receipt as Torch.

    Native random operations are refused: these ResNet graphs are deterministic,
    and host RNG preservation must not pretend to restore opaque native streams.
    """
    backend = adapter.backend_name
    if backend not in {'tf', 'paddle'}:
        raise TypeError(f'Native calibration is unavailable for {backend}')
    boundaries = tuple(boundaries)
    if not boundaries or len(set(boundaries)) != len(boundaries):
        raise ValueError('Calibration requires distinct admitted boundaries')
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.runtime.auxiliary_state import commit_auxiliary_state
    from splitfleet.runtime.owned_auxiliary_state import statistics_for_stage

    started = time.perf_counter()
    state = adapter.export_ndarrays(model)
    initial_hash = model_state_hash(model, adapter)
    python_rng, numpy_rng, torch_rng = random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(model, getattr(model, 'training', None), getattr(model, '_splitfleet_training', None))]
    parameters = {}
    if backend == 'paddle':
        parameters = dict(model.named_parameters())
        modes.extend((module, module.training, getattr(module, '_splitfleet_training', None))
                     for _, module in model.named_sublayers())
    gradients = {name: None if value.grad is None else value.grad.detach().clone()
                 for name, value in parameters.items()}
    stage = 'suffix' if source == 'server_shape_matched_current_deployment' else 'prefix'
    records = []
    try:
        adapter.set_training(model, True)
        with native_device_scope(backend, device):
            for cut in boundaries:
                adapter.load_ndarrays(model, state)
                handle = make_handle(cut)
                if handle.plan.boundary != cut:
                    raise ValueError('Calibration must use an exact admitted canonical boundary')
                if any(any(token in str(node.op_type).lower() for token in ('random', 'rng', 'dropout', 'bernoulli'))
                       for node in handle.runtime.trace_graph.nodes):
                    raise ValueError('Native stochastic calibration requires a native RNG restoration contract')
                ownership = state_ownership(handle, 'calibration')
                owners = dict(zip(ownership['names'], ownership['owners'], strict=True))
                native = adapter._state(model)
                trainable = {name for name, value in native.items()
                             if (value.trainable if backend == 'tf' else not value.stop_gradient)}
                owned = {name: value for name, value in native.items() if name in trainable and owners[name] == stage}
                optimizer = None
                if optimizer_fn is not None and owned:
                    optimizer = optimizer_fn(_OwnedParameters(model, owned))
                samples = []
                for batch_index in range(2):
                    if backend == 'paddle':
                        model.clear_gradients()
                    args = bind_model_inputs(inputs.args, adapter, model=model)
                    _synchronize(backend, device, args)
                    begin = time.perf_counter()
                    boundary = handle.backend.run_prefix(*args, input_kwargs=dict(inputs.kwargs), training=True)
                    _synchronize(backend, device, boundary.tensors)
                    forward_ms = (time.perf_counter() - begin) * 1000
                    begin = time.perf_counter()
                    loss, returned = handle.backend.train_suffix(boundary, targets, loss_fn=loss_fn,
                        optimizer=optimizer if stage == 'suffix' else None)
                    all_returned = returned
                    auxiliary = handle.backend.train_suffix_auxiliary
                    if auxiliary is not None:
                        commit_auxiliary_state(model, statistics_for_stage(model, auxiliary, ownership, 'suffix'))
                    _synchronize(backend, device, (loss, returned))
                    service_ms = (time.perf_counter() - begin) * 1000
                    begin = time.perf_counter()
                    prefix = handle.backend.backward_prefix(boundary, boundary_grads=returned,
                        optimizer=optimizer if stage == 'prefix' else None)
                    if auxiliary is not None:
                        commit_auxiliary_state(model, statistics_for_stage(model, auxiliary, ownership, 'prefix'))
                    _synchronize(backend, device, (prefix, adapter._state(model)))
                    backward_ms = (time.perf_counter() - begin) * 1000
                    if batch_index == 1:
                        samples.append((forward_ms, backward_ms, service_ms))
                if not _finite(backend, (loss, all_returned, prefix, [value.grad for value in parameters.values()])):
                    raise ValueError('Calibration loss or native gradients are nonfinite')
                forward_ms, backward_ms, service_ms = np.median(samples, axis=0).tolist()
                records.append({'boundary': cut, 'graph_signature': handle.plan.graph_signature,
                    'feature_abi_id': handle.plan.feature_abi_id, 'client_forward_ms': forward_ms,
                    'client_backward_ms': backward_ms, 'local_tail_service_ms': service_ms,
                    'loss': adapter.scalar_value(loss), 'measured_batches': 1,
                    'optimizer_steps': 2 if optimizer is not None else 0,
                    'optimizer_stage': stage if optimizer_fn is not None else None})
                del handle, optimizer, boundary, returned, all_returned, prefix, auxiliary, loss
    finally:
        adapter.load_ndarrays(model, state)
        for name, parameter in parameters.items():
            parameter.grad = gradients[name]
        for module, training, explicit_training in modes:
            if training is not None:
                module.training = training
            if explicit_training is not None:
                module._splitfleet_training = explicit_training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    final_hash = model_state_hash(model, adapter)
    if final_hash != initial_hash or not torch.equal(torch_rng, torch.get_rng_state()):
        raise ValueError('Native calibration did not preserve model and RNG state')
    return {'schema': 'splitfleet.training-calibration', 'source': source, 'device': str(device),
        'framework_backend': backend, 'model_hash_before': initial_hash, 'model_hash_after': final_hash,
        'state_and_torch_rng_preserved': True, 'state_and_rng_preserved': True,
        'rng_scope': 'Python/NumPy/Torch restored; native stochastic graph operations rejected',
        'optimizer_steps': sum(row['optimizer_steps'] for row in records), 'persistent_optimizer_steps': 0,
        'warmup_batches': 1, 'elapsed_sec': time.perf_counter() - started, 'records': records,
        'interpretation': 'Current native split execution; temporary owned SGD and state/RNG restored; '
                          'not network or global round timing'}
