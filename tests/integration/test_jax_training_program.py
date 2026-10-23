"""Cached native JAX derivatives must keep live data, state and ABI checks."""
from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.parametrize('disable_jit', [False, True])
def test_cached_native_gradients_keep_live_parameters_targets_auxiliary_and_batch_shapes(disable_jit):
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    from splitfleet.autosplit import AutoSplitSession
    from tests.integration.test_auxiliary_split_state import _case

    with jax.disable_jit(disable_jit):
        case = _case('jax')
        case.model._splitfleet_jax_training_program_cache = True
        handle = AutoSplitSession().prepare_runtime(case.model, case.inputs,
            batch_axes=case.axes, dynamic_batch=(1, 16))
        cut = next(node for node in handle.runtime.trace_graph.nodes if node.op_type == 'dot_general')
        handle = handle.backend.repartition('after:' + cut.canonical_id)
        observed = []

        def loss(output, targets):
            value = case.loss(output, targets)
            jax.debug.callback(lambda logits: observed.append(np.asarray(logits).copy()),
                               output['logits'], ordered=True)
            return value

        rng = np.random.default_rng(93)
        for step, batch in enumerate((16, 12, 16), 1):
            images = jnp.asarray(rng.normal(size=(batch, 2, 2, 3)).astype(np.float32))
            targets = jnp.asarray(rng.normal(size=(batch, 3)).astype(np.float32))

            def reference(params):
                output = case.model(params, images)
                return case.loss(output, targets), output

            (expected_loss, expected), expected_gradients = jax.value_and_grad(
                reference, has_aux=True)(case.model.params)
            boundary = handle.backend.run_prefix(case.model.params, images, training=True)
            value, gradients = handle.backend.train_suffix(boundary, targets, loss_fn=loss)
            actual = handle.backend.backward_prefix(boundary, gradients)
            auxiliary = handle.backend.train_suffix_auxiliary
            jax.block_until_ready((value, actual, auxiliary))
            jax.effects_barrier()
            np.testing.assert_allclose(value, expected_loss, atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(observed[-1], expected['logits'], atol=1e-5, rtol=1e-4)
            for name in expected_gradients:
                np.testing.assert_allclose(actual['inputs'][0][name], expected_gradients[name],
                                           atol=1e-5, rtol=1e-4)
            for name, fields in auxiliary.items():
                for key, field in fields.items():
                    np.testing.assert_allclose(field, expected['batch_statistics'][name][key],
                                               atol=1e-5, rtol=1e-4)
            assert case.count() == step - 1
            case.model.params.update({name: value - np.float32(.01) * actual['inputs'][0][name]
                                      for name, value in case.model.params.items()})
            case.model.apply_statistics(auxiliary)
            assert case.count() == step and len(observed) == step

        # B=16 is reused even though its weights, images and labels changed.
        assert len(handle.runtime._splitfleet_jax_training_programs) == 4
        malformed = replace(boundary, metadata={**boundary.metadata, 'split_id': 'wrong-cut'})
        with pytest.raises(Exception, match='split_id'):
            handle.backend.train_suffix(malformed, targets, loss_fn=loss)
        malformed = replace(boundary, metadata={**boundary.metadata, 'supports_prefix_backward': False})
        with pytest.raises(Exception, match='training|backward'):
            handle.backend.backward_prefix(malformed, gradients)

        # Production removes resident parameter gradients before returning the
        # cotangents. A different key set must never reuse a full-root program.
        from splitfleet.autosplit.boundary import to_torchlens_boundary
        key = next(iter(gradients))
        partial = {key: gradients[key]}
        expected_partial = handle.runtime.backward_prefix(to_torchlens_boundary(boundary), partial)
        actual_partial = handle.backend.backward_prefix(boundary, partial)
        for name in expected_partial['inputs'][0]:
            np.testing.assert_allclose(actual_partial['inputs'][0][name],
                                       expected_partial['inputs'][0][name], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('stage', ['prefix', 'suffix'])
def test_jax_calibration_returns_only_transported_gradients_and_restores_state(stage):
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.backends.utils import adapter_for
    from splitfleet.runtime.functional_sgd import OwnedFunctionalSGD, resident_parameter_bindings
    from splitfleet.server.placement.greedy.native_calibration import calibrate_native_split
    from splitfleet.tasks import ModelInputs

    class Model:
        training = True
        _splitfleet_jax_training_program_cache = True

        def __init__(self):
            self.params = {'a': jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 20,
                           'b': jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 30}

        def __call__(self, params, images):
            return (images @ params['a']) @ params['b']

        def state_tensors(self):
            return dict(self.params)

        def set_params(self, params):
            self.params.update(params)

        def load_state(self, state):
            self.params.update({name: jnp.asarray(value) for name, value in state.items()})

    model = Model()
    images = jnp.asarray([[1., 2., 3.], [-1., 1., .5]], dtype=jnp.float32)
    labels = jnp.asarray([[.2, .3, .1, .4, .5], [.1, .4, .2, .3, .6]], dtype=jnp.float32)
    handle = AutoSplitSession().prepare_runtime(model, (model.params, images),
        batch_axes={'/args/1': 0}, dynamic_batch=(1, 16))
    cut = next(n for n in handle.runtime.trace_graph.nodes if n.op_type == 'dot_general')
    seen = []

    def make_handle(boundary):
        h = handle.backend.repartition(boundary)
        native = h.backend.backward_prefix
        ownership = state_ownership(h, 'calibration')
        resident = resident_parameter_bindings(h, ownership)
        assert resident

        def observe(value, *, boundary_grads, optimizer):
            assert not set(boundary_grads).intersection(resident)
            seen.append(1)
            return native(value, boundary_grads, optimizer=optimizer)

        h.backend.backward_prefix = observe
        return h

    before = {name: np.asarray(value).copy() for name, value in model.params.items()}
    receipt = calibrate_native_split(model, ModelInputs((model.params, images)), labels,
        boundaries=['after:' + cut.canonical_id], make_handle=make_handle,
        loss_fn=lambda output, target: jnp.mean((output - target) ** 2), device='cpu',
        source='server_shape_matched_current_deployment' if stage == 'suffix' else 'client',
        optimizer_fn=lambda value: OwnedFunctionalSGD(value, .05),
        adapter=adapter_for(model, (model.params, images)))
    assert len(seen) == 2 and receipt['state_and_rng_preserved']
    assert receipt['optimizer_steps'] == 2 and receipt['persistent_optimizer_steps'] == 0
    for name, value in before.items():
        np.testing.assert_array_equal(model.params[name], value)
