"""Native replay and SGD must use each replica's independent variables."""

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.parametrize('device', ['/CPU:0', '/GPU:0'])
def test_rebound_replica_matches_native_sgd_without_changing_source(device):
    tf = pytest.importorskip('tensorflow')
    if device.startswith('/GPU') and not tf.config.list_physical_devices('GPU'):
        pytest.skip('A real TensorFlow GPU is required')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.autosplit.tensorflow_clone import clone_tensorflow_runtime_handle

    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__(name='replica_model')
            self.training = True
            self.first = tf.keras.layers.Dense(5, activation='tanh', name='first')
            self.last = tf.keras.layers.Dense(3, name='last')

        def call(self, value):
            return self.last(self.first(value))

        def canonical_variables(self):
            return {str(index): value for index, value in enumerate(self.variables)}

        def _splitfleet_replica_contract(self):
            return type(self), self.training

        def __deepcopy__(self, memo):
            clone = Model()
            clone(tf.zeros((1, 4)))
            clone.set_weights(self.get_weights())
            clone.training = self.training
            memo[id(self)] = clone
            return clone

    with tf.device(device):
        inputs = tf.constant(np.arange(12, dtype=np.float32).reshape(3, 4)/13)
        targets = tf.constant(np.arange(9, dtype=np.float32).reshape(3, 3)/17)
        source, target, reference = Model(), Model(), Model()
        for model in (source, target, reference):
            model(inputs)
        target.set_weights([value + .2 for value in source.get_weights()])
        reference.set_weights(target.get_weights())
        original = source.get_weights()
        handle = prepare_torchlens_runtime(source, (inputs,), boundary='50%',
            batch_axes={'/args/0': 0}, dynamic_batch=(1, 8))
        clone = clone_tensorflow_runtime_handle(handle, target)
        assert clone is not None
        assert clone.plan.metadata['_reused_capture']
        assert clone.runtime.graph_identity == handle.runtime.graph_identity
        for batch in (1, 3, 8):
            values = tf.repeat(inputs[:1], batch, axis=0)
            output = clone.backend.run_suffix(clone.backend.run_prefix(values))
            np.testing.assert_allclose(output.numpy(), target(values).numpy(), atol=1e-5, rtol=1e-4)
        def loss_fn(logits, truth):
            return tf.reduce_mean((logits-truth)**2)
        with tf.GradientTape() as tape:
            loss = loss_fn(reference(inputs), targets)
        gradients = tape.gradient(loss, reference.trainable_variables)
        tf.keras.optimizers.SGD(.01).apply_gradients(zip(gradients, reference.trainable_variables))
        optimizer = tf.keras.optimizers.SGD(.01)
        optimizer.build(target.trainable_variables)
        boundary = clone.backend.run_prefix(inputs, training=True)
        _, roots = clone.backend.train_suffix(boundary, targets, loss_fn=loss_fn, optimizer=optimizer)
        clone.backend.backward_prefix(boundary, roots, optimizer=optimizer)
        for actual, expected in zip(target.get_weights(), reference.get_weights(), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        for actual, expected in zip(source.get_weights(), original, strict=True):
            np.testing.assert_array_equal(actual, expected)
        target.training = False
        assert clone_tensorflow_runtime_handle(handle, target) is None


@pytest.mark.integration
def test_full_resnet50_replica_matches_native_update_and_batchnorm_state():
    tf = pytest.importorskip('tensorflow')
    if not tf.config.list_physical_devices('GPU'):
        pytest.skip('A real TensorFlow GPU is required')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
    from experiments.resnet50_quality.data import load_cifar
    from experiments.resnet50_quality.model_contract import canonical_state
    from tests.integration.resnet50_reference.native import (
        build_model, configure_backend, export_state, native_batch, optimizer, training_mode, train_step,
    )
    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.autosplit.tensorflow_clone import clone_tensorflow_runtime_handle

    configure_backend('tensorflow')
    data = load_cifar('data/cifar-10-batches-py')
    ids = data.train_indices[:16]
    images, labels = native_batch('tensorflow', data.images(ids), data.train_labels[ids], 'cuda:0')
    initial, different = canonical_state(233), canonical_state(234)
    source = build_model('tensorflow', initial, 'cuda:0')
    target = build_model('tensorflow', different, 'cuda:0')
    reference = build_model('tensorflow', different, 'cuda:0')
    for model in (source, target, reference):
        training_mode('tensorflow', model, True)
        model._splitfleet_tensorflow_function_forward = True
    handle = prepare_torchlens_runtime(source, (images,), boundary='50%',
        batch_axes={'/args/0': 0}, dynamic_batch=(1, 16))
    assert handle.runtime.batch_validation['status'] == 'passed'
    cut = [node for node in handle.runtime.trace_graph.nodes
           if node.module_path == 'conv1' and node.op_type.lower() == 'conv2d'][-1]
    handle = handle.backend.repartition('after:' + cut.canonical_id)
    clone = clone_tensorflow_runtime_handle(handle, target)
    assert clone is not None
    native_optimizer = optimizer('tensorflow', reference, .05)
    split_optimizer = optimizer('tensorflow', target, .05)
    prefix_optimizer = optimizer('tensorflow', target, .05)
    from splitfleet.tasks.losses import sparse_cross_entropy

    def loss_fn(output, truth):
        return sparse_cross_entropy(output['logits'], truth)

    for batch in (16, 4):
        train_step('tensorflow', reference, images[:batch], labels[:batch], native_optimizer=native_optimizer)
        boundary = clone.backend.run_prefix(images[:batch], training=True)
        _, roots = clone.backend.train_suffix(boundary, labels[:batch],
            loss_fn=loss_fn, optimizer=split_optimizer)
        clone.backend.backward_prefix(boundary, roots, optimizer=prefix_optimizer)
        statistics = clone.backend.train_suffix_auxiliary
        assert len(statistics) == 53
        target.apply_statistics(statistics)
        actual, expected = export_state('tensorflow', target), export_state('tensorflow', reference)
        for name in actual:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-5, rtol=1e-4, err_msg=name)
    for name, value in export_state('tensorflow', source).items():
        np.testing.assert_array_equal(value, initial[name], err_msg=name)
    assert handle.runtime._splitfleet_shared_suffix_functions.compile_count == 2
    from splitfleet.runtime.tensorflow_forward import _ReboundSuffixTraining
    assert all(isinstance(value, _ReboundSuffixTraining)
               for value in clone.runtime._splitfleet_native_training_functions.values())


@pytest.mark.integration
@pytest.mark.parametrize('device', ['/CPU:0', '/GPU:1'])
def test_private_replicas_share_native_gradient_programs_across_wire_and_threads(device):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event

    tf = pytest.importorskip('tensorflow')
    if device == '/GPU:1' and len(tf.config.list_physical_devices('GPU')) < 2:
        pytest.skip('Two real TensorFlow GPUs are required')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.autosplit.tensorflow_clone import clone_tensorflow_runtime_handle
    from splitfleet.runtime.tensorflow_forward import _ReboundSuffixTraining
    from splitfleet.transport import decode_boundary, encode_boundary
    from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary

    class Model(tf.keras.Model):
        _splitfleet_tensorflow_function_forward = True

        def __init__(self):
            super().__init__(name='shared_program_private_state')
            self.training = True
            self.first = tf.keras.layers.Dense(5, activation='tanh', name='first')
            self.last = tf.keras.layers.Dense(3, name='last')

        def call(self, value):
            return self.last(self.first(value))

        def canonical_variables(self):
            return {str(index): value for index, value in enumerate(self.variables)}

        def _splitfleet_replica_contract(self):
            return type(self), self.training

        def __deepcopy__(self, memo):
            result = Model()
            result(tf.zeros((1, 4)))
            result.set_weights(self.get_weights())
            memo[id(self)] = result
            return result

    def loss_fn(logits, truth):
        return tf.reduce_mean((logits - truth)**2)

    with tf.device(device):
        sample = tf.constant(np.arange(32, dtype=np.float32).reshape(8, 4)/37)
        source = Model()
        source(sample)
        original = source.get_weights()
        base = prepare_torchlens_runtime(source, sample[:3], boundary='50%',
            batch_axes={'/args/0': 0}, dynamic_batch=(1, 8))
        base = base.backend.repartition(base.plan.boundary)
        replicas = []
        for index in range(3):
            model, oracle = Model(), Model()
            model(sample)
            oracle(sample)
            model.set_weights([value + (index+1)/10 for value in original])
            oracle.set_weights(model.get_weights())
            handle = clone_tensorflow_runtime_handle(base, model)
            assert handle is not None
            opt = tf.keras.optimizers.SGD(.01)
            opt.build(model.trainable_variables)
            replicas.append((handle, oracle, opt, tf.keras.optimizers.SGD(.01)))
        from splitfleet.backends.utils import adapter_for
        from splitfleet.common import ServerModelFitIns
        from splitfleet.common.constants import AUTOSPLIT_PLAN_ID_CONFIG_KEY
        from splitfleet.runtime.tensorflow_capture_scope import tensorflow_execution_scope
        from splitfleet.server.server_model.autosplit_tail_server_model import AutoSplitTailServerModel
        from splitfleet.server.stage_runtime.manager import StageRuntimeManager

        manager = StageRuntimeManager()
        manager.bind_runtime_handle(base)
        tail = AutoSplitTailServerModel(runtime_manager=manager, model=source,
            sample_inputs=(sample[:3],), loss_fn=loss_fn, defer_model=True,
            device='cuda:1' if device == '/GPU:1' else 'cpu')
        ins = ServerModelFitIns(parameters=adapter_for(source, (sample[:3],)).export_ndarrays(source),
            config={AUTOSPLIT_PLAN_ID_CONFIG_KEY: base.plan.plan_id}, sid='prepared')
        entered, release = Event(), Event()

        def active_native_execution():
            with tensorflow_execution_scope():
                entered.set()
                assert release.wait(15)

        with ThreadPoolExecutor(max_workers=2) as pool:
            native = pool.submit(active_native_execution)
            assert entered.wait(10)
            preparing = pool.submit(tail.configure_fit, ins)
            try:
                preparing.result(timeout=10)
                assert tail.runtime_handle.plan.metadata['_reused_capture']
            finally:
                release.set()
            native.result(timeout=10)
        # A genuine fallback capture must still wait for active native work.
        from concurrent.futures import TimeoutError as FutureTimeout
        from splitfleet.runtime.tensorflow_capture_scope import _GATE

        uncached = AutoSplitTailServerModel(runtime_manager=manager, model=source,
            sample_inputs=(sample[:3],), loss_fn=loss_fn,
            device='cuda:1' if device == '/GPU:1' else 'cpu')
        uncached.model._splitfleet_replica_contract = lambda: 'requires_fresh_capture'
        entered.clear()
        release.clear()
        with ThreadPoolExecutor(max_workers=2) as pool:
            native = pool.submit(active_native_execution)
            assert entered.wait(10)
            preparing = pool.submit(uncached.configure_fit, ins)
            try:
                with _GATE.condition:
                    assert _GATE.condition.wait_for(lambda: _GATE.waiting_writers > 0, timeout=10)
                with pytest.raises(FutureTimeout):
                    preparing.result(timeout=.05)
            finally:
                release.set()
            native.result(timeout=10)
            preparing.result(timeout=15)
        assert not uncached.runtime_handle.plan.metadata.get('_reused_capture', False)
        rendezvous = Barrier(3)
        for step, batch in enumerate((3, 1, 8)):
            calls = []
            for index, (handle, oracle, opt, oracle_opt) in enumerate(replicas):
                values = sample[:batch] + index/13
                targets = tf.constant(np.full((batch, 3), index/7, np.float32))
                with tf.GradientTape() as tape:
                    expected_loss = loss_fn(oracle(values), targets)
                gradients = tape.gradient(expected_loss, oracle.trainable_variables)
                oracle_opt.apply_gradients(zip(gradients, oracle.trainable_variables))
                local = handle.backend.run_prefix(values, training=True)
                envelope = boundary_to_envelope(local, round_id=1, client_id=str(index), step_id=str(step),
                    plan_id=handle.plan.plan_id, split_id=local.metadata['split_id'],
                    canonical_graph_hash=local.metadata['graph_shape_hash'],
                    boundary_schema_hash=handle.plan.feature_abi_id, model_version=1)
                remote = envelope_to_boundary(decode_boundary(encode_boundary(envelope)), handle.runtime,
                    'cuda:1' if device == '/GPU:1' else 'cpu')
                calls.append((handle, local, remote, targets, opt, expected_loss))

            def suffix(call):
                handle, _, remote, targets, opt, expected = call
                rendezvous.wait(timeout=15)
                actual, roots = handle.backend.train_suffix(remote, targets, loss_fn=loss_fn, optimizer=opt)
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-4)
                return roots

            with ThreadPoolExecutor(max_workers=3) as pool:
                returned = list(pool.map(suffix, calls))
            for (handle, local, _, _, opt, _), roots in zip(calls, returned, strict=True):
                handle.backend.backward_prefix(local, roots, optimizer=opt)
            for handle, oracle, _, _ in replicas:
                for actual, expected in zip(handle.model.get_weights(), oracle.get_weights(), strict=True):
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
                assert all(isinstance(value, _ReboundSuffixTraining)
                           for value in handle.runtime._splitfleet_native_training_functions.values())
        assert base.runtime._splitfleet_shared_suffix_functions.compile_count == 3
        # An undeclared loss resource must retain the ordinary private function.
        handle, oracle, opt, oracle_opt = replicas[0]
        multiplier = tf.Variable(1.3, trainable=False)

        def resource_loss(logits, truth):
            return multiplier * loss_fn(logits, truth)

        targets = tf.zeros((3, 3))
        with tf.GradientTape() as tape:
            expected = resource_loss(oracle(sample[:3]), targets)
        gradients = tape.gradient(expected, oracle.trainable_variables)
        oracle_opt.apply_gradients(zip(gradients, oracle.trainable_variables))
        boundary = handle.backend.run_prefix(sample[:3], training=True)
        _, roots = handle.backend.train_suffix(boundary, targets, loss_fn=resource_loss, optimizer=opt)
        handle.backend.backward_prefix(boundary, roots, optimizer=opt)
        for actual, expected in zip(handle.model.get_weights(), oracle.get_weights(), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        assert any(not isinstance(value, _ReboundSuffixTraining)
                   for value in handle.runtime._splitfleet_native_training_functions.values())
        assert base.runtime._splitfleet_shared_suffix_functions.compile_count == 3
        assert len(base.runtime._splitfleet_shared_suffix_functions.rejected) == 1
        for actual, expected in zip(source.get_weights(), original, strict=True):
            np.testing.assert_array_equal(actual, expected)
