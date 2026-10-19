"""Compare compiled dispatch with the original native SGD and BN assignments."""

import numpy as np
import pytest

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


@pytest.mark.integration
@pytest.mark.parametrize('device', ['/CPU:0', '/GPU:0'])
def test_native_optimizer_and_batchnorm_updates_match_eager_execution(request, device):
    if _run_isolated(request):
        return
    tf = pytest.importorskip('tensorflow')
    if device.startswith('/GPU') and not tf.config.list_physical_devices('GPU'):
        pytest.skip('A real TensorFlow GPU is required')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
    from tests.integration.resnet50_reference.tf_model import CanonicalBatchNorm
    from splitfleet.runtime.tensorflow_stack_scope import call_tensorflow_function
    from splitfleet.runtime.tensorflow_updates import CompiledNativeOptimizer

    with tf.device(device):
        left = [tf.Variable(np.arange(15, dtype=np.float32).reshape(3, 5)/17)]
        right = [tf.Variable(variable.numpy()) for variable in left]
        eager = tf.keras.optimizers.SGD(learning_rate=.05, momentum=0.)
        compiled = CompiledNativeOptimizer(tf.keras.optimizers.SGD(learning_rate=.05, momentum=0.))
        before = right[0].numpy().copy()
        for step in range(3):
            gradients = [tf.constant(np.full((3, 5), (step+1)/19, dtype=np.float32))]
            eager.apply_gradients(zip(gradients, left, strict=True))
            compiled.apply_gradients(zip(gradients, right, strict=True))
        np.testing.assert_array_equal(left[0].numpy(), right[0].numpy())
        assert not np.array_equal(before, right[0].numpy())
        assert int(eager.iterations.numpy()) == int(compiled.iterations.numpy()) == 3
        eager.learning_rate = compiled.learning_rate = .01
        eager.apply_gradients(zip(gradients, left, strict=True))
        compiled.apply_gradients(zip(gradients, right, strict=True))
        np.testing.assert_array_equal(left[0].numpy(), right[0].numpy())
        assert int(compiled.iterations.numpy()) == 4
        with pytest.raises(RuntimeError, match='variable bindings changed'):
            compiled.apply_gradients([(gradients[0], tf.Variable(before))])

        reference = CanonicalBatchNorm(5, name='reference')
        tested = CanonicalBatchNorm(5, name='tested')
        reference.build((None, 2, 2, 5))
        tested.build((None, 2, 2, 5))
        commit = tf.function(tested.apply_statistics, autograph=False)
        for count in (16, 8, 4):
            values = {'mean': tf.constant(np.arange(5, dtype=np.float32)/11),
                      'variance': tf.constant(np.arange(5, dtype=np.float32)/13+1),
                      'count': tf.constant(count, dtype=tf.int64)}
            reference.apply_statistics(values)
            call_tensorflow_function(commit, values)
        for left_value, right_value in zip(reference.weights, tested.weights, strict=True):
            np.testing.assert_array_equal(left_value.numpy(), right_value.numpy())
        assert int(tested.num_batches_tracked.numpy()) == 3
        with pytest.raises(tf.errors.InvalidArgumentError):
            call_tensorflow_function(commit, {**values, 'count': tf.constant(1, dtype=tf.int64)})
        assert int(tested.num_batches_tracked.numpy()) == 3


@pytest.mark.integration
@pytest.mark.parametrize('device', ['/CPU:0', '/GPU:1'])
def test_batchnorm_commit_programs_share_code_and_preserve_private_state(request, device):
    if _run_isolated(request):
        return
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import gc
    import weakref

    tf = pytest.importorskip('tensorflow')
    if device.startswith('/GPU') and len(tf.config.list_physical_devices('GPU')) < 2:
        pytest.skip('Two real TensorFlow GPUs are required')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
    from tests.integration.resnet50_reference.tf_model import CanonicalBatchNorm, ResNet50

    class Owner:
        def __init__(self, layers):
            self.layers = layers
            self.device = device
            self._splitfleet_statistics_functions = {}

        def batch_norm_layers(self):
            return self.layers

    with tf.device(device):
        pairs = []
        for replica in range(3):
            actual, expected = {}, {}
            for name, channels in [('first', 3), ('second', 5)]:
                actual[name] = CanonicalBatchNorm(channels, name=name)
                expected[name] = CanonicalBatchNorm(channels, name=name)
                for layer in (actual[name], expected[name]):
                    layer.build((None, 2, 2, channels))
                    layer.running_mean.assign(tf.fill((channels,), (replica+1)/7))
                    layer.num_batches_tracked.assign(replica)
            pairs.append((Owner(actual), expected))

        barrier = Barrier(len(pairs))

        def train(entry):
            replica, (owner, oracle) = entry
            with tf.device(device):
                for step, count in enumerate((16, 8, 4)):
                    values = {name: {'mean': tf.fill((layer.channels,), (replica+step+1)/11),
                        'variance': tf.fill((layer.channels,), (replica+step+2)/13),
                        'count': tf.constant(count, tf.int64)} for name, layer in owner.layers.items()}
                    barrier.wait(timeout=30)
                    ResNet50.apply_statistics(owner, values)
                    for name, layer in oracle.items():
                        layer.apply_statistics(values[name])
                return values

        with ThreadPoolExecutor(max_workers=3) as executor:
            values = list(executor.map(train, enumerate(pairs)))
        functions = [next(iter(owner._splitfleet_statistics_functions.values())) for owner, _ in pairs]
        assert all(function is functions[0] for function in functions)
        assert functions[0].experimental_get_tracing_count() == 1
        bindings = tuple((layer.running_mean.value, layer.running_var.value,
                          layer.num_batches_tracked.value) for layer in pairs[0][0].layers.values())
        concrete = functions[0].get_concrete_function(bindings, values[0])
        assert not any(value.dtype == tf.resource for value in concrete.captured_inputs)
        for owner, oracle in pairs:
            for name, layer in owner.layers.items():
                for actual, expected in zip(layer.weights, oracle[name].weights, strict=True):
                    np.testing.assert_array_equal(actual.numpy(), expected.numpy())

        owner, oracle = pairs[0]
        untouched = [value.numpy().copy() for value in owner.layers['second'].weights]
        ResNet50.apply_statistics(owner, values[0], names={'first'})
        oracle['first'].apply_statistics(values[0]['first'])
        for actual, expected in zip(owner.layers['second'].weights, untouched, strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected)
        before = [value.numpy().copy() for value in owner.layers['first'].weights]
        invalid = {'first': {**values[0]['first'], 'count': tf.constant(1, tf.int64)}}
        with pytest.raises(tf.errors.InvalidArgumentError):
            ResNet50.apply_statistics(owner, invalid)
        for actual, expected in zip(owner.layers['first'].weights, before, strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected)
        reference = weakref.ref(owner)
        buffer = weakref.ref(owner.layers['first'].running_mean.value)
        pairs.clear()
        del owner, oracle, bindings
        gc.collect()
        assert reference() is None
        assert buffer() is None
