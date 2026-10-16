"""Full FP32 Keras ResNet50-V1.5 with the experiment's portable state.

Keras' standard BatchNormalization updates its running variance differently
from the experiment contract.  The layer below keeps normalization and state
updates explicit, including the integer count of actual training batches.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from threading import RLock
from types import SimpleNamespace
from typing import Any

import numpy as np
import tensorflow as tf

from experiments.resnet50_quality.model_contract import (
    BLOCKS,
    INPUT_SIZE,
    NUM_CLASSES,
    PLANES,
    SCHEMA,
    TRAINABLE_PARAMETERS,
    batch_norm_associations,
    validate_state,
)

_STATISTICS_FUNCTION_LOCK = RLock()


@lru_cache(maxsize=64)
def _statistics_function(device, layout):
    """Share a native BN commit program; all mutable resources are arguments."""

    @tf.function(autograph=False)
    def commit(bindings, statistics):
        with tf.device(device):
            for (name, channels, retention), resources in zip(layout, bindings, strict=True):
                state = SimpleNamespace(name=name, channels=channels, retention=retention,
                    running_mean=resources[0], running_var=resources[1], num_batches_tracked=resources[2])
                CanonicalBatchNorm.apply_statistics(state, statistics[name])

    return commit


def _resolve_device(device: str) -> str:
    """Require a real requested device instead of relying on soft placement."""
    requested = str(device).strip().lower()
    if requested in {"cpu", "cpu:0", "/cpu:0"}:
        resolved = "/CPU:0"
    elif requested.startswith(("cuda:", "gpu:", "/gpu:")):
        try:
            index = int(requested.rsplit(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid TensorFlow device {device!r}") from exc
        if index < 0:
            raise ValueError(f"Invalid TensorFlow device {device!r}")
        resolved = f"/GPU:{index}"
    else:
        raise ValueError(f"Unsupported TensorFlow device {device!r}")
    available = {"/" + item.name.rsplit("device:", 1)[-1].lstrip("/")
                 for item in tf.config.list_logical_devices(resolved[1:4])}
    if resolved not in available:
        raise RuntimeError(f"Requested TensorFlow device {resolved} is unavailable")
    return resolved


class CanonicalBatchNorm(tf.keras.layers.Layer):
    """NHWC BatchNorm with PyTorch-compatible normalization and buffers."""

    def __init__(self, channels: int, *, name: str):
        super().__init__(name=name, dtype="float32", autocast=False)
        self.channels = int(channels)
        self.epsilon = 1e-5
        self.retention = 0.9

    def build(self, input_shape):
        if len(input_shape) != 4 or int(input_shape[-1]) != self.channels:
            raise ValueError(f"{self.name} requires NHWC input with {self.channels} channels")
        self.gamma = self.add_weight(
            name="weight", shape=(self.channels,), dtype="float32", initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="bias", shape=(self.channels,), dtype="float32", initializer="zeros",
            trainable=True,
        )
        self.running_mean = self.add_weight(
            name="running_mean", shape=(self.channels,), dtype="float32", initializer="zeros",
            trainable=False,
        )
        self.running_var = self.add_weight(
            name="running_var", shape=(self.channels,), dtype="float32", initializer="ones",
            trainable=False,
        )
        self.num_batches_tracked = self.add_weight(
            name="num_batches_tracked", shape=(), dtype="int64", initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)

    def _statistics(self, inputs):
        # Reduce only spatial/batch axes; every channel sees m=N*H*W samples.
        shape = tf.shape(inputs, out_type=tf.int64)
        samples = shape[0] * shape[1] * shape[2]
        known = tuple(inputs.shape[:3])
        if all(dimension is not None for dimension in known) and np.prod(known) <= 1:
            raise ValueError("BatchNorm training requires more than one value per channel")
        mean = tf.reduce_mean(inputs, axis=(0, 1, 2))
        centered = inputs - mean
        variance = tf.reduce_mean(centered * centered, axis=(0, 1, 2))
        return {"mean": mean, "variance": variance, "count": samples}

    def apply_statistics(self, statistics):
        """Commit one real batch's population statistics to this layer."""
        mean = tf.convert_to_tensor(statistics["mean"], dtype=tf.float32)
        variance = tf.convert_to_tensor(statistics["variance"], dtype=tf.float32)
        samples = tf.convert_to_tensor(statistics["count"])
        if tuple(mean.shape) != (self.channels,) or tuple(variance.shape) != (self.channels,):
            raise ValueError(f"{self.name} statistics have an incorrect channel shape")
        if tuple(samples.shape) != () or not samples.dtype.is_integer:
            raise ValueError(f"{self.name} sample count must be an integer scalar")
        # Assert's diagnostic string tensors only have CPU kernels. Keep the
        # guard explicit when the training runtime disables soft placement.
        with tf.device('/CPU:0'):
            check = tf.debugging.assert_greater(samples, tf.cast(1, samples.dtype))
        with tf.control_dependencies([] if check is None else [check]):
            count = tf.cast(samples, tf.float32)
            unbiased_variance = variance * (count / (count - 1.0))
            old = tf.constant(self.retention, dtype=tf.float32)
            new = tf.constant(1.0 - self.retention, dtype=tf.float32)
            self.running_mean.assign(old * self.running_mean + new * tf.stop_gradient(mean))
            self.running_var.assign(
                old * self.running_var + new * tf.stop_gradient(unbiased_variance),
            )
            self.num_batches_tracked.assign_add(tf.constant(1, dtype=tf.int64))

    def _training_call(self, inputs, *, defer_statistics=False):
        statistics = self._statistics(inputs)
        if not defer_statistics:
            self.apply_statistics(statistics)
        output = tf.nn.batch_normalization(
            inputs, statistics["mean"], statistics["variance"],
            # Keras' implicit offset arithmetic can bypass raw-op capture.
            # Read the underlying variables explicitly so native replay and
            # its GradientTape consume live beta/gamma resource reads.
            tf.identity(self.beta.value), tf.identity(self.gamma.value), self.epsilon,
        )
        return (output, statistics) if defer_statistics else output

    def _evaluation_call(self, inputs):
        return tf.nn.batch_normalization(
            inputs, tf.identity(self.running_mean.value), tf.identity(self.running_var.value),
            tf.identity(self.beta.value), tf.identity(self.gamma.value), self.epsilon,
        )

    def call(self, inputs, training=False, defer_statistics=False):
        if inputs.dtype != tf.float32:
            raise TypeError("Canonical BatchNorm requires FP32 inputs")
        if training is None:
            training = False
        if isinstance(training, (bool, np.bool_)):
            if training:
                return self._training_call(inputs, defer_statistics=defer_statistics)
            output = self._evaluation_call(inputs)
            return (output, {}) if defer_statistics else output
        if defer_statistics:
            # Tensor-valued mode flags require an identical output structure
            # in both graph branches.  A count of zero marks evaluation stats.
            return tf.cond(tf.cast(training, tf.bool),
                           lambda: self._training_call(inputs, defer_statistics=True),
                           lambda: (self._evaluation_call(inputs), {
                               "mean": tf.identity(self.running_mean),
                               "variance": tf.identity(self.running_var),
                               "count": tf.constant(0, tf.int64),
                           }))
        return tf.cond(tf.cast(training, tf.bool),
                       lambda: self._training_call(inputs),
                       lambda: self._evaluation_call(inputs))


def _conv(channels: int, size: int, *, stride: int = 1, name: str):
    return tf.keras.layers.Conv2D(
        channels, size, strides=stride, padding="valid", data_format="channels_last",
        use_bias=False, kernel_initializer="zeros", dtype="float32", name=name,
    )


class _SymmetricMaxPool(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="maxpool", dtype="float32")

    def call(self, inputs):
        # The pool follows ReLU, so zero and -inf padding give identical
        # logits and upstream gradients (zero activations have zero ReLU VJP).
        padded = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)))
        return tf.nn.max_pool2d(padded, ksize=3, strides=2, padding="VALID", data_format="NHWC")


class _Projection(tf.keras.layers.Layer):
    def __init__(self, channels: int, stride: int):
        super().__init__(name="downsample", dtype="float32")
        self.conv = _conv(channels, 1, stride=stride, name="0")
        self.bn = CanonicalBatchNorm(channels, name="1")

    @property
    def layers(self):
        return [self.conv, self.bn]

    def build(self, input_shape):
        self.conv.build(input_shape)
        self.bn.build(self.conv.compute_output_shape(input_shape))
        super().build(input_shape)

    def call(self, inputs, training=False, defer_statistics=False):
        return self.bn(self.conv(inputs), training=training, defer_statistics=defer_statistics)


class _Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, input_channels: int, planes: int, stride: int, *, name: str):
        super().__init__(name=name, dtype="float32")
        self.planes = planes
        self.stride = stride
        self.conv1 = _conv(planes, 1, name="conv1")
        self.bn1 = CanonicalBatchNorm(planes, name="bn1")
        self.relu1 = tf.keras.layers.ReLU(name="relu1", dtype="float32")
        self.pad2 = tf.keras.layers.ZeroPadding2D(1, name="pad2", dtype="float32")
        self.conv2 = _conv(planes, 3, stride=stride, name="conv2")
        self.bn2 = CanonicalBatchNorm(planes, name="bn2")
        self.relu2 = tf.keras.layers.ReLU(name="relu2", dtype="float32")
        self.conv3 = _conv(planes * self.expansion, 1, name="conv3")
        self.bn3 = CanonicalBatchNorm(planes * self.expansion, name="bn3")
        if input_channels != planes * self.expansion or stride != 1:
            self.downsample = _Projection(planes * self.expansion, stride)
        else:
            self.downsample = None
        self.relu3 = tf.keras.layers.ReLU(name="relu3", dtype="float32")

    def build(self, input_shape):
        self.conv1.build(input_shape)
        shape = self.conv1.compute_output_shape(input_shape)
        self.bn1.build(shape)
        padded_shape = self.pad2.compute_output_shape(shape)
        self.conv2.build(padded_shape)
        shape = self.conv2.compute_output_shape(padded_shape)
        self.bn2.build(shape)
        self.conv3.build(shape)
        self.bn3.build(self.conv3.compute_output_shape(shape))
        if self.downsample is not None:
            self.downsample.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        batch, height, width, _ = input_shape
        return (batch, None if height is None else (height + self.stride - 1) // self.stride,
                None if width is None else (width + self.stride - 1) // self.stride,
                self.planes * self.expansion)

    def call(self, inputs, training=False, defer_statistics=False):
        identity = inputs
        statistics = {}

        def normalize(layer, value):
            result = layer(value, training=training, defer_statistics=defer_statistics)
            if defer_statistics:
                result, batch_statistics = result
                if batch_statistics:
                    statistics[layer.name] = batch_statistics
            return result

        output = self.relu1(normalize(self.bn1, self.conv1(inputs)))
        output = self.relu2(normalize(self.bn2, self.conv2(self.pad2(output))))
        output = normalize(self.bn3, self.conv3(output))
        if self.downsample is not None:
            identity = self.downsample(inputs, training=training, defer_statistics=defer_statistics)
            if defer_statistics:
                identity, batch_statistics = identity
                if batch_statistics:
                    statistics["downsample.1"] = batch_statistics
        output = self.relu3(output + identity)
        return (output, statistics) if defer_statistics else output


class _Stage(tf.keras.layers.Layer):
    def __init__(self, input_channels: int, planes: int, blocks: int, stride: int, *, name: str):
        super().__init__(name=name, dtype="float32")
        self.blocks = [
            _Bottleneck(input_channels if index == 0 else planes * 4,
                        planes, stride if index == 0 else 1, name=str(index))
            for index in range(blocks)
        ]

    def build(self, input_shape):
        shape = input_shape
        for block in self.blocks:
            block.build(shape)
            shape = block.compute_output_shape(shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for block in self.blocks:
            shape = block.compute_output_shape(shape)
        return shape

    def call(self, inputs, training=False, defer_statistics=False):
        output = inputs
        statistics = {}
        for index, block in enumerate(self.blocks):
            output = block(output, training=training, defer_statistics=defer_statistics)
            if defer_statistics:
                output, batch_statistics = output
                statistics.update({f"{index}.{name}": value for name, value in batch_statistics.items()})
        return (output, statistics) if defer_statistics else output


class ResNet50(tf.keras.Model):
    """Complete ResNet50-V1.5; stage names remain visible to TorchLens."""

    def _splitfleet_replica_contract(self):
        return (type(self), self.device, self.training, self.defer_statistics,
                bool(getattr(self, '_splitfleet_training', self.training)),
                tuple((type(layer), layer.get_config())
                      for layer in self._flatten_layers(include_self=False)))

    def state_associations(self):
        return batch_norm_associations()

    def __init__(self, *, device: str, defer_statistics: bool = False):
        super().__init__(name="resnet50", dtype="float32")
        self.device = device
        self.training = False
        self.defer_statistics = bool(defer_statistics)
        self._splitfleet_tensorflow_compiled_updates = True
        object.__setattr__(self, '_splitfleet_statistics_functions', {})
        self.input_spec = tf.keras.layers.InputSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3))
        self.pad1 = tf.keras.layers.ZeroPadding2D(3, name="pad1", dtype="float32")
        self.conv1 = _conv(64, 7, stride=2, name="conv1")
        self.bn1 = CanonicalBatchNorm(64, name="bn1")
        self.relu = tf.keras.layers.ReLU(name="relu", dtype="float32")
        self.maxpool = _SymmetricMaxPool()
        input_channels = 64
        for stage, (blocks, planes) in enumerate(zip(BLOCKS, PLANES, strict=True), 1):
            setattr(self, f"layer{stage}", _Stage(
                input_channels, planes, blocks, 1 if stage == 1 else 2, name=f"layer{stage}",
            ))
            input_channels = planes * 4
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D(
            data_format="channels_last", name="avgpool", dtype="float32",
        )
        self.fc = tf.keras.layers.Dense(
            NUM_CLASSES, use_bias=True, kernel_initializer="zeros", bias_initializer="zeros",
            name="fc", dtype="float32",
        )

    def build(self, input_shape):
        if tuple(input_shape[1:]) != (INPUT_SIZE, INPUT_SIZE, 3):
            raise ValueError("ResNet50 requires NHWC inputs with shape Bx64x64x3")
        with tf.device(self.device):
            padded_shape = self.pad1.compute_output_shape(input_shape)
            self.conv1.build(padded_shape)
            self.bn1.build(self.conv1.compute_output_shape(padded_shape))
            shape = (input_shape[0], 16, 16, 64)
            for stage in range(1, 5):
                layer = getattr(self, f"layer{stage}")
                layer.build(shape)
                shape = layer.compute_output_shape(shape)
            self.fc.build((input_shape[0], shape[-1]))
        super().build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = getattr(self, "_splitfleet_training", self.training)
        with tf.device(self.device):
            output = self.bn1(self.conv1(self.pad1(inputs)), training=training,
                              defer_statistics=self.defer_statistics)
            statistics = {}
            if self.defer_statistics:
                output, batch_statistics = output
                if batch_statistics:
                    statistics["bn1"] = batch_statistics
            output = self.maxpool(self.relu(output))
            for stage in range(1, 5):
                output = getattr(self, f"layer{stage}")(
                    output, training=training, defer_statistics=self.defer_statistics,
                )
                if self.defer_statistics:
                    output, batch_statistics = output
                    statistics.update({f"layer{stage}.{name}": value
                                       for name, value in batch_statistics.items()})
            logits = self.fc(self.avgpool(output))
            return {"logits": logits, "batch_statistics": statistics} if self.defer_statistics else logits

    def __deepcopy__(self, memo: dict[int, Any]):
        clone = build_model(export_state(self), device=self.device,
                            defer_statistics=self.defer_statistics)
        memo[id(self)] = clone
        clone.training = bool(self.training)
        if hasattr(self, "_splitfleet_training"):
            clone._splitfleet_training = bool(self._splitfleet_training)
        clone._splitfleet_tensorflow_function_forward = bool(
            getattr(self, '_splitfleet_tensorflow_function_forward', False))
        return clone

    def batch_norm_layers(self) -> dict[str, CanonicalBatchNorm]:
        layers = {"bn1": self.bn1}
        for stage in range(1, 5):
            for index, block in enumerate(getattr(self, f"layer{stage}").blocks):
                prefix = f"layer{stage}.{index}"
                for part in range(1, 4):
                    layers[f"{prefix}.bn{part}"] = getattr(block, f"bn{part}")
                if block.downsample is not None:
                    layers[prefix + ".downsample.1"] = block.downsample.bn
        return layers

    def apply_statistics(self, statistics: Mapping[str, Any], names=None) -> None:
        """Commit a real batch once, optionally for only stage-owned BN names.

        ``names`` contains canonical BN prefixes, such as ``layer2.0.bn1``.
        Collection/capture/profiling must never invoke this commit operation.
        """
        layers = self.batch_norm_layers()
        selected = set(statistics) if names is None else set(names)
        unknown = set(statistics) - set(layers)
        if unknown or selected - set(statistics):
            raise ValueError(f"Unknown or missing BatchNorm statistics: {sorted(unknown | (selected - set(statistics)))}")
        if not selected:
            return
        key = tuple(prefix for prefix in layers if prefix in selected)

        def commit(values):
            with tf.device(self.device):
                for prefix in key:
                    layers[prefix].apply_statistics(values[prefix])

        if tf.inside_function():
            commit(statistics)
            return
        layout = tuple((name, layers[name].channels, layers[name].retention) for name in key)
        cache_key = self.device, layout
        function = self._splitfleet_statistics_functions.get(cache_key)
        if function is None:
            with _STATISTICS_FUNCTION_LOCK:
                function = _statistics_function(*cache_key)
            self._splitfleet_statistics_functions[cache_key] = function
        # TensorFlow traces Variable arguments by their type/shape and alias
        # pattern. Replicas use the same program with their own live resources;
        # neither the cache nor the function closes over a client model.
        bindings = tuple((layers[name].running_mean.value, layers[name].running_var.value,
                          layers[name].num_batches_tracked.value) for name in key)
        from splitfleet.runtime.tensorflow_stack_scope import call_tensorflow_function
        call_tensorflow_function(function, bindings, {name: statistics[name] for name in key})

    def canonical_variables(self) -> dict[str, Any]:
        """Return each native variable once, in canonical schema order."""
        variables: dict[str, Any] = {"conv1.weight": self.conv1.kernel}

        def batch_norm(prefix, layer):
            variables.update({
                prefix + ".weight": layer.gamma,
                prefix + ".bias": layer.beta,
                prefix + ".running_mean": layer.running_mean,
                prefix + ".running_var": layer.running_var,
                prefix + ".num_batches_tracked": layer.num_batches_tracked,
            })

        batch_norm("bn1", self.bn1)
        for stage in range(1, 5):
            for index, block in enumerate(getattr(self, f"layer{stage}").blocks):
                prefix = f"layer{stage}.{index}"
                for part in range(1, 4):
                    variables[f"{prefix}.conv{part}.weight"] = getattr(block, f"conv{part}").kernel
                    batch_norm(f"{prefix}.bn{part}", getattr(block, f"bn{part}"))
                if block.downsample is not None:
                    variables[prefix + ".downsample.0.weight"] = block.downsample.layers[0].kernel
                    batch_norm(prefix + ".downsample.1", block.downsample.layers[1])
        variables["fc.weight"] = self.fc.kernel
        variables["fc.bias"] = self.fc.bias
        return variables


def build_model(state: Mapping[str, np.ndarray], device: str = "cpu", *,
                defer_statistics: bool = False) -> ResNet50:
    """Build a full native model and load every parameter and BN buffer."""
    validate_state(state)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    resolved = _resolve_device(device)
    with tf.device(resolved):
        model = ResNet50(device=resolved, defer_statistics=defer_statistics)
        model.build((None, INPUT_SIZE, INPUT_SIZE, 3))
        variables = model.canonical_variables()
        for row in SCHEMA:
            value = np.asarray(state[row.name])
            if len(row.shape) == 4:
                value = value.transpose(2, 3, 1, 0)
            elif row.name == "fc.weight":
                value = value.T
            variables[row.name].assign(value)
    count = sum(int(np.prod(variable.shape)) for variable in model.trainable_variables)
    if count != TRAINABLE_PARAMETERS:
        raise AssertionError(f"Unexpected Keras ResNet50 trainable parameter count: {count}")
    return model


def export_state(model: ResNet50) -> dict[str, np.ndarray]:
    """Export copied OIHW/linear state, including scalar int64 BN counters."""
    variables = model.canonical_variables()
    state: dict[str, np.ndarray] = {}
    for row in SCHEMA:
        value = np.asarray(variables[row.name].numpy())
        if len(row.shape) == 4:
            value = value.transpose(3, 2, 0, 1)
        elif row.name == "fc.weight":
            value = value.T
        # np.array preserves zero-dimensional counters, unlike ascontiguousarray.
        state[row.name] = np.array(value, copy=True, order="C")
    validate_state(state)
    return state
