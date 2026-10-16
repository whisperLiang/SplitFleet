"""Native FP32 Paddle ResNet50-V1.5 with the portable state schema.

Paddle's stock BatchNorm updates its running variance with population variance.
The explicit layer below instead follows the experiment's unbiased running
variance rule, while retaining population variance in the training forward.
``defer_statistics`` exposes those updates as native auxiliary tensor outputs
for split replay to commit once after the actual batch, without buffer mutation
during capture. Both modes train BatchNorm's affine parameters normally.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from experiments.resnet50_quality.model_contract import (
    BLOCKS,
    NAMES,
    NUM_CLASSES,
    PLANES,
    batch_norm_associations,
    validate_state,
)


class CanonicalBatchNorm2D(nn.Layer):
    """Trainable NCHW BatchNorm with explicit, persistent canonical buffers."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.epsilon = 1e-5
        self.retention = 0.9
        self._statistics_sink = None
        self._statistics_name = None
        self.weight = self.create_parameter(
            [channels], dtype="float32", default_initializer=nn.initializer.Constant(1.0)
        )
        self.bias = self.create_parameter(
            [channels], dtype="float32", is_bias=True,
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.register_buffer("running_mean", paddle.zeros([channels], dtype="float32"))
        self.register_buffer("running_var", paddle.ones([channels], dtype="float32"))
        self.register_buffer("num_batches_tracked", paddle.zeros([], dtype="int64"))

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError("CanonicalBatchNorm2D requires NCHW input with matching channels")
        if inputs.dtype != paddle.float32:
            raise TypeError("CanonicalBatchNorm2D requires FP32 input")

        # Channel vectors broadcast directly in NHWC. This avoids ambiguous
        # leading singleton BN-statistic shapes in native dynamic replay.
        value = paddle.transpose(inputs, [0, 2, 3, 1])

        if self.training:
            observations = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
            if observations <= 1:
                raise ValueError("Training BatchNorm requires more than one observation per channel")
            mean = paddle.mean(value, axis=[0, 1, 2])
            variance = paddle.mean(paddle.square(value - mean), axis=[0, 1, 2])
            statistics = {
                "mean": mean,
                "variance": variance,
            }
            if self._statistics_sink is None:
                self.apply_statistics({**statistics, "count": observations})
            else:
                # A tensor-derived count also works when replay changes B. A
                # Python literal from the capture batch would update variance
                # incorrectly on the final, smaller minibatch.
                statistics["count"] = paddle.sum(
                    paddle.ones_like(inputs[:, :1, :, :], dtype="int64")
                )
                self._statistics_sink[self._statistics_name] = statistics
        else:
            mean = self.running_mean
            variance = self.running_var

        normalized = (value - mean) / paddle.sqrt(variance + self.epsilon)
        return paddle.transpose(normalized * self.weight + self.bias, [0, 3, 1, 2])

    def validate_statistics(self, statistics):
        mean = statistics["mean"]
        variance = statistics["variance"]
        if not isinstance(mean, paddle.Tensor) or not isinstance(variance, paddle.Tensor):
            raise TypeError("Paddle BatchNorm statistics must be native tensors")
        if tuple(mean.shape) != (self.channels,) or tuple(variance.shape) != (self.channels,):
            raise ValueError("Paddle BatchNorm statistics have incorrect channel shapes")
        if mean.dtype != paddle.float32 or variance.dtype != paddle.float32:
            raise TypeError("Paddle BatchNorm statistics must be FP32")
        count = statistics["count"]
        if isinstance(count, paddle.Tensor):
            if tuple(count.shape) != () or count.dtype != paddle.int64:
                raise TypeError("Paddle BatchNorm count must be a scalar int64 tensor")
            count = int(count.item())
        elif not isinstance(count, (int, np.integer)):
            raise TypeError("Paddle BatchNorm count must be an integer")
        if count <= 1:
            raise ValueError("Training BatchNorm requires more than one observation per channel")
        return mean, variance, count

    def apply_statistics(self, statistics):
        """Commit one actual training batch while preserving all buffer objects."""
        mean, variance, count = self.validate_statistics(statistics)
        with paddle.no_grad():
            self.running_mean.set_value(
                self.retention * self.running_mean + (1.0 - self.retention) * mean.detach()
            )
            self.running_var.set_value(
                self.retention * self.running_var
                + (1.0 - self.retention) * variance.detach() * (count / (count - 1))
            )
            self.num_batches_tracked.set_value(self.num_batches_tracked + 1)


def _conv(in_channels: int, out_channels: int, kernel_size: int, *, stride: int = 1, padding: int = 0):
    return nn.Conv2D(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        bias_attr=False, dtype="float32", data_format="NCHW",
        weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0)),
    )


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1):
        super().__init__()
        out_channels = planes * self.expansion
        self.conv1 = _conv(in_channels, planes, 1)
        self.bn1 = CanonicalBatchNorm2D(planes)
        self.conv2 = _conv(planes, planes, 3, stride=stride, padding=1)
        self.bn2 = CanonicalBatchNorm2D(planes)
        self.conv3 = _conv(planes, out_channels, 1)
        self.bn3 = CanonicalBatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Sequential(
                _conv(in_channels, out_channels, 1, stride=stride),
                CanonicalBatchNorm2D(out_channels),
            )
            if stride != 1 or in_channels != out_channels else None
        )
        self.stride = stride

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        identity = inputs if self.downsample is None else self.downsample(inputs)
        value = self.relu(self.bn1(self.conv1(inputs)))
        value = self.relu(self.bn2(self.conv2(value)))
        value = self.bn3(self.conv3(value))
        # Out-of-place addition also fits Paddle's capture conventions.
        return self.relu(value + identity)


class CanonicalLinear(nn.Layer):
    """Paddle linear execution with weight stored in canonical [out, in] order."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = self.create_parameter(
            [out_features, in_features], dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.bias = self.create_parameter(
            [out_features], dtype="float32", is_bias=True,
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        return F.linear(inputs, paddle.transpose(self.weight, [1, 0]), self.bias)


class ResNet50(nn.Layer):
    """Full [3, 4, 6, 3] Bottleneck network with ten output logits."""

    def state_associations(self):
        return batch_norm_associations()

    def __init__(self):
        super().__init__()
        self.defer_statistics = False
        self.conv1 = _conv(3, 64, 7, stride=2, padding=3)
        self.bn1 = CanonicalBatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        channels = 64
        for stage, (blocks, planes) in enumerate(zip(BLOCKS, PLANES, strict=True), 1):
            sequence = [Bottleneck(channels, planes, stride=1 if stage == 1 else 2)]
            channels = planes * Bottleneck.expansion
            sequence.extend(Bottleneck(channels, planes) for _ in range(1, blocks))
            self.add_sublayer(f"layer{stage}", nn.Sequential(*sequence))
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc = CanonicalLinear(channels, NUM_CLASSES)
        for name, layer in self.named_sublayers():
            if isinstance(layer, CanonicalBatchNorm2D):
                layer._statistics_name = name

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor | dict[str, object]:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 64, 64) or inputs.shape[0] <= 0:
            raise ValueError("ResNet50 requires nonempty NCHW batches shaped [B, 3, 64, 64]")
        if inputs.dtype != paddle.float32:
            raise TypeError("ResNet50 requires FP32 input")
        statistics = {}
        norms = [layer for layer in self.sublayers() if isinstance(layer, CanonicalBatchNorm2D)]
        if self.defer_statistics and self.training:
            for layer in norms:
                layer._statistics_sink = statistics
        try:
            value = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
            value = self.layer1(value)
            value = self.layer2(value)
            value = self.layer3(value)
            value = self.layer4(value)
            logits = self.fc(paddle.flatten(self.avgpool(value), start_axis=1))
        finally:
            for layer in norms:
                layer._statistics_sink = None
        if self.defer_statistics:
            return {"logits": logits, "batch_statistics": statistics}
        return logits

    def apply_statistics(self, statistics, names=None):
        """Commit selected canonical BN prefixes owned by one executed stage.

        Split execution must call this once per batch for each owning BN layer,
        after executing its native training operation. Captures and probes may
        discard these auxiliary outputs without advancing running statistics.
        """
        norms = {
            name: layer for name, layer in self.named_sublayers()
            if isinstance(layer, CanonicalBatchNorm2D)
        }
        selected = set(statistics) if names is None else set(names)
        if not set(statistics).issubset(norms) or not selected.issubset(norms):
            raise ValueError("Unknown canonical Paddle BatchNorm statistics prefix")
        if not selected.issubset(statistics):
            raise ValueError("Missing canonical Paddle BatchNorm statistics for selected layers")
        # Validate the complete selected update before mutating any buffer.
        for name in selected:
            norms[name].validate_statistics(statistics[name])
        for name, layer in norms.items():
            if name in selected:
                layer.apply_statistics(statistics[name])


def build_model(state: Mapping[str, np.ndarray], device: str = "cpu") -> paddle.nn.Layer:
    """Create an independent native model on the requested native device."""
    validate_state(state)
    if device == "cpu":
        native_device = "cpu"
    elif device.startswith("cuda:") and device[5:].isdigit():
        index = int(device[5:])
        if not paddle.is_compiled_with_cuda():
            raise RuntimeError(f"Paddle {device} requires a CUDA-enabled Paddle build")
        if index >= paddle.device.cuda.device_count():
            raise RuntimeError(f"Paddle {device} is unavailable: no CUDA device is visible at index {index}")
        native_device = f"gpu:{index}"
    else:
        raise ValueError("Paddle ResNet50 device must be 'cpu' or 'cuda:<nonnegative index>'")

    # Explicit FP32 constructors avoid changing the process-wide default dtype;
    # the device guard likewise restores the caller's default device afterwards.
    with paddle.device.device_guard(native_device):
        model = ResNet50()
        model.set_state_dict({
            name: paddle.to_tensor(np.asarray(state[name]).copy()) for name in NAMES
        })
    return model


def export_state(model: paddle.nn.Layer) -> dict[str, np.ndarray]:
    """Return an independent canonical snapshot including all BatchNorm state."""
    native_state = model.state_dict()
    if set(native_state) != set(NAMES):
        raise ValueError("Paddle ResNet50 state names differ from the canonical schema")
    from splitfleet.backends.utils import native_device_scope, paddle_model_device
    with native_device_scope("paddle", paddle_model_device(model)):
        state = {
            name: np.asarray(native_state[name].detach().cpu().numpy()).copy() for name in NAMES
        }
    validate_state(state)
    return state
