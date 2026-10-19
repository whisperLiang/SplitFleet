"""Complete native Tinygrad ResNet50-V1.5 with canonical, owned BN state."""

import numpy as np
from tinygrad import Tensor, dtypes

from experiments.resnet50_quality.model_contract import (
    BLOCKS, INPUT_SIZE, SCHEMA, batch_norm_associations, validate_state,
)


class ResNet50:
    def __init__(self, state, device="cpu"):
        self.device = device.upper()
        self.training = False
        self.defer_statistics = True
        self.state = {}
        self.load_state(state)

    def load_state(self, state):
        validate_state(state)
        for row in SCHEMA:
            value = Tensor(np.array(state[row.name], copy=True), device=self.device).realize()
            if row.name in self.state:
                self.state[row.name].replace(value)
            else:
                self.state[row.name] = value.is_param_(row.trainable)

    def state_tensors(self):
        return dict(self.state)

    def state_associations(self):
        return batch_norm_associations()

    def train(self, training=True):
        self.training = bool(training)
        return self

    def _conv(self, images, name, *, stride=1, padding=0):
        return images.conv2d(self.state[name + ".weight"], stride=stride, padding=padding)

    def _batch_norm(self, images, name, statistics):
        if self.training:
            # Derive the count from the native batch so replay can change B.
            observations = images[:, :1]
            count = ((observations != 0).cast(dtypes.int64)
                     + (observations == 0).cast(dtypes.int64)).sum()
            denominator = count.cast(dtypes.float32)
            mean = images.sum((0, 2, 3)) / denominator
            centered = images - mean.reshape(1, -1, 1, 1)
            variance = centered.square().sum((0, 2, 3)) / denominator
            statistics[name] = {"mean": mean.detach(), "variance": variance.detach(), "count": count}
        else:
            mean, variance = self.state[name + ".running_mean"], self.state[name + ".running_var"]
            centered = images - mean.reshape(1, -1, 1, 1)
        scale = self.state[name + ".weight"] * (variance + 1e-5).rsqrt()
        return centered * scale.reshape(1, -1, 1, 1) + self.state[name + ".bias"].reshape(1, -1, 1, 1)

    def __call__(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE) or images.shape[0] < 1:
            raise ValueError("ResNet50 requires nonempty NCHW batches shaped Bx3x64x64")
        if images.dtype != dtypes.float32:
            raise TypeError("ResNet50 requires FP32 images")
        statistics = {}
        value = self._batch_norm(self._conv(images, "conv1", stride=2, padding=3), "bn1", statistics).relu()
        value = value.pad((1, 1, 1, 1), value=float("-inf")).max_pool2d(3, stride=2)
        for stage, blocks in enumerate(BLOCKS, 1):
            for index in range(blocks):
                name = f"layer{stage}.{index}"
                stride = 2 if stage > 1 and index == 0 else 1
                identity = value
                hidden = self._batch_norm(self._conv(value, name + ".conv1"), name + ".bn1", statistics).relu()
                hidden = self._batch_norm(self._conv(hidden, name + ".conv2", stride=stride, padding=1),
                                          name + ".bn2", statistics).relu()
                hidden = self._batch_norm(self._conv(hidden, name + ".conv3"), name + ".bn3", statistics)
                if name + ".downsample.0.weight" in self.state:
                    identity = self._batch_norm(self._conv(identity, name + ".downsample.0", stride=stride),
                                                name + ".downsample.1", statistics)
                value = (hidden + identity).relu()
        logits = value.mean((2, 3)).linear(self.state["fc.weight"].T, self.state["fc.bias"])
        return {"logits": logits, "batch_statistics": statistics}

    def apply_statistics(self, statistics, names=None):
        selected = {name: values for name, values in statistics.items()
                    if names is None or name in names}
        updates = {}
        for name, values in selected.items():
            mean, variance, count = (values[key].numpy() for key in ("mean", "variance", "count"))
            expected = tuple(self.state[name + ".running_mean"].shape)
            if mean.shape != expected or variance.shape != expected or mean.dtype != np.float32 or variance.dtype != np.float32:
                raise ValueError(f"Invalid BatchNorm statistics for {name}")
            if count.shape != () or count.dtype != np.int64 or int(count) <= 1:
                raise ValueError(f"Invalid BatchNorm count for {name}")
            if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0).any():
                raise FloatingPointError(f"Nonfinite or negative BatchNorm statistics for {name}")
            for part, batch in (("running_mean", mean), ("running_var", variance * np.float32(int(count) / (int(count) - 1)))):
                key = name + "." + part
                updates[key] = np.float32(.9) * self.state[key].numpy() + np.float32(.1) * batch
            key = name + ".num_batches_tracked"
            updates[key] = self.state[key].numpy() + np.int64(1)
        for name, value in updates.items():
            self.state[name].replace(Tensor(value, device=self.device).realize())


def build_model(state, device="cpu"):
    from splitfleet.runtime.tinygrad_precision import configure_device
    configure_device(device.upper())
    return ResNet50(state, device)


def export_state(model):
    state = {name: value.numpy().copy() for name, value in model.state_tensors().items()}
    validate_state(state)
    return state
