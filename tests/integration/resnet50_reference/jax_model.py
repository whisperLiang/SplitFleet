"""Native functional FP32 JAX ResNet50-V1.5 with deferred BatchNorm state.

Only ``params`` is differentiated.  Training forwards return population batch
statistics as auxiliary outputs; the owner commits them with
``apply_statistics`` once the actual minibatch has completed.  No forward or
gradient tracing operation mutates running statistics or batch counters.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from torchlens.backends.jax.modules import encode_module_scope

from experiments.resnet50_quality.model_contract import (
    BLOCKS,
    INPUT_SIZE,
    PLANES,
    SCHEMA,
    TRAINABLE_NAMES,
    TRAINABLE_PARAMETERS,
    batch_norm_associations,
    validate_state,
)

# JAX otherwise truncates the canonical int64 BN counters.  All floating
# arrays, reductions, coefficients and kernel accumulators remain explicit FP32.
jax.config.update("jax_enable_x64", True)

_PARAMETER_SCHEMA = {row.name: row for row in SCHEMA if row.trainable}
_BN_CHANNELS = {
    row.name.removesuffix(".running_mean"): row.shape[0]
    for row in SCHEMA if row.name.endswith(".running_mean")
}


def _resolve_device(device: str) -> tuple[str, Any]:
    requested = str(device).strip().lower()
    if requested in {"cpu", "cpu:0"}:
        backend, index, canonical = "cpu", 0, "cpu"
    elif requested.startswith(("cuda:", "gpu:")):
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid JAX device {device!r}") from exc
        if index < 0:
            raise ValueError(f"Invalid JAX device {device!r}")
        backend, canonical = "gpu", f"cuda:{index}"
    else:
        raise ValueError(f"Unsupported JAX device {device!r}")
    try:
        devices = jax.devices(backend)
    except RuntimeError as exc:
        raise RuntimeError(f"Requested JAX device {canonical!r} is unavailable") from exc
    if index >= len(devices):
        raise RuntimeError(f"Requested JAX device {canonical!r} is unavailable")
    return canonical, devices[index]


def _conv(images, weight, *, stride=1, padding=0):
    return jax.lax.conv_general_dilated(
        images, weight, window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )


def _relu(images):
    # Select gives the same zero derivative at x=0 as Torch ReLU.  The JAX
    # maximum primitive's equal-input derivative would instead be one half.
    return jax.lax.select(images > np.float32(0), images, jnp.zeros_like(images))


def _channel(value, shape):
    """Keep the complete batch/spatial broadcast explicit for native replay.

    Rewriting a singleton batch broadcast to B changes its transpose from one
    joint reduction to separate spatial/batch reductions.  A full activation
    shape gives native execution and dynamic replay the same FP32 reduction.
    """
    return jax.lax.broadcast_in_dim(value, shape, (1,))


class ResNet50:
    """Callable ``model(params, NCHW_images)`` with canonical OIHW parameters.

    ``training`` controls only which statistics normalize the activations.
    Evaluation reads the current ``buffers`` on every untransformed call.
    An evaluation JIT wrapper must be retraced after replacing buffers, since
    JAX closes over these external arrays at trace time.
    """

    def __init__(self, state: Mapping[str, np.ndarray], *, device: str):
        self._splitfleet_jax_training_program_cache = True
        self.device, self.jax_device = _resolve_device(device)
        self.training = False
        self.params: dict[str, Any] = {}
        self.buffers: dict[str, Any] = {}
        self.batch_norm_names = tuple(_BN_CHANNELS)
        self.load_state(state)
        self.trainable_parameter_count = sum(int(value.size) for value in self.params.values())
        if self.trainable_parameter_count != TRAINABLE_PARAMETERS:
            raise AssertionError("Unexpected JAX ResNet50 trainable parameter count")

    def load_state(self, state: Mapping[str, np.ndarray]) -> None:
        """Load all canonical state while retaining both host dict identities."""
        validate_state(state)
        native = {
            row.name: jax.device_put(np.array(state[row.name], copy=True), self.jax_device)
            for row in SCHEMA
        }
        self.params.update({row.name: native[row.name] for row in SCHEMA if row.trainable})
        self.buffers.update({row.name: native[row.name] for row in SCHEMA if not row.trainable})

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Replace parameter leaves in place so previously bound inputs stay live."""
        if set(params) != set(TRAINABLE_NAMES):
            raise ValueError("JAX trainable parameter names differ from the canonical schema")
        for name, row in _PARAMETER_SCHEMA.items():
            value = params[name]
            if isinstance(value, jax.core.Tracer):
                raise RuntimeError("set_params must run outside a JAX transformation")
            if tuple(value.shape) != row.shape or np.dtype(value.dtype) != np.dtype("float32"):
                raise ValueError(f"JAX parameter shape or dtype differs at {name}")
        self.params.update({name: jax.device_put(params[name], self.jax_device) for name in TRAINABLE_NAMES})

    def state_tensors(self) -> dict[str, Any]:
        """Expose native parameters, float buffers and int64 counters in SCHEMA order."""
        return {
            row.name: (self.params if row.trainable else self.buffers)[row.name]
            for row in SCHEMA
        }

    def state_associations(self):
        return batch_norm_associations()

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def _batch_norm(self, images, params, name, statistics, *, training):
        if training:
            samples = images.shape[0] * images.shape[2] * images.shape[3]
            if samples <= 1:
                raise ValueError("BatchNorm training requires more than one value per channel")
            count = jnp.sum(jnp.ones_like(images[:, :1, :, :], dtype=jnp.int64), dtype=jnp.int64)
            denominator = count.astype(jnp.float32)
            # Retain m in native replay rather than capture a B=1 reciprocal.
            mean = jnp.sum(images, axis=(0, 2, 3), dtype=jnp.float32) / denominator
            centered = images - _channel(mean, images.shape)
            variance = jnp.sum(centered * centered, axis=(0, 2, 3), dtype=jnp.float32) / denominator
            statistics[name] = {
                # has_aux keeps state effects outside the differentiated CE.
                # A stop_gradient alias here can rebind a live normalization
                # operand during native eager capture and erase its derivative.
                "mean": mean,
                "variance": variance,
                # A native shaped reduction lets replay rewrite the batch
                # extent for a final short minibatch.  A scalar m literal
                # would retain the batch size used to capture the graph.
                "count": count,
            }
        else:
            mean = self.buffers[name + ".running_mean"]
            variance = self.buffers[name + ".running_var"]
            centered = images - _channel(mean, images.shape)
        scale = params[name + ".weight"] * jax.lax.rsqrt(variance + np.float32(1e-5))
        return centered * _channel(scale, images.shape) + _channel(params[name + ".bias"], images.shape)

    def __call__(self, params: Mapping[str, Any], images):
        if tuple(images.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE) or images.ndim != 4:
            raise ValueError("ResNet50 requires NCHW inputs with shape Bx3x64x64")
        if images.shape[0] < 1:
            raise ValueError("ResNet50 requires a nonempty batch")
        if np.dtype(images.dtype) != np.dtype("float32"):
            raise TypeError("JAX ResNet50 requires FP32 images")
        training = bool(getattr(self, "_splitfleet_training", self.training))
        statistics: dict[str, dict[str, Any]] = {}
        with jax.named_scope(encode_module_scope("stem")):
            output = _conv(images, params["conv1.weight"], stride=2, padding=3)
            output = _relu(self._batch_norm(output, params, "bn1", statistics, training=training))
            output = jax.lax.reduce_window(
                # A host scalar selects reduce_window_max.  Creating a JAX
                # scalar here selects generic reduce_window during tracing,
                # whose max reducer lacks reverse-mode support in JAX 0.6.2.
                output, np.float32(-np.inf), jax.lax.max,
                window_dimensions=(1, 1, 3, 3), window_strides=(1, 1, 2, 2),
                padding=((0, 0), (0, 0), (1, 1), (1, 1)),
            )
        for stage, (blocks, _) in enumerate(zip(BLOCKS, PLANES, strict=True), 1):
            with jax.named_scope(encode_module_scope(f"layer{stage}")):
                for index in range(blocks):
                    prefix = f"layer{stage}.{index}"
                    stride = 2 if stage > 1 and index == 0 else 1
                    with jax.named_scope(encode_module_scope(prefix)):
                        identity = output
                        hidden = _conv(output, params[prefix + ".conv1.weight"])
                        hidden = _relu(self._batch_norm(hidden, params, prefix + ".bn1", statistics, training=training))
                        hidden = _conv(hidden, params[prefix + ".conv2.weight"], stride=stride, padding=1)
                        hidden = _relu(self._batch_norm(hidden, params, prefix + ".bn2", statistics, training=training))
                        hidden = _conv(hidden, params[prefix + ".conv3.weight"])
                        hidden = self._batch_norm(hidden, params, prefix + ".bn3", statistics, training=training)
                        if prefix + ".downsample.0.weight" in params:
                            identity = _conv(identity, params[prefix + ".downsample.0.weight"], stride=stride)
                            identity = self._batch_norm(identity, params, prefix + ".downsample.1", statistics, training=training)
                        output = _relu(hidden + identity)
        with jax.named_scope(encode_module_scope("head")):
            pooled = jnp.mean(output, axis=(2, 3), dtype=jnp.float32)
            logits = jax.lax.dot_general(
                pooled, params["fc.weight"], dimension_numbers=(((1,), (1,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
            ) + params["fc.bias"]
        return {"logits": logits, "batch_statistics": statistics}

    def apply_statistics(self, stats: Mapping[str, Mapping[str, Any]], names: Iterable[str] | None = None) -> None:
        """Commit one batch's BN effects for all or selected canonical BN prefixes.

        ``variance`` is the population batch variance; ``count`` is N*H*W at
        that BN.  Running variance receives its unbiased m/(m-1) correction.
        Prefix and suffix owners can commit disjoint subsets of one output.
        Every explicitly selected BN prefix must have supplied statistics.
        This host operation must be called outside gradient/JIT tracing.
        """
        unknown = set(stats) - set(_BN_CHANNELS)
        if unknown:
            raise ValueError(f"Unknown BatchNorm statistics: {sorted(unknown)}")
        selected = set(stats) if names is None else ({names} if isinstance(names, str) else set(names))
        unknown = selected - set(_BN_CHANNELS)
        if unknown:
            raise ValueError(f"Unknown BatchNorm names: {sorted(unknown)}")
        missing = selected - set(stats)
        if missing:
            raise ValueError(f"Missing BatchNorm statistics: {sorted(missing)}")
        selected_stats = {name: stats[name] for name in self.batch_norm_names if name in selected}
        if any(isinstance(value, jax.core.Tracer) for values in selected_stats.values() for value in values.values()):
            raise RuntimeError("apply_statistics must run outside a JAX transformation")
        host_stats = jax.device_get(selected_stats)
        # Validate the complete selected batch before committing any state.
        for name, values in host_stats.items():
            if set(values) != {"mean", "variance", "count"}:
                raise ValueError(f"Incomplete BatchNorm statistics for {name}")
            for key in ("mean", "variance"):
                value = np.asarray(values[key])
                if value.shape != (_BN_CHANNELS[name],) or value.dtype != np.float32:
                    raise ValueError(f"BatchNorm {key} shape or dtype differs at {name}")
                if not np.isfinite(value).all() or (key == "variance" and (value < 0).any()):
                    raise FloatingPointError(f"Invalid BatchNorm {key} at {name}")
            count = np.asarray(values["count"])
            if count.shape != () or count.dtype.kind not in {"i", "u"} or int(count) <= 1:
                raise ValueError(f"BatchNorm count must be an integer greater than one at {name}")
        updates = {}
        for name, values in selected_stats.items():
            count = int(np.asarray(host_stats[name]["count"]))
            mean = jax.device_put(values["mean"], self.jax_device)
            variance = jax.device_put(values["variance"], self.jax_device)
            unbiased = variance * np.float32(count / (count - 1))
            updates[name + ".running_mean"] = np.float32(0.9) * self.buffers[name + ".running_mean"] + np.float32(0.1) * mean
            updates[name + ".running_var"] = np.float32(0.9) * self.buffers[name + ".running_var"] + np.float32(0.1) * unbiased
            updates[name + ".num_batches_tracked"] = self.buffers[name + ".num_batches_tracked"] + np.int64(1)
        self.buffers.update(updates)

    def __deepcopy__(self, memo):
        clone = build_model(export_state(self), device=self.device)
        memo[id(self)] = clone
        clone.training = bool(self.training)
        if hasattr(self, "_splitfleet_training"):
            clone._splitfleet_training = bool(self._splitfleet_training)
        return clone


def build_model(state: Mapping[str, np.ndarray], device: str = "cpu") -> ResNet50:
    return ResNet50(state, device=device)


def export_state(model: ResNet50) -> dict[str, np.ndarray]:
    """Export copied canonical arrays, retaining scalar int64 BN counters."""
    native = jax.device_get(model.state_tensors())
    state = {row.name: np.array(native[row.name], copy=True, order="C") for row in SCHEMA}
    validate_state(state)
    return state
