"""Lazy framework dispatch for standard supervised task losses."""

from __future__ import annotations

from typing import Any, Mapping


def tensor_backend(value: Any) -> str:
    module = type(value).__module__.lower()
    for prefix, name in (("torch", "torch"), ("tensorflow", "tf"), ("keras", "tf"),
                         ("jax", "jax"), ("jaxlib", "jax"), ("paddle", "paddle"), ("tinygrad", "tinygrad")):
        if module.startswith(prefix):
            return name
    raise TypeError(f"Unsupported tensor backend for {type(value).__name__}; supply a custom loss_fn.")


def extract_logits(outputs: Any, key: str = "logits") -> Any:
    if isinstance(outputs, Mapping):
        if key not in outputs:
            raise ValueError(f"Model output has no {key!r} field.")
        return outputs[key]
    if hasattr(outputs, key):
        return getattr(outputs, key)
    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise ValueError("Model returned no logits.")
        return outputs[0]
    return outputs


def sparse_cross_entropy(logits: Any, targets: Any, *, ignore_index: int = -100) -> Any:
    """Mean sparse CE for [N, C, ...] logits, preserving the native autograd graph."""
    backend = tensor_backend(logits)
    if targets is None:
        raise ValueError("Supervised cross entropy requires targets.")
    if backend == "torch":
        from torch.nn.functional import cross_entropy
        labels = targets.long()
        valid_count = (labels != ignore_index).sum().clamp_min(1)
        return cross_entropy(logits, labels, ignore_index=ignore_index, reduction="sum") / valid_count
    if backend == "paddle":
        from paddle.nn.functional import cross_entropy
        return cross_entropy(logits, targets.astype("int64"), axis=1, ignore_index=ignore_index)
    if backend == "jax":
        import jax.numpy as jnp
        from jax.nn import log_softmax
        labels = jnp.asarray(targets, dtype=jnp.int32)
        mask = labels != ignore_index
        safe_labels = jnp.where(mask, labels, 0)
        scores = jnp.moveaxis(log_softmax(logits, axis=1), 1, -1)
        losses = -jnp.take_along_axis(scores, safe_labels[..., None], axis=-1)[..., 0]
        return jnp.sum(jnp.where(mask, losses, 0)) / jnp.maximum(jnp.sum(mask), 1)
    if backend == "tf":
        import tensorflow as tf
        labels = tf.cast(targets, tf.int64)
        mask = tf.not_equal(labels, ignore_index)
        rank = len(logits.shape)
        scores = tf.transpose(logits, [0, *range(2, rank), 1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.where(mask, labels, tf.zeros_like(labels)), logits=scores,
        )
        return tf.math.divide_no_nan(tf.reduce_sum(tf.where(mask, losses, 0)),
                                     tf.reduce_sum(tf.cast(mask, losses.dtype)))
    if backend == "tinygrad":
        order = (0, *range(2, len(logits.shape)), 1)
        return logits.permute(*order).sparse_categorical_crossentropy(targets, ignore_index=ignore_index)
    raise AssertionError(backend)
