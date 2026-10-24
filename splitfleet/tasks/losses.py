"""Lazy framework dispatch for standard supervised task losses."""

from __future__ import annotations

from typing import Any, Mapping


def tensor_backend(value: Any) -> str:
    module = type(value).__module__.lower()
    for prefix, name in (("torch", "torch"), ("tensorflow", "tf"), ("keras", "tf"),
                         ("paddle", "paddle")):
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
        if logits.ndim > 2:
            # Spatial CUDA NLL loss uses a nondeterministic reduction. Treat
            # each spatial position as one sample without changing sparse CE
            # or its ignore-index normalization, using the deterministic 2D path.
            classes = logits.shape[1]
            logits = logits.movedim(1, -1).reshape(-1, classes)
            labels = labels.reshape(-1)
        return cross_entropy(logits, labels, ignore_index=ignore_index, reduction="sum") / valid_count
    if backend == "paddle":
        from paddle.nn.functional import cross_entropy
        return cross_entropy(logits, targets.astype("int64"), axis=1, ignore_index=ignore_index)
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
    raise AssertionError(backend)
