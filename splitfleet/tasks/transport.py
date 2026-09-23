"""Nested task batches carried as one numeric array through Flower NumPy RPCs."""

from __future__ import annotations

import numpy as np

from splitfleet.tasks.base import ModelInputs, TaskBatch


def encode_task_batch(batch: TaskBatch, *, backend: str = "torch") -> np.ndarray:
    from splitfleet.backends import BACKEND_ADAPTERS
    from splitfleet.backends.utils import move_value
    from splitfleet.transport import encode_bundle_wire
    adapter = BACKEND_ADAPTERS.create(backend)
    tree = {"args": batch.inputs.args, "kwargs": dict(batch.inputs.kwargs),
            "targets": batch.targets, "num_examples": batch.num_examples}
    tree = move_value(tree, adapter, "cpu")
    return np.frombuffer(encode_bundle_wire(tree, backend=backend), dtype=np.uint8).copy()


def decode_task_batch(payload: np.ndarray, *, device: str = "cpu", backend: str = "torch") -> TaskBatch:
    from splitfleet.transport import decode_boundary, decode_bundle_wire
    array = np.asarray(payload)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError("Task batch payload must be a one-dimensional uint8 array.")
    raw = array.tobytes()
    if decode_boundary(raw).backend != backend:
        raise ValueError("Task batch tensor backend must match the server model backend.")
    tree = decode_bundle_wire(raw, device)
    return TaskBatch(ModelInputs(tree["args"], tree["kwargs"]), tree["targets"], tree["num_examples"])
