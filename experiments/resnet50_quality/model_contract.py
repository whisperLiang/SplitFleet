"""Canonical PyTorch ResNet50-V1.5 state for paired initializations and checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

BLOCKS = (3, 4, 6, 3)
PLANES = (64, 128, 256, 512)
NUM_CLASSES = 10
INPUT_SIZE = 64
TRAINABLE_PARAMETERS = 23_528_522


@dataclass(frozen=True)
class StateTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    trainable: bool
    stage: int


def state_schema() -> tuple[StateTensor, ...]:
    result = []

    def parameter(name, shape, stage):
        result.append(StateTensor(name, shape, "float32", True, stage))

    def batch_norm(name, channels, stage):
        parameter(name + ".weight", (channels,), stage)
        parameter(name + ".bias", (channels,), stage)
        result.extend([
            StateTensor(name + ".running_mean", (channels,), "float32", False, stage),
            StateTensor(name + ".running_var", (channels,), "float32", False, stage),
            StateTensor(name + ".num_batches_tracked", (), "int64", False, stage),
        ])

    parameter("conv1.weight", (64, 3, 7, 7), 0)
    batch_norm("bn1", 64, 0)
    input_channels = 64
    for stage, (blocks, planes) in enumerate(zip(BLOCKS, PLANES, strict=True), 1):
        for block in range(blocks):
            prefix = f"layer{stage}.{block}"
            output_channels = planes * 4
            parameter(prefix + ".conv1.weight", (planes, input_channels, 1, 1), stage)
            batch_norm(prefix + ".bn1", planes, stage)
            parameter(prefix + ".conv2.weight", (planes, planes, 3, 3), stage)
            batch_norm(prefix + ".bn2", planes, stage)
            parameter(prefix + ".conv3.weight", (output_channels, planes, 1, 1), stage)
            batch_norm(prefix + ".bn3", output_channels, stage)
            if input_channels != output_channels or (stage > 1 and block == 0):
                parameter(prefix + ".downsample.0.weight", (output_channels, input_channels, 1, 1), stage)
                batch_norm(prefix + ".downsample.1", output_channels, stage)
            input_channels = output_channels
    parameter("fc.weight", (NUM_CLASSES, input_channels), 5)
    parameter("fc.bias", (NUM_CLASSES,), 5)
    if sum(int(np.prod(row.shape)) for row in result if row.trainable) != TRAINABLE_PARAMETERS:
        raise AssertionError("ResNet50 canonical parameter schema is inconsistent")
    return tuple(result)


SCHEMA = state_schema()
NAMES = tuple(row.name for row in SCHEMA)
TRAINABLE_NAMES = tuple(row.name for row in SCHEMA if row.trainable)


def batch_norm_associations():
    """Declare each canonical BN buffer's affine parameter owners."""
    return {row.name: (row.name.rsplit(".", 1)[0] + ".weight", row.name.rsplit(".", 1)[0] + ".bias")
            for row in SCHEMA if not row.trainable}


def canonical_state(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = {}
    for row in SCHEMA:
        if row.name.endswith("num_batches_tracked"):
            value = np.asarray(0, dtype=np.int64)
        elif row.name.endswith("running_var") or (row.name.endswith(".weight") and len(row.shape) == 1):
            value = np.ones(row.shape, dtype=np.float32)
        elif row.name.endswith(".weight"):
            scale = np.sqrt(2.0 / (row.shape[0] * row.shape[2] * row.shape[3])) if len(row.shape) == 4 else 0.01
            value = (rng.standard_normal(row.shape) * scale).astype(np.float32)
        else:
            value = np.zeros(row.shape, dtype=np.float32)
        state[row.name] = value
    return state


def validate_state(state):
    if set(state) != set(NAMES):
        raise ValueError("ResNet50 state names differ from the canonical schema")
    for row in SCHEMA:
        value = np.asarray(state[row.name])
        if value.shape != row.shape or str(value.dtype) != row.dtype:
            raise ValueError(f"ResNet50 state schema differs at {row.name}")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite ResNet50 state at {row.name}")


def state_hash(state):
    validate_state(state)
    digest = hashlib.sha256()
    for row in SCHEMA:
        digest.update(json.dumps([row.name, row.shape, row.dtype], separators=(",", ":")).encode())
        digest.update(np.asarray(state[row.name]).tobytes(order="C"))
    return digest.hexdigest()


def arrays_to_state(arrays):
    if len(arrays) != len(SCHEMA):
        raise ValueError("ResNet50 state tensor count mismatch")
    result = dict(zip(NAMES, arrays, strict=True))
    validate_state(result)
    return result
