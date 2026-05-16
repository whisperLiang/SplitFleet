"""Serialization helpers for autosplit plans and stage payloads.

Optimized for performance with:
- Modern zipfile serialization format
- Optional lz4 compression
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Any, Dict, Optional

import lz4.frame
import torch

from splitfleet.autosplit.types import PlacementPlan

# Compression header byte
_COMPRESSION_LZ4 = 0x01


def dumps_torch_object(
    value: Any,
    *,
    compress: bool = False,
    compression_level: int = 6,
) -> bytes:
    """Serialize a Python object containing tensors.

    Args:
        value: Object to serialize (can contain tensors).
        compress: Whether to apply lz4 compression.
        compression_level: Compression level (0-16).

    Returns:
        Serialized bytes with optional compression header.
    """
    buffer = io.BytesIO()
    torch.save(value, buffer, _use_new_zipfile_serialization=True)
    data = buffer.getvalue()

    if not compress:
        return data

    compressed = lz4.frame.compress(data, compression_level=compression_level)
    return bytes([_COMPRESSION_LZ4]) + compressed


def loads_torch_object(
    value: bytes,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Deserialize a Python object containing tensors.

    Automatically detects and decompresses if compression header present.

    Args:
        value: Serialized bytes.
        map_location: Device to map tensors to.

    Returns:
        Deserialized object.
    """
    if not value:
        raise ValueError("Cannot deserialize empty bytes")

    # Check for compression header byte
    if value[0] == _COMPRESSION_LZ4:
        data = lz4.frame.decompress(value[1:])
    else:
        data = value

    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location=map_location, weights_only=False)


def dump_model_state(
    model,
    *,
    compress: bool = False,
) -> bytes:
    """Serialize a model state dict for worker synchronization.

    Args:
        model: PyTorch model with state_dict() method.
        compress: Whether to compress the state dict.

    Returns:
        Serialized state dict bytes.
    """
    return dumps_torch_object(model.state_dict(), compress=compress)


def load_model_state(
    model,
    state_bytes: bytes,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load a serialized model state dict into a model.

    Args:
        model: PyTorch model to load state into.
        state_bytes: Serialized state dict bytes.
        map_location: Device to map tensors to.
    """
    if not state_bytes:
        return
    state_dict = loads_torch_object(state_bytes, map_location=map_location)
    model.load_state_dict(state_dict)


def serialize_plan_descriptor(placement_plan: PlacementPlan) -> bytes:
    """Serialize the portable subset of a placement plan."""

    descriptor = {
        "plan_id": placement_plan.plan_id,
        "backend": "ariadne",
        "graph_signature": placement_plan.graph_signature,
        "split_id": placement_plan.split_id,
        "boundary": placement_plan.boundary,
        "mode": placement_plan.mode,
        "stage_count": placement_plan.stage_count,
        "client_stage_count": 1,
        "stage_to_worker": dict(placement_plan.stage_to_worker),
        "score": placement_plan.score,
        "constraints": asdict(placement_plan.constraints),
        "objective": asdict(placement_plan.objective),
        "metadata": {
            key: value
            for key, value in placement_plan.metadata.items()
            if not key.startswith("_")
        },
    }
    return json.dumps(descriptor, sort_keys=True).encode("utf-8")


def deserialize_plan_descriptor(payload: bytes) -> Dict[str, Any]:
    """Parse a serialized plan descriptor."""

    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))
