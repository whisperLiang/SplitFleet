"""Serialization helpers for autosplit plans and stage payloads."""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Any, Dict

import torch

from slbd.autosplit.types import PlacementPlan


def dumps_torch_object(value: Any) -> bytes:
    """Serialize a Python object containing tensors."""

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def loads_torch_object(value: bytes, *, map_location: str | torch.device = "cpu") -> Any:
    """Deserialize a Python object containing tensors."""

    buffer = io.BytesIO(value)
    return torch.load(buffer, map_location=map_location, weights_only=False)


def dump_model_state(model) -> bytes:
    """Serialize a model state dict for worker synchronization."""

    return dumps_torch_object(model.state_dict())


def load_model_state(model, state_bytes: bytes, *, map_location: str | torch.device = "cpu") -> None:
    """Load a serialized model state dict into a model."""

    if not state_bytes:
        return
    state_dict = loads_torch_object(state_bytes, map_location=map_location)
    model.load_state_dict(state_dict)


def serialize_plan_descriptor(placement_plan: PlacementPlan) -> bytes:
    """Serialize the portable subset of a placement plan."""

    descriptor = {
        "plan_id": placement_plan.plan_id,
        "model_name": placement_plan.partition_plan.model_name,
        "graph_signature": placement_plan.partition_plan.graph_signature,
        "cutoffs": list(placement_plan.partition_plan.metadata.get("cutoffs", [])),
        "stage_to_worker": dict(placement_plan.stage_to_worker),
        "score": placement_plan.score,
        "constraints": asdict(placement_plan.constraints),
        "objective": asdict(placement_plan.objective),
        "metadata": dict(placement_plan.metadata),
    }
    return json.dumps(descriptor, sort_keys=True).encode("utf-8")


def deserialize_plan_descriptor(payload: bytes) -> Dict[str, Any]:
    """Parse a serialized plan descriptor."""

    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))
