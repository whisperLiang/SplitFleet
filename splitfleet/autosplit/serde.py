"""Portable JSON serialization for split placement descriptors."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from splitfleet.autosplit.types import SplitPlacementPlan


def serialize_plan_descriptor(placement_plan: SplitPlacementPlan) -> bytes:
    """Serialize the portable subset of a placement plan."""

    descriptor = {
        "plan_id": placement_plan.plan_id,
        "engine": placement_plan.engine,
        "backend": placement_plan.backend,
        "runtime_backend": placement_plan.runtime_backend,
        "graph_signature": placement_plan.graph_signature,
        "split_id": placement_plan.split_id,
        "boundary": placement_plan.boundary,
        "mode": placement_plan.mode,
        "candidate_id": placement_plan.candidate_id,
        "boundary_tensor_labels": list(placement_plan.boundary_tensor_labels),
        "payload_bytes": placement_plan.payload_bytes,
        "feature_abi_id": placement_plan.feature_abi_id,
        "runtime_contract": dict(placement_plan.runtime_contract),
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


def deserialize_plan_descriptor(payload: bytes) -> dict[str, Any]:
    """Parse a serialized plan descriptor."""

    descriptor = json.loads(payload.decode("utf-8"))
    if not isinstance(descriptor, dict):
        raise ValueError("Split placement descriptor must be a JSON object")
    for field_name in ("engine", "backend", "runtime_backend", "plan_id", "split_id"):
        if not descriptor.get(field_name):
            raise ValueError(f"Split placement descriptor is missing {field_name!r}")
    if descriptor["engine"] != "torchlens" or descriptor["backend"] != "torchlens":
        raise ValueError("Only explicit TorchLens placement descriptors are supported")
    return descriptor
