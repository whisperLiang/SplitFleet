"""Local boundary payloads and TorchLens replay conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchlens.split import ReplayBoundary


@dataclass
class BoundarySpec:
    boundary: str
    boundary_tensor_labels: list[str]


@dataclass
class BoundaryPayload:
    tensors: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    batch_size: int | None = None
    spec: BoundarySpec | None = None


def from_torchlens_boundary(boundary: ReplayBoundary) -> BoundaryPayload:
    metadata = {
        **boundary.metadata,
        "backend": boundary.backend,
        "_torchlens_spec": dict(boundary.spec),
    }
    return BoundaryPayload(
        tensors=dict(boundary.tensors),
        metadata=metadata,
        batch_size=int(metadata["runtime_batch_size"]),
        spec=BoundarySpec(metadata["split_id"], list(boundary.spec)),
    )


def to_torchlens_boundary(payload: BoundaryPayload) -> ReplayBoundary:
    spec = payload.metadata.get("_torchlens_spec")
    if not spec:
        raise ValueError("BoundaryPayload requires native TorchLens spec metadata.")
    metadata = dict(payload.metadata)
    # A remote suffix owns an independently updated model replica. Graph,
    # schema and model-version contracts are validated at the wire boundary.
    metadata.pop("state_fingerprint", None)
    return ReplayBoundary(
        backend=metadata["backend"],
        tensors=dict(payload.tensors),
        spec=dict(spec),
        metadata=metadata,
    )
