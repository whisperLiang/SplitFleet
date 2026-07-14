"""Conversion between local TorchLens payloads and stable wire envelopes."""

from __future__ import annotations

from typing import Any

from splitfleet.autosplit.boundary import BoundaryPayload, BoundarySpec
from splitfleet.backends import BACKEND_ADAPTERS
from splitfleet.transport.envelopes import (
    BoundaryEnvelope,
    GradientEnvelope,
)


def boundary_to_envelope(
    payload: BoundaryPayload,
    *,
    round_id: int,
    client_id: str,
    step_id: str,
    plan_id: str,
    split_id: str,
    canonical_graph_hash: str,
    boundary_schema_hash: str,
    model_version: int,
) -> BoundaryEnvelope:
    backend = str(payload.metadata.get("backend", "torch"))
    adapter = BACKEND_ADAPTERS.create(backend)
    return BoundaryEnvelope(
        tensors=tuple(adapter.encode_tensor(name, value) for name, value in payload.tensors.items()),
        engine="torchlens",
        backend=backend,
        round_id=round_id,
        client_id=client_id,
        step_id=step_id,
        plan_id=plan_id,
        split_id=split_id,
        canonical_graph_hash=canonical_graph_hash,
        boundary_schema_hash=boundary_schema_hash,
        model_version=model_version,
        batch_size=int(payload.batch_size or 0),
    )


def envelope_to_boundary(envelope: BoundaryEnvelope, runtime: Any, device: Any) -> BoundaryPayload:
    if envelope.backend != str(runtime.adapter.name):
        raise ValueError("Boundary backend does not match suffix runtime")
    adapter = BACKEND_ADAPTERS.create(envelope.backend)
    tensors = {item.tensor_id: adapter.decode_tensor(item, device) for item in envelope.tensors}
    return BoundaryPayload(
        tensors=tensors,
        metadata={
            "_torchlens_spec": dict(runtime.boundary_spec),
            "split_id": envelope.split_id,
            "graph_shape_hash": envelope.canonical_graph_hash,
            "batch_size": envelope.batch_size,
            "backend": envelope.backend,
        },
        batch_size=envelope.batch_size,
        spec=BoundarySpec(envelope.split_id, list(tensors)),
    )


def gradients_to_envelope(boundary: BoundaryEnvelope, gradients: dict[str, Any]) -> GradientEnvelope:
    adapter = BACKEND_ADAPTERS.create(boundary.backend)
    return GradientEnvelope(
        tensors=tuple(adapter.encode_tensor(name, value) for name, value in gradients.items()),
        backend=boundary.backend,
        round_id=boundary.round_id,
        client_id=boundary.client_id,
        step_id=boundary.step_id,
        plan_id=boundary.plan_id,
        split_id=boundary.split_id,
        model_version=boundary.model_version,
    )


def envelope_to_gradients(envelope: GradientEnvelope, device: Any) -> dict[str, Any]:
    adapter = BACKEND_ADAPTERS.create(envelope.backend)
    return {item.tensor_id: adapter.decode_tensor(item, device) for item in envelope.tensors}
