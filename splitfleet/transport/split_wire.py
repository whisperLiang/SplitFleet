"""Conversion between local TorchLens payloads and stable wire envelopes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from splitfleet.backends import BACKEND_ADAPTERS
from splitfleet.transport.envelopes import (
    BoundaryEnvelope,
    GradientEnvelope,
)

if TYPE_CHECKING:
    from splitfleet.autosplit.boundary import BoundaryPayload


# Only portable replay information crosses the wire. Prefix autograd context
# and the fingerprint of a process-local model replica must stay local.
REPLAY_METADATA_KEYS = (
    "runtime_batch_size", "shape_program_hash", "batch_symbol", "profile_hash",
    "device_policy", "state_prefix_kind",
)


def replay_metadata(metadata: Any) -> dict[str, Any]:
    result = {}
    for key in REPLAY_METADATA_KEYS:
        if key in metadata:
            value = metadata[key]
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise TypeError(f"Replay metadata {key!r} must be a JSON scalar.")
            result[key] = value
    return result


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
    try:
        backend = str(payload.metadata["backend"])
    except KeyError as exc:
        raise ValueError("Boundary payload is missing its backend.") from exc
    if payload.batch_size is None:
        raise ValueError("Boundary payload is missing its batch size.")
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
        batch_size=int(payload.batch_size),
        metadata=replay_metadata(payload.metadata),
    )


def envelope_to_boundary(envelope: BoundaryEnvelope, runtime: Any, device: Any) -> BoundaryPayload:
    from splitfleet.autosplit.boundary import BoundaryPayload, BoundarySpec

    if envelope.backend != str(runtime.adapter.name):
        raise ValueError("Boundary backend does not match suffix runtime")
    adapter = BACKEND_ADAPTERS.create(envelope.backend)
    tensors = {item.tensor_id: adapter.decode_tensor(item, device) for item in envelope.tensors}
    return BoundaryPayload(
        tensors=tensors,
        metadata={
            **replay_metadata(envelope.metadata),
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
