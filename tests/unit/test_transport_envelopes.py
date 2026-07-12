from __future__ import annotations

import pytest
import torch

from splitfleet.runtime import PrefixContextStore
from splitfleet.transport import (
    BoundaryEnvelope,
    GradientEnvelope,
    decode_boundary,
    decode_gradients,
    decode_tensor,
    encode_boundary,
    encode_gradients,
    encode_tensor,
    EnvelopeLimits,
    TensorEnvelope,
)


@pytest.mark.parametrize("compression", ["none", "lz4"])
def test_boundary_envelope_round_trip_is_pickle_free(compression: str) -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    envelope = BoundaryEnvelope(
        tensors=(encode_tensor("activation", source),),
        round_id=4,
        client_id="client-1",
        step_id="step-8",
        split_id="split-a",
        canonical_graph_hash="graph",
        boundary_schema_hash="schema",
        batch_size=3,
        metadata={"state_schema_hash": "state", "_local_context": object()},
    )

    payload = encode_boundary(envelope, compression=compression)
    restored = decode_boundary(payload)

    assert payload.startswith(b"SFW1")
    assert restored.step_id == "step-8"
    assert restored.metadata == {"state_schema_hash": "state"}
    assert torch.equal(decode_tensor(restored.tensors[0]), source)


def test_gradient_envelope_detects_corruption() -> None:
    envelope = GradientEnvelope(tensors=(encode_tensor("grad", torch.ones(2)),))
    payload = bytearray(encode_gradients(envelope))
    payload[-1] ^= 1

    with pytest.raises(ValueError, match="checksum"):
        decode_gradients(bytes(payload))


def test_prefix_context_store_consumes_context_once_and_cleans_round() -> None:
    store = PrefixContextStore()
    first = object()
    store.put(2, "a", "1", first)
    store.put(2, "b", "1", object())
    store.put(3, "a", "1", object())

    assert store.pop(2, "a", "1") is first
    with pytest.raises(KeyError, match="No pending"):
        store.pop(2, "a", "1")
    assert store.discard_round(2) == 1
    assert len(store) == 1


def test_decode_enforces_resource_limits_and_tensor_shape_bytes() -> None:
    envelope = BoundaryEnvelope(tensors=(encode_tensor("x", torch.ones(4)),))
    payload = encode_boundary(envelope)
    with pytest.raises(ValueError, match="wire size"):
        decode_boundary(payload, limits=EnvelopeLimits(max_wire_bytes=8))

    malformed = TensorEnvelope("x", (8,), "float32", b"\x00" * 4)
    with pytest.raises(ValueError, match="payload length"):
        decode_tensor(malformed)


def test_decode_rejects_decompressed_payload_over_limit() -> None:
    envelope = BoundaryEnvelope(tensors=(encode_tensor("x", torch.ones(128)),))
    payload = encode_boundary(envelope, compression="lz4")
    limits = EnvelopeLimits(max_payload_bytes=16, max_wire_bytes=len(payload) + 1)
    with pytest.raises(ValueError, match="decompressed size|payload exceeds"):
        decode_boundary(payload, limits=limits)
