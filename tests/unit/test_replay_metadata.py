"""Only replay schema metadata belongs on the remote tensor boundary."""

import pytest
import torch

from splitfleet.autosplit.boundary import BoundaryPayload
from splitfleet.transport import decode_boundary, encode_boundary
from splitfleet.transport.split_wire import boundary_to_envelope, replay_metadata


def test_wire_preserves_shape_binding_without_serializing_local_context():
    payload = BoundaryPayload(
        tensors={"feature": torch.ones(2, 4)}, batch_size=2,
        metadata={
            "backend": "torch", "runtime_batch_size": 2,
            "shape_program_hash": "shape-hash", "profile_hash": "profile-hash",
            "batch_symbol": "B", "state_prefix_kind": "training",
            "state_fingerprint": "client-local-state", "prefix_boundary_tensors": object(),
        },
    )
    envelope = boundary_to_envelope(
        payload, round_id=1, client_id="client", step_id="step", plan_id="plan",
        split_id="split", canonical_graph_hash="graph", boundary_schema_hash="schema",
        model_version=1,
    )
    restored = decode_boundary(encode_boundary(envelope))
    assert restored.metadata == {
        "runtime_batch_size": 2, "shape_program_hash": "shape-hash",
        "profile_hash": "profile-hash", "batch_symbol": "B", "state_prefix_kind": "training",
    }


def test_replay_metadata_rejects_nonportable_shape_binding():
    with pytest.raises(TypeError, match="JSON scalar"):
        replay_metadata({"runtime_batch_size": torch.tensor(2)})
