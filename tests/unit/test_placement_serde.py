from __future__ import annotations

import pytest

from splitfleet.autosplit.serde import deserialize_plan_descriptor, serialize_plan_descriptor
from splitfleet.autosplit.types import SplitPlacementPlan


def test_placement_descriptor_keeps_portable_metadata_and_explicit_abi() -> None:
    plan = SplitPlacementPlan(
        plan_id="plan",
        split_id="split",
        graph_signature="graph",
        boundary="after:relu:1",
        mode="inference",
        prefix_worker_id="client",
        suffix_worker_id="server",
        score=1.0,
        feature_abi_id="abi",
        runtime_contract={"feature_abi_id": "abi"},
        metadata={"task": "text_classification", "_runtime_handle": object()},
    )

    descriptor = deserialize_plan_descriptor(serialize_plan_descriptor(plan))

    assert descriptor["feature_abi_id"] == "abi"
    assert descriptor["runtime_contract"] == {"feature_abi_id": "abi"}
    assert descriptor["metadata"] == {"task": "text_classification"}
    assert descriptor["stage_to_worker"] == {"prefix": "client", "suffix": "server"}


@pytest.mark.parametrize("payload", [b"", b"[]", b"null", b"{}"])
def test_invalid_placement_descriptor_cannot_become_an_empty_plan(payload: bytes) -> None:
    with pytest.raises(ValueError):
        deserialize_plan_descriptor(payload)
