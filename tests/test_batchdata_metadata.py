from __future__ import annotations

from splitfleet.common.typing import BatchData, ControlCode
from splitfleet.proto import server_model_pb2


def test_batchdata_metadata_defaults_to_empty_dict() -> None:
    payload = BatchData(data={}, control_code=ControlCode.OK)
    assert payload.metadata == {}


def test_proto_batchdata_exposes_metadata_field() -> None:
    proto = server_model_pb2.BatchData(
        method="forward",
        metadata={"plan_id": "plan-1", "stage_id": "stage-2"},
    )
    assert proto.metadata["plan_id"] == "plan-1"
    assert proto.metadata["stage_id"] == "stage-2"
