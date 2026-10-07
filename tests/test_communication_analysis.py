import pytest

from experiments.analysis.communication import communication_summary


def test_available_combined_counters_are_kept_separate_and_missing_total_is_unknown():
    result = {"fit_records": [{"metrics": {"upload_bytes": 100, "download_bytes": 25, "state_upload_bytes": 30}},
                              {"metrics": {"upload_bytes": 200, "download_bytes": 50, "state_upload_bytes": 60}}],
              "server_fit_records": [{"metrics": {"state_bytes": 70}}]}
    report = communication_summary(result)
    assert report["measured_counters"]["activation_and_target_upload_bytes"] == 300
    assert report["measured_counters"]["client_state_upload_raw_tensor_bytes"] == 90
    assert report["activation_upload_bytes"] is None
    assert report["total_communication_bytes"] is None
    assert not report["complete"]


def test_partial_counter_coverage_does_not_undercount_as_complete():
    report = communication_summary({"fit_records": [{"metrics": {"upload_bytes": 100}}, {"metrics": {}}]})
    assert report["measured_counters"]["activation_and_target_upload_bytes"] is None


def test_negative_or_fractional_bytes_are_rejected():
    for value in [-1, 1.5, True]:
        with pytest.raises(ValueError):
            communication_summary({"fit_records": [{"metrics": {"upload_bytes": value}}]})


def complete_result():
    return {"method": "splitfleet", "communication_accounting": "training_client_application_buffers_v1",
        "rounds": 1, "expected_clients": 1, "fit_failures": [],
        "fit_records": [{"round_id": 1, "cid": "a", "metrics": {"activation_upload_bytes": 100,
            "target_upload_bytes": 20, "request_metadata_upload_bytes": 10, "gradient_download_bytes": 50,
            "response_metadata_download_bytes": 15, "model_state_upload_bytes": 200}}],
        "state_download_records": [{"round_id": 1, "cid": "a", "model_state_download_bytes": 300}]}


def test_separated_serialized_application_buffers_sum_only_with_all_receipts():
    result = complete_result()
    report = communication_summary(result)
    assert report["model_state_synchronization_bytes"] == 500
    assert report["total_communication_bytes"] == 695 and report["complete"]
    assert report["physical_network_traffic_bytes"] is None
    result["fit_failures"] = [{"reason": "failed"}]
    assert communication_summary(result)["total_communication_bytes"] is None
    result["fit_failures"] = []
    result["expected_clients"] = 2
    assert not communication_summary(result)["complete"]


def test_missing_category_and_duplicate_receipts_are_not_accepted_as_complete():
    result = complete_result()
    result["fit_records"][0]["metrics"].pop("gradient_download_bytes")
    assert communication_summary(result)["total_communication_bytes"] is None
    result["fit_records"] *= 2
    with pytest.raises(ValueError, match="duplicate"): communication_summary(result)


def test_native_fl_records_real_state_transfers_without_boundary_traffic():
    result = complete_result()
    result["method"] = "fedavg"
    result["fit_records"][0]["metrics"] = {"model_state_upload_bytes": 200}
    report = communication_summary(result)
    assert report["activation_upload_bytes"] == report["gradient_download_bytes"] == 0
    assert report["total_communication_bytes"] == 500 and report["complete"]


def test_matching_receipts_from_the_wrong_round_do_not_complete_the_budget():
    result = complete_result()
    result["fit_records"][0]["round_id"] = 2
    result["state_download_records"][0]["round_id"] = 2
    assert not communication_summary(result)["complete"]


def test_client_counts_actual_boundary_target_gradient_and_metadata_buffers():
    import json
    from types import SimpleNamespace
    import torch
    from splitfleet.backends import BACKEND_ADAPTERS
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient, _RoundTelemetry
    from splitfleet.common.typing import BatchData, ControlCode
    from splitfleet.transport import BoundaryEnvelope, encode_gradients
    from splitfleet.transport.split_wire import gradients_to_envelope

    adapter = BACKEND_ADAPTERS.create("torch")
    envelope = BoundaryEnvelope(tensors=(adapter.encode_tensor("hidden", torch.ones(2, 3)),),
        engine="torchlens", backend="torch", round_id=1, client_id="a", step_id="s", plan_id="p",
        split_id="after:hidden", canonical_graph_hash="g", boundary_schema_hash="b", model_version=1, batch_size=2)
    gradient = encode_gradients(gradients_to_envelope(envelope, {"hidden": torch.ones(2, 3)}))
    response = BatchData(data={"gradients": gradient, "metadata": json.dumps({"loss": .5, "num_examples": 2}).encode()},
                         metadata={}, control_code=ControlCode.OK)
    requests = []
    def tail(request, **kwargs):
        requests.append(request)
        return response
    client = AutoSplitSplitLearningClient.__new__(AutoSplitSplitLearningClient)
    client.backend_adapter = adapter
    client.server_model_proxy = SimpleNamespace(train_tail=tail)
    counters = _RoundTelemetry()
    client._call_tail(method_name="train_tail", boundary=envelope, targets=torch.zeros(2, dtype=torch.long),
                      num_examples=2, telemetry=counters)
    request = requests[0]
    assert counters.activation_upload_bytes == len(request.data["boundary"])
    assert counters.target_upload_bytes == len(request.data["targets"])
    assert counters.request_metadata_upload_bytes == len(request.data["metadata"])
    assert counters.gradient_download_bytes == len(gradient)
    assert counters.response_metadata_download_bytes == len(response.data["metadata"])
    assert counters.gradient_download_bytes + counters.response_metadata_download_bytes == counters.download_bytes
