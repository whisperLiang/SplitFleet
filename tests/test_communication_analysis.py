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
