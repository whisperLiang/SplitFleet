"""Validate bootstrap identity and actual per-client fit feedback."""
import copy
import json

import pytest

from experiments.orchestrate_physical_multitask import _validate_transport_learning
from experiments.orchestrate_physical_multitask import _native_monitor_receipt
from splitfleet.common.network_probe import PROBE_MODE, estimate_network
from tests.unit.placement_fixtures import _network_probe_receipt


def result():
    return {"placement_controller": "Greedy", "cost_initialization_receipt": {
        "transport_learning_mode": "affine_bootstrap_with_actual_round_feedback",
        "online_feedback_updates_retained": True},
        "round_diagnostics": {str(rid): {"assignment": {"a": "cut", "b": "cut"},
            "feedback_clients": ["a", "b"]} for rid in (1, 2)}}


def test_actual_feedback_covers_every_planned_client():
    assert _validate_transport_learning(result(), rounds=2) == []
    value = result()
    value["round_diagnostics"]["2"]["feedback_clients"].pop()
    assert _validate_transport_learning(value, rounds=2) == [
        "round 2: actual placement feedback is incomplete"]


@pytest.mark.parametrize("observed", [["a", "a"], ["a", "unexpected"], []])
def test_duplicate_or_unplanned_feedback_cannot_replace_actual_clients(observed):
    value = result()
    value["round_diagnostics"]["1"]["feedback_clients"] = observed
    assert _validate_transport_learning(value, rounds=2) == [
        "round 1: actual placement feedback is incomplete"]


def test_bootstrap_protocol_and_online_update_receipts_are_required():
    value = result()
    value["cost_initialization_receipt"]["transport_learning_mode"] = "legacy_bootstrap_residual"
    assert _validate_transport_learning(value, rounds=2) == [
        "Greedy transport calibration receipt is inconsistent"]
    value = result()
    value["cost_initialization_receipt"].pop("online_feedback_updates_retained")
    assert _validate_transport_learning(value, rounds=2) == ["Greedy online feedback receipt is missing"]


@pytest.mark.parametrize("fault", [None, "missing", "stale", "configured"])
def test_active_mode_requires_complete_measured_current_round_context(fault):
    value = result()
    value["cost_initialization_receipt"]["network_context_mode"] = PROBE_MODE
    for rid in (1, 2):
        diagnostic = value["round_diagnostics"][str(rid)]
        diagnostic["network_probes"] = {cid: _network_probe_receipt(
            rid, upload_mbps=20, download_mbps=80, rtt_ms=7) for cid in ("a", "b")}
        diagnostic["network_contexts"] = {cid: estimate_network(probe, round_id=rid)
            for cid, probe in diagnostic["network_probes"].items()}
    if fault == "missing":
        del value["round_diagnostics"]["2"]["network_probes"]["a"]
    elif fault == "stale":
        value["round_diagnostics"]["2"]["network_probes"]["a"]["round_id"] = 1
    elif fault == "configured":
        value["round_diagnostics"]["2"]["network_contexts"]["a"]["upload_mbps"] = 100
    errors = _validate_transport_learning(value, rounds=2)
    assert bool(errors) is (fault is not None)


@pytest.mark.parametrize("change", ["none", "missing_ready", "duplicate_ready", "fault"])
def test_native_monitor_requires_ready_and_one_actual_clean_child_exit(change):
    ready = {"event": "native_trap_ready", "pid": 123}
    end = {"event": "monitored_child_exit", "returncode": 0,
           "native_fault_observed": False, "trap_ready": True}
    rows = [ready, end]
    if change == "missing_ready":
        rows.pop(0)
    if change == "duplicate_ready":
        rows.append(ready)
    if change == "fault":
        rows.append({"event": "native_fault_captured", "signal": 11})
    receipt = _native_monitor_receipt("ordinary model log\n" + "\n".join(json.dumps(row) for row in rows))
    assert receipt["valid"] is (change == "none")
