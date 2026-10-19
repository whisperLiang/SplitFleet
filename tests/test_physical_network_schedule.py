"""The physical training strategies must actually apply registered TCP phases."""

import copy
import json
from types import SimpleNamespace

import pytest

from experiments.common.network_schedule import RoundNetworkSchedule
from experiments.physical_multitask import RecordingFedAvg, RecordingSplit
from experiments.placement_stress import network_phases, validate_network_execution


@pytest.mark.parametrize("strategy_type", [RecordingSplit, RecordingFedAvg])
def test_training_applies_current_phase_before_native_configuration(strategy_type, monkeypatch, tmp_path):
    phases = network_phases("dynamic", 12)
    control = tmp_path / "control.json"
    schedule = RoundNetworkSchedule(phases, control)
    schedule.publish_initialization()
    strategy = strategy_type.__new__(strategy_type)
    strategy.network_phase_provider = schedule
    strategy.state_download_records = []
    observed = []
    def reject_configured_context(provider):
        raise AssertionError("Network pacing settings must not be passed to the policy")

    strategy.placement_policy = SimpleNamespace(bind_network_context=reject_configured_context)

    def native_configuration(self, round_id, parameters, clients):
        state = json.loads(control.read_text())
        assert state["round_id"] == round_id
        observed.append(state)
        return []

    monkeypatch.setattr(strategy_type.__bases__[0], "configure_fit", native_configuration)
    for round_id in range(1, 13):
        assert strategy.configure_fit(round_id, None, None) == []
    assert [row["upload_mbps"] for row in observed] == [1024.] * 4 + [128.] * 4 + [1024.] * 4
    proxy = {"physical_transfer": True, "round_counters": {
        str(row["round_id"]): {**row, "upload_bytes": 100, "download_bytes": 200} for row in observed}}
    validate_network_execution({"network_schedule": schedule.receipt()}, proxy, phases, 12)


@pytest.mark.parametrize("strategy_type", [RecordingSplit, RecordingFedAvg])
def test_unscheduled_training_configuration_still_works(strategy_type, monkeypatch):
    strategy = strategy_type.__new__(strategy_type)
    strategy.state_download_records = []
    monkeypatch.setattr(strategy_type.__bases__[0], "configure_fit", lambda *args: [])
    assert strategy.configure_fit(1, None, None) == []


@pytest.fixture
def network_receipts(tmp_path):
    phases = network_phases("dynamic", 12)
    schedule = RoundNetworkSchedule(phases, tmp_path / "control.json")
    schedule.publish_initialization()
    for rid in range(1, 13):
        schedule.publish(rid)
    counters = {str(row["round_id"]): {**row, "upload_bytes": 10, "download_bytes": 20}
                for row in schedule.applied}
    return {"network_schedule": schedule.receipt()}, {"physical_transfer": True, "round_counters": counters}, phases


def test_registered_phases_without_execution_are_rejected(network_receipts):
    result, proxy, phases = copy.deepcopy(network_receipts)
    result["network_schedule"]["applied"] = result["network_schedule"]["applied"][:1]
    proxy["round_counters"] = {"0": proxy["round_counters"]["0"]}
    with pytest.raises(ValueError, match="not applied before every training round"):
        validate_network_execution(result, proxy, phases, 12)


@pytest.mark.parametrize("fault", ["wrong_rate", "missing_round", "no_bytes"])
def test_actual_proxy_evidence_is_checked(network_receipts, fault):
    result, proxy, phases = copy.deepcopy(network_receipts)
    if fault == "wrong_rate":
        proxy["round_counters"]["5"]["upload_mbps"] = 1024.
    elif fault == "missing_round":
        del proxy["round_counters"]["5"]
    else:
        proxy["round_counters"]["5"]["upload_bytes"] = 0
    with pytest.raises(ValueError, match="round 5"):
        validate_network_execution(result, proxy, phases, 12)
