"""The paired 50-client campaign executes only full 100-round runs."""

import sys

import pytest

from experiments.resnet50_quality import campaign
from experiments.resnet50_quality.hardware import hardware_profile
from experiments.resnet50_quality.queue import read_json, save_json
from experiments.resnet50_quality.status import current_status


@pytest.fixture
def prepared(tmp_path, monkeypatch, quality_inputs):
    from experiments import www2027_study
    from experiments.resnet50_quality import hardware

    def freeze(_workspace, output):
        runtime = output / "runtime_snapshot"
        runtime.mkdir()
        manifest = {"source_identity": "current-fixture", "source_hashes": {}}
        save_json(output / "runtime_manifest.json", manifest)
        return runtime, manifest

    monkeypatch.setattr(www2027_study, "freeze_runtime", freeze)
    monkeypatch.setattr(hardware, "gpu_snapshot", lambda: {})
    hardware_file = tmp_path / "hardware.json"
    save_json(hardware_file, hardware_profile(cpus=[16, 17]))
    output = tmp_path / "campaign"
    result = campaign.prepare_campaign(output=output, protocol=quality_inputs.protocol,
        hardware=hardware_file, data_root=quality_inputs.data_root)
    return output, result


def test_campaign_runs_both_paired_seeds_with_fifty_clients_and_one_budget(prepared):
    output, result = prepared
    assert result["planned_runs"] == 4 and result["seeds"] == [233, 234]
    queue = read_json(result["queue"])
    assert [(job["seed"], job["method"]) for job in queue["jobs"]] == [
        (233, "fedavg"), (233, "splitfleet"), (234, "splitfleet"), (234, "fedavg")]
    assert all(job["clients"] == 50 and job["rounds"] == 100
               and job["examples_per_run"] == 4_500_000 for job in queue["jobs"])
    assert all(job["argv"][0] == sys.executable for job in queue["jobs"])
    assert queue["all_rounds"] == 100 and queue["failure_policy"] == "stop"
    observed = current_status(output)
    assert observed["campaign_completed_runs"] == 0 and observed["campaign_planned_runs"] == 4
    assert observed["active_round_budget"] == 100


@pytest.mark.parametrize("passes", [False, True])
def test_campaign_records_complete_or_failed_training_without_starting_another_phase(prepared, monkeypatch, passes):
    output, result = prepared
    calls = []

    def run(path):
        state = read_json(path)
        calls.append(state["all_rounds"])
        state.update(status="passed" if passes else "failed", completed_runs=4 if passes else 1)
        for index, job in enumerate(state["jobs"]):
            job["status"] = "passed" if passes or index == 0 else "pending"
        save_json(path, state)
        return 0 if passes else 1

    monkeypatch.setattr(campaign, "run_controller", run)
    assert campaign.execute_campaign(output / "campaign.json") == (0 if passes else 1)
    assert calls == [100]
    assert read_json(output / "campaign.json")["status"] == ("passed" if passes else "failed")
    with pytest.raises(FileExistsError):
        campaign.execute_campaign(output / "campaign.json")


@pytest.mark.parametrize("invalid_plan", ["out_of_order", "unverified_passed", "old_population"])
def test_invalid_plan_is_rejected_before_training(prepared, invalid_plan):
    output, result = prepared
    state = read_json(result["queue"])
    if invalid_plan == "out_of_order":
        state["jobs"].reverse()
    elif invalid_plan == "unverified_passed":
        state["jobs"][0]["status"] = "passed"
    else:
        state["jobs"][0]["clients"] = 100
    save_json(result["queue"], state)
    with pytest.raises(ValueError, match="seed order"):
        campaign.execute_campaign(output / "campaign.json")
    assert not (output / "campaign_claim.json").exists()
