"""A native failure must not erase completed physical-round evidence."""
import json
from types import SimpleNamespace

import pytest

from experiments.common.physical_progress import PhysicalRoundRecorder
from experiments.physical_multitask import RecordingSplit
from splitfleet.server.strategy import AutoSplitStrategy


def strategy_receipts():
    return SimpleNamespace(placement_policy=SimpleNamespace(round_diagnostics={1: {"assignment": {"a": "cut"}}}),
        fit_records=[{"round_id": 1, "cid": "a", "metrics": {"coordinator_fit_rpc_ms": 100}}],
        fit_failures=[], server_fit_records=[{"round_id": 1, "metrics": {"service_ms": 2}}],
        evaluation_records=[{"round_id": 1, "metrics": {"map50": .9}, "elapsed_sec": 1.25}],
        state_download_records=[{"round_id": 1, "model_state_download_bytes": 120}])


def test_completed_round_survives_next_round_serialization_failure(tmp_path):
    recorder = PhysicalRoundRecorder(tmp_path / "result.json", metadata={"seed": 5}, measurement_started_ns=123)
    strategy = strategy_receipts()
    recorder.save_round(strategy, 1)
    first = (recorder.directory / "round_001.json").read_bytes()
    receipt = json.loads(first)
    assert receipt["evaluation_records"][0]["elapsed_sec"] == 1.25
    assert receipt["measurement_started_monotonic_ns"] == 123
    assert receipt["fit_records"] == strategy.fit_records
    assert receipt["round_diagnostics"]["assignment"] == {"a": "cut"}
    strategy.evaluation_records.append({"round_id": 2, "metrics": {"map50": float("nan")}})
    with pytest.raises(ValueError):
        recorder.save_round(strategy, 2)
    assert (recorder.directory / "round_001.json").read_bytes() == first
    assert json.loads((recorder.directory / "latest.json").read_text())["last_completed_round"] == 1
    assert not (recorder.directory / "round_002.json").exists()
    assert receipt["complete_job"] is False


def test_last_phase_marks_exact_failing_preparation(tmp_path, monkeypatch):
    recorder = PhysicalRoundRecorder(tmp_path / "result.json", metadata={}, measurement_started_ns=1)
    strategy = RecordingSplit.__new__(RecordingSplit)
    strategy.round_recorder, strategy._recording_round = recorder, 11
    def fails(*args, **kwargs):
        raise RuntimeError("native preparation failed")
    monkeypatch.setattr(AutoSplitStrategy, "get_or_create_placement_plan", fails)
    with pytest.raises(RuntimeError, match="native preparation"):
        strategy.get_or_create_placement_plan("after:op:1")
    records = [json.loads(line) for line in (recorder.directory / "events.jsonl").read_text().splitlines()]
    assert records[-1]["event"] == "placement_prepare_start"
    assert records[-1]["round_id"] == 11 and records[-1]["boundary"] == "after:op:1"


def test_selection_journal_captures_assignment_before_runtime_preparation(tmp_path, monkeypatch):
    recorder = PhysicalRoundRecorder(tmp_path / "result.json", metadata={}, measurement_started_ns=1)
    strategy = RecordingSplit.__new__(RecordingSplit)
    strategy.round_recorder, strategy._recording_round = recorder, 11
    strategy._round_placement_assignments = {}
    monkeypatch.setattr(AutoSplitStrategy, "_plan_round_placements", lambda *args, **kwargs: {"a": "cut"})
    assert strategy._plan_round_placements(11, ["a"], training=True) == {"a": "cut"}
    records = [json.loads(line) for line in (recorder.directory / "events.jsonl").read_text().splitlines()]
    assert records[-1]["assignment"] == {"a": "cut"}
    assert records[-1]["event"] == "placement_selection_end"
