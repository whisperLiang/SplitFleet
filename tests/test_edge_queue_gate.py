"""A partially published admission ledger never bypasses or kills the gate."""

from argparse import Namespace
import json

import pytest

from experiments import run_edge_standard_study as queue


@pytest.mark.parametrize("model_count", [1, 4])
def test_partial_admission_write_waits_for_complete_passed_snapshot(tmp_path, monkeypatch, model_count):
    admission = tmp_path / "admission"
    admission.mkdir()
    (admission / "runtime_snapshot").mkdir()
    (admission / "runtime_manifest.json").write_text(json.dumps({"source_identity": "fixed"}))
    ledger = admission / "coordinator_attempts.json"
    ledger.write_text("")
    wait_for = tmp_path / "prior.json"
    wait_for.write_text(json.dumps({"status": "completed"}))
    deployment = tmp_path / "devices.json"
    deployment.write_text(json.dumps({"hosts": [{"id": "fixture", "ssh": "fixture",
        "python": "/fixture/python", "workdir": "/fixture"}]}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"deployment": str(deployment),
        "resources": {"models": {f"fixture-{i}": {} for i in range(model_count)}}}))
    pauses = []

    def publish_after_read(delay):
        pauses.append(delay)
        ledger.write_text(json.dumps([{"status": "passed"} for _ in range(3 * model_count)]))

    def reached_admission(*args, **kwargs):
        assert len(json.loads(ledger.read_text())) == 3 * model_count
        raise RuntimeError("fixture reached device admission")

    monkeypatch.setattr(queue.time, "sleep", publish_after_read)
    monkeypatch.setattr(queue, "remote", reached_admission)
    root = tmp_path / "queue"
    with pytest.raises(RuntimeError, match="fixture reached device admission"):
        queue.run(Namespace(root=root, admission=admission, plan=plan, wait_for=wait_for))
    state = json.loads((root / "queue.json").read_text())
    assert pauses == [1] and state["incomplete_gate_reads"] == 1
    assert state["attempts"] == []
    assert state["failure_policy"] == "record failure and stop; no automatic retry"
