"""The CLI parent closes the running status even when Ray kills its child."""

import json
from types import SimpleNamespace

import pytest

from experiments.common import evidence
from experiments.virtual_sfl import __main__ as cli


@pytest.mark.parametrize("child_status", ["running", "failed"])
def test_failed_child_preserves_completed_metrics_and_existing_failure(tmp_path, monkeypatch, child_status):
    output = tmp_path / "run"
    row = {"round_id": 1, "successful_clients": 100, "train_loss": .25}
    child_report = {"status": child_status, "reason": "original child failure"}

    def begin(path, config):
        path.mkdir()
        (path / "runtime_snapshot").mkdir()
        (path / "result.json").write_text(json.dumps(child_report))
        (path / "round_metrics.jsonl").write_text(json.dumps(row) + '\n{"round_id":')
        (path / "metrics.jsonl").write_text("existing child metrics")

    monkeypatch.setattr(evidence, "begin_run", begin)
    monkeypatch.setattr("splitfleet.virtual.runtime.check_simulation_dependencies", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="exit 1"):
        cli.frozen_run(SimpleNamespace(as_dict=lambda: {}), output)
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "failed"
    if child_status == "running":
        assert report["rounds_completed"] == 1
        assert report["child_exit_code"] == 1
        assert json.loads((output / "metrics.jsonl").read_text()) == row
    else:
        assert report == child_report
        assert (output / "metrics.jsonl").read_text() == "existing child metrics"
