"""A frozen child cannot change inputs or overwrite an attempted measurement."""

import json

import pytest

from experiments.common import evidence


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    output = tmp_path / "run"
    evidence.begin_run(output, {"seed": 2027, "repeats": 3})
    monkeypatch.setattr(evidence, "__file__", str(output / "runtime_snapshot/experiments/common/evidence.py"))
    return output


def test_child_execution_is_claimed_once_even_if_it_crashes(frozen_run):
    assert evidence.claim_frozen_run(frozen_run)["seed"] == 2027
    # A child that dies before finish_run still has status=running. Its marker
    # preserves the attempt and refuses a second execution over partial data.
    with pytest.raises(FileExistsError):
        evidence.claim_frozen_run(frozen_run)


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_terminal_run_is_never_overwritten(frozen_run, status):
    receipt = frozen_run / "result.json"
    receipt.write_text(json.dumps({"status": status, "measured": 123}))
    before = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        evidence.claim_frozen_run(frozen_run)
    assert receipt.read_bytes() == before


@pytest.mark.parametrize("change,reason", [("config", "configuration"), ("source", "runtime source"),
                                          ("dependency", "TorchLens source")])
def test_frozen_inputs_must_match_recorded_bytes(frozen_run, monkeypatch, change, reason):
    if change == "config":
        path = frozen_run / "config.json"
        config = json.loads(path.read_text())
        config["seed"] += 1
        path.write_text(json.dumps(config))
    elif change == "source":
        (frozen_run / "runtime_snapshot/splitfleet/__init__.py").write_text("changed = True\n")
    else:
        monkeypatch.setattr("experiments.analysis.partition_coverage._torchlens_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match=reason):
        evidence.claim_frozen_run(frozen_run)
    assert not (frozen_run / "execution_attempt.json").exists()
