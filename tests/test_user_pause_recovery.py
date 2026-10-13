"""Controlled stops cannot disguise failures or rewrite incomplete seed blocks."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.analysis.user_pause_recovery import validate_user_pause


@pytest.fixture
def paused(tmp_path):
    root = tmp_path / "original"
    run = "bert_s10505"
    ledger = root / "ledger.json"
    attempts = root / "physical" / run / "attempts.json"
    pause = tmp_path / "pause.json"
    receipt = tmp_path / "recovery.json"
    bound = {}

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        bound[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    save(ledger, {"jobs": [{"run_id": run, "stage": "bert", "seed": 10505,
        "status": "failed", "exit_code": -2, "started_unix": 10, "finished_unix": 30}]})
    save(attempts, [{"status": "completed"}, {"status": "running"}])
    save(pause, {"status": "paused_by_user", "reason": "User explicitly requested: 暂时停止实验",
        "population_root": str(root), "active_run_id": run, "owned_transport_closed": True,
        "experiment_dispatch_stopped": True, "automatic_restart_authorized": False,
        "requested_unix": 20, "signal_sent_unix": 21,
        "after_cleanup_snapshots": {str(p): {"sha256": bound[str(p)]} for p in (ledger, attempts)}})
    save(receipt, {"schema": "splitfleet.controlled-user-pause-recovery.v1",
        "interruption_origin": "explicit_user_request", "original_statuses_modified": False,
        "pause_receipt": str(pause), "original_root": str(root), "run_id": run,
        "stage": "bert", "seed": 10505, "interrupted_attempts": 1, "unpaired_successes_retained": 1})
    return receipt, bound, save


def test_user_pause_preserves_running_status_and_does_not_require_fake_boot(paused):
    receipt, bound, _ = paused
    assert validate_user_pause(receipt, bound) == 1


def test_user_pause_requires_original_authorization_receipt_even_when_rehashed(paused):
    receipt, bound, save = paused
    path = Path(json.loads(receipt.read_text())["pause_receipt"])
    value = json.loads(path.read_text())
    value["reason"] = "A slow or low-quality baseline was stopped"
    save(path, value)
    with pytest.raises(ValueError, match="original controlled-stop receipt"):
        validate_user_pause(receipt, bound)


def test_user_pause_cannot_hide_an_actual_method_failure(paused):
    receipt, bound, save = paused
    value = json.loads(receipt.read_text())
    path = Path(value["original_root"]) / "physical" / value["run_id"] / "attempts.json"
    attempts = json.loads(path.read_text())
    attempts[1]["status"] = "failed"
    save(path, attempts)
    with pytest.raises(ValueError, match="preserved stop snapshot"):
        validate_user_pause(receipt, bound)


def test_user_pause_cannot_relabel_an_ordinary_process_exit_as_sigint(paused):
    receipt, bound, save = paused
    value = json.loads(receipt.read_text())
    path = Path(value["original_root"]) / "ledger.json"
    ledger = json.loads(path.read_text())
    ledger["jobs"][0]["exit_code"] = 1
    save(path, ledger)
    with pytest.raises(ValueError, match="recorded user SIGINT"):
        validate_user_pause(receipt, bound)


def test_user_pause_requires_the_unpaired_success_denominator(paused):
    receipt, bound, save = paused
    value = json.loads(receipt.read_text())
    value["unpaired_successes_retained"] = 0
    save(receipt, value)
    with pytest.raises(ValueError, match="count changed"):
        validate_user_pause(receipt, bound)
