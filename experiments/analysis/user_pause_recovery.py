"""Verify a recorded user stop without treating it as a coordinator reboot."""
from pathlib import Path
import hashlib
import json


def validate_user_pause(path, bound):
    path = Path(path)

    def read_bound(filename):
        target = Path(filename)
        expected = bound.get(str(target))
        if expected is None or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("User-pause recovery artifact is not bound or changed: " + str(target))
        return json.loads(target.read_text())

    receipt = read_bound(path)
    if (receipt.get("schema") != "splitfleet.controlled-user-pause-recovery.v1"
            or receipt.get("interruption_origin") != "explicit_user_request"
            or receipt.get("original_statuses_modified") is not False):
        raise ValueError("User-pause recovery classification is invalid")
    pause = read_bound(receipt["pause_receipt"])
    root = Path(receipt["original_root"])
    if (pause.get("status") != "paused_by_user"
            or pause.get("reason") != "User explicitly requested: 暂时停止实验"
            or Path(pause["population_root"]).resolve() != root.resolve()
            or pause.get("active_run_id") != receipt["run_id"]
            or pause.get("owned_transport_closed") is not True
            or pause.get("experiment_dispatch_stopped") is not True
            or pause.get("automatic_restart_authorized") is not False):
        raise ValueError("A user-pause recovery requires the original controlled-stop receipt")
    ledger_path = root / "ledger.json"
    ledger = read_bound(ledger_path)
    jobs = [job for job in ledger["jobs"] if job["run_id"] == receipt["run_id"]]
    if len(jobs) != 1:
        raise ValueError("User-pause recovery must identify exactly one original block")
    job = jobs[0]
    if (job.get("status") != "failed" or job.get("exit_code") != -2
            or not job["started_unix"] < pause["signal_sent_unix"] <= job["finished_unix"]
            or pause["signal_sent_unix"] < pause["requested_unix"]
            or job["stage"] != receipt["stage"] or job["seed"] != receipt["seed"]):
        raise ValueError("The original block was not interrupted by the recorded user SIGINT")
    attempts_path = root / "physical" / job["run_id"] / "attempts.json"
    attempts = read_bound(attempts_path)
    for original in (ledger_path, attempts_path):
        snapshot = pause["after_cleanup_snapshots"].get(str(original))
        if not snapshot or snapshot["sha256"] != bound[str(original)]:
            raise ValueError("User-pause ledger/attempts differ from the preserved stop snapshot")
    interrupted = sum(a["status"] == "running" for a in attempts)
    completed = sum(a["status"] == "completed" for a in attempts)
    if (interrupted != receipt["interrupted_attempts"] or interrupted != 1
            or any(a["status"] == "failed" for a in attempts)
            or completed != receipt["unpaired_successes_retained"]):
        raise ValueError("User-pause interruption or unpaired-success count changed")
    return interrupted
