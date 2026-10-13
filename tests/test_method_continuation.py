import json
from pathlib import Path

import pytest

from experiments.analysis.method_continuation import expected_remaining_command, sha, validate_phase_continuations


@pytest.fixture
def continuation(tmp_path):
    original = tmp_path / "original"
    execution = tmp_path / "execution"
    original.mkdir(); execution.mkdir()
    run_id = "rfdetr_nano_budget8_s10505"
    old = original / "physical" / run_id
    old.mkdir(parents=True)
    retained = {"scheme": "splitfed_fixed50", "status": "completed"}
    (old / "attempts.json").write_text(json.dumps([retained]))
    result = old / "object_detection_splitfed_fixed50" / "result.json"
    result.parent.mkdir(); result.write_text('{"seed": 10505}')
    for root in (original, execution):
        source = root / "runtime_snapshot" / "execution.py"
        source.parent.mkdir(); source.write_text("unchanged training source\n")
        (root / "runtime_manifest.json").write_text(json.dumps({"source_hashes": {"execution.py": sha(source)}}))
    command = ["python", "--seed", "10505", "--learning-rate", "1e-05", "--output-root", str(original / "physical"),
               "--log-root", str(original / "logs"), "--fixed-boundaries", "50%", "75%", "25%", "--time-budget-sec", "300"]
    receipt_path = tmp_path / "continuation.json"
    receipt = {"schema": "splitfleet.user-directed-method-continuation.v1",
        "user_instruction": "原失败记录不需要保留，直接从RF-DETR 种子 10505 的 Fixed75开始重跑后续实验",
        "original_phase_root": str(original), "execution_root": str(execution), "run_id": run_id,
        "runtime_source_identity": "same-source", "discarded_failed_method_attempts": 1,
        "retained_schemes": ["splitfed_fixed50"], "retained_completed_attempts": [retained],
        "remaining_schemes": ["splitfed_fixed75", "splitfed_fixed25", "splitfleet", "fedavg", "fedasync", "fedbuff", "fedcompass"],
        "original_command": command, "remaining_command": expected_remaining_command(command, execution),
        "created_unix": 10, "bound_artifacts_sha256": {str(result): sha(result)}}
    job = {"run_id": run_id, "seed": 10505, "stage": "rfdetr_nano_budget8", "status": "running", "command": command,
           "method_continuation_receipt": str(receipt_path)}
    execution_ledger = {"started_unix": 11, "runtime_source_identity": "same-source",
                       "jobs": [{"command": receipt["remaining_command"]}]}
    (execution / "ledger.json").write_text(json.dumps(execution_ledger))

    def bind():
        receipt_path.write_text(json.dumps(receipt))
        job["method_continuation_receipt_sha256"] = sha(receipt_path)

    bind()
    return original, execution, result, receipt, job, bind


def run(fixture):
    root, _, _, _, job, _ = fixture
    return validate_phase_continuations(root, {"source_identity": "same-source"}, {"jobs": [job]})


def test_explicit_method_resume_preserves_failure_count_and_success(continuation):
    receipts, artifacts = run(continuation)
    assert receipts[0]["discarded_failed_method_attempts"] == 1
    assert str(continuation[2]) in artifacts


def test_training_arguments_cannot_change(continuation):
    receipt = continuation[3]
    receipt["remaining_command"][receipt["remaining_command"].index("--learning-rate") + 1] = "0.001"
    continuation[5]()
    with pytest.raises(ValueError, match="training argument"):
        run(continuation)


def test_retained_fixed50_cannot_be_replaced(continuation):
    continuation[2].write_text('{"seed": 10504}')
    with pytest.raises(ValueError, match="Retained success"):
        run(continuation)


def test_training_source_cannot_be_changed(continuation):
    (continuation[1] / "runtime_snapshot/execution.py").write_text("different training\n")
    with pytest.raises(ValueError, match="training snapshot"):
        run(continuation)


def test_discarding_logs_cannot_hide_failed_attempt_count(continuation):
    continuation[3]["discarded_failed_method_attempts"] = 0
    continuation[5]()
    with pytest.raises(ValueError, match="authorized original block"):
        run(continuation)


def test_retroactive_method_selection_rejected(continuation):
    continuation[3]["created_unix"] = 12
    continuation[5]()
    with pytest.raises(ValueError, match="before its outcomes"):
        run(continuation)


def test_seed_substitution_rejected(continuation):
    continuation[4]["seed"] = 10506
    with pytest.raises(ValueError, match="authorized original block"):
        run(continuation)


def test_completed_block_cannot_omit_remaining_methods(continuation):
    continuation[4]["status"] = "completed"
    with pytest.raises(ValueError, match="seven remaining successes"):
        run(continuation)
