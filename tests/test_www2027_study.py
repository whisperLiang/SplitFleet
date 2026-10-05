"""Evidence gates for paired seed blocks and immutable study execution."""

import json
from pathlib import Path

from experiments import www2027_study as study


def plan_for(seeds):
    return {"stages": [{"name": "timing", "seeds": seeds, "tasks": ["image_classification"],
                        "calibration": False, "rounds": 10, "train_samples": 240,
                        "test_samples": 200, "optimizer": "adam", "learning_rate": 1e-4,
                        "batch_size": 4, "image_model": "resnet50_pretrained"}],
            "minimum_complete_pairs_for_primary_claim": 12, "quality_margin": {"image_classification": .01},
            "bootstrap_samples": 100, "analysis_seed": 2026}


def write_block(root, seed, *, source="same", omit=None, barrier=True):
    folder = root / "physical" / f"timing_s{seed}"
    folder.mkdir(parents=True)
    study.save_json(folder / "source_hashes.json", {"file": source})
    rows = []
    for scheme in study.SCHEMES:
        if scheme == omit:
            continue
        run = folder / scheme
        run.mkdir()
        method = "splitfed_fixed" if scheme.startswith("splitfed_fixed") else scheme
        fixed = scheme.removeprefix("splitfed_fixed") + "%" if method == "splitfed_fixed" else None
        result = {"seed": seed, "rounds": 10, "train_size": 240, "test_size": 200,
                  "optimizer": "adam", "learning_rate": 1e-4, "batch_size": 4, "image_model": "resnet50_pretrained",
                  "server_duration_sec": 5 if scheme == "splitfleet" else 10,
                  "final_model_hash": "model", "method": method, "fixed_boundary": fixed,
                  "online_cost_learning": method == "splitfleet"}
        study.save_json(run / "result.json", result)
        study.save_json(run / "validation_report.json", {"barrier_verified": barrier})
        rows.append({"task": "image_classification", "run_dir": str(run),
                     "final_metrics": {"accuracy": .7}})
    study.save_json(folder / "summary.json", rows)


def fake_audit(folder):
    rows = json.loads((folder / "summary.json").read_text())
    return {"scheme_count": len(rows), "runs": rows}


def test_partial_population_cannot_support_primary_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    for seed in [1, 2]:
        write_block(tmp_path, seed)
    report = study.analyze(plan_for(list(range(1, 13))), tmp_path)["reports"][0]
    assert report["paired_n"] == 2
    assert all(not effect["primary_claim_supported"] for effect in report["comparisons"])


def test_all_planned_seed_pairs_are_required_and_can_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    for seed in range(1, 13):
        write_block(tmp_path, seed)
    report = study.analyze(plan_for(list(range(1, 13))), tmp_path)["reports"][0]
    assert report["paired_n"] == 12
    assert all(effect["primary_claim_supported"] for effect in report["comparisons"])


def test_missing_baseline_rejects_entire_pair_and_keeps_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    write_block(tmp_path, 1)
    write_block(tmp_path, 2, omit="fedavg")
    study.save_json(tmp_path / "physical" / "timing_s2" / "attempts.json",
                    [{"status": "failed", "scheme": "fedavg", "error": "oom"}])
    result = study.analyze(plan_for([1, 2]), tmp_path)
    assert result["reports"][0]["complete_seeds"] == [1]
    assert result["exclusions"][0]["seed"] == 2
    assert result["attempts"][1]["failed_schemes"] == 1
    assert result["attempts"][1]["failed_attempts"][0]["error"] == "oom"


def test_source_change_and_missing_receipt_reject_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    write_block(tmp_path, 1)
    write_block(tmp_path, 2, source="modified")
    write_block(tmp_path, 3, barrier=False)
    result = study.analyze(plan_for([1, 2, 3]), tmp_path)
    assert result["reports"][0]["complete_seeds"] == [1]
    assert {entry["seed"] for entry in result["exclusions"]} == {2, 3}


def test_runtime_snapshot_does_not_follow_workspace_changes(tmp_path):
    workspace = tmp_path / "workspace"
    for folder in ("experiments", "splitfleet"):
        (workspace / folder).mkdir(parents=True)
        (workspace / folder / "__init__.py").write_text("original = True\n")
    runtime, record = study.freeze_runtime(workspace, tmp_path / "run")
    (workspace / "splitfleet" / "__init__.py").write_text("original = False\n")
    assert (runtime / "splitfleet" / "__init__.py").read_text() == "original = True\n"
    assert set(record["source_hashes"]) == {"experiments/__init__.py", "splitfleet/__init__.py"}


def nano_primary_block(tmp_path):
    write_block(tmp_path, 1)
    plan = plan_for([1])
    stage = plan["stages"][0]
    stage.update(tasks=["object_detection"], image_model="rfdetr_nano",
                 model_id="rfdetr_nano", checkpoint_sha256="nano-checkpoint")
    plan["quality_margin"] = {"object_detection": .01}
    plan["split_state_exchange"] = "owned"
    folder = tmp_path / "physical/timing_s1"
    rows = json.loads((folder / "summary.json").read_text())
    for row in rows:
        row.update(task="object_detection", final_metrics={"map50": .7})
        path = Path(row["run_dir"]) / "result.json"
        result = json.loads(path.read_text())
        result.update(image_model="rfdetr_nano", model_id="rfdetr_nano",
                      pretrain_checkpoint_sha256="nano-checkpoint", owned_state_exchange=True)
        study.save_json(path, result)
    study.save_json(folder / "summary.json", rows)
    return plan


def test_primary_nano_uses_stage_checkpoint_and_keeps_matched_exchange(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    plan = nano_primary_block(tmp_path)
    assert "checkpoint_sha256" not in plan
    analysis = study.analyze(plan, tmp_path)
    assert analysis["reports"][0]["complete_seeds"] == [1]
    assert not analysis["exclusions"]


def test_primary_nano_rejects_wrong_checkpoint_or_unmatched_exchange(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    plan = nano_primary_block(tmp_path)
    path = tmp_path / "physical/timing_s1/splitfed_fixed25/result.json"
    result = json.loads(path.read_text())
    for field, value, reason in [("pretrain_checkpoint_sha256", "wrong", "checkpoint"),
                                ("owned_state_exchange", False, "state exchange")]:
        study.save_json(path, {**result, field: value})
        analysis = study.analyze(plan, tmp_path)
        assert analysis["reports"][0]["paired_n"] == 0
        assert reason in analysis["exclusions"][0]["reason"]
