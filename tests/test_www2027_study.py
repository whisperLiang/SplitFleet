"""Evidence gates for paired seed blocks and immutable study execution."""

import json
import hashlib
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


def freeze_plan(root, plan):
    study.save_json(root / "frozen_plan.json", plan)
    study.save_json(root / "ledger.json", {
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "jobs": [{"stage": stage["name"], "seed": seed, "status": "completed"}
                 for stage in plan["stages"] for seed in stage["seeds"]],
    })


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
    plan = plan_for(list(range(1, 13)))
    freeze_plan(tmp_path, plan)
    report = study.analyze(plan, tmp_path)["reports"][0]
    assert report["paired_n"] == 12
    assert all(effect["primary_claim_supported"] for effect in report["comparisons"])


def test_success_files_cannot_erase_a_failed_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    write_block(tmp_path, 1)
    study.save_json(tmp_path / "physical/timing_s1/attempts.json", [
        {"status": "failed", "scheme": "fedavg", "error": "oom"},
        {"status": "completed", "scheme": "fedavg"},
    ])
    report = study.analyze(plan_for([1]), tmp_path)
    assert report["reports"][0]["complete_seeds"] == []
    assert report["time_to_quality"] == report["communication"] == []
    assert report["attempts"][0]["failed_attempts"][0]["error"] == "oom"
    assert "attempt" in report["exclusions"][0]["reason"]


def test_failed_or_duplicate_ledger_jobs_reject_success_files(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    write_block(tmp_path, 1)
    plan = plan_for([1])
    freeze_plan(tmp_path, plan)
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    ledger["jobs"][0]["status"] = "failed"
    study.save_json(tmp_path / "ledger.json", ledger)
    assert study.analyze(plan, tmp_path)["reports"][0]["paired_n"] == 0
    ledger["jobs"].append({"stage": "timing", "seed": 1, "status": "completed"})
    study.save_json(tmp_path / "ledger.json", ledger)
    assert "duplicate" in study.analyze(plan, tmp_path)["exclusions"][0]["reason"]


def test_posthoc_seed_removal_or_missing_plan_binding_cannot_support_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    for seed in range(1, 13):
        write_block(tmp_path, seed)
    supplied = plan_for(list(range(1, 13)))
    unbound = study.analyze(supplied, tmp_path)
    assert not unbound["plan_bound_to_execution"]
    assert all(not effect["primary_claim_supported"] for effect in unbound["reports"][0]["comparisons"])
    freeze_plan(tmp_path, plan_for(list(range(1, 14))))
    shortened = study.analyze(supplied, tmp_path)
    assert shortened["reports"][0]["paired_n"] == 12
    assert not shortened["plan_bound_to_execution"]
    assert all(not effect["complete_planned_population"] for effect in shortened["reports"][0]["comparisons"])


def test_modified_frozen_threshold_file_does_not_match_execution_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    write_block(tmp_path, 1)
    plan = plan_for([1])
    plan["analysis_config"] = {"quality_metric": "accuracy", "time_to_quality_thresholds": [.6]}
    freeze_plan(tmp_path, plan)
    # Editing both supplied and saved plans cannot change the digest already
    # recorded by the execution ledger.
    plan["analysis_config"]["time_to_quality_thresholds"] = [.5]
    study.save_json(tmp_path / "frozen_plan.json", plan)
    analysis = study.analyze(plan, tmp_path)
    assert not analysis["plan_bound_to_execution"]
    assert not analysis["ttq_thresholds_bound_to_frozen_plan"]
    assert all(row["status"] == "unavailable" for row in analysis["time_to_quality"])


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


def test_ttq_exports_frozen_thresholds_and_keeps_incomplete_pairs_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    plan = plan_for([1, 2])
    plan["analysis_config"] = {"quality_metric": "accuracy", "time_to_quality_thresholds": [.6]}
    study.save_json(tmp_path / "frozen_plan.json", plan)
    write_block(tmp_path, 1)
    write_block(tmp_path, 2, omit="fedavg")
    for path in (tmp_path / "physical/timing_s1").glob("*/result.json"):
        result = json.loads(path.read_text())
        result["evaluation_records"] = [{"round_id": 1, "elapsed_sec": 3, "metrics": {"accuracy": .7}}]
        study.save_json(path, result)
    report = study.analyze(plan, tmp_path)
    assert len(report["time_to_quality"]) == len(study.SCHEMES)
    assert all(row["seed"] == 1 and row["time_sec"] == 3 for row in report["time_to_quality"])
    assert all(row["complete_paired_n"] == 1 for row in report["paired_ttq_summary"])
    assert all(not row["complete"] for row in report["communication"])
    plan["analysis_config"]["time_to_quality_thresholds"] = [.5]
    changed = study.analyze(plan, tmp_path)
    assert all(row["status"] == "unavailable" and row["time_sec"] is None for row in changed["time_to_quality"])


def text_ttq_block(root, seed):
    write_block(root, seed)
    folder = root / "physical" / f"timing_s{seed}"
    rows = json.loads((folder / "summary.json").read_text())
    for row in rows:
        row.update(task="text_classification", final_metrics={"accuracy": .92, "macro_f1": .78})
        path = Path(row["run_dir"]) / "result.json"
        result = json.loads(path.read_text())
        result["evaluation_records"] = [
            {"round_id": 1, "elapsed_sec": 2, "metrics": {"accuracy": .79, "macro_f1": .65}},
            {"round_id": 2, "elapsed_sec": 4, "metrics": {"accuracy": .85, "macro_f1": .72}},
            {"round_id": 3, "elapsed_sec": 8, "metrics": {"accuracy": .92, "macro_f1": .78}},
        ]
        study.save_json(path, result)
    study.save_json(folder / "summary.json", rows)


def text_ttq_plan(seeds):
    plan = plan_for(seeds)
    plan["stages"][0]["tasks"] = ["text_classification"]
    plan["quality_margin"] = {"text_classification": .01}
    plan["analysis_config"] = {
        "tasks": {"text_classification": {
            "quality_metric": "accuracy", "time_to_quality_thresholds": [.8, .9],
        }},
    }
    return plan


def test_text_ttq_uses_frozen_accuracy_without_excluding_macro_f1_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    for seed in (1, 2):
        text_ttq_block(tmp_path, seed)
    plan = text_ttq_plan([1, 2])
    freeze_plan(tmp_path, plan)
    analysis = study.analyze(plan, tmp_path)
    assert not analysis["exclusions"]
    assert analysis["plan_bound_to_execution"] and analysis["ttq_thresholds_bound_to_frozen_plan"]
    report = analysis["reports"][0]
    assert report["complete_seeds"] == [1, 2] and report["primary_metric"] == "macro_f1"
    assert all(scheme["mean_quality"] == .78 for scheme in report["schemes"].values())
    assert all(not comparison["primary_claim_supported"] for comparison in report["comparisons"])
    assert len(analysis["time_to_quality"]) == 2 * 2 * len(study.SCHEMES)
    assert all(row["metric"] == "accuracy" and row["status"] == "reached"
               and row["time_sec"] == { .8: 4, .9: 8 }[row["threshold"]]
               for row in analysis["time_to_quality"])


def test_frozen_ttq_metric_must_be_present_in_recorded_evaluations(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "summarize", fake_audit)
    text_ttq_block(tmp_path, 1)
    plan = text_ttq_plan([1])
    freeze_plan(tmp_path, plan)
    path = tmp_path / "physical/timing_s1/fedavg/result.json"
    result = json.loads(path.read_text())
    del result["evaluation_records"][0]["metrics"]["accuracy"]
    study.save_json(path, result)
    analysis = study.analyze(plan, tmp_path)
    assert analysis["reports"][0]["paired_n"] == 0
    assert analysis["exclusions"][0]["seed"] == 1
    assert "accuracy" in analysis["exclusions"][0]["reason"]
    assert not analysis["time_to_quality"] and not analysis["communication"]


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
