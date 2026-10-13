"""An explicit continuation cannot erase failures or mix task environments."""
import hashlib
import json

import pytest

from experiments.analysis import fedcompass_study as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def amended(tmp_path, monkeypatch):
    methods = ["fedavg", "fedcompass", "splitfed_fixed25", "splitfed_fixed50", "splitfed_fixed75", "splitfleet"]
    seeds = [10501, 10502, 10503, 10504, 10505]
    stages = [{"name": task, "tasks": [task], "seeds": seeds, "time_budget_sec": 300}
              for task in audit.PRIMARY]
    original = tmp_path / "original"
    continuation = tmp_path / "continuation"
    root = tmp_path / "publication"
    plan = {"stages": stages, "method_order": methods, "cosplit_hyperparameters": {"alpha_edge": 1.0}}
    original_hash = write(original / "frozen_plan.json", plan)
    next_plan = {**plan, "stages": stages[1:]}
    next_hash = write(continuation / "frozen_plan.json", next_plan)
    for folder, selected in ((original, stages[:1]), (continuation, stages[1:])):
        jobs = []
        for stage in selected:
            for seed in seeds:
                run_id = f"{stage['name']}_s{seed}"
                jobs.append({"stage": stage["name"], "seed": seed, "run_id": run_id, "status": "completed"})
                write(folder / "physical" / run_id / "attempts.json",
                      [{"status": "completed", "scheme": method} for method in methods])
        write(folder / "ledger.json", {"started_unix": 20, "jobs": jobs})
    failure = {"stage": stages[1]["name"], "seed": seeds[0], "run_id": "failed", "status": "failed"}
    original_ledger = json.loads((original / "ledger.json").read_text())
    original_ledger.update(started_unix=1)
    original_ledger["jobs"].append(failure)
    write(original / "ledger.json", original_ledger)
    write(original / "physical/failed/attempts.json", [{"status": "failed", "scheme": "splitfed_fixed25"}])
    manifest = {"schema": "splitfleet.fedcompass-amended-cohort.v1", "created_unix": 10,
        "phase_combination": "disjoint_tasks_only", "expected_primary_jobs": 120,
        "bound_artifacts_sha256": {str(original / "frozen_plan.json"): original_hash},
        "original_plan_path": str(original / "frozen_plan.json"), "runtime_source_identity": "source",
        "paired_seeds": seeds, "methods": methods, "retained_original_failed_jobs": 1,
        "phases": [{"root": str(original), "frozen_plan_sha256": original_hash,
                    "selected_stages": [stages[0]["name"]]},
                   {"root": str(continuation), "frozen_plan_sha256": next_hash,
                    "selected_stages": [s["name"] for s in stages[1:]]}]}
    write(root / "cohort_manifest.json", manifest)

    def phase(folder, **options):
        selected = options["selected_stages"]
        return {"runtime_source_identity": "source", "study_root": str(folder),
            "tasks": [{"task": task, "planned_seeds": seeds, "paired_n": 5} for task in selected],
            "methods": methods, "planned_jobs": 30 * len(selected), "completed_jobs": 30 * len(selected),
            "complete_planned_population": True, "source_artifacts_sha256": {},
            "cosplit_run_metadata_correction": None, "transport_metadata_correction": None,
            **{key: [] for key in ("observations", "checkpoints", "time_to_quality", "paired_ttq_summary",
                                  "pending_blocks_excluded_from_partial_inspection")}}
    monkeypatch.setattr(audit, "summarize_phase", phase)
    return root, continuation, manifest, phase


def test_failure_remains_visible_when_disjoint_completed_tasks_are_combined(amended):
    root, _, _, _ = amended
    evidence = audit.summarize(root)
    assert evidence["planned_jobs"] == evidence["completed_jobs"] == 120
    assert evidence["all_attempted_failed_jobs"] == 1
    assert evidence["retained_excluded_blocks"][0]["primary_endpoint_observations_used"] is False


def test_outcome_time_manifest_cannot_be_declared_prospective(amended):
    root, _, manifest, _ = amended
    manifest["created_unix"] = 21
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="before resumed execution"):
        audit.summarize(root)


def test_dependency_amendment_cannot_change_deadlines(amended):
    root, continuation, manifest, _ = amended
    path = continuation / "frozen_plan.json"
    plan = json.loads(path.read_text())
    plan["stages"][0]["time_budget_sec"] = 600
    manifest["phases"][1]["frozen_plan_sha256"] = write(path, plan)
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="paired protocol"):
        audit.summarize(root)


def test_task_cannot_merge_seeds_from_different_dependency_phases(amended, monkeypatch):
    root, continuation, _, original_phase = amended

    def mixed(folder, **options):
        result = original_phase(folder, **options)
        if folder == continuation:
            result["tasks"][0]["task"] = next(iter(audit.PRIMARY))
        return result
    monkeypatch.setattr(audit, "summarize_phase", mixed)
    with pytest.raises(ValueError, match="overlaps phases"):
        audit.summarize(root)


def test_further_failed_attempt_cannot_hide_inside_continuation(amended):
    root, continuation, _, _ = amended
    write(continuation / "ledger.json", {"started_unix": 20, "jobs": [
        {"stage": "text_classification", "seed": 10501, "run_id": "failure2", "status": "failed"}]})
    write(continuation / "physical/failure2/attempts.json", [{"status": "failed"}])
    with pytest.raises(ValueError, match="further explicit amendment"):
        audit.summarize(root)


def test_endpoint_table_cannot_exceed_retained_success_receipts(amended):
    root, continuation, _, _ = amended
    path = next((continuation / "physical").glob("*/attempts.json"))
    attempts = json.loads(path.read_text())
    write(path, attempts[:-1])
    with pytest.raises(ValueError, match="Completed-attempt denominator"):
        audit.summarize(root)


def fixed_reboot(amended, monkeypatch):
    root, continuation, manifest, original_phase = amended
    methods = [m for m in manifest["methods"] if m != "fedcompass"]
    manifest.update(expected_primary_jobs=100, methods=methods,
                    retained_original_failed_jobs=0, retained_original_interrupted_jobs=1)
    for phase in manifest["phases"]:
        folder = audit.Path(phase["root"])
        path = folder / "frozen_plan.json"
        plan = json.loads(path.read_text())
        plan.update(method_order=methods, primary_objective="fixed_rounds_sync_time_and_quality")
        for stage in plan["stages"]:
            stage.pop("time_budget_sec")
            stage["rounds"] = 10
        phase["frozen_plan_sha256"] = write(path, plan)
        if str(path) in manifest["bound_artifacts_sha256"]:
            manifest["bound_artifacts_sha256"][str(path)] = phase["frozen_plan_sha256"]
        for path in (folder / "physical").glob("*/attempts.json"):
            attempts = [a for a in json.loads(path.read_text()) if a["scheme"] != "fedcompass"]
            for attempt in attempts:
                if attempt["status"] == "failed":
                    attempt["status"] = "running"
            write(path, attempts)
    receipt = root / "interruption.json"
    manifest["interruption_receipt"] = str(receipt)
    manifest["bound_artifacts_sha256"][str(receipt)] = write(receipt, {
        "interrupted_attempts": 1, "previous_boot_id": "old", "current_boot_id": "new"})
    write(root / "cohort_manifest.json", manifest)

    def phase(folder, **options):
        value = original_phase(folder, **options)
        value.update(methods=methods, planned_jobs=25 * len(options["selected_stages"]),
                     completed_jobs=25 * len(options["selected_stages"]))
        return value
    monkeypatch.setattr(audit, "summarize_phase", phase)
    return root, continuation, manifest


def test_fixed_round_reboot_retains_complete_tasks_and_interruption(amended, monkeypatch):
    root, _, _ = fixed_reboot(amended, monkeypatch)
    value = audit.summarize(root)
    assert value["completed_jobs"] == value["planned_jobs"] == 100
    assert value["all_attempted_interrupted_jobs"] == 1
    assert value["all_attempted_failed_jobs"] == 0


def test_reboot_cannot_be_declared_without_bound_interruption_evidence(amended, monkeypatch):
    root, _, manifest = fixed_reboot(amended, monkeypatch)
    manifest["bound_artifacts_sha256"].pop(manifest["interruption_receipt"])
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="not hash bound"):
        audit.summarize(root)


def test_fixed_round_continuation_cannot_switch_to_deadline_objective(amended, monkeypatch):
    root, continuation, manifest = fixed_reboot(amended, monkeypatch)
    path = continuation / "frozen_plan.json"
    plan = json.loads(path.read_text())
    plan["primary_objective"] = "quality_at_common_deadline"
    manifest["phases"][1]["frozen_plan_sha256"] = write(path, plan)
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="changed training"):
        audit.summarize(root)
