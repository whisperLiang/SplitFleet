"""Boot recovery preserves whole pairs and forbids selective replacements."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.analysis import fedcompass_study as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(params=[5, 8])
def recovered(tmp_path, monkeypatch, request):
    seeds = list(range(10501, 10506))
    tasks = list(audit.PRIMARY)
    methods = ["fedavg", "splitfed_fixed25", "splitfed_fixed50", "splitfed_fixed75", "splitfleet"]
    fixed = request.param == 5
    if not fixed:
        methods = ["fedavg", "fedasync", "fedbuff", "fedcompass", *methods[1:]]
    plan = {"source_identity": "source", "primary_objective": "fixed_rounds_sync_time_and_quality",
        "method_order": ["fedavg", "splitfed_fixed", "splitfleet"], "analysis_seed": 2026,
        "bootstrap_samples": 100, "stages": [{"name": t, "tasks": [t], "image_model": t,
                                              "seeds": seeds, "rounds": 10} for t in tasks]}
    if not fixed:
        plan.update(primary_objective="fixed_budget_training_speed",
                    method_order=["fedavg", "fedasync", "fedbuff", "fedcompass", "splitfed_fixed", "splitfleet"],
                    time_budget_sec=300)
    first, next_phase, publication = [tmp_path / n for n in ("first", "next", "publication")]
    selections = [{tasks[0]: seeds, tasks[1]: seeds[:2]},
                  {tasks[1]: seeds[2:], tasks[2]: seeds, tasks[3]: seeds}]
    phases, bound = [], {}
    for folder, selection, start in zip((first, next_phase), selections, (1, 20)):
        path = folder / "frozen_plan.json"
        digest = write(path, plan)
        bound[str(path)] = digest
        phases.append({"root": str(folder), "frozen_plan_sha256": digest,
                       "selected_seeds": selection, "prospective_continuation": start == 20})
        jobs = []
        for task, chosen in selection.items():
            for seed in chosen:
                run_id = f"{task}_s{seed}"
                jobs.append({"run_id": run_id, "stage": task, "seed": seed, "status": "completed"})
                write(folder / "physical" / run_id / "attempts.json",
                      [{"scheme": m, "status": "completed"} for m in methods])
        write(folder / "ledger.json", {"started_unix": start, "jobs": jobs})
    manifest = {"schema": "splitfleet.paired-block-recovery.v1", "created_unix": 10,
        "phase_combination": "disjoint_complete_paired_blocks", "expected_primary_jobs": 20 * len(methods),
        "original_plan_path": str(first / "frozen_plan.json"), "runtime_source_identity": "source",
        "methods": methods, "retained_original_failed_jobs": 0, "retained_original_interrupted_jobs": 0,
        "bound_artifacts_sha256": bound, "phases": phases}
    write(publication / "cohort_manifest.json", manifest)

    def phase(folder, **options):
        rows = []
        for task, chosen in options["selected_seeds"].items():
            for seed in chosen:
                for i, method in enumerate(methods):
                    rows.append({"model": task, "task": task, "method": method, "seed": seed,
                        "quality_pct": 50. + i + (seed-10501)/10., "server_duration_sec": 100.-i,
                        "global_updates": 10, "incorporated_optimizer_steps": 600,
                        "completed_optimizer_steps_including_drain": 600, "incorporated_examples": 2400,
                        "application_bytes_including_drain": 1000})
        return {"runtime_source_identity": "source", "shared_execution_source_identity": "shared",
            "study_root": str(folder), "observations": rows,
            "source_artifacts_sha256": {}, "checkpoints": [], "time_to_quality": [],
            "pending_blocks_excluded_from_partial_inspection": [], "complete_planned_population": True,
            "completed_jobs": len(rows), "methods": methods,
            "protocol": "fixed_rounds_sync" if fixed else "common_wall_clock"}
    monkeypatch.setattr(audit, "summarize_phase", phase)
    return publication, first, next_phase, manifest


def test_recovery_recombines_raw_pairs_into_original_five_seed_tasks(recovered):
    root, _, _, manifest = recovered
    result = audit.summarize(root)
    assert result["completed_jobs"] == result["planned_jobs"] == manifest["expected_primary_jobs"]
    assert all(t["paired_n"] == 5 for t in result["tasks"])
    assert all(len(t["time_effects"]) == (4 if len(manifest["methods"]) == 5 else 0) for t in result["tasks"])
    assert all(len(t["quality_effects"]) == len(manifest["methods"]) - 1 for t in result["tasks"])


def test_recovery_cannot_overlap_seed_blocks(recovered):
    root, _, _, manifest = recovered
    task = list(audit.PRIMARY)[1]
    manifest["phases"][1]["selected_seeds"][task].append(10502)
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="overlaps"):
        audit.summarize(root)


def test_recovery_cannot_replace_an_earlier_complete_pair(recovered):
    root, first, _, manifest = recovered
    task = list(audit.PRIMARY)[1]
    path = first / "ledger.json"
    ledger = json.loads(path.read_text())
    run_id = f"{task}_s10503"
    ledger["jobs"].append({"run_id": run_id, "stage": task, "seed": 10503, "status": "completed"})
    write(path, ledger)
    write(first / "physical" / run_id / "attempts.json", [{"status": "completed"}] * len(manifest["methods"]))
    with pytest.raises(ValueError, match="cannot be replaced"):
        audit.summarize(root)


def test_recovery_cannot_bind_selection_after_resumed_outcomes(recovered):
    root, _, _, manifest = recovered
    manifest["created_unix"] = 21
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="before outcomes"):
        audit.summarize(root)


def test_recovery_cannot_change_training_settings_even_when_rehashed(recovered):
    root, _, next_phase, manifest = recovered
    path = next_phase / "frozen_plan.json"
    plan = json.loads(path.read_text())
    plan["learning_rate"] = 0.1
    digest = write(path, plan)
    manifest["phases"][1]["frozen_plan_sha256"] = digest
    manifest["bound_artifacts_sha256"][str(path)] = digest
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="canonical training"):
        audit.summarize(root)


def add_excluded_failure_phase(recovered):
    root, first, _, manifest = recovered
    folder = root.parent / "failed"
    plan = json.loads((first / "frozen_plan.json").read_text())
    bound = manifest["bound_artifacts_sha256"]
    for name, value in (("frozen_plan.json", plan), ("executed_plan.json", plan),
                        ("runtime_manifest.json", {"source_identity": "source"})):
        path = folder / name
        bound[str(path)] = write(path, value)
    task = list(audit.PRIMARY)[1]
    run_id = f"{task}_s10503"
    ledger = {"started_unix": 1, "jobs": [{"stage": task, "seed": 10503,
        "run_id": run_id, "status": "failed"}]}
    bound[str(folder / "ledger.json")] = write(folder / "ledger.json", ledger)
    path = folder / "physical" / run_id / "attempts.json"
    bound[str(path)] = write(path, [{"status": "completed"}, {"status": "completed"}, {"status": "failed"}])
    manifest["phases"].insert(1, {"root": str(folder), "selected_seeds": {},
        "frozen_plan_sha256": bound[str(folder / "frozen_plan.json")]})
    receipt = folder / "failure_receipt.json"
    bound[str(receipt)] = write(receipt, {"schema": "splitfleet.operational-failure-recovery.v1",
                                        "failed_method_attempts": 1})
    manifest["operational_failure_receipts"] = [str(receipt)]
    manifest["retained_original_failed_jobs"] = 1
    write(root / "cohort_manifest.json", manifest)
    return root, folder, manifest


def test_recovery_retains_failed_phase_without_using_unpaired_outcomes(recovered):
    root, _, _ = add_excluded_failure_phase(recovered)
    result = audit.summarize(root)
    assert result["completed_jobs"] == result["planned_jobs"]
    assert result["all_attempted_completed_jobs"] == result["planned_jobs"] + 2
    assert result["excluded_completed_jobs"] == 2
    assert result["all_attempted_failed_jobs"] == 1


def test_recovery_requires_bound_metadata_for_excluded_phase(recovered):
    root, folder, manifest = add_excluded_failure_phase(recovered)
    manifest["bound_artifacts_sha256"].pop(str(folder / "runtime_manifest.json"))
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="Excluded phase artifacts"):
        audit.summarize(root)


def test_recovery_requires_failure_count_receipt(recovered):
    root, _, manifest = add_excluded_failure_phase(recovered)
    manifest["operational_failure_receipts"] = []
    write(root / "cohort_manifest.json", manifest)
    with pytest.raises(ValueError, match="Failed methods lack"):
        audit.summarize(root)


def test_recovery_rejects_changed_shared_execution_even_with_same_runtime_label(recovered, monkeypatch):
    root, _, next_phase, _ = recovered
    original = audit.summarize_phase
    def changed(folder, **options):
        value = original(folder, **options)
        if folder == next_phase:
            value["shared_execution_source_identity"] = "changed"
        return value
    monkeypatch.setattr(audit, "summarize_phase", changed)
    with pytest.raises(ValueError, match="shared execution"):
        audit.summarize(root)
