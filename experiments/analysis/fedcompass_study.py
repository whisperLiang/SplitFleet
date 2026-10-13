"""Audited quality-at-deadline evidence for the physical FedCompass comparison.

The final metric evaluates the last model committed before the deadline, even
when draining outstanding RPCs and evaluating that model finish afterwards.
Completed work and application bytes include that drain; neither is presented
as work/traffic observed exclusively within the training budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    # Raw-artifact/statistical auditing needs CPU only. Avoid GPU runtime
    # initialization and extra math workers alongside measured training.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from experiments.analysis.communication import communication_summary
from experiments.common.statistics import holm_adjust, paired_effect
from experiments.physical_multitask import scheme_name
from experiments.summarize_physical_multitask import PRIMARY
from experiments.www2027_study import analyze, save_json, schemes_for


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_published_endpoints(evidence_path: Path) -> dict:
    """Check published successes after unused attempt records were removed.

    The frozen evidence fixes the original population and attempt counts. This
    audit verifies every published result and retained source; it does not
    reconstruct deleted failures or exclusions.
    """
    evidence_path = evidence_path.resolve()
    value = json.loads(evidence_path.read_text())
    workspace = Path(__file__).resolve().parents[2]
    receipt_path = workspace / "paper/evidence/manuscript_cleanup_20261013/receipt.json"
    if not receipt_path.exists():
        current = summarize(Path(value["study_root"]))
        for key in ("observations", "tasks", "checkpoints", "methods", "time_to_quality",
                    "runtime_source_identity", "shared_execution_source_identity"):
            if value[key] != json.loads(json.dumps(current[key])):
                raise ValueError("Published evidence differs from its raw audit: " + key)
        return {"scope": "complete_original_attempt_population", "published_jobs": len(value["observations"])}
    receipt = json.loads(receipt_path.read_text())
    relative = str(evidence_path.relative_to(workspace))
    if receipt["published_evidence_sha256"].get(relative) != digest(evidence_path):
        raise ValueError("Published evidence is not bound to the cleanup receipt")
    removed = {str(workspace / row["path"]): row["sha256"] for row in receipt["removed_files"] if "sha256" in row}
    analysis_files = {str(workspace / name) for name in value["analysis_source_sha256"]}
    checked = pruned = 0
    updated_analysis = []
    for name, expected in value["source_artifacts_sha256"].items():
        path = Path(name)
        if name in removed:
            if removed[name] != expected or path.exists():
                raise ValueError("Deleted source does not match its recorded hash: " + name)
            pruned += 1
        elif path.is_file():
            actual = digest(path)
            if actual != expected:
                if name not in analysis_files:
                    raise ValueError("Retained execution artifact changed: " + name)
                updated_analysis.append(name)
            checked += 1
        else:
            raise ValueError("Unrecorded missing execution artifact: " + name)
    methods = value["methods"]
    seeds = [10501, 10502, 10503, 10504, 10505]
    models = {task["model"] for task in value["tasks"]}
    expected_population = {(model, method, seed) for model in models for method in methods for seed in seeds}
    observed = [(row["model"], row["method"], row["seed"]) for row in value["observations"]]
    if (len(models) != 4 or len(observed) != len(set(observed)) or set(observed) != expected_population
            or len(observed) != value["planned_jobs"] or len(observed) != value["completed_jobs"]
            or not value["complete_planned_population"] or value["cohorts_pooled"]):
        raise ValueError("Published endpoints do not cover the frozen paired population")
    fixed_rounds = value["protocol"] == "fixed_rounds_sync"
    paired_inputs = {}
    for row in value["observations"]:
        result_path = Path(row["run_dir"]) / "result.json"
        # A deletion receipt can never replace a published success.
        if not result_path.is_file() or digest(result_path) != row["result_sha256"]:
            raise ValueError("Missing or changed published endpoint: " + str(result_path))
        raw = json.loads(result_path.read_text())
        final = raw["evaluation_records"][-1]
        if (raw["fit_failures"] or raw["model_id"] != row["model"] or raw["task"] != row["task"]
                or raw["seed"] != row["seed"] or scheme_name(raw["method"], raw["fixed_boundary"]) != row["method"]
                or final["metrics"][row["metric"]] != row["quality"]
                or 100 * row["quality"] != row["quality_pct"]
                or raw["server_duration_sec"] != row["server_duration_sec"]):
            raise ValueError("Published endpoint differs from the completed raw result")
        for key in ("initial_model_hash", "final_model_hash", "data_content_hash", "partition_hash",
                    "worker_ids", "train_size", "test_size"):
            if raw[key] != row[key]:
                raise ValueError("Published training identity differs: " + key)
        pairing = tuple(raw[key] for key in ("initial_model_hash", "data_content_hash", "partition_hash"))
        group = row["model"], row["seed"]
        if group in paired_inputs and pairing != paired_inputs[group]:
            raise ValueError("Methods do not share initialization and partitions within a seed")
        paired_inputs[group] = pairing
        if fixed_rounds:
            if raw["rounds"] != 10 or final["round_id"] != 10 or raw.get("time_budget_sec") is not None:
                raise ValueError("Fixed-round success did not complete its planned work")
        else:
            budget = row["time_budget_sec"]
            commit = raw["last_committed_model_elapsed_sec"]
            if (final.get("evaluation_kind") != "final_frozen_model" or not 0 <= commit < budget
                    or final.get("model_version") != raw["global_updates"]
                    or final.get("model_committed_elapsed_sec") != commit):
                raise ValueError("Budget endpoint is not the model committed before the deadline")
    return {"scope": "published_successes_and_retained_sources_after_cleanup",
            "published_jobs": len(observed), "retained_sources_checked": checked,
            "deleted_sources_verified_against_cleanup_receipt": pruned,
            "updated_analysis_files": updated_analysis,
            "deleted_attempt_population_reaudited": False}


def summarize(root: Path, *, require_complete: bool = True) -> dict:
    root = root.resolve()
    if (root / "replacement_population.json").exists():
        from experiments.analysis.population_replacement import validate_replacement
        corrected, retirement, artifacts = validate_replacement(root)
        value = summarize(corrected, require_complete=require_complete)
        value.update(study_root=str(root), executed_population_root=str(corrected),
                     retired_populations=[retirement], cohorts_pooled=False)
        value["source_artifacts_sha256"].update(artifacts)
        return value
    if (root / "cohort_manifest.json").exists():
        if json.loads((root / "cohort_manifest.json").read_text()).get("schema") == "splitfleet.paired-block-recovery.v1":
            return summarize_block_recovery(root, require_complete=require_complete)
        return summarize_amendment(root, require_complete=require_complete)
    return summarize_phase(root, require_complete=require_complete)


def summarize_phase(root: Path, *, require_complete: bool = True,
                    selected_stages: list[str] | None = None,
                    selected_seeds: dict[str, list[int]] | None = None) -> dict:
    root = root.resolve()
    plan = json.loads((root / "frozen_plan.json").read_text())
    ledger = json.loads((root / "ledger.json").read_text())
    from experiments.analysis.method_continuation import validate_phase_continuations
    method_continuations, continuation_artifacts = validate_phase_continuations(root, plan, ledger)
    audited = analyze(plan, root)
    stages = plan["stages"]
    if selected_stages is not None:
        if (not selected_stages or len(selected_stages) != len(set(selected_stages))
                or not set(selected_stages) <= {stage["name"] for stage in stages}):
            raise ValueError("Invalid amendment stage selection")
        stages = [stage for stage in stages if stage["name"] in selected_stages]
        audited["reports"] = [row for row in audited["reports"] if row["stage"] in selected_stages]
        audited["exclusions"] = [row for row in audited["exclusions"] if row["stage"] in selected_stages]
        audited["time_to_quality"] = [row for row in audited["time_to_quality"] if row["stage"] in selected_stages]
        from experiments.analysis.time_to_quality import summarize_paired_ttq
        audited["paired_ttq_summary"] = summarize_paired_ttq(audited["time_to_quality"])
    if selected_seeds is not None:
        if set(selected_seeds) != {s["name"] for s in stages}:
            raise ValueError("Paired-block selection does not match selected stages")
        for stage, report in zip(stages, audited["reports"], strict=True):
            seeds = selected_seeds[stage["name"]]
            if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= set(stage["seeds"]):
                raise ValueError("Invalid paired-block seed selection")
            report["planned_seeds"] = seeds
            report["complete_seeds"] = [s for s in report["complete_seeds"] if s in seeds]
            report["paired_n"] = len(report["complete_seeds"])
        stages = [{**s, "seeds": selected_seeds[s["name"]]} for s in stages]
        for key in ("exclusions", "time_to_quality"):
            audited[key] = [r for r in audited[key] if r.get("seed") in selected_seeds.get(r.get("stage"), [])]
        from experiments.analysis.time_to_quality import summarize_paired_ttq
        audited["paired_ttq_summary"] = summarize_paired_ttq(audited["time_to_quality"])
    schemes = schemes_for(plan)
    fixed_rounds = plan.get("primary_objective") == "fixed_rounds_sync_time_and_quality"
    if fixed_rounds and (set(schemes) != {"fedavg", "splitfed_fixed25", "splitfed_fixed50", "splitfed_fixed75", "splitfleet"}
                         or any(s.get("time_budget_sec", plan.get("time_budget_sec")) is not None for s in stages)):
        raise ValueError("Fixed-round evidence requires the five synchronous methods without a deadline")
    if not fixed_rounds and ("fedcompass" not in schemes or "fedprox" in schemes):
        raise ValueError("This evidence requires the prospectively matched FedCompass cohort")
    pending = {(job["stage"], job["seed"]) for job in ledger["jobs"] if job["status"] == "running"}
    pending_exclusions = [row for row in audited["exclusions"] if (row["stage"], row["seed"]) in pending]
    fatal_exclusions = [row for row in audited["exclusions"] if row not in pending_exclusions]
    if (not audited["plan_bound_to_execution"] or fatal_exclusions
            or (require_complete and pending_exclusions)):
        raise ValueError(f"Execution audit failed: {audited['exclusions']}")
    jobs = Counter((job["stage"], job["seed"], job["status"]) for job in ledger["jobs"])
    complete = ((ledger["status"] == "completed" or selected_stages is not None)
        and all(report["complete_seeds"] == sorted(report["planned_seeds"])
                for report in audited["reports"])
        and all(jobs[(stage["name"], seed, "completed")] == 1
                for stage in stages for seed in stage["seeds"]))
    if require_complete and not complete:
        raise ValueError("Publication evidence requires every prospectively planned paired seed")
    artifacts = {str(root / name): digest(root / name) for name in (
        "frozen_plan.json", "executed_plan.json", "runtime_manifest.json", "ledger.json")}
    artifacts.update(continuation_artifacts)
    runtime = json.loads((root / "runtime_manifest.json").read_text())
    frozen_hashes = runtime["source_hashes"]
    identity = hashlib.sha256(json.dumps(frozen_hashes, sort_keys=True).encode()).hexdigest()
    if identity != runtime["source_identity"]:
        raise ValueError("Frozen runtime source identity does not match its manifest")
    for name, expected in frozen_hashes.items():
        path = root / "runtime_snapshot" / name
        if digest(path) != expected:
            raise ValueError(f"Frozen runtime file changed: {name}")
        artifacts[str(path)] = expected
    from experiments.analysis.transport_source_amendment import comparison_source
    training_identity, source_amendment, amendment_artifacts = comparison_source(root, runtime, plan, ledger)
    artifacts.update(amendment_artifacts)
    from experiments.analysis.baseline_source_amendment import baseline_source, shared_execution_identity
    baseline_amendment, baseline_artifacts, guard_hashes = baseline_source(root, runtime, plan, ledger)
    artifacts.update(baseline_artifacts)
    shared_identity = shared_execution_identity(root, runtime, guard_hashes)
    correction_path = root / "transport_metadata_correction.json"
    correction = json.loads(correction_path.read_text()) if correction_path.exists() else None
    if correction:
        if (correction["original_frozen_plan_sha256"] != digest(root / "frozen_plan.json")
                or not correction["metadata_only"] or correction["training_parameters_changed"]
                or correction["runs_restarted"]):
            raise ValueError("Transport metadata correction changes the primary protocol")
        artifacts[str(correction_path)] = digest(correction_path)
    cosplit_correction_path = root / "cosplit_run_metadata_correction.json"
    if cosplit_correction_path.exists():
        artifacts[str(cosplit_correction_path)] = digest(cosplit_correction_path)
    analysis_sources = {}
    workspace = Path(__file__).resolve().parents[2]
    for name in ("experiments/analysis/fedcompass_study.py", "experiments/www2027_study.py",
                 "experiments/common/statistics.py", "experiments/analysis/communication.py",
                 "experiments/analysis/time_to_quality.py", "experiments/summarize_physical_multitask.py",
                 "experiments/analysis/transport_source_amendment.py",
                 "experiments/analysis/baseline_source_amendment.py",
                 "experiments/analysis/population_replacement.py"):
        analysis_sources[name] = digest(workspace / name)
        artifacts[str(workspace / name)] = analysis_sources[name]
    if method_continuations:
        name = "experiments/analysis/method_continuation.py"
        analysis_sources[name] = digest(workspace / name)
        artifacts[str(workspace / name)] = analysis_sources[name]
    rows, checkpoints, tasks = [], [], []
    for stage, report in zip(stages, audited["reports"], strict=True):
        if len(stage["tasks"]) != 1 or stage["calibration"]:
            raise ValueError("Expected one primary task per stage")
        task = stage["tasks"][0]
        budget = stage.get("time_budget_sec", plan.get("time_budget_sec"))
        if (not fixed_rounds and not budget) or report["task"] != task:
            raise ValueError("A frozen wall-clock budget and matching task are required")
        metric = PRIMARY[task]
        observations = {method: [] for method in schemes}
        for seed in report["complete_seeds"]:
            folder = root / "physical" / f"{stage['name']}_s{seed}"
            attempts = json.loads((folder / "attempts.json").read_text())
            if (len(attempts) != len(schemes) or any(a["status"] != "completed" for a in attempts)
                    or Counter(a["scheme"] for a in attempts) != Counter(schemes)):
                raise ValueError("Complete endpoints require one successful attempt per planned method")
            for path in folder.glob("*.json"):
                artifacts[str(path)] = digest(path)
            for name, expected in frozen_hashes.items():
                path = folder / "source_snapshot" / name
                if digest(path) != expected:
                    raise ValueError(f"Executed seed snapshot changed: {path}")
                artifacts[str(path)] = expected
            block = json.loads((folder / "summary.json").read_text())
            population = []
            for job in block:
                run_dir = Path(job["run_dir"])
                result_path = run_dir / "result.json"
                result = json.loads(result_path.read_text())
                method = scheme_name(result["method"], result["fixed_boundary"])
                population.append(method)
                for path in run_dir.glob("*.json"):
                    artifacts[str(path)] = digest(path)
                final = result["evaluation_records"][-1]
                commit = result.get("last_committed_model_elapsed_sec")
                if not fixed_rounds and (final.get("evaluation_kind") != "final_frozen_model"
                        or final.get("model_version") != result["global_updates"]
                        or final.get("model_committed_elapsed_sec") != commit
                        or not 0 <= commit < budget
                        or final["metrics"] != job["final_metrics"]):
                    raise ValueError("Endpoint does not evaluate the last model committed before the deadline")
                if fixed_rounds and (result["rounds"] != stage["rounds"]
                        or final["round_id"] != stage["rounds"] or final["metrics"] != job["final_metrics"]
                        or result.get("time_budget_sec") is not None):
                    raise ValueError("Fixed-round endpoint does not finish every planned round")
                fits = result["fit_records"]
                accepted = fits if fixed_rounds else [fit for fit in fits if fit.get("update_status") == "aggregated"]
                comm = communication_summary(result)
                if not comm["complete"]:
                    raise ValueError("Application-buffer accounting does not cover every completed assignment")
                row = {
                    "model": stage["image_model"], "task": task, "method": method,
                    "seed": seed, "metric": metric, "quality": final["metrics"][metric],
                    "quality_pct": 100 * final["metrics"][metric], "time_budget_sec": budget,
                    "server_duration_sec": result["server_duration_sec"],
                    "model_committed_elapsed_sec": commit,
                    "final_evaluation_elapsed_sec": final["elapsed_sec"],
                    "global_updates": result.get("global_updates", result["rounds"]),
                    "completed_assignments_including_drain": len(fits),
                    "incorporated_optimizer_steps": sum(fit["metrics"]["num_batches"] for fit in accepted),
                    "completed_optimizer_steps_including_drain": sum(fit["metrics"]["num_batches"] for fit in fits),
                    "incorporated_examples": sum(fit["num_examples"] for fit in accepted),
                    "completed_examples_including_drain": sum(fit["num_examples"] for fit in fits),
                    "update_status_counts": dict(Counter(fit.get("update_status", "aggregated") for fit in fits)),
                    "application_bytes_including_drain": comm["total_communication_bytes"],
                    "activation_upload_bytes": comm["activation_upload_bytes"],
                    "gradient_download_bytes": comm["gradient_download_bytes"],
                    "model_state_synchronization_bytes": comm["model_state_synchronization_bytes"],
                    "data_content_hash": result["data_content_hash"],
                    "partition_hash": result["partition_hash"],
                    "initial_model_hash": result["initial_model_hash"],
                    "final_model_hash": result["final_model_hash"],
                    "checkpoint_sha256": result["pretrain_checkpoint_sha256"],
                    "worker_ids": result["worker_ids"], "train_size": result["train_size"],
                    "test_size": result["test_size"], "run_dir": str(run_dir),
                    "result_sha256": digest(result_path),
                }
                rows.append(row)
                observations[method].append(row)
                for checkpoint in result["evaluation_records"]:
                    checkpoints.append({"model": row["model"], "task": task, "method": method,
                        "seed": seed, "elapsed_sec": checkpoint["elapsed_sec"],
                        "snapshot_elapsed_sec": checkpoint.get("snapshot_elapsed_sec", checkpoint["elapsed_sec"]),
                        "model_version": checkpoint.get("model_version", checkpoint["round_id"]),
                        "quality_pct": 100 * checkpoint["metrics"][metric],
                        "within_time_budget": checkpoint.get("within_time_budget", True),
                        "evaluation_kind": checkpoint.get("evaluation_kind", "round_end")})
            if Counter(population) != Counter(schemes):
                raise ValueError("Duplicate or missing method in the matched seed block")
        summaries = []
        for method, samples in observations.items():
            values = {}
            for key in ("quality_pct", "global_updates", "incorporated_optimizer_steps",
                        "completed_optimizer_steps_including_drain", "incorporated_examples",
                        "application_bytes_including_drain", "server_duration_sec"):
                data = [row[key] for row in samples]
                values[key + "_mean"] = float(np.mean(data)) if data else None
                values[key + "_sd"] = float(np.std(data, ddof=1)) if len(data) >= 2 else None
            summaries.append({"model": stage["image_model"], "task": task, "method": method,
                "metric": metric, "time_budget_sec": budget, "paired_n": len(samples), **values})
        effects, time_effects = [], []
        if report["paired_n"] >= 2:
            for method in schemes:
                if method == "splitfleet":
                    continue
                left = sorted(observations["splitfleet"], key=lambda row: row["seed"])
                right = sorted(observations[method], key=lambda row: row["seed"])
                if [row["seed"] for row in left] != [row["seed"] for row in right]:
                    raise ValueError("Unpaired endpoint observations")
                effect = paired_effect([row["quality_pct"] for row in left],
                    [row["quality_pct"] for row in right], seed=plan["analysis_seed"],
                    bootstrap_samples=plan["bootstrap_samples"]).to_dict()
                effects.append({"reference": "splitfleet", "comparator": method,
                    "endpoint": "quality_difference_percentage_points", **effect})
                if fixed_rounds:
                    timing = paired_effect([row["server_duration_sec"] for row in left],
                        [row["server_duration_sec"] for row in right], seed=plan["analysis_seed"],
                        bootstrap_samples=plan["bootstrap_samples"]).to_dict()
                    time_effects.append({"reference": "splitfleet", "comparator": method,
                        "endpoint": "fixed_rounds_time_difference_seconds", **timing})
            adjusted = holm_adjust(effect["sign_flip_p_value"] for effect in effects)
            for effect, p_value in zip(effects, adjusted, strict=True):
                effect["holm_adjusted_p_value"] = p_value
            for effect, p_value in zip(time_effects, holm_adjust(e["sign_flip_p_value"] for e in time_effects), strict=True):
                effect["holm_adjusted_p_value"] = p_value
        tasks.append({"model": stage["image_model"], "task": task, "time_budget_sec": budget,
            "planned_seeds": stage["seeds"], "complete_seeds": report["complete_seeds"],
            "paired_n": report["paired_n"], "summaries": summaries, "quality_effects": effects,
            "time_effects": time_effects, "fixed_work_time_comparable": fixed_rounds})
    return {"schema": "splitfleet.fedcompass-physical-evidence.v1",
        "study_root": str(root), "complete_planned_population": complete,
        "pending_blocks_excluded_from_partial_inspection": pending_exclusions,
        "planned_jobs": sum(len(stage["seeds"]) * len(stage["tasks"]) * len(schemes) for stage in stages),
        "completed_jobs": len(rows), "methods": list(schemes), "tasks": tasks,
        "runtime_source_identity": runtime["source_identity"],
        "runtime_source_identities": [runtime["source_identity"]],
        "training_source_identity": training_identity,
        "shared_execution_source_identity": shared_identity,
        "shared_execution_source_files": len(runtime["source_hashes"])-1,
        "baseline_source_amendments": [baseline_amendment] if baseline_amendment else [],
        "transport_source_amendments": [source_amendment] if source_amendment else [],
        "observations": rows, "checkpoints": checkpoints, "source_artifacts_sha256": artifacts,
        "analysis_source_sha256": analysis_sources, "transport_metadata_correction": correction,
        "cosplit_run_metadata_correction": audited["cosplit_run_metadata_correction"],
        "cohorts_pooled": False,
        "method_continuations": method_continuations,
        "discarded_failed_method_attempts": sum(r["discarded_failed_method_attempts"] for r in method_continuations),
        "protocol": "fixed_rounds_sync" if fixed_rounds else "common_wall_clock",
        "endpoint_definition": ("Time and quality after every client completes all ten full one-epoch synchronous rounds, including all tail batches."
            if fixed_rounds else "Quality of the last model committed strictly before the frozen server-clock deadline; final evaluation may finish after draining outstanding assignments."),
        "work_definition": ("Every configured client completes ten full one-epoch synchronous rounds, including all tail batches. Completed and incorporated work coincide; application buffers exclude the declared non-training traffic and transport overhead."
            if fixed_rounds else "Incorporated steps are from assignments aggregated before the deadline. Completed steps and application buffers include all drained assignments; within-budget executed work and physical wire bytes are not inferred."),
        "checkpoint_definition": ("Actual evaluation completion times after each complete synchronous round; no initial evaluation."
            if fixed_rounds else "Actual evaluation completion times. Minimum cadence is requested; synchronous evaluations wait for complete rounds, while asynchronous methods may evaluate with outstanding client work. No interpolated threshold crossings."),
        "statistics": f"Seed is the paired unit; mean and sample SD. Percentile paired bootstrap intervals are descriptive. Exact two-sided sign flips, Holm correction over {len(schemes)-1} SplitFleet comparisons within each task/endpoint.",
        "inference_limit": "With five pairs, the smallest possible exact two-sided sign-flip p is 0.0625. This cohort cannot support p<0.05 significance, full-dataset convergence or universal superiority.",
        "paired_ttq_summary": audited["paired_ttq_summary"],
        "time_to_quality": audited["time_to_quality"]}


def summarize_amendment(root: Path, *, require_complete: bool = True) -> dict:
    """Combine disjoint task populations bound before amended outcomes exist.

    A failed, unpaired block is retained in the original phase and never
    supplies observations to its amended replacement. No task combines
    seeds from different execution phases or dependency environments.
    """
    path = root / "cohort_manifest.json"
    manifest = json.loads(path.read_text())
    expected_jobs = manifest.get("expected_primary_jobs")
    if (manifest.get("schema") != "splitfleet.fedcompass-amended-cohort.v1"
            or manifest.get("phase_combination") != "disjoint_tasks_only"
            or expected_jobs not in (100, 120)):
        raise ValueError("Invalid amended primary population")
    for name, expected in manifest["bound_artifacts_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Amendment source changed: {name}")
    phases, seen_tasks = [], set()
    original_plan = json.loads(Path(manifest["original_plan_path"]).read_text())
    if (expected_jobs == 100 and (original_plan.get("primary_objective") != "fixed_rounds_sync_time_and_quality"
            or set(manifest["methods"]) != {"fedavg", "splitfed_fixed25", "splitfed_fixed50", "splitfed_fixed75", "splitfleet"})):
        raise ValueError("The 100-job amendment requires the fixed-round synchronous population")
    original_stages = {stage["name"]: stage for stage in original_plan["stages"]}
    training_keys = ("method_order", "dirichlet_alpha", "split_state_exchange", "evaluation_interval_sec",
        "fedcompass_alpha", "fedcompass_latest_time_factor", "fedcompass_rpc_timeout_sec",
        "fedcompass_staleness_exponent", "optimizer_policy", "analysis_config", "analysis_seed",
        "bootstrap_samples", "minimum_complete_pairs_for_primary_claim", "primary_objective",
        "async_local_steps", "fedasync_alpha", "fedasync_staleness_exponent", "fedasync_proximal_mu",
        "fedbuff_buffer_size", "fedbuff_server_learning_rate")
    for phase in manifest["phases"]:
        phase_root = Path(phase["root"])
        frozen = phase_root / "frozen_plan.json"
        if digest(frozen) != phase["frozen_plan_sha256"]:
            raise ValueError("Execution plan differs from its prospective amendment")
        phase_plan = json.loads(frozen.read_text())
        ledger = json.loads((phase_root / "ledger.json").read_text())
        selection_created = manifest["created_unix"]
        binding_path = phase.get("selection_binding_manifest")
        if binding_path:
            # A later transport amendment may retain already complete tasks
            # from the preceding, prospectively bound execution phase. The
            # immutable ancestor supplies that phase's original selection date.
            if binding_path not in manifest["bound_artifacts_sha256"]:
                raise ValueError("Earlier phase-selection manifest is not hash bound")
            ancestor = json.loads(Path(binding_path).read_text())
            matches = [p for p in ancestor.get("phases", []) if p["root"] == str(phase_root)]
            if (ancestor.get("schema") != manifest["schema"] or len(matches) != 1
                    or ancestor.get("runtime_source_identity") != manifest["runtime_source_identity"]
                    or ancestor.get("methods") != manifest["methods"]
                    or ancestor.get("paired_seeds") != manifest["paired_seeds"]
                    or matches[0]["frozen_plan_sha256"] != phase["frozen_plan_sha256"]
                    or not set(phase["selected_stages"]) <= set(matches[0]["selected_stages"])
                    or not ancestor["created_unix"] < manifest["created_unix"]):
                raise ValueError("Earlier phase selection differs from its recorded amendment")
            selection_created = ancestor["created_unix"]
        if (phase_root != Path(manifest["phases"][0]["root"])
                and not selection_created < ledger["started_unix"]):
            raise ValueError("Amendment selection was not bound before resumed execution")
        if any(phase_plan.get(key) != original_plan.get(key) for key in training_keys):
            raise ValueError("Dependency amendment changed training or analysis settings")
        for stage in phase_plan["stages"]:
            if stage["name"] not in phase["selected_stages"]:
                continue
            expected = original_stages[stage["name"]]
            if ({key: value for key, value in stage.items() if key != "deployment"}
                    != {key: value for key, value in expected.items() if key != "deployment"}):
                raise ValueError("Dependency amendment changed a task's paired protocol")
        original_controller = {key: value for key, value in original_plan["cosplit_hyperparameters"].items()
                               if key not in ("seed", "state_path")}
        phase_controller = {key: value for key, value in phase_plan["cosplit_hyperparameters"].items()
                            if key not in ("seed", "state_path")}
        if phase_controller != original_controller:
            raise ValueError("Dependency amendment changed controller hyperparameters")
        value = summarize_phase(phase_root, require_complete=require_complete,
                                selected_stages=phase["selected_stages"])
        if value["runtime_source_identity"] != manifest["runtime_source_identity"]:
            raise ValueError("Amendment changed the frozen training source")
        for task in value["tasks"]:
            if (task["task"] in seen_tasks or task["planned_seeds"] != manifest["paired_seeds"]
                    or (require_complete and task["paired_n"] != 5)):
                raise ValueError("Task population overlaps phases or loses planned paired seeds")
            seen_tasks.add(task["task"])
        if value["methods"] != manifest["methods"]:
            raise ValueError("Amendment changed the comparator population")
        phases.append(value)
    if seen_tasks != set(PRIMARY) or sum(p["planned_jobs"] for p in phases) != expected_jobs:
        raise ValueError("The amendment does not cover the four frozen task populations")
    failed = []
    total_completed = total_failed = 0
    for phase in manifest["phases"]:
        phase_root = Path(phase["root"])
        ledger = json.loads((phase_root / "ledger.json").read_text())
        for job in ledger["jobs"]:
            attempts_path = phase_root / "physical" / job["run_id"] / "attempts.json"
            if not attempts_path.exists():
                continue
            attempts = json.loads(attempts_path.read_text())
            total_completed += sum(a["status"] == "completed" for a in attempts)
            total_failed += sum(a["status"] == "failed" for a in attempts)
            if job["stage"] not in phase["selected_stages"]:
                failed.append({"phase_root": str(phase_root), "stage": job["stage"],
                    "seed": job["seed"], "block_status": job["status"], "attempts": attempts,
                    "primary_endpoint_observations_used": False})
    if total_failed != manifest["retained_original_failed_jobs"]:
        raise ValueError("An additional failure requires a further explicit amendment")
    interrupted_count = sum(a["status"] == "running" for block in failed for a in block["attempts"])
    if interrupted_count != manifest.get("retained_original_interrupted_jobs", 0):
        raise ValueError("Interrupted attempts require their own explicit amendment receipt")
    if interrupted_count:
        receipt_path = manifest.get("interruption_receipt")
        if receipt_path not in manifest["bound_artifacts_sha256"]:
            raise ValueError("Coordinator interruption receipt is not hash bound")
        receipt = json.loads(Path(receipt_path).read_text())
        if (receipt.get("interrupted_attempts") != interrupted_count
                or receipt.get("previous_boot_id") == receipt.get("current_boot_id")):
            raise ValueError("Coordinator interruption receipt does not match retained attempts")
    included_count = sum(phase["completed_jobs"] for phase in phases)
    excluded_count = sum(a["status"] == "completed" for block in failed for a in block["attempts"])
    if require_complete and total_completed != included_count + excluded_count:
        raise ValueError("Completed-attempt denominator differs from included and excluded observations")
    result = dict(phases[0])
    for key in ("tasks", "observations", "checkpoints", "time_to_quality", "paired_ttq_summary",
                "pending_blocks_excluded_from_partial_inspection"):
        result[key] = [row for phase in phases for row in phase[key]]
    result.update(study_root=str(root),
        complete_planned_population=all(phase["complete_planned_population"] for phase in phases),
        planned_jobs=expected_jobs, completed_jobs=sum(phase["completed_jobs"] for phase in phases),
        execution_phases=[{"study_root": phase["study_root"],
            "planned_jobs": phase["planned_jobs"], "completed_jobs": phase["completed_jobs"],
            "tasks": [task["task"] for task in phase["tasks"]]} for phase in phases],
        amendment=manifest, retained_excluded_blocks=failed,
        all_attempted_completed_jobs=total_completed, all_attempted_failed_jobs=total_failed,
        all_attempted_interrupted_jobs=interrupted_count,
        excluded_completed_jobs=excluded_count,
        pending_completed_jobs_not_yet_paired=total_completed-included_count-excluded_count,
        cosplit_run_metadata_corrections=[phase["cosplit_run_metadata_correction"] for phase in phases],
        transport_metadata_corrections=[phase["transport_metadata_correction"] for phase in phases])
    artifacts = {name: expected for phase in phases for name, expected in phase["source_artifacts_sha256"].items()}
    artifacts.update(manifest["bound_artifacts_sha256"])
    artifacts[str(path)] = digest(path)
    for block in failed:
        folder = Path(block["phase_root"]) / "physical" / f"{block['stage']}_s{block['seed']}"
        for item in folder.rglob("*.json"):
            artifacts[str(item)] = digest(item)
        log_root = Path(block["phase_root"]) / "logs" / f"{block['stage']}_s{block['seed']}"
        for item in log_root.rglob("*.log"):
            artifacts[str(item)] = digest(item)
    result["source_artifacts_sha256"] = artifacts
    return result


def task_statistics(plan, stage, methods, rows, complete_seeds):
    """Recompute paired statistics from complete blocks, never phase means."""
    task = stage["tasks"][0]
    budget = stage.get("time_budget_sec", plan.get("time_budget_sec"))
    fixed = plan.get("primary_objective") == "fixed_rounds_sync_time_and_quality"
    groups = {m: sorted([r for r in rows if r["model"] == stage["image_model"] and r["method"] == m],
                        key=lambda r: r["seed"]) for m in methods}
    summaries = []
    for method, samples in groups.items():
        values = {}
        for key in ("quality_pct", "global_updates", "incorporated_optimizer_steps",
                    "completed_optimizer_steps_including_drain", "incorporated_examples",
                    "application_bytes_including_drain", "server_duration_sec"):
            data = [r[key] for r in samples]
            values[key + "_mean"] = float(np.mean(data)) if data else None
            values[key + "_sd"] = float(np.std(data, ddof=1)) if len(data) >= 2 else None
        summaries.append({"model": stage["image_model"], "task": task, "method": method,
                          "metric": PRIMARY[task], "time_budget_sec": budget,
                          "paired_n": len(samples), **values})
    quality, timing = [], []
    if len(complete_seeds) >= 2:
        left = groups["splitfleet"]
        for method in methods:
            if method == "splitfleet":
                continue
            right = groups[method]
            if [r["seed"] for r in left] != [r["seed"] for r in right]:
                raise ValueError("Recovery statistics require matched complete seed blocks")
            for key, effects, endpoint in (("quality_pct", quality, "quality_difference_percentage_points"),
                    ("server_duration_sec", timing, "fixed_rounds_time_difference_seconds")):
                if key == "server_duration_sec" and not fixed:
                    continue
                effect = paired_effect([r[key] for r in left], [r[key] for r in right],
                    seed=plan["analysis_seed"], bootstrap_samples=plan["bootstrap_samples"]).to_dict()
                effects.append({"reference": "splitfleet", "comparator": method,
                                "endpoint": endpoint, **effect})
        for effects in (quality, timing):
            for effect, p in zip(effects, holm_adjust(e["sign_flip_p_value"] for e in effects), strict=True):
                effect["holm_adjusted_p_value"] = p
    return {"model": stage["image_model"], "task": task, "time_budget_sec": budget,
            "planned_seeds": stage["seeds"], "complete_seeds": complete_seeds,
            "paired_n": len(complete_seeds), "summaries": summaries, "quality_effects": quality,
            "time_effects": timing, "fixed_work_time_comparable": fixed}


def summarize_block_recovery(root, *, require_complete=True):
    """Resume the same population at whole paired blocks after host reboot."""
    path = root / "cohort_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("phase_combination") != "disjoint_complete_paired_blocks":
        raise ValueError("Recovery must retain complete paired blocks")
    for name, expected in manifest["bound_artifacts_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Recovery binding changed: " + name)
    receipts = []
    for name in manifest.get("interruption_receipts", []):
        if name not in manifest["bound_artifacts_sha256"]:
            raise ValueError("Recovery interruption receipt is not bound")
        receipt = json.loads(Path(name).read_text())
        if not receipt.get("previous_boot_id") or receipt.get("previous_boot_id") == receipt.get("current_boot_id"):
            raise ValueError("Recovery requires an actual coordinator boot change")
        receipts.append(receipt)
    controlled_interrupted = 0
    if manifest.get("controlled_interruption_receipts"):
        from experiments.analysis.user_pause_recovery import validate_user_pause
        for name in manifest["controlled_interruption_receipts"]:
            controlled_interrupted += validate_user_pause(name, manifest["bound_artifacts_sha256"])
    if sum(r.get("interrupted_attempts", 0) for r in receipts) + controlled_interrupted != manifest["retained_original_interrupted_jobs"]:
        raise ValueError("Interrupted methods lack matching reboot or user-pause receipts")
    failed_receipts = []
    for name in manifest.get("operational_failure_receipts", []):
        if name not in manifest["bound_artifacts_sha256"]:
            raise ValueError("Operational failure receipt is not bound")
        receipt = json.loads(Path(name).read_text())
        if receipt.get("schema") != "splitfleet.operational-failure-recovery.v1":
            raise ValueError("Operational failure receipt schema is invalid")
        failed_receipts.append(receipt)
    if sum(r.get("failed_method_attempts", 0) for r in failed_receipts) != manifest["retained_original_failed_jobs"]:
        raise ValueError("Failed methods lack matching operational failure receipts")
    plan = json.loads(Path(manifest["original_plan_path"]).read_text())
    methods = list(schemes_for(plan))
    if methods != manifest["methods"] or plan["source_identity"] != manifest["runtime_source_identity"]:
        raise ValueError("Recovery changes methods or frozen source")
    expected_blocks = {(s["name"], seed) for s in plan["stages"] for seed in s["seeds"]}
    if len(expected_blocks) * len(methods) != manifest["expected_primary_jobs"]:
        raise ValueError("Recovery changes the original population size")
    seen, phases, excluded = set(), [], []
    total_completed = total_failed = interrupted = 0
    for phase in manifest["phases"]:
        folder = Path(phase["root"])
        phase_plan = json.loads((folder / "frozen_plan.json").read_text())
        canonical_stages = {s["name"]: s for s in plan["stages"]}
        ignored_plan_fields = {"stages"}
        source_amendment_path = folder / "source_amendment.json"
        if source_amendment_path.exists():
            if (str(source_amendment_path) not in manifest["bound_artifacts_sha256"]
                    or digest(source_amendment_path) != phase.get("source_amendment_sha256")):
                raise ValueError("Recovery transport source amendment is not bound")
            amendment = json.loads(source_amendment_path.read_text())
            if (amendment["original_source_identity"] != manifest["runtime_source_identity"]
                    or amendment["amended_source_identity"] != phase_plan["source_identity"]
                    or Path(amendment["amended_source_root"]) != Path(phase_plan["source_root"])):
                raise ValueError("Recovery transport source amendment changes its source binding")
            ignored_plan_fields |= {"source_root", "source_identity"}
        if (digest(folder / "frozen_plan.json") != phase["frozen_plan_sha256"]
                or {k: v for k, v in phase_plan.items() if k not in ignored_plan_fields} != {k: v for k, v in plan.items() if k not in ignored_plan_fields}
                or any(s != canonical_stages.get(s["name"]) for s in phase_plan["stages"])):
            raise ValueError("Recovery changes a canonical training or analysis setting")
        ledger_path = folder / "ledger.json"
        if not ledger_path.exists():
            if require_complete:
                raise ValueError("Recovery continuation has not started")
            # A prospectively bound, unstarted phase still owns its blocks.
            ledger = {"jobs": []}
        else:
            ledger = json.loads(ledger_path.read_text())
        selection = phase["selected_seeds"]
        blocks = {(stage, seed) for stage, seeds in selection.items() for seed in seeds}
        if not blocks <= expected_blocks or seen & blocks:
            raise ValueError("Recovery overlaps or substitutes paired seed blocks")
        seen |= blocks
        if phase.get("prospective_continuation") and ledger.get("started_unix") is not None:
            if not manifest["created_unix"] < ledger["started_unix"]:
                raise ValueError("Recovery selection was not bound before outcomes")
        if ledger_path.exists() and selection:
            value = summarize_phase(folder, require_complete=require_complete,
                selected_stages=list(selection), selected_seeds=selection)
            if value.get("training_source_identity", value["runtime_source_identity"]) != manifest["runtime_source_identity"]:
                raise ValueError("Recovery runtime changed")
            phases.append(value)
        elif ledger_path.exists():
            # A phase containing no complete pair remains in failure accounting.
            # Its immutable metadata is bound; none of its model observations
            # contribute to summaries or inference.
            for name in ("frozen_plan.json", "ledger.json", "executed_plan.json", "runtime_manifest.json"):
                if str(folder / name) not in manifest["bound_artifacts_sha256"]:
                    raise ValueError("Excluded phase artifacts must be bound")
            runtime = json.loads((folder / "runtime_manifest.json").read_text())
            if runtime["source_identity"] != manifest["runtime_source_identity"]:
                raise ValueError("Excluded recovery runtime changed")
        for job in ledger["jobs"]:
            attempt_path = folder / "physical" / job["run_id"] / "attempts.json"
            attempts = json.loads(attempt_path.read_text()) if attempt_path.exists() else []
            total_completed += sum(a["status"] == "completed" for a in attempts)
            total_failed += sum(a["status"] == "failed" for a in attempts)
            if (job["stage"], job["seed"]) not in blocks:
                if job["status"] == "completed" and len(attempts) == len(methods) and all(a["status"] == "completed" for a in attempts):
                    raise ValueError("An earlier complete paired block cannot be replaced")
                interrupted += sum(a["status"] == "running" for a in attempts)
                excluded.append({"phase_root": str(folder), "run_id": job["run_id"],
                    "stage": job["stage"], "seed": job["seed"], "attempts": attempts,
                    "primary_endpoint_observations_used": False})
    if seen != expected_blocks or not phases:
        raise ValueError("Recovery must cover every original paired seed block")
    if total_failed != manifest["retained_original_failed_jobs"] or interrupted != manifest["retained_original_interrupted_jobs"]:
        raise ValueError("A new failure/interruption requires another explicit recovery binding")
    rows = [r for p in phases for r in p["observations"]]
    complete_keys = Counter((r["model"], r["seed"]) for r in rows)
    if any(n != len(methods) for n in complete_keys.values()):
        raise ValueError("Recovery cannot merge methods across attempts")
    tasks = [task_statistics(plan, s, methods, rows,
             sorted(seed for seed in s["seeds"] if (s["image_model"], seed) in complete_keys)) for s in plan["stages"]]
    complete = len(rows) == manifest["expected_primary_jobs"] and all(t["paired_n"] == 5 for t in tasks)
    if require_complete and not complete:
        raise ValueError("Recovery publication requires all original five paired seeds")
    excluded_count = sum(a["status"] == "completed" for b in excluded for a in b["attempts"])
    if require_complete and total_completed != len(rows) + excluded_count:
        raise ValueError("Recovery successful-attempt denominator changed")
    result = dict(phases[0])
    from experiments.analysis.time_to_quality import summarize_paired_ttq
    ttq = [r for p in phases for r in p["time_to_quality"]]
    artifacts = {name: sha for p in phases for name, sha in p["source_artifacts_sha256"].items()}
    artifacts.update(manifest["bound_artifacts_sha256"])
    artifacts[str(path)] = digest(path)
    identities = sorted({p["runtime_source_identity"] for p in phases})
    if len({p["shared_execution_source_identity"] for p in phases}) != 1:
        raise ValueError("Recovery changes the shared execution implementation")
    amendments = [a for p in phases for a in p.get("transport_source_amendments", [])]
    method_continuations = [a for p in phases for a in p.get("method_continuations", [])]
    discarded_failed = sum(a["discarded_failed_method_attempts"] for a in method_continuations)
    result.update(study_root=str(root), planned_jobs=manifest["expected_primary_jobs"],
        completed_jobs=len(rows), complete_planned_population=complete, tasks=tasks, observations=rows,
        checkpoints=[r for p in phases for r in p["checkpoints"]], time_to_quality=ttq,
        paired_ttq_summary=summarize_paired_ttq(ttq), amendment=manifest, retained_excluded_blocks=excluded,
        all_attempted_completed_jobs=total_completed, all_attempted_failed_jobs=total_failed + discarded_failed,
        discarded_failed_method_attempts=discarded_failed, method_continuations=method_continuations,
        all_attempted_interrupted_jobs=interrupted, excluded_completed_jobs=excluded_count,
        pending_completed_jobs_not_yet_paired=total_completed-len(rows)-excluded_count,
        runtime_source_identity=identities[0] if len(identities) == 1 else None,
        runtime_source_identities=identities, training_source_identity=manifest["runtime_source_identity"],
        transport_source_amendments=amendments,
        execution_phases=[{"study_root": p["study_root"], "completed_jobs": p["completed_jobs"],
            "runtime_source_identity": p["runtime_source_identity"]} for p in phases],
        source_artifacts_sha256=artifacts, cohorts_pooled=False)
    if controlled_interrupted:
        result["analysis_source_sha256"]["experiments/analysis/user_pause_recovery.py"] = digest(Path(__file__).with_name("user_pause_recovery.py"))
    if method_continuations:
        result["analysis_source_sha256"]["experiments/analysis/method_continuation.py"] = digest(Path(__file__).with_name("method_continuation.py"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Audit a new experiment with complete original attempt records")
    source.add_argument("--published-evidence", type=Path,
                        help="Audit retained published successes after removal of unused attempts")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.published_evidence:
        result = audit_published_endpoints(args.published_evidence)
        if args.out:
            if args.out.exists():
                raise FileExistsError("Preserve the earlier audit receipt; choose a fresh output")
            save_json(args.out, result)
        print(json.dumps(result, sort_keys=True))
        return
    result = summarize(args.root, require_complete=not args.allow_partial)
    output = args.out or args.root / "paper_evidence.json"
    save_json(output, result)
    print(f"{result['completed_jobs']}/{result['planned_jobs']} audited jobs: {output}")


if __name__ == "__main__":
    main()
