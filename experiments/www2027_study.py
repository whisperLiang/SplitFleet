"""Serial seed blocks and audited paired analysis for the four-task study.

Failed attempts remain in the ledger;
the executor stops on failure and never silently retries or replaces a seed.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from experiments.common.statistics import holm_adjust, paired_effect
from experiments.physical_multitask import FIXED_BOUNDARIES, METHODS, scheme_name
from experiments.summarize_physical_multitask import PRIMARY, summarize
from experiments.analysis.time_to_quality import time_to_quality, summarize_paired_ttq
from experiments.analysis.communication import communication_summary

SCHEMES = ("fedavg", "fedcompass", "splitfed_fixed25", "splitfed_fixed50", "splitfed_fixed75", "splitfleet")
LEGACY_SCHEMES = ("fedavg", "fedprox", "splitfed_fixed25", "splitfed_fixed50", "splitfed_fixed75", "splitfleet")


def schemes_for(plan):
    methods = plan.get("method_order", METHODS)
    return tuple(scheme_name(method, boundary) for method in (*methods,)
                 for boundary in (FIXED_BOUNDARIES if method == "splitfed_fixed" else (None,)))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def cosplit_metadata_correction(root: Path, plan: dict) -> dict | None:
    """Validate an explicit correction of copied run identity, never tuning."""
    path = root / "cosplit_run_metadata_correction.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    frozen = root / "frozen_plan.json"
    runtime = json.loads((root / "runtime_manifest.json").read_text())
    if (value.get("schema") != "splitfleet.cosplit-run-metadata-correction.v1"
            or value.get("metadata_only") is not True
            or value.get("training_parameters_changed") is not False
            or value.get("runs_restarted") is not False
            or value.get("original_frozen_plan_sha256") != hashlib.sha256(frozen.read_bytes()).hexdigest()
            or value.get("runtime_source_identity") != runtime["source_identity"]
            or value.get("frozen_physical_module_sha256") != runtime["source_hashes"]["experiments/physical_multitask.py"]
            or value.get("corrected_field_rules") != {
                "seed": "paired_seed_from_frozen_stage",
                "state_path": "result_path_with_bandit_json_suffix"}
            or value.get("original_fields") != {
                key: plan.get("cosplit_hyperparameters", {}).get(key) for key in ("seed", "state_path")}):
        raise ValueError("Invalid CoSplit run-metadata correction")
    return value


def validate_cosplit_config(actual: dict, expected: dict, *, seed: int,
                            result_path: Path, correction: dict | None = None,
                            require_run_identity: bool = False) -> None:
    if require_run_identity:
        if actual.get("seed") != seed:
            raise ValueError("CoSplit-UCB parameter seed differs from the paired seed")
        if actual.get("state_path") != str(result_path.with_suffix(".bandit.json")):
            raise ValueError("CoSplit-UCB parameter state_path differs from the isolated output")
    for name, value in expected.items():
        if correction and name == "seed":
            value = seed
        elif correction and name == "state_path":
            value = str(result_path.with_suffix(".bandit.json"))
        if actual.get(name) != value:
            raise ValueError(f"CoSplit-UCB parameter {name} differs from the frozen plan")


def freeze_runtime(workspace: Path, root: Path) -> tuple[Path, dict]:
    """Freeze once for the whole study, including independent seed blocks."""
    import os
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    runtime = root / "runtime_snapshot"
    runtime.mkdir(parents=True, exist_ok=False)
    for folder in ("splitfleet", "experiments"):
        shutil.copytree(workspace / folder, runtime / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(runtime.rglob("*.py"))}
    packages = {}
    for package in ("torch", "torchvision", "torchlens", "flwr", "numpy", "rfdetr",
                    "tensorflow", "jax", "paddlepaddle", "timm", "transformers"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    from experiments.analysis.partition_coverage import _torchlens_source_identity
    artifacts = {name: hashlib.sha256((workspace / name).read_bytes()).hexdigest()
                 for name in ("torchlens-2.34.1-py3-none-any.whl", "uv.lock", "pyproject.toml")
                 if (workspace / name).is_file()}
    record = {"source_hashes": hashes,
              "source_identity": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
              "torchlens_source_sha256": _torchlens_source_identity(),
              "dependency_artifacts_sha256": artifacts,
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": packages}
    save_json(root / "runtime_manifest.json", record)
    return runtime, record


def command_for(plan: dict, stage: dict, index: int, root: Path) -> tuple[list[str], str]:
    seed = stage["seeds"][index]
    run_id = f"{stage['name']}_s{seed}"
    methods = list(plan.get("method_order", METHODS))
    methods = methods[index % len(methods):] + methods[:index % len(methods)]
    cuts = list(FIXED_BOUNDARIES)
    cuts = cuts[index % len(cuts):] + cuts[:index % len(cuts)]
    tasks = stage["tasks"]
    tasks = tasks[index % len(tasks):] + tasks[:index % len(tasks)]
    command = [sys.executable, "-m", "experiments.orchestrate_physical_multitask",
               "--deployment", stage.get("deployment", plan["deployment"]), "--run-id", run_id,
               "--output-root", str(root / "physical"), "--log-root", str(root / "logs"),
               "--data-root", plan["data_root"], "--seed", str(seed),
               "--tasks", *tasks, "--methods", *methods, "--fixed-boundaries", *cuts,
               "--model" if stage.get("model_id") else "--image-model", stage["image_model"],
               "--train-samples", str(stage["train_samples"]),
               "--rounds", str(stage["rounds"]), "--batch-size", str(stage["batch_size"]),
               "--dirichlet-alpha", str(plan["dirichlet_alpha"]),
               "--optimizer", stage["optimizer"], "--learning-rate", str(stage["learning_rate"]),
               "--timeout", str(stage["timeout_per_scheme_sec"])]
    if stage["test_samples"] is not None:
        command += ["--test-samples", str(stage["test_samples"])]
    if plan.get("split_state_exchange"):
        command += ["--split-state-exchange", plan["split_state_exchange"]]
    if plan.get("save_model", False):
        command.append("--save-model")
    if plan.get("keep_input_bundles", False):
        command.append("--keep-input-bundles")
    for name in ("time_budget_sec", "evaluation_interval_sec", "fedcompass_min_steps", "fedcompass_max_steps",
                 "fedcompass_latest_time_factor", "fedcompass_alpha", "fedcompass_staleness_exponent",
                 "fedcompass_rpc_timeout_sec", "async_local_steps", "fedasync_alpha",
                 "fedasync_staleness_exponent", "fedasync_proximal_mu", "fedbuff_buffer_size",
                 "fedbuff_server_learning_rate"):
        value = stage.get(name, plan.get(name))
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if stage["image_model"] == "rfdetr_nano" and not stage.get("model_id"):
        command += ["--pretrain-weights", plan["checkpoint"]]
    if stage.get("model_id"):
        command += ["--pretrain-weights", stage["checkpoint"]]
        if stage.get("tokenizer_path"):
            command += ["--tokenizer-path", stage["tokenizer_path"]]
    return command, run_id


def analyze(plan: dict, root: Path) -> dict:
    schemes = schemes_for(plan)
    reports = []
    exclusions = []
    ledger = json.loads((root / "ledger.json").read_text()) if (root / "ledger.json").exists() else {}
    run_metadata_correction = cosplit_metadata_correction(root, plan)
    runtime = json.loads((root / "runtime_manifest.json").read_text()) if (root / "runtime_manifest.json").exists() else None
    source_identity = runtime["source_hashes"] if runtime else None
    jobs = {(job["stage"], job["seed"]): job for job in ledger.get("jobs", [])}
    job_counts = Counter((job["stage"], job["seed"]) for job in ledger.get("jobs", []))
    attempts = []
    ttq_rows, communication_rows = [], []
    frozen_file = root / "frozen_plan.json"
    frozen = json.loads(frozen_file.read_text()) if frozen_file.exists() else None
    frozen_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() if frozen is not None else None
    ledger_plan_matches = frozen_hash is not None and ledger.get("plan_sha256") == frozen_hash
    plan_bound_to_execution = ledger_plan_matches and frozen == plan
    thresholds_frozen = (frozen is not None and frozen.get("analysis_config") == plan.get("analysis_config")
        and ("plan_sha256" not in ledger or ledger_plan_matches))
    for stage in plan["stages"]:
        by_task = {task: {} for task in stage["tasks"]}
        for seed in stage["seeds"]:
            folder = root / "physical" / f"{stage['name']}_s{seed}"
            attempt_file = folder / "attempts.json"
            attempted = json.loads(attempt_file.read_text()) if attempt_file.exists() else []
            attempts.append({"stage": stage["name"], "seed": seed,
                             "block_status": jobs.get((stage["name"], seed), {}).get("status", "not_started"),
                             "planned_schemes": len(stage["tasks"]) * len(schemes),
                             "completed_schemes": sum(a["status"] == "completed" for a in attempted),
                             "failed_schemes": sum(a["status"] == "failed" for a in attempted),
                             "running_schemes": sum(a["status"] == "running" for a in attempted),
                             "not_attempted_schemes": len(stage["tasks"]) * len(schemes) - len(attempted),
                             "failed_attempts": [a for a in attempted if a["status"] == "failed"]})
            if not (folder / "summary.json").exists():
                continue
            try:
                if any(attempt.get("status") != "completed" for attempt in attempted):
                    raise ValueError("seed block contains a failed, running or unknown attempt")
                job_key = (stage["name"], seed)
                if job_counts[job_key] > 1:
                    raise ValueError("execution ledger contains duplicate seed attempts")
                if job_key in jobs and jobs[job_key].get("status") != "completed":
                    raise ValueError("execution ledger does not mark the seed block completed")
                block_ttq, block_communication = [], []
                hashes = json.loads((folder / "source_hashes.json").read_text())
                if source_identity is not None and hashes != source_identity:
                    raise ValueError("seed block differs from the study's frozen runtime")
                audited = summarize(folder)
                if audited["scheme_count"] != len(schemes) or len(audited["runs"]) != len(stage["tasks"]) * len(schemes):
                    raise ValueError("incomplete planned scheme block")
                for task in stage["tasks"]:
                    observations = {}
                    for row in json.loads((folder / "summary.json").read_text()):
                        if row["task"] != task:
                            continue
                        run_dir = Path(row["run_dir"])
                        result = json.loads((run_dir / "result.json").read_text())
                        validation = json.loads((run_dir / "validation_report.json").read_text())
                        if not validation.get("barrier_verified") or "server_duration_sec" not in result:
                            raise ValueError("missing barrier receipt or monotonic server endpoint")
                        timed_budget = stage.get("time_budget_sec", plan.get("time_budget_sec"))
                        async_run = result["method"] in ("fedcompass", "fedasync", "fedbuff")
                        actual_updates = result.get("global_updates") if async_run else result["rounds"]
                        if (result["seed"] != seed or (timed_budget is None and actual_updates != stage["rounds"])
                                or result["train_size"] != stage["train_samples"]
                                or result["optimizer"] != stage["optimizer"]
                                or result["learning_rate"] != stage["learning_rate"]
                                or result["batch_size"] != stage["batch_size"]
                                or result["image_model"] != stage["image_model"]):
                            raise ValueError("result differs from frozen execution budget")
                        if timed_budget is not None and (result.get("time_budget_sec") != timed_budget
                                or result.get("evaluation_interval_sec") != stage.get("evaluation_interval_sec", plan.get("evaluation_interval_sec", 30.0))):
                            raise ValueError("result differs from the frozen wall-clock budget")
                        if result["method"] == "fedcompass":
                            config = result.get("fedcompass_config") or {}
                            defaults = {"min_steps": 20, "max_steps": 100, "latest_time_factor": 1.2,
                                        "alpha": .9, "staleness_exponent": .5}
                            keys = {"min_steps": "min_local_steps", "max_steps": "max_local_steps"}
                            if any(config.get(keys.get(name, name)) != stage.get("fedcompass_" + name,
                                    plan.get("fedcompass_" + name, default)) for name, default in defaults.items()):
                                raise ValueError("FedCompass parameters differ from the frozen plan")
                        if result["method"] in ("fedasync", "fedbuff"):
                            config = result.get("async_fl_config") or {}
                            expected_config = {"method": result["method"],
                                "local_steps": stage.get("async_local_steps", plan.get("async_local_steps", 20)),
                                "alpha": stage.get("fedasync_alpha", plan.get("fedasync_alpha", .9)),
                                "staleness_exponent": stage.get("fedasync_staleness_exponent", plan.get("fedasync_staleness_exponent", .5)),
                                "proximal_mu": stage.get("fedasync_proximal_mu", plan.get("fedasync_proximal_mu", .005)),
                                "buffer_size": stage.get("fedbuff_buffer_size", plan.get("fedbuff_buffer_size", 2)),
                                "server_learning_rate": stage.get("fedbuff_server_learning_rate", plan.get("fedbuff_server_learning_rate", 1.0))}
                            if config != expected_config:
                                raise ValueError("Asynchronous parameters differ from the frozen plan")
                        if stage["test_samples"] is not None and result["test_size"] != stage["test_samples"]:
                            raise ValueError("test population differs from frozen budget")
                        if (stage["image_model"] == "rfdetr_nano" and not stage.get("model_id")
                                and result["pretrain_checkpoint_sha256"] != plan["checkpoint_sha256"]):
                            raise ValueError("pretraining checkpoint differs")
                        if stage.get("model_id") and (
                                result.get("model_id") != stage["model_id"] or
                                result["pretrain_checkpoint_sha256"] != stage["checkpoint_sha256"]):
                            raise ValueError("primary architecture or pretrained checkpoint differs")
                        if (plan.get("split_state_exchange") == "owned"
                                and result["method"] in ("splitfed_fixed", "splitfleet")
                                and result.get("owned_state_exchange") is not True):
                            raise ValueError("SFL state exchange differs from the matched protocol")
                        expected_evaluation = ("initial, wall-clock cadence and final frozen model"
                                               if timed_budget is not None or async_run else plan.get("evaluation_policy"))
                        if (expected_evaluation and result.get("evaluation_policy") != expected_evaluation):
                            raise ValueError("evaluation budget differs from the matched protocol")
                        expected_order = ("coordinator_arrival_and_group_deadline" if result["method"] == "fedcompass"
                            else "coordinator_arrival" if async_run else plan.get("aggregation_order"))
                        if (expected_order and result.get("aggregation_order") != expected_order):
                            raise ValueError("aggregation order differs from the matched protocol")
                        if result["method"] == "splitfleet":
                            if result.get("online_cost_learning") is not True:
                                raise ValueError("Placement feedback receipt is missing")
                            if "greedy_config" in result:
                                actual_config = result["greedy_config"]
                                expected = plan.get("greedy_hyperparameters", {})
                                if result.get("placement_controller") != "Greedy":
                                    raise ValueError("Default placement controller differs")
                                for key, value in expected.items():
                                    if actual_config.get(key) != value:
                                        raise ValueError(f"Greedy parameter {key} differs from the frozen plan")
                                if actual_config.get("seed") != seed:
                                    raise ValueError("Greedy seed differs from the paired seed")
                            else:
                                # Historical audits retain the recorded policy identity.
                                actual_config = result.get("cosplit_config") or {}
                                validate_cosplit_config(actual_config, plan.get("cosplit_hyperparameters", {}),
                                    seed=seed, result_path=run_dir / "result.json", correction=run_metadata_correction,
                                    require_run_identity=plan.get("cosplit_identity_policy") == "derive_and_verify_per_run")
                        worker_ids = stage.get("worker_ids", plan.get("worker_ids"))
                        if worker_ids and (
                                result.get("worker_ids") != worker_ids or
                                result.get("expected_clients") != stage.get("workers", plan["workers"])):
                            raise ValueError("physical worker topology differs from the matched protocol")
                        scheme = scheme_name(result["method"], result["fixed_boundary"])
                        observations[scheme] = {
                            "time_sec": float(result["server_duration_sec"]),
                            "quality": float(row["final_metrics"][PRIMARY[task]]),
                            "resolved_cut": result.get("resolved_fixed_boundary"),
                            "final_model_hash": result["final_model_hash"],
                            "fixed_work_time_comparable": not async_run and timed_budget is None,
                        }
                        analysis_config = plan.get("analysis_config", {})
                        quality_config = analysis_config.get("tasks", {}).get(task, analysis_config)
                        metric = quality_config.get("quality_metric", PRIMARY[task])
                        for threshold in quality_config.get("time_to_quality_thresholds", []):
                            # TTQ follows the metric frozen in the plan. It may
                            # differ from the final-quality comparison metric
                            # (e.g. text accuracy versus macro-F1).
                            if not math.isfinite(threshold):
                                raise ValueError("TTQ requires finite frozen thresholds")
                            evaluations = result.get("evaluation_records", [])
                            if timed_budget is not None:
                                evaluations = [row for row in evaluations if row.get("elapsed_sec", float("inf")) <= timed_budget]
                            value = time_to_quality(evaluations, metric=metric, threshold=threshold) if thresholds_frozen else {
                                "metric": metric, "threshold": threshold, "status": "unavailable", "time_sec": None,
                                "reason": "Threshold configuration is not bound to the execution-time frozen plan"}
                            block_ttq.append({"stage": stage["name"], "task": task, "seed": seed, "scheme": scheme,
                                              "valid": True, "complete_paired_seed": True, **value})
                        block_communication.append({"stage": stage["name"], "task": task, "seed": seed,
                            "scheme": scheme, "complete_paired_seed": True, **communication_summary(result)})
                    if set(observations) != set(schemes):
                        raise ValueError("missing comparator")
                    if any(not math.isfinite(v["time_sec"]) or v["time_sec"] <= 0
                           or not math.isfinite(v["quality"]) for v in observations.values()):
                        raise ValueError("invalid timing or quality value")
                    by_task[task][seed] = observations
                if source_identity is None:
                    source_identity = hashes
                ttq_rows.extend(block_ttq)
                communication_rows.extend(block_communication)
            except (ValueError, KeyError, OSError) as exc:
                # Exclude the entire seed block; successful methods must not
                # turn a failed block into a favourable complete-case claim.
                for observations in by_task.values():
                    observations.pop(seed, None)
                exclusions.append({"stage": stage["name"], "seed": seed, "reason": str(exc)})
        for task, observations in by_task.items():
            seeds = sorted(observations)
            item = {"stage": stage["name"], "task": task, "calibration": stage["calibration"],
                    "planned_seeds": stage["seeds"], "complete_seeds": seeds,
                    "paired_n": len(seeds), "primary_metric": PRIMARY[task],
                    "schemes": {}, "comparisons": []}
            for scheme in schemes:
                times = [observations[seed][scheme]["time_sec"] for seed in seeds]
                quality = [observations[seed][scheme]["quality"] for seed in seeds]
                item["schemes"][scheme] = {
                    "mean_time_sec": sum(times) / len(times) if times else None,
                    "mean_quality": sum(quality) / len(quality) if quality else None,
                    "observations": [{"seed": seed, **observations[seed][scheme]} for seed in seeds],
                }
            if len(seeds) >= 2 and not stage["calibration"]:
                time_effects, quality_effects = [], []
                baselines = [scheme for scheme in schemes if scheme != "splitfleet"]
                for baseline in baselines:
                    time_effects.append(paired_effect(
                        [observations[s]["splitfleet"]["time_sec"] for s in seeds],
                        [observations[s][baseline]["time_sec"] for s in seeds],
                        bootstrap_samples=plan["bootstrap_samples"], seed=plan["analysis_seed"],
                    ).to_dict())
                    quality_effects.append(paired_effect(
                        [observations[s]["splitfleet"]["quality"] for s in seeds],
                        [observations[s][baseline]["quality"] for s in seeds],
                        bootstrap_samples=plan["bootstrap_samples"], seed=plan["analysis_seed"],
                    ).to_dict())
                adjusted = holm_adjust(effect["sign_flip_p_value"] for effect in time_effects)
                for index, baseline in enumerate(baselines):
                    time_effect, quality_effect = time_effects[index], quality_effects[index]
                    complete = (plan_bound_to_execution and len(seeds) == len(stage["seeds"])
                        and len(seeds) >= plan["minimum_complete_pairs_for_primary_claim"]
                        and all((stage["name"], seed) in jobs for seed in seeds))
                    comparable = all(observations[s][baseline]["fixed_work_time_comparable"]
                                     and observations[s]["splitfleet"]["fixed_work_time_comparable"] for s in seeds)
                    faster = comparable and time_effect["confidence_interval_95"][1] < 0 and adjusted[index] < 0.05
                    noninferior = quality_effect["confidence_interval_95"][0] > -plan["quality_margin"][task]
                    item["comparisons"].append({
                        "baseline": baseline, "time_effect": time_effect, "quality_effect": quality_effect,
                        "holm_time_p": adjusted[index], "faster_supported": faster,
                        "fixed_work_time_comparable": comparable,
                        "quality_noninferiority_interval_pass": noninferior,
                        "complete_planned_population": complete,
                        "joint_time_quality_supported": complete and faster and noninferior,
                        "primary_claim_definition": plan.get("primary_objective", "speed_with_quality_noninferiority"),
                        "primary_claim_supported": complete and faster and (
                            plan.get("primary_objective") == "fixed_budget_training_speed" or noninferior),
                    })
            reports.append(item)
    return {"schema": "splitfleet.www2027-study-analysis.v1", "updated_unix": time.time(),
            "reports": reports, "exclusions": exclusions, "attempts": attempts,
            "selected_stages": ledger.get("selected_stages"), "execution_status": ledger.get("status", "unknown"),
            "time_to_quality": ttq_rows, "paired_ttq_summary": summarize_paired_ttq(ttq_rows),
            "ttq_thresholds_bound_to_frozen_plan": thresholds_frozen,
            "plan_bound_to_execution": plan_bound_to_execution,
            "cosplit_run_metadata_correction": run_metadata_correction,
            "communication": communication_rows,
            "inference_unit": "seed; stages and tasks analyzed separately",
            "limitations": ["Percentile bootstrap is approximate, particularly with few seeds",
                            "Sign-flip test assumes exchangeability of paired difference signs",
                            "Fixed-budget observations do not establish full-dataset convergence",
                            "No adaptive-SFL algorithm reproduction or measured privacy guarantee"],
            "submission_ready": False}


def write_report(analysis: dict, root: Path) -> None:
    save_json(root / "analysis.json", analysis)
    save_json(root / "time_to_quality.json", {"rows": analysis["time_to_quality"], "summaries": analysis["paired_ttq_summary"],
        "thresholds_bound_to_frozen_plan": analysis["ttq_thresholds_bound_to_frozen_plan"]})
    save_json(root / "communication.json", analysis["communication"])
    lines = ["# WWW2027 当前执行结果", "", "由真实结果与验证收据自动生成。校准不纳入推断；未完成不表示零耗时或零失败。", "",
             "| 阶段 | 任务 | 完成 seeds / 计划 | 方法 | 平均服务器秒 | 平均任务指标 |",
             "|---|---|---:|---|---:|---:|"]
    for report in analysis["reports"]:
        for scheme, values in report["schemes"].items():
            if values["mean_time_sec"] is None:
                continue
            lines.append(f"| {report['stage']} | {report['task']} | {report['paired_n']} / {len(report['planned_seeds'])} | {scheme} | {values['mean_time_sec']:.3f} | {values['mean_quality']:.6f} |")
    lines += ["", "未达到完整 seed 数的结果仅作阶段性观察。完整结果、配对区间、Holm 校正、失败和排除原因见 analysis.json 与 ledger.json。", ""]
    (root / "report.md").write_text("\n".join(lines))


def execute(plan: dict, root: Path, stages: list[str]) -> None:
    ledger_path = root / "ledger.json"
    if ledger_path.exists():
        raise FileExistsError("An execution ledger already exists; use a new amended execution directory")
    plan_bytes = json.dumps(plan, sort_keys=True).encode()
    ledger = {"schema": "splitfleet.www2027-execution-ledger.v1", "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
              "status": "running", "selected_stages": stages, "jobs": [], "started_unix": time.time()}
    save_json(root / "frozen_plan.json", plan)
    runtime, runtime_record = freeze_runtime(Path(plan.get("source_root", Path(plan["data_root"]).parent)), root)
    if plan.get("source_identity") and runtime_record["source_identity"] != plan["source_identity"]:
        raise ValueError("The queued primary study source no longer matches its admission snapshot")
    execution_plan = copy.deepcopy(plan)
    inputs = root / "inputs"
    inputs.mkdir()
    for key in ("deployment",):
        if key not in plan:
            continue
        target = inputs / (key + ".json")
        shutil.copyfile(plan[key], target)
        execution_plan[key] = str(target)
    from experiments.unified_multitask.edge_models import file_sha256
    if "checkpoint" in plan and file_sha256(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint no longer matches the frozen plan")
    copied_deployments = {}
    for stage in execution_plan["stages"]:
        if not stage.get("model_id") or stage["name"] not in stages:
            continue
        if file_sha256(stage["checkpoint"]) != stage["checkpoint_sha256"]:
            raise ValueError("Primary pretrained checkpoint no longer matches its frozen plan")
        name = stage["model_id"]
        if stage.get("deployment"):
            if name not in copied_deployments:
                target = inputs / (name + ".deployment.json")
                shutil.copyfile(stage["deployment"], target)
                copied_deployments[name] = str(target)
            stage["deployment"] = copied_deployments[name]
    ledger["runtime_source_identity"] = runtime_record["source_identity"]
    save_json(root / "executed_plan.json", execution_plan)
    save_json(ledger_path, ledger)
    env = dict(os.environ, PYTHONPATH=str(runtime))
    for stage in execution_plan["stages"]:
        if stage["name"] not in stages:
            continue
        for index, seed in enumerate(stage["seeds"]):
            command, run_id = command_for(execution_plan, stage, index, root)
            log_path = root / "logs" / (run_id + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            job = {"stage": stage["name"], "seed": seed, "run_id": run_id, "command": command,
                   "status": "running", "started_unix": time.time(), "log": str(log_path)}
            ledger["jobs"].append(job)
            save_json(ledger_path, ledger)
            print(f"start {run_id}", flush=True)
            with log_path.open("w") as stream:
                process = subprocess.run(command, cwd=runtime, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, check=False)
            job.update(exit_code=process.returncode, finished_unix=time.time(),
                       status="completed" if process.returncode == 0 else "failed")
            save_json(ledger_path, ledger)
            write_report(analyze(plan, root), root)
            print(f"{job['status']} {run_id}", flush=True)
            if process.returncode:
                ledger.update(status="failed", finished_unix=time.time())
                save_json(ledger_path, ledger)
                raise RuntimeError(f"{run_id} failed; original attempt preserved at {log_path}")
    ledger.update(status="completed", finished_unix=time.time())
    save_json(ledger_path, ledger)
    write_report(analyze(plan, root), root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "analyze"):
        cmd = sub.add_parser(mode)
        cmd.add_argument("--plan", type=Path, required=True)
        cmd.add_argument("--root", type=Path, required=True)
        if mode == "run":
            cmd.add_argument("--stages", nargs="+")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "run":
        stages = args.stages or [stage["name"] for stage in plan["stages"]]
        if not set(stages) <= {stage["name"] for stage in plan["stages"]}:
            raise ValueError("unknown execution stage")
        execute(plan, root, stages)
    else:
        write_report(analyze(plan, root), root)


if __name__ == "__main__":
    main()
