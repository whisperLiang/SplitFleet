"""Reanalyse the frozen DeepLab cohort for the two-panel motivation figure.

This is a read-only, descriptive reuse of P04, not a new physical experiment.
Use the common client fit timestamps, never whole-job time or prefix-only time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = ROOT / "results/www2027_study_20261003/remaining_tasks_serial_v1/deeplabv3_resnet50_queue_v1/study/deeplabv3_resnet50/timing"
DEFAULT_REGISTRY = ROOT / "paper/evidence/storyline_redesign_20261007/derived/evidence.json"
DEVICES = {
    "cpu1": ("orin140-cpu", "orin140", "cpu", 0),
    "cpu2": ("orin118-cpu", "orin118", "cpu", 2),
    "cpu3": ("orin238-cpu", "orin238", "cpu", 4),
    "gpu1": ("orin140-gpu", "orin140", "cuda:0", 1),
    "gpu2": ("orin118-gpu", "orin118", "cuda:0", 3),
    "gpu3": ("orin238-gpu", "orin238", "cuda:0", 5),
}
METHODS = ("fedavg", "splitfed_fixed25", "splitfed_fixed50", "splitfed_fixed75")
SCHEMA = "splitfleet.deeplab-motivation-reanalysis.v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synchronization_cost_model(rows: list[dict]) -> dict:
    """Derive equal-batch slack; never label it observed barrier waiting.

    Compute maxima inside each simultaneous round before averaging seeds.
    Maxima of client means lose changes in which client has the highest cost.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["seed"], row["round"]), []).append(row)
    client_rows, round_rows = [], []
    for (method, seed, round_id), group in sorted(grouped.items()):
        if len(group) != 6 or {row["label"] for row in group} != set(DEVICES):
            raise ValueError("A synchronization comparison needs the same complete six-client round")
        costs = [row["sec_per_batch"] for row in group]
        if any(not math.isfinite(value) or value <= 0 for value in costs):
            raise ValueError("Invalid normalized cost in synchronization comparison")
        critical = max(costs)
        waits = [critical - value for value in costs]
        round_rows.append({"method": method, "seed": seed, "round": round_id,
                           "requested_cut": group[0]["requested_cut"],
                           "critical_cost_sec_per_batch": critical,
                           "mean_implied_wait_sec_per_batch": mean(waits),
                           "max_implied_wait_sec_per_batch": max(waits),
                           "sum_implied_wait_sec_per_batch": sum(waits)})
        for row in group:
            client_rows.append({"method": method, "seed": seed, "round": round_id,
                                "label": row["label"], "requested_cut": row["requested_cut"],
                                "cost_sec_per_batch": row["sec_per_batch"],
                                "critical_cost_sec_per_batch": critical,
                                "implied_wait_sec_per_batch": critical - row["sec_per_batch"],
                                "highest_cost": row["sec_per_batch"] == critical})
    seeds = sorted({row["seed"] for row in rows})
    seed_rows, condition_rows = [], []
    for method in METHODS:
        for seed in seeds:
            matched = [row for row in round_rows if row["method"] == method and row["seed"] == seed]
            if len(matched) != 10 or {row["round"] for row in matched} != set(range(1, 11)):
                raise ValueError("A seed must retain all ten original synchronized rounds")
            seed_rows.append({"method": method, "seed": seed, "requested_cut": matched[0]["requested_cut"],
                              "critical_cost_sec_per_batch": mean(row["critical_cost_sec_per_batch"] for row in matched),
                              "mean_implied_wait_sec_per_batch": mean(row["mean_implied_wait_sec_per_batch"] for row in matched),
                              "max_implied_wait_sec_per_batch": mean(row["max_implied_wait_sec_per_batch"] for row in matched),
                              "rounds": 10})
        matched = [row for row in seed_rows if row["method"] == method]
        condition_rows.append({"method": method, "requested_cut": matched[0]["requested_cut"],
                               "mean_critical_cost_sec_per_batch": mean(row["critical_cost_sec_per_batch"] for row in matched),
                               "seed_critical_costs_sec_per_batch": [row["critical_cost_sec_per_batch"] for row in matched],
                               "mean_implied_wait_sec_per_batch": mean(row["mean_implied_wait_sec_per_batch"] for row in matched),
                               "mean_max_implied_wait_sec_per_batch": mean(row["max_implied_wait_sec_per_batch"] for row in matched),
                               "seeds": seeds, "n_seeds": len(seeds)})
    per_client = []
    for method in METHODS:
        for label in DEVICES:
            means, waits, fractions = [], [], []
            for seed in seeds:
                matched = [row for row in client_rows if row["method"] == method and row["label"] == label and row["seed"] == seed]
                means.append(mean(row["cost_sec_per_batch"] for row in matched))
                waits.append(mean(row["implied_wait_sec_per_batch"] for row in matched))
                fractions.append(mean(row["highest_cost"] for row in matched))
            per_client.append({"method": method, "label": label,
                               "mean_cost_sec_per_batch": mean(means),
                               "mean_implied_wait_sec_per_batch": mean(waits),
                               "seed_costs_sec_per_batch": means, "seed_waits_sec_per_batch": waits,
                               "highest_cost_fraction": mean(fractions)})
    baseline = {row["seed"]: row["critical_cost_sec_per_batch"] for row in seed_rows if row["method"] == "fedavg"}
    baseline_condition = next(row for row in condition_rows if row["method"] == "fedavg")
    for row in condition_rows:
        matched = [item for item in seed_rows if item["method"] == row["method"]]
        changes = [100 * (item["critical_cost_sec_per_batch"] / baseline[item["seed"]] - 1) for item in matched]
        row["mean_paired_critical_change_pct_vs_local"] = mean(changes)
        row["seed_critical_changes_pct_vs_local"] = changes
        row["ratio_of_mean_critical_costs_vs_local"] = row["mean_critical_cost_sec_per_batch"] / baseline_condition["mean_critical_cost_sec_per_batch"]
    return {"schema": "splitfleet.equal-batch-synchronization-cost.v1",
            "scope": "derived_equal_batch_comparison_not_observed_wait",
            "cost_definition": "original client fit interval / executed batch count",
            "critical_definition": "tau = max client cost inside each original seed/round; then average rounds within seed and equally average seeds",
            "wait_definition": "w_i = tau - t_i; slack under a common-start equal-batch comparison, not recorded physical stall",
            "actual_synchronization_wait_measured": False,
            "no_cross_host_timestamp_subtraction": True,
            "conditions": condition_rows, "per_seed_condition": seed_rows,
            "per_client_condition": per_client, "raw_round_costs": round_rows,
            "raw_client_slack": client_rows,
            "limitations": ["Unequal epoch lengths were normalized; this is a counterfactual equal-batch comparison",
                            "No coordinator-clock fit-completion receipts exist for the original P04 jobs",
                            "75% can increase critical cost versus Local while implied waiting is still below Local",
                            "Derived slack is neither measured client barrier waiting nor suffix-server queueing"]}


def analyse(cohort: Path, registry_path: Path, output: Path) -> dict:
    cohort, registry_path, output = (p.resolve() for p in (cohort, registry_path, output))
    if output.exists():
        raise FileExistsError("Select a fresh analysis directory; frozen results are read-only")
    registry = json.loads(registry_path.read_text())
    registered = {str(Path(row["path"]).resolve()): row["sha256"] for row in registry["sources"]}
    source_hashes = {str(registry_path): digest(registry_path)}

    def read(path: Path, *, require_registered=False):
        actual = digest(path)
        expected = registered.get(str(path))
        if require_registered and expected is None:
            raise ValueError(f"The historical evidence registry does not bind {path.name}")
        if expected is not None and actual != expected:
            raise ValueError(f"A frozen input changed: {path.name}")
        source_hashes[str(path)] = actual
        return json.loads(path.read_text())

    plan = read(cohort / "frozen_plan.json")
    runtime = read(cohort / "runtime_manifest.json")
    if plan["source_identity"] != runtime["source_identity"]:
        raise ValueError("Frozen protocol and executed source identities differ")
    stage, = plan["stages"]
    if (stage["model_id"], stage["batch_size"], stage["rounds"], stage["seeds"]) != (
        "deeplabv3_resnet50", 2, 10, [7401, 7402]
    ):
        raise ValueError("This reanalysis is scoped to the original two-seed DeepLab cohort")
    workers = {details[0] for details in DEVICES.values()}
    if set(stage["worker_ids"]) != workers:
        raise ValueError("Expected exactly six CPU/GPU workers on three physical hosts")
    rows, seed_rows, references, cuts = [], [], {}, {}
    for seed in stage["seeds"]:
        for method in METHODS:
            arm_dir = cohort / "physical" / f"deeplabv3_resnet50_timing_s{seed}" / f"semantic_segmentation_{method}"
            result = read(arm_dir / "result.json", require_registered=True)
            validation = read(arm_dir / "validation_report.json")
            if not validation["valid"] or result["fit_failures"]:
                raise ValueError("Only complete, validated arms can provide the plotted observations")
            if (result["model_id"], result["task"], result["seed"], result["batch_size"], result["rounds"],
                result["optimizer"], result["learning_rate"], result["local_epochs"]) != (
                "deeplabv3_resnet50", "semantic_segmentation", seed, 2, 10, "adam", .0001, 1
            ):
                raise ValueError("Model/task/work budget differs from the declared design")
            if set(result["worker_ids"]) != workers or result["server_device"] != "cuda:0":
                raise ValueError("An arm used an unexpected worker population or suffix device")
            paired = {key: result[key] for key in (
                "initial_model_hash", "data_content_hash", "partition_hash", "partition_sizes",
                "pretrain_checkpoint_sha256", "model_metadata", "math_policy"
            )}
            if seed in references and references[seed] != paired:
                raise ValueError("Initialization/data/partition/execution policies differ within a seed")
            references[seed] = paired
            nominal = None if method == "fedavg" else method.removeprefix("splitfed_fixed") + "%"
            canonical = "full_local" if nominal is None else result["fixed_cut_resolution"][nominal]
            if nominal is not None:
                if result["resolved_fixed_boundary"] != canonical:
                    raise ValueError("A static arm did not execute the pre-resolved cut")
                if nominal in cuts and cuts[nominal] != canonical:
                    raise ValueError("Canonical cut identity changed between seeds")
                cuts[nominal] = canonical
            if len(result["fit_records"]) != 60:
                raise ValueError("Missing client-round receipts")
            for label, (identity, host, device, index) in DEVICES.items():
                fits = [fit for fit in result["fit_records"] if fit["metrics"]["logical_client_id"] == identity]
                if sorted(fit["round_id"] for fit in fits) != list(range(1, 11)):
                    raise ValueError("A worker does not have all ten original measured rounds")
                client_rows = []
                for fit in sorted(fits, key=lambda item: item["round_id"]):
                    metric = fit["metrics"]
                    examples = result["partition_sizes"][str(index)]
                    batches = math.ceil(examples / result["batch_size"])
                    if (metric["device"], metric["client_index"], metric["boundary"], metric["num_examples"],
                        fit["num_examples"], metric["num_batches"], metric["partition_hash"], metric.get("round_id", fit["round_id"])) != (
                        device, index, canonical, examples, examples, batches, result["partition_hash"], fit["round_id"]
                    ):
                        raise ValueError("Client identity, executed work, cut or data receipts disagree")
                    if metric.get("skipped_batches", 0) or metric.get("skipped_examples", 0):
                        raise ValueError("Skipped training work cannot be treated as a completed condition")
                    start, finish = metric["fit_started_unix_ns"], metric["fit_finished_unix_ns"]
                    if not isinstance(start, int) or not isinstance(finish, int) or not 0 < start < finish:
                        raise ValueError("Invalid within-client fit interval")
                    duration = (finish - start) / 1e9
                    if not math.isfinite(metric["fit_duration_sec"]) or metric["fit_duration_sec"] <= 0:
                        raise ValueError("Invalid recorded internal training duration")
                    row = {"seed": seed, "method": method, "panel": "a" if nominal is None else "b",
                           "label": label, "physical_host": host, "logical_client_id": identity,
                           "device": device, "round": fit["round_id"], "requested_cut": nominal or "full_local",
                           "canonical_cut": canonical, "examples": examples, "batches": batches,
                           "partial_final_batch": examples % 2 != 0, "client_fit_sec": duration,
                           "sec_per_batch": duration / batches,
                           "internal_fit_duration_sec": metric["fit_duration_sec"],
                           "source_result": str(arm_dir / "result.json")}
                    rows.append(row)
                    client_rows.append(row)
                seed_rows.append({"seed": seed, "method": method, "panel": client_rows[0]["panel"],
                                  "label": label, "requested_cut": nominal or "full_local", "canonical_cut": canonical,
                                  "mean_sec_per_batch": mean(row["sec_per_batch"] for row in client_rows),
                                  "rounds": len(client_rows), "examples_per_round": examples, "batches_per_round": batches})
    conditions = []
    for method in METHODS:
        for label in DEVICES:
            matched = [row for row in seed_rows if row["method"] == method and row["label"] == label]
            conditions.append({"method": method, "label": label, "panel": matched[0]["panel"],
                               "requested_cut": matched[0]["requested_cut"], "canonical_cut": matched[0]["canonical_cut"],
                               "mean_sec_per_batch": mean(row["mean_sec_per_batch"] for row in matched),
                               "seed_means_sec_per_batch": [row["mean_sec_per_batch"] for row in matched],
                               "seeds": [row["seed"] for row in matched], "n_seeds": len(matched)})
    full = {row["label"]: row["mean_sec_per_batch"] for row in conditions if row["panel"] == "a"}
    best = {label: min((row for row in conditions if row["panel"] == "b" and row["label"] == label),
                      key=lambda row: row["mean_sec_per_batch"])["requested_cut"] for label in DEVICES}
    summary = {"schema": SCHEMA, "status": "completed", "scope": "descriptive_reanalysis_of_P04",
               "new_physical_training": False, "post_hoc_figure_selection": True,
               "model": "DeepLabV3-ResNet50", "dataset": "Oxford-IIIT Pet", "image_size": 320,
               "batch_size": 2, "optimizer": "Adam", "learning_rate": .0001,
               "cohort": str(cohort), "source_identity": runtime["source_identity"],
               "source_hashes": source_hashes, "source_registry_sha256": digest(registry_path),
               "device_mapping": {label: {"logical_client_id": ident, "physical_host": host, "device": device}
                                  for label, (ident, host, device, _) in DEVICES.items()},
               "fixed_cut_resolution": cuts, "seeds": stage["seeds"], "rounds_per_seed": 10,
               "warmup_exclusion": "none; all original rounds retained, no retrospective warm-up deletion",
               "primary_replication_unit": "paired seed", "completed_jobs": 8,
               "client_round_records": len(rows), "per_seed_condition": seed_rows, "conditions": conditions,
               "endpoint": "Within each client's fit_started_unix_ns to fit_finished_unix_ns, after readiness wait and before returned-state transmission; includes local preparation/export and split activation/gradient RPC; excludes federation download/upload, readiness waiting, aggregation and evaluation",
               "clock": "within-host recorded time.time_ns; no subtraction between physical hosts",
               "normalization": "client fit seconds divided by actual executed batch count per round; ten rounds averaged inside each seed, then two seed means equally weighted",
               "limitations": ["Unequal client partitions; a final batch can contain one rather than two examples",
                               "CPU and GPU workers run concurrently on each host; shared memory/thermals may contribute",
                               "All split workers share the suffix server; split latency includes RPC and contention",
                               "Two original seeds, ten rounds; this figure is not an independent replication of P04",
                               "Resource isolation, network/server interventions and equal-quality acceleration are not established"],
               "observations": {"local_sec_per_batch": full,
                                "same_host_cpu_over_gpu": {str(i): full[f"cpu{i}"] / full[f"gpu{i}"] for i in (1, 2, 3)},
                                "tested_minimum_cut": best},
               "human_scientific_verification": "pending", "submission_ready": False}
    summary["synchronization_cost_model"] = synchronization_cost_model(rows)
    summary["analysis_code_sha256"] = digest(Path(__file__))
    # Refuse changed inputs before publishing a derived artifact.
    if any(digest(Path(path)) != expected for path, expected in source_hashes.items()):
        raise ValueError("An input changed during the read-only analysis")
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / "raw_client_rounds.csv", rows)
    save_csv(output / "per_seed_condition.csv", seed_rows)
    save_csv(output / "condition_summary.csv", [{k: v for k, v in row.items() if not isinstance(v, list)} for row in conditions])
    model = summary["synchronization_cost_model"]
    save_csv(output / "synchronization_round_costs.csv", model["raw_round_costs"])
    save_csv(output / "implied_client_wait.csv", model["raw_client_slack"])
    save_csv(output / "synchronization_seed_summary.csv", model["per_seed_condition"])
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.cohort, args.source_registry, args.output)
    print(json.dumps({key: result[key] for key in ("status", "scope", "completed_jobs", "client_round_records", "observations")}, indent=2))


if __name__ == "__main__":
    main()
