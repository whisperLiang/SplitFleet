"""Report complete paired training accuracy and measured wall time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from .efficiency import (
    measured_training_time,
    paired_efficiency_statistics,
    validate_timing_pair,
)
from .convergence import convergence_comparison, convergence_metrics, plot_comparison
from .native import BACKENDS
from .protocol import SEEDS, hash_file, validate_main_protocol, validate_worker_protocol
from .queue import read_json, save_json


def completed_run(path, *, backend, method, seed):
    if backend not in BACKENDS:
        raise ValueError("This experiment requires PyTorch with 50 clients")
    path = Path(path)
    config = read_json(path / "config.json")
    manifest = read_json(path / "run_manifest.json")
    validation = read_json(path / "quality_validation.json")
    protocol_path = Path(config['workload']['protocol']) / 'protocol.json'
    protocol = read_json(protocol_path)
    validate_worker_protocol(protocol)
    main_rounds, expected_examples = validate_main_protocol(protocol)
    if config.get('rounds') != main_rounds or hash_file(protocol_path) != manifest.get('protocol_sha256'):
        raise ValueError('Completed run differs from its frozen main budget or protocol')
    steps_per_round = BACKENDS[backend] * ((45_000 // BACKENDS[backend] + 15) // 16)
    if (manifest.get("status") != "passed" or manifest.get("stage") != "main"
            or manifest.get("backend") != backend or manifest.get("method") != method
            or manifest.get("seed") != seed or manifest.get("clients") != BACKENDS[backend]
            or validation.get("status") != "passed" or validation.get("examples_seen") != expected_examples
            or validation.get("training_tasks") != BACKENDS[backend] * main_rounds
            or validation.get("optimization_steps") != steps_per_round * main_rounds):
        raise ValueError(f"Run is not a completed main experiment: {path}")
    resources = config["resources"]
    actual_workers = validation.get("worker_count")
    declared_workers = resources.get("max_concurrent_clients")
    policy = resources.get("worker_count_policy")
    if (type(actual_workers) is not int or actual_workers < 1
            or actual_workers != declared_workers
            or policy not in {"one_per_client", "pooled"}
            or (policy == "one_per_client" and actual_workers != BACKENDS[backend])
            or (policy == "pooled" and actual_workers > BACKENDS[backend])):
        raise ValueError(f"Run did not record a valid actual worker population: {path}")
    with (path / "round_metrics.csv").open() as stream:
        rounds = list(csv.DictReader(stream))
    if ([int(row["round"]) for row in rounds] != list(range(1, main_rounds + 1))
            or any(int(row["examples_seen"]) != (index + 1) * 45_000 for index, row in enumerate(rounds))
            or any(int(row["successful_clients"]) != BACKENDS[backend] for row in rounds)):
        raise ValueError("Main round/sample budget is incomplete")
    if any(int(row["optimization_steps"]) != steps_per_round for row in rounds):
        raise ValueError("Main SGD step budget is incomplete")
    metrics = read_json(path / "test_metrics.json")
    for checkpoint in ("final", "best_validation"):
        row = metrics[checkpoint]
        if (row["samples"] != 10_000 or not np.isfinite(row["loss"])
                or not np.isfinite(row["top1"]) or not 0 <= row["top1"] <= 1):
            raise ValueError("Final/best checkpoints require the complete finite test evaluation")
    identities, suffix_commits = {}, {}
    with (path / "client_metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["status"] == "suffix_committed":
                key = (row["partition_id"], row["round_id"])
                if key in suffix_commits:
                    raise ValueError("Duplicated suffix commit")
                suffix_commits[key] = row["metrics"]
                continue
            if row["status"] != "success":
                raise ValueError("A failed logical client cannot enter a complete main pair")
            key = (row["partition_id"], row["round_id"])
            if key in identities:
                raise ValueError("Duplicated client/round identity")
            identities[key] = (row["num_examples"], row["metrics"]["num_batches"],
                               row["metrics"]["data_batch_hash"])
    expected_identities = {(client, round_id) for client in range(BACKENDS[backend])
                           for round_id in range(1, main_rounds + 1)}
    if identities.keys() != expected_identities:
        raise ValueError("Client identity manifest is incomplete")
    if method == "splitfleet":
        if suffix_commits.keys() != identities.keys():
            raise ValueError("A successful prefix has no committed private suffix")
        for key, values in identities.items():
            commit = suffix_commits[key]
            if (commit["num_examples"] != values[0] or commit["suffix_optimization_steps"] != values[1]
                    or commit["suffix_changed_tensors"] <= 0):
                raise ValueError("Private suffix training budget differs from its prefix")
    timing = measured_training_time(path)
    convergence = convergence_metrics(rounds, protocol["convergence"], timing["wall_seconds"])
    return {"manifest": manifest, "test": metrics, "rounds": rounds, "identities": identities,
            "config": config, "timing": timing, "convergence": convergence}


def paired_statistics(fedavg, splitfleet, *, margin_pp=1.0):
    fedavg, splitfleet = np.asarray(fedavg, dtype=np.float64), np.asarray(splitfleet, dtype=np.float64)
    if fedavg.shape != (len(SEEDS),) or splitfleet.shape != (len(SEEDS),):
        raise ValueError("The frozen primary comparison needs all two paired seeds")
    if not np.isfinite(fedavg).all() or not np.isfinite(splitfleet).all():
        raise ValueError("Training outcome contains nonfinite accuracy")
    differences = 100 * (splitfleet - fedavg)
    mean, standard_deviation = float(differences.mean()), float(differences.std(ddof=1))
    half_width = float(t.ppf(.975, len(differences) - 1) * standard_deviation / np.sqrt(len(differences)))
    interval = [mean - half_width, mean + half_width]
    return {"fedavg_top1_mean": float(fedavg.mean()), "fedavg_top1_sample_sd": float(fedavg.std(ddof=1)),
            "splitfleet_top1_mean": float(splitfleet.mean()),
            "splitfleet_top1_sample_sd": float(splitfleet.std(ddof=1)),
            "paired_delta_pp": differences.tolist(), "mean_delta_pp": mean,
            "paired_95pct_t_interval_pp": interval, "degrees_of_freedom": len(SEEDS) - 1,
            "equivalence_margin_pp": margin_pp,
            "interval_within_margin": interval[0] > -margin_pp and interval[1] < margin_pp,
            "assumption": "The paired differences are independent and approximately normally distributed; two seeds give limited precision."}


def summarize_queue(queue_path, output):
    paths = list(queue_path) if isinstance(queue_path, (list, tuple)) else [queue_path]
    paths = [Path(path).resolve() for path in paths]
    output = Path(output).resolve()
    queues = [read_json(path) for path in paths]
    jobs = [job for queue in queues for job in queue["jobs"]]
    if any(job["backend"] not in BACKENDS for job in jobs):
        raise ValueError("Only the current PyTorch / 50-client experiment can be summarized")
    selected = dict.fromkeys(job['backend'] for job in jobs)
    excluded = {backend: reason for queue in queues
                for backend, reason in queue.get('excluded_backends', {}).items() if backend not in selected}
    identities = [(job["backend"], job.get("seed"), job.get("method")) for job in jobs]
    if len(identities) != len(set(identities)):
        raise ValueError("Two queues contain the same experiment identity")
    output.mkdir(parents=True, exist_ok=False)
    rows, comparisons = [], {}
    for backend in selected:
        backend_jobs = [job for job in jobs if job["backend"] == backend]
        completed = sum(job["status"] == "passed" for job in backend_jobs)
        planned_runs = 2 * len(SEEDS)
        row = {"backend": backend, "clients": BACKENDS[backend], "planned_runs": planned_runs,
               "completed_runs": completed, "status": "incomplete"}
        if len(backend_jobs) != planned_runs or completed != planned_runs:
            rows.append(row)
            continue
        outcomes, paired = {}, []
        for seed in SEEDS:
            for method in ("fedavg", "splitfleet"):
                job = next(job for job in backend_jobs if job["seed"] == seed and job["method"] == method)
                outcomes[seed, method] = completed_run(job["output"], backend=backend, method=method, seed=seed)
            left, right = outcomes[seed, "fedavg"], outcomes[seed, "splitfleet"]
            if (left["manifest"]["initial_hash"] != right["manifest"]["initial_hash"]
                    or left["manifest"]["protocol_sha256"] != right["manifest"]["protocol_sha256"]
                    or left["identities"] != right["identities"]):
                raise ValueError("Methods do not share canonical initialization and augmented batch identities")
            validate_timing_pair(left, right)
            paired.append({"seed": seed, **{method: outcomes[seed, method]["test"]
                                            for method in ("fedavg", "splitfleet")},
                           "training_times": {method: outcomes[seed, method]["timing"]
                                              for method in ("fedavg", "splitfleet")}})
        round_counts = {outcome['config']['rounds'] for outcome in outcomes.values()}
        protocols = {outcome['manifest']['protocol_sha256'] for outcome in outcomes.values()}
        if len(round_counts) != 1 or len(protocols) != 1:
            raise ValueError('All paired seeds must use the same frozen training budget and protocol')
        main_rounds = next(iter(round_counts))
        statistics = paired_statistics([outcomes[seed, "fedavg"]["test"]["final"]["top1"] for seed in SEEDS],
                                       [outcomes[seed, "splitfleet"]["test"]["final"]["top1"] for seed in SEEDS])
        efficiency = paired_efficiency_statistics(
            [outcomes[seed, "fedavg"]["timing"]["wall_seconds"] for seed in SEEDS],
            [outcomes[seed, "splitfleet"]["timing"]["wall_seconds"] for seed in SEEDS])
        convergence = convergence_comparison(outcomes, SEEDS)
        objective_met = convergence["both_converged"] and efficiency["speedup_interval_above_one"]
        comparisons[backend] = {"clients": BACKENDS[backend], "seeds": list(SEEDS), "rounds": main_rounds,
                                "primary": statistics, "efficiency": efficiency,
                                "convergence": convergence,
                                "both_converged_and_faster": objective_met,
                                "accuracy_close_and_faster": statistics["interval_within_margin"] and efficiency["speedup_interval_above_one"],
                                "paired_tests": paired}
        row.update(status="passed", rounds=main_rounds, **{key: value for key, value in statistics.items()
                                       if key in {"fedavg_top1_mean", "fedavg_top1_sample_sd", "splitfleet_top1_mean",
                                                  "splitfleet_top1_sample_sd", "mean_delta_pp", "interval_within_margin"}})
        row["paired_95pct_t_interval_pp"] = json.dumps(statistics["paired_95pct_t_interval_pp"])
        row.update({key: efficiency[key] for key in (
            "fedavg_wall_mean_seconds", "splitfleet_wall_mean_seconds", "geometric_mean_speedup",
            "geometric_time_reduction_pct", "speedup_interval_above_one")})
        row["paired_95pct_t_interval_speedup"] = json.dumps(efficiency["paired_95pct_t_interval_speedup"])
        row["both_converged_and_faster"] = objective_met
        curve_rows = []
        for method in ("fedavg", "splitfleet"):
            for index in range(main_rounds):
                accuracy = np.array([float(outcomes[seed, method]["rounds"][index]["val_top1"]) for seed in SEEDS])
                loss = np.array([float(outcomes[seed, method]["rounds"][index]["val_ce"]) for seed in SEEDS])
                elapsed = np.array([float(outcomes[seed, method]["rounds"][index]["elapsed_wall_sec"]) for seed in SEEDS])
                if not np.isfinite(accuracy).all() or not np.isfinite(loss).all():
                    raise ValueError("Validation curves are incomplete or nonfinite")
                curve_rows.append({"method": method, "round": index + 1, "examples_seen": (index + 1) * 45_000,
                                   "val_top1_mean": float(accuracy.mean()), "val_top1_sd": float(accuracy.std(ddof=1)),
                                   "val_ce_mean": float(loss.mean()), "val_ce_sd": float(loss.std(ddof=1)),
                                   "elapsed_wall_sec_mean": float(elapsed.mean())})
        write_csv(output / f"{backend}_validation_curves.csv", curve_rows)
        write_csv(output / "paired_validation_curves.csv", [
            {"seed": seed, "method": method, **point}
            for (seed, method), outcome in outcomes.items() for point in outcome["rounds"]])
        write_csv(output / "time_to_target.csv", convergence["runs"])
        plot_comparison(outcomes, SEEDS, output,
                        target=convergence["runs"][0]["target_val_top1"])
        rows.append(row)
    write_csv(output / "backend_summary.csv", rows)
    report = {"status": "passed" if selected and len(comparisons) == len(selected) else "incomplete",
              "execution_modes": sorted({queue.get('execution_mode', 'standard_admission') for queue in queues}),
              "prerequisite_validation": sorted({queue.get('prerequisite_validation', 'standard_admission') for queue in queues}),
              "excluded_backends": excluded,
              "scope": "PyTorch ResNet50, 50 clients, 100 full rounds, paired FedAvg/SplitFleet convergence and measured time",
              "objective": "Both methods meet the frozen convergence criterion for both seeds and the paired training speedup interval is above 1",
              "queues": [str(path) for path in paths], "backends": rows, "comparisons": comparisons,
              "main_results_available": bool(comparisons),
              "completed_backend_comparisons": len(comparisons)}
    save_json(output / "summary.json", report)
    return report


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_queue(args.queue, args.output)
    print(json.dumps({"status": report["status"], "completed_backend_comparisons": report["completed_backend_comparisons"]}))


if __name__ == "__main__":
    main()
