"""Compare complete paired runs only after verifying their shared inputs."""

import csv
import json
from pathlib import Path


def summarize_comparison(directory):
    directory = Path(directory)
    methods = ("splitfleet", "fedavg")
    inputs, curves, timing_rows = {}, [], []
    for method in methods:
        run = directory / method
        read = lambda name: json.loads((run / name).read_text())
        validation, config = read("validation.json"), read("config.json")
        if validation["status"] != "passed":
            raise ValueError("Comparison requires both complete, validated runs")
        config.pop("config_sha256", None)
        if config["workload"].pop("training_method", "splitfleet") != method:
            raise ValueError("Comparison method does not match its frozen configuration")
        partition = read("partition_manifest.json")
        inputs[method] = {
            "config": config,
            "partitions": partition["clients"], "data": partition["data"],
            "initial_model_hash": validation["initial_model_hash"],
            "runtime_manifest": read("runtime_manifest.json"),
        }
        rounds = [json.loads(line) for line in (run / "round_metrics.jsonl").read_text().splitlines()]
        evaluations = {r["round_id"]: r for r in read("model_metrics.json")["evaluations"]}
        if [r["round_id"] for r in rounds] != list(range(1, config["rounds"] + 1)):
            raise ValueError("Comparison requires the complete declared round budget")
        for row in rounds:
            if row["successful_clients"] != config["virtual_clients"]["total"] or row.get("failed_clients", 0):
                raise ValueError("Comparison contains an incomplete training round")
            evaluation = evaluations[row["round_id"]]
            curves.append({
                "method": method, "round": row["round_id"],
                "train_loss": row["train_loss"], "evaluation_loss": evaluation["metrics"]["loss"],
                "evaluation_accuracy": evaluation["metrics"]["accuracy"],
                "round_training_wall_sec": row["round_wall_sec"],
                "evaluation_elapsed_wall_sec": evaluation["elapsed_wall_sec"],
            })
        invocation = read("invocation_timing.json")
        if invocation["child_exit_code"] != 0:
            raise ValueError("Comparison requires successful frozen child invocations")
        final = evaluations[config["rounds"]]["metrics"]
        timing_rows.append({"method": method, "rounds": config["rounds"],
                            "clients": config["virtual_clients"]["total"],
                            "total_wall_sec": invocation["wall_sec"],
                            "final_evaluation_accuracy": final["accuracy"],
                            "final_evaluation_loss": final["loss"]})
    if inputs["splitfleet"] != inputs["fedavg"]:
        raise ValueError("Paired model initialization, partitions, configurations or frozen sources differ")
    for filename, rows in (("convergence_comparison.csv", curves), ("training_time_comparison.csv", timing_rows)):
        with (directory / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"status": "passed", "shared_inputs_verified": True, "runs": timing_rows,
               "fedavg_seconds_over_splitfleet_seconds": timing_rows[1]["total_wall_sec"] / timing_rows[0]["total_wall_sec"],
               "timing_scope": "Frozen child invocation including initialization, evaluation, checkpoints and cleanup; excludes queue wait"}
    (directory / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
