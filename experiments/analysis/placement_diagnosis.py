"""Audit measured placement costs without treating predictions as new timings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def diagnose(result):
    fits = result["fit_records"]
    if not fits or result.get("fit_failures"):
        raise ValueError("Diagnosis requires a completed, nonfailed training run")
    batches = sum(row["metrics"]["num_batches"] for row in fits)
    if batches <= 0:
        raise ValueError("Diagnosis requires actual training batches")
    diagnostics = list(result.get("round_diagnostics", {}).values())
    joint = [row for row in diagnostics if "predicted_client_completion_ms" in row]
    planned = [row for parent in diagnostics for row in (
        [parent] if "planning_compute" in parent else list(parent.get("per_client", {}).values()))
        if "planning_compute" in row]

    def weighted(name):
        present = [row["metrics"] for row in fits if row["metrics"].get(name) is not None]
        count = sum(row["num_batches"] for row in present)
        return sum(row[name] * row["num_batches"] for row in present) / count if count else None

    queue = (sum(row["metrics"]["suffix_server_queue_sec"] for row in fits) * 1000 / batches
             if all("suffix_server_queue_sec" in row["metrics"] for row in fits) else None)
    service = weighted("server_service_mean_ms")
    forward, backward = weighted("client_forward_mean_ms"), weighted("client_backward_mean_ms")
    capacity_binding = sum((row.get("server_capacity_lower_bound_ms") or 0) >=
                           max(row["predicted_client_completion_ms"].values()) for row in joint)
    rpc_spans = []
    for current in sorted({row["round_id"] for row in fits}):
        rows = [row for row in fits if row["round_id"] == current]
        values = [row["metrics"].get("coordinator_fit_rpc_ms") for row in rows]
        if all(value is not None for value in values):
            rpc_spans.append(max(values) / 1000)
    return {
        "model": result["model_id"], "seed": result["seed"],
        "controller": result.get("placement_controller", "CoSplit-UCB"),
        "clients": len(result["worker_ids"]), "server_capacity": result["server_training_concurrency"],
        "job_sec": result["server_duration_sec"], "batches": batches,
        "server_service_ms": service, "queue_ms_per_batch": queue,
        "queue_service_ratio": queue / service if queue is not None and service else None,
        "transport_ms": weighted("network_roundtrip_mean_ms"),
        "prefix_compute_ms": forward + backward if forward is not None and backward is not None else None,
        "active_server_jobs": weighted("server_load_active_jobs"),
        "gpu_utilization": weighted("server_load_gpu_utilization"),
        "queue_length": weighted("server_load_queue_length"),
        "planning_sec": sum(row["planning_compute"]["total_sec"] for row in planned) if planned else None,
        "cost_prediction_sec": sum(row["planning_compute"]["estimate_sec"] for row in planned) if planned else None,
        "capacity_binding_rounds": capacity_binding, "joint_scored_rounds": len(joint),
        "exploration_actions": sum(len(row.get("exploration", [])) for row in joint),
        "upper_mean_ratio_median": median(row["baseline_upper_makespan_ms"] / row["baseline_makespan_ms"]
                                          for row in joint) if joint else None,
        "changed_cut_fits": sum(row["metrics"].get("switch_ms", 0) > 0 for row in fits),
        "rpc_max_sum_sec": sum(rpc_spans) if len(rpc_spans) == result["rounds"] else None,
        "training_wall_sum_sec": sum(row["actual_round_wall_ms"] / 1000 for row in joint)
            if joint and all("actual_round_wall_ms" in row for row in joint) else None,
        "evaluation_sec": sum(row["duration_sec"] for row in result["evaluation_records"])
            if all("duration_sec" in row for row in result["evaluation_records"]) else None,
        "network_shaping": result.get("network_schedule", {}).get("bandwidth_shaping"),
        "state_download_bytes": sum(row["model_state_download_bytes"] for row in result.get("state_download_records", []))
            if result.get("state_download_records") else None,
        "state_upload_bytes": sum(row["metrics"]["model_state_upload_bytes"] for row in fits)
            if all("model_state_upload_bytes" in row["metrics"] for row in fits) else None,
        "activation_upload_bytes": sum(row["metrics"]["activation_upload_bytes"] for row in fits)
            if all("activation_upload_bytes" in row["metrics"] for row in fits) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_root.glob("physical/**/result.json"))
    if not paths:
        raise ValueError("No physical result files found")
    rows, sources = [], []
    for path in paths:
        raw = path.read_bytes()
        rows.append(diagnose(json.loads(raw)))
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    summaries = []
    for model, controller, capacity in sorted({(row["model"], row["controller"], row["server_capacity"]) for row in rows}):
        group = [row for row in rows if (row["model"], row["controller"], row["server_capacity"]) == (model, controller, capacity)]
        summaries.append({"model": model, "controller": controller, "server_capacity": capacity, "n": len(group), **{
            key: mean(row[key] for row in group) if all(isinstance(row[key], (int, float)) for row in group) else None
            for key in group[0] if key not in ("model", "seed", "controller", "server_capacity")}})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnosis.json").write_text(json.dumps({
        "schema": "splitfleet.placement-diagnosis.v1", "sources": sources, "runs": rows,
        "summary": summaries, "scope": "Descriptive audit; historical methods use different source snapshots",
    }, indent=2, allow_nan=False) + "\n")
    with (args.output / "runs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
