"""Audit asynchronous receipts without imposing synchronous-round semantics."""

from __future__ import annotations

import math


def validate_fedcompass(result, *, bundle, workers, barrier_records):
    errors = []
    ids = [worker["identity"] for worker in workers]
    by_id = {worker["identity"]: worker for worker in workers}
    for key in ("task", "seed", "data_content_hash", "partition_hash", "initial_model_hash",
                "image_model", "pretrain_checkpoint_sha256", "batch_size", "train_size", "test_size"):
        expected = bundle.get(key)
        if key == "task":
            expected = bundle["task"]
        if result.get(key) != expected:
            errors.append(f"FedCompass {key} differs from the frozen bundle")
    for key in ("model_id", "model_metadata", "num_parameters", "trainable_parameters"):
        if key in bundle and result.get(key) != bundle[key]:
            errors.append(f"FedCompass {key} differs from the frozen bundle")
    if bundle.get("worker_ids") is not None and bundle["worker_ids"] != ids:
        errors.append("FedCompass workers differ from the frozen bundle")
    if result.get("method") != "fedcompass" or result.get("aggregation_protocol") != "semi_asynchronous":
        errors.append("FedCompass semi-asynchronous protocol receipt is missing")
    if result.get("expected_clients") != len(ids) or result.get("worker_ids") != ids:
        errors.append("FedCompass configured worker population differs")
    if result.get("server_fit_records") or result.get("owned_state_exchange"):
        errors.append("FedCompass produced split-server execution")
    if result.get("fit_failures") or result.get("final_model_hash") == result.get("initial_model_hash"):
        errors.append("FedCompass failed or did not update its global model")
    assignments = bundle.get("assignments", {})
    if sorted(i for positions in assignments.values() for i in positions) != list(range(bundle["train_size"])):
        errors.append("FedCompass partitions are not disjoint and exhaustive")
    receipts = barrier_records or []
    readiness = len(receipts) == 1 and receipts[0].get("round_id") == 1 and set(receipts[0].get("ready_ids", [])) == set(ids)
    if readiness:
        row = receipts[0]
        arrivals = row.get("ready_monotonic_ns", {})
        start, end = row.get("release_started_monotonic_ns", 0), row.get("release_finished_monotonic_ns", 0)
        readiness = (set(arrivals) == set(ids) and min(arrivals.values()) > 0
                     and max(arrivals.values()) <= start <= end)
    if not readiness or result.get("fit_barrier") != "initial_readiness_only":
        errors.append("FedCompass must have exactly one complete initial readiness barrier")
    config = result.get("fedcompass_config") or {}
    try:
        from experiments.baselines.fedcompass import FedCompassConfig
        FedCompassConfig(**config)
    except (TypeError, ValueError):
        errors.append("FedCompass configuration receipt is invalid")
    dispatches = result.get("dispatch_records", [])
    fits = result.get("fit_records", [])
    planned = {row["dispatch_id"]: row for row in dispatches}
    if (not fits or len(planned) != len(dispatches)
            or len(fits) != result.get("completed_dispatches")
            or {row.get("dispatch_id") for row in fits} != set(planned) or len(fits) != len(planned)):
        errors.append("FedCompass dispatch/arrival accounting is incomplete")
    contributions = {cid: 0 for cid in ids}
    process_ids = {cid: set() for cid in ids}
    for row in fits:
        metrics = row.get("metrics", {})
        cid = metrics.get("logical_client_id")
        job = planned.get(row.get("dispatch_id"), {})
        if cid not in by_id or cid != job.get("client_id"):
            errors.append("FedCompass client identity differs from its assignment")
            continue
        contributions[cid] += 1
        process_ids[cid].add(metrics.get("pid"))
        worker = by_id[cid]
        if metrics.get("client_index") != worker["index"] or metrics.get("device") != worker["device"]:
            errors.append(f"FedCompass {cid} used the wrong worker or device")
        steps = job.get("local_steps", 0)
        size = bundle["partition_sizes"][str(worker["index"])]
        batches_per_epoch = math.ceil(size / bundle["batch_size"])
        expected_examples = ((steps // batches_per_epoch) * size
                             + min((steps % batches_per_epoch) * bundle["batch_size"], size))
        if (not config.get("min_local_steps", 1) <= steps <= config.get("max_local_steps", 0)
                or metrics.get("num_batches") != steps or metrics.get("assigned_local_steps") != steps
                or metrics.get("model_version") != job.get("model_version")
                or metrics.get("partition_hash") != bundle["partition_hash"]):
            errors.append(f"FedCompass {cid} assignment or partition receipt is invalid")
        if (row.get("num_examples", 0) != expected_examples
                or row.get("num_examples") != metrics.get("num_examples")):
            errors.append(f"FedCompass {cid} sample accounting is invalid")
        if (metrics.get("fit_started_unix_ns", 0) <= 0
                or metrics.get("fit_finished_unix_ns", 0) <= metrics.get("fit_started_unix_ns", 0)):
            errors.append(f"FedCompass {cid} fit timestamps are invalid")
        if row.get("update_status") not in ("aggregated", "discarded_at_stop", "unaggregated_at_stop", "superseded_late_buffer"):
            errors.append("FedCompass contains an unfinished update receipt")
        for key in ("task_loss", "fit_duration_sec"):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                errors.append(f"FedCompass {cid} has invalid {key}")
    if any(count == 0 for count in contributions.values()):
        errors.append("FedCompass initial configured-client contributions are missing")
    if any(len(pids) != 1 or next(iter(pids), None) is None for pids in process_ids.values()):
        errors.append("FedCompass client process identity changed or is missing")
    for worker in workers:
        host_id = worker["host"]["id"]
        cpu, gpu = process_ids.get(host_id + "-cpu"), process_ids.get(host_id + "-gpu")
        if cpu and gpu and cpu == gpu:
            errors.append(f"FedCompass {host_id} has no distinct CPU/GPU processes")
    aggregations = result.get("aggregation_records", [])
    if ([row.get("model_version") for row in aggregations] != list(range(1, result.get("global_updates", 0) + 1))
            or result.get("rounds") != result.get("global_updates")):
        errors.append("FedCompass model-version accounting is invalid")
    total = sum(bundle["partition_sizes"].values())
    weights = {cid: bundle["partition_sizes"][str(i)] / total for i, cid in enumerate(ids)}
    if result.get("fedcompass_client_weights") != weights:
        errors.append("FedCompass client weighting differs from the matched data objective")
    applied = [dispatch for row in aggregations for dispatch in row.get("dispatch_ids", [])]
    aggregated_ids = [row.get("dispatch_id") for row in fits if row.get("update_status") == "aggregated"]
    if len(applied) != len(set(applied)) or set(applied) != set(aggregated_ids):
        errors.append("FedCompass updates were duplicated or omitted from aggregation receipts")
    budget = result.get("time_budget_sec")
    for row in fits:
        if row.get("update_status") != "aggregated":
            continue
        try:
            delay = row["aggregated_model_version"] - 1 - row["model_version"]
            expected = config["alpha"] * weights[row["metrics"]["logical_client_id"]] * (1 + delay) ** -config["staleness_exponent"]
            valid = delay >= 0 and row["staleness"] == delay and math.isclose(row["aggregation_weight"], expected)
            if budget is not None:
                valid = valid and row["aggregation_elapsed_sec"] < budget
            if not valid:
                errors.append("FedCompass staleness, weight or timed endpoint is invalid")
        except (KeyError, TypeError, ValueError):
            errors.append("FedCompass aggregation receipt is invalid")
    evaluations = result.get("evaluation_records", [])
    if not evaluations or evaluations[0].get("model_version") != 0:
        errors.append("FedCompass initial evaluation is missing")
    previous = -1.0
    for row in evaluations:
        now = row.get("elapsed_sec", -1)
        if not math.isfinite(now) or now < previous or not row.get("metrics"):
            errors.append("FedCompass evaluation timing or metrics are invalid")
        for value in row.get("metrics", {}).values():
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append("FedCompass task metric is invalid")
        previous = now
    return {"schema": "splitfleet.fedcompass-validation.v1", "valid": not errors,
            "errors": errors, "warnings": [], "workers": len(ids), "worker_ids": ids,
            "barrier_verified": readiness, "initial_readiness_verified": readiness,
            "aggregation_protocol": "semi_asynchronous", "fit_interval_overlap_sec": {},
            "client_contribution_counts": contributions, "global_updates": result.get("global_updates"),
            "completed_dispatches": len(fits)}
