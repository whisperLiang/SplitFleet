"""Validate FedAsync/FedBuff identity, local work, buffering and deadline receipts."""
from __future__ import annotations

import math
from collections import Counter

from experiments.baselines.async_fl import AsyncFLConfig


def validate_async_fl(result, *, bundle, workers, barrier_records):
    errors = []
    ids = [worker["identity"] for worker in workers]
    by_id = {worker["identity"]: worker for worker in workers}
    method = result.get("method")
    if method not in ("fedasync", "fedbuff") or result.get("aggregation_protocol") != "asynchronous":
        errors.append("Asynchronous algorithm identity is missing")
    for name in ("task", "seed", "partition_hash", "initial_model_hash", "data_content_hash",
                 "train_size", "test_size", "batch_size", "image_model", "pretrain_checkpoint_sha256",
                 "model_id", "model_metadata", "num_parameters", "trainable_parameters"):
        if name in bundle and result.get(name) != bundle[name]:
            errors.append(f"Async {name} differs from the frozen bundle")
    if result.get("worker_ids") != ids or result.get("expected_clients") != len(ids):
        errors.append("Async worker population differs")
    if (result.get("server_fit_records") or result.get("owned_state_exchange")
            or result.get("fit_failures") or result.get("final_model_hash") == result.get("initial_model_hash")):
        errors.append("Async run failed, used split execution or has no model update")
    if sorted(i for positions in bundle.get("assignments", {}).values() for i in positions) != list(range(bundle["train_size"])):
        errors.append("Async partitions are not disjoint and exhaustive")
    receipts = barrier_records or []
    readiness = False
    if len(receipts) == 1 and receipts[0].get("round_id") == 1:
        row = receipts[0]
        arrivals = row.get("ready_monotonic_ns", {})
        start, end = row.get("release_started_monotonic_ns", 0), row.get("release_finished_monotonic_ns", 0)
        readiness = (set(row.get("ready_ids", [])) == set(ids) and set(arrivals) == set(ids)
            and min(arrivals.values(), default=0) > 0 and max(arrivals.values(), default=0) <= start <= end)
    if not readiness or result.get("fit_barrier") != "initial_readiness_only":
        errors.append("Async run requires exactly one initial readiness barrier")
    raw_config = result.get("async_fl_config", {})
    try:
        config = AsyncFLConfig(**raw_config)
        if config.method != method:
            raise ValueError("Wrong configuration method")
    except (TypeError, ValueError) as exc:
        errors.append(f"Invalid async configuration: {exc}")
        return {"valid": False, "errors": errors, "barrier_verified": readiness}
    dispatches = result.get("dispatch_records", [])
    planned = {row["dispatch_id"]: row for row in dispatches}
    fits = result.get("fit_records", [])
    if (not fits or len(planned) != len(dispatches) or len(fits) != len(planned)
            or len(fits) != result.get("completed_dispatches")
            or Counter(r.get("dispatch_id") for r in fits) != Counter(planned.keys())):
        errors.append("Async dispatch/fit accounting is incomplete")
    contributions, pids = Counter(), {cid: set() for cid in ids}
    for fit in fits:
        metrics = fit.get("metrics", {})
        cid = metrics.get("logical_client_id")
        job = planned.get(fit.get("dispatch_id"), {})
        if cid not in by_id or cid != job.get("client_id"):
            errors.append("Async assignment identity differs")
            continue
        worker = by_id[cid]
        contributions[cid] += 1
        pids[cid].add(metrics.get("pid"))
        size = bundle["partition_sizes"][str(worker["index"])]
        steps = config.local_steps
        epoch_batches = math.ceil(size / bundle["batch_size"])
        examples = (steps // epoch_batches) * size + min((steps % epoch_batches) * bundle["batch_size"], size)
        if (metrics.get("client_index") != worker["index"] or metrics.get("device") != worker["device"]
                or job.get("local_steps") != steps or metrics.get("num_batches") != steps
                or metrics.get("assigned_local_steps") != steps
                or metrics.get("partition_hash") != bundle["partition_hash"]
                or metrics.get("model_version") != job.get("model_version")
                or fit.get("num_examples") != examples or metrics.get("num_examples") != examples
                or metrics.get("proximal_mu") != (config.proximal_mu if method == "fedasync" else 0.0)):
            errors.append("Async local work, optimizer or partition receipt differs")
        if metrics.get("fit_started_unix_ns", 0) <= 0 or metrics.get("fit_finished_unix_ns", 0) <= metrics.get("fit_started_unix_ns", 0):
            errors.append("Async fit timestamps are invalid")
        for name in ("task_loss", "fit_duration_sec"):
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                errors.append(f"Async {name} is invalid")
        if fit.get("update_status") not in ("aggregated", "unaggregated_at_stop", "discarded_at_stop"):
            errors.append("Async fit has no completed update status")
    if any(contributions[cid] < 1 or len(pids[cid]) != 1 or None in pids[cid] for cid in ids):
        errors.append("Async client contribution/process identity is missing or changed")
    for worker in workers:
        host = worker["host"]["id"]
        if pids.get(host + "-cpu") and pids.get(host + "-gpu") and pids[host + "-cpu"] == pids[host + "-gpu"]:
            errors.append("Async CPU/GPU workers share one process")
    aggregations = result.get("aggregation_records", [])
    if not aggregations or [a.get("model_version") for a in aggregations] != list(range(1, result.get("global_updates", 0) + 1)):
        errors.append("Async committed model versions are incomplete")
    applied = [i for a in aggregations for i in a.get("dispatch_ids", [])]
    if Counter(applied) != Counter(f["dispatch_id"] for f in fits if f.get("update_status") == "aggregated") or len(applied) != len(set(applied)):
        errors.append("Async aggregation omits or duplicates a dispatch")
    fit_by_id = {f["dispatch_id"]: f for f in fits}
    budget = result.get("time_budget_sec")
    for aggregation in aggregations:
        count = 1 if method == "fedasync" else config.buffer_size
        if len(aggregation.get("dispatch_ids", [])) != count or aggregation.get("buffer_size") != count:
            errors.append("Async aggregation has an invalid buffer population")
        for dispatch_id in aggregation.get("dispatch_ids", []):
            fit = fit_by_id.get(dispatch_id, {})
            try:
                delay = aggregation["model_version"] - 1 - fit["model_version"]
                weight = (config.alpha * (1 + delay) ** -config.staleness_exponent if method == "fedasync"
                          else config.server_learning_rate / config.buffer_size)
                if (delay < 0 or fit["staleness"] != delay or not math.isclose(fit["aggregation_weight"], weight)
                        or fit["aggregated_model_version"] != aggregation["model_version"]
                        or fit["aggregation_elapsed_sec"] != aggregation["elapsed_sec"]
                        or (budget is not None and aggregation["elapsed_sec"] >= budget)):
                    errors.append("Async staleness, coefficient or deadline differs")
            except (KeyError, TypeError, ValueError):
                errors.append("Async aggregation receipt is incomplete")
    evaluations = result.get("evaluation_records", [])
    if not evaluations or evaluations[0].get("model_version") != 0:
        errors.append("Async initial evaluation is missing")
    previous = -1.0
    for evaluation in evaluations:
        now = evaluation.get("elapsed_sec", -1)
        if not math.isfinite(now) or now < previous or not evaluation.get("metrics"):
            errors.append("Async evaluation timing or metric is invalid")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in evaluation.get("metrics", {}).values()):
            errors.append("Async task quality is invalid")
        previous = now
    return {"schema": "splitfleet.async-fl-validation.v1", "valid": not errors,
        "errors": errors, "warnings": [], "worker_ids": ids, "workers": len(ids),
        "barrier_verified": readiness, "initial_readiness_verified": readiness,
        "aggregation_protocol": "asynchronous",
        "fit_interval_overlap_sec": {}, "client_contribution_counts": dict(contributions),
        "global_updates": result.get("global_updates")}
