"""Audit measured communication counters without inventing missing categories."""

from __future__ import annotations

import math


def _byte_count(value, key):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or value < 0:
        raise ValueError(f"{key} must contain nonnegative integer byte counts")
    return int(value)


def _sum_complete(records, key):
    if not records or any(key not in row["metrics"] for row in records):
        return None
    values = [row["metrics"][key] for row in records]
    return sum(_byte_count(value, key) for value in values)


def communication_summary(result):
    fits = result.get("fit_records", [])
    suffix = result.get("server_fit_records", [])
    measured = {
        "activation_and_target_upload_bytes": _sum_complete(fits, "upload_bytes"),
        "gradient_and_response_download_bytes": _sum_complete(fits, "download_bytes"),
        "client_state_upload_raw_tensor_bytes": _sum_complete(fits, "state_upload_bytes"),
        "suffix_replica_state_raw_tensor_bytes": _sum_complete(suffix, "state_bytes"),
    }
    report = {
        "measured_counters": measured,
        "activation_upload_bytes": None,
        "target_upload_bytes": None,
        "gradient_download_bytes": None,
        "model_state_synchronization_wire_bytes": None,
        "total_communication_bytes": None,
        "complete": False,
        "interpretation": "Combined application payloads and raw state tensors use different accounting domains; do not add them as total network traffic. Activation/target separation, gradient/metadata separation, all state download transfers, and transport overhead are not recorded.",
    }
    if result.get("communication_accounting") != "training_client_application_buffers_v1":
        return report
    full_model = result.get("method") in ("fedavg", "fedprox")
    for key in ("activation_upload_bytes", "target_upload_bytes", "request_metadata_upload_bytes",
                "gradient_download_bytes", "response_metadata_download_bytes"):
        report[key] = 0 if full_model else _sum_complete(fits, key)
    report["model_state_upload_bytes"] = _sum_complete(fits, "model_state_upload_bytes")
    downloads = result.get("state_download_records", [])
    report["model_state_download_bytes"] = (
        sum(_byte_count(row["model_state_download_bytes"], "model_state_download_bytes") for row in downloads)
        if downloads and all("model_state_download_bytes" in row for row in downloads) else None)
    state = [report["model_state_upload_bytes"], report["model_state_download_bytes"]]
    report["model_state_synchronization_bytes"] = sum(state) if all(value is not None for value in state) else None
    identities = lambda records: [(row["round_id"], row["cid"]) for row in records]
    uploaded, configured = identities(fits), identities(downloads)
    if len(set(uploaded)) != len(uploaded) or len(set(configured)) != len(configured):
        raise ValueError("Communication records contain duplicate round/client receipts")
    budget = result.get("rounds", 0) * result.get("expected_clients", 0)
    by_round = {}
    for round_id, cid in uploaded:
        by_round.setdefault(round_id, set()).add(cid)
    rounds = result.get("rounds", 0)
    expected_clients = result.get("expected_clients", 0)
    round_population_complete = (isinstance(rounds, int) and not isinstance(rounds, bool) and rounds > 0
        and isinstance(expected_clients, int) and not isinstance(expected_clients, bool) and expected_clients > 0
        and set(by_round) == set(range(1, rounds + 1))
        and all(len(clients) == expected_clients for clients in by_round.values()))
    population_complete = bool(budget and len(fits) == budget and set(uploaded) == set(configured)
                               and round_population_complete and not result.get("fit_failures"))
    categories = [report[key] for key in ("activation_upload_bytes", "target_upload_bytes",
        "request_metadata_upload_bytes", "gradient_download_bytes", "response_metadata_download_bytes",
        "model_state_synchronization_bytes")]
    report["complete"] = population_complete and all(value is not None for value in categories)
    report["total_communication_bytes"] = sum(categories) if report["complete"] else None
    report["accounting_domain"] = "training_client_application_buffers_v1"
    report["population_complete"] = population_complete
    report["physical_network_traffic_bytes"] = None
    report["interpretation"] = (
        "Actual serialized Boundary/Gradient envelopes, target/request/response data buffers and Flower .npy "
        "parameter buffers over all completed training client rounds. Excludes initial/calibration/evaluation "
        "traffic, Flower config/metric buffers, protobuf/transport headers, TLS, retries and coordinator-local "
        "suffix replica copies. Total is complete only in this declared application-buffer domain; physical "
        "network traffic and model synchronization including transport overhead remain unmeasured.")
    return report
