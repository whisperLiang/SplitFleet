"""Audit measured communication counters without inventing missing categories."""

from __future__ import annotations


def _sum_complete(records, key):
    if not records or any(key not in row["metrics"] for row in records):
        return None
    values = [row["metrics"][key] for row in records]
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in values):
        raise ValueError(f"{key} must contain nonnegative byte counts")
    return sum(int(value) for value in values)


def communication_summary(result):
    fits = result.get("fit_records", [])
    suffix = result.get("server_fit_records", [])
    measured = {
        "activation_and_target_upload_bytes": _sum_complete(fits, "upload_bytes"),
        "gradient_and_response_download_bytes": _sum_complete(fits, "download_bytes"),
        "client_state_upload_raw_tensor_bytes": _sum_complete(fits, "state_upload_bytes"),
        "suffix_replica_state_raw_tensor_bytes": _sum_complete(suffix, "state_bytes"),
    }
    return {
        "measured_counters": measured,
        "activation_upload_bytes": None,
        "target_upload_bytes": None,
        "gradient_download_bytes": None,
        "model_state_synchronization_wire_bytes": None,
        "total_communication_bytes": None,
        "complete": False,
        "interpretation": "Combined application payloads and raw state tensors use different accounting domains; do not add them as total network traffic. Activation/target separation, gradient/metadata separation, all state download transfers, and transport overhead are not recorded.",
    }
