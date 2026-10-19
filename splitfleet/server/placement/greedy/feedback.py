"""Round feedback validation and robust aggregation."""

from __future__ import annotations

from dataclasses import fields
from typing import Sequence

import numpy as np

from .types import ExecutionProfileKey, PlacementFeedback


_MEASUREMENTS = (
    "client_forward_ms",
    "client_backward_ms",
    "network_upload_ms",
    "network_download_ms",
    "network_roundtrip_ms",
    "state_exchange_ms",
    "server_service_ms",
    "switch_ms",
    "completion_ms",
    "client_peak_memory_mb",
    "server_peak_memory_mb",
    "transport_upload_bytes",
    "transport_download_bytes",
    "transport_request_overhead_bytes",
    "transport_response_overhead_bytes",
)


def _median(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not present:
        return None
    if any(value < 0 for value in present):
        raise ValueError("placement timing/memory feedback must be non-negative")
    return float(np.median(np.asarray(present, dtype=np.float64)))


def aggregate_feedback(feedback: Sequence[PlacementFeedback]) -> list[PlacementFeedback]:
    """Produce one median observation per client/boundary/round."""

    groups: dict[tuple[int, str, str], list[PlacementFeedback]] = {}
    valid_names = {item.name for item in fields(PlacementFeedback)}
    if not set(_MEASUREMENTS) <= valid_names:
        raise AssertionError("PlacementFeedback schema is incomplete")
    for value in feedback:
        if not isinstance(value, PlacementFeedback):
            raise TypeError("observe_round expects PlacementFeedback values")
        groups.setdefault((int(value.round_id), str(value.client_id), str(value.boundary)), []).append(value)
    aggregated: list[PlacementFeedback] = []
    for (round_id, client_id, boundary), values in sorted(groups.items()):
        if len({value.success for value in values}) > 1:
            raise ValueError("cannot combine successful and failed feedback for one action")
        profiles = {
            ExecutionProfileKey.from_value(value.execution_profile).stable_id
            for value in values
            if value.execution_profile is not None
        }
        if len(profiles) > 1:
            raise ValueError("cannot combine different execution profiles for one action")
        kwargs = {name: _median([getattr(value, name) for value in values]) for name in _MEASUREMENTS}
        completion_scopes = {value.completion_scope for value in values if value.completion_ms is not None}
        completion_scope = next(iter(completion_scopes)) if len(completion_scopes) == 1 else None
        if len(completion_scopes) > 1:
            kwargs["completion_ms"] = None
        loads = [value.server_load for value in values if value.server_load is not None]
        server_load = None
        if loads:
            server_load = {
                name: measured
                for name in set().union(*(load.keys() for load in loads))
                if (measured := _median([load.get(name) for load in loads])) is not None
            }
        aggregated.append(
            PlacementFeedback(
                round_id=round_id,
                client_id=client_id,
                boundary=boundary,
                num_examples=sum(max(int(value.num_examples), 0) for value in values),
                num_batches=sum(max(int(value.num_batches), 0) for value in values),
                success=all(value.success for value in values),
                execution_profile=next(iter(profiles)) if profiles else None,
                server_load=server_load,
                completion_scope=completion_scope,
                **kwargs,
            )
        )
    return aggregated


__all__ = ["aggregate_feedback"]
