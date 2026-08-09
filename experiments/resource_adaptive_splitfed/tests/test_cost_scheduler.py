from __future__ import annotations

from experiments.resource_adaptive_splitfed.split_cost_model import SplitCostModel, SplitCostPrediction
from experiments.resource_adaptive_splitfed.split_scheduler import ResourceAdaptiveSplitScheduler


def prediction(
    key: str,
    total: float,
    *,
    client: float,
    server: float,
    memory: float = 100.0,
    feasible: bool = True,
):
    return SplitCostPrediction(
        split_key=key,
        predicted_round_ms=total,
        predicted_client_compute_ms=client,
        predicted_network_ms=0.0,
        predicted_server_compute_ms=server,
        predicted_server_queue_ms=0.0,
        predicted_client_peak_memory_mb=memory,
        predicted_server_gpu_time_ms=server,
        predicted_switch_ms=0.0,
        feasible=feasible,
        infeasible_reason=None if feasible else "client_memory",
    )


def test_scheduler_rejects_memory_infeasible_candidate() -> None:
    scheduler = ResourceAdaptiveSplitScheduler(use_hysteresis=False)
    selected = scheduler.select_splits(
        [{"client_id": "weak"}],
        {"weak": {}},
        {"max_server_concurrency": 1},
        {"weak": [prediction("stem", 1, client=1, server=0, feasible=False), prediction("layer2", 4, client=3, server=1)]},
    )
    assert selected == {"weak": "layer2"}


def test_global_scheduler_avoids_early_cut_server_congestion() -> None:
    scheduler = ResourceAdaptiveSplitScheduler(use_hysteresis=False)
    clients = [{"client_id": str(index)} for index in range(4)]
    choices = {
        str(index): [
            prediction("stem", 5, client=1, server=4),
            prediction("full_local", 8, client=8, server=0),
        ]
        for index in range(4)
    }
    selected = scheduler.select_splits(
        clients,
        {str(index): {} for index in range(4)},
        {"max_server_concurrency": 1},
        choices,
    )
    assert 0 < sum(value == "stem" for value in selected.values()) < 4
    assert any(value == "full_local" for value in selected.values())


def test_hysteresis_blocks_small_or_too_frequent_switches() -> None:
    scheduler = ResourceAdaptiveSplitScheduler(
        min_relative_improvement_to_switch=0.10,
        min_rounds_between_switches=3,
        use_hysteresis=True,
    )
    scheduler.current_splits["c"] = "layer2"
    scheduler.last_switch_round["c"] = 2
    options = {"c": [prediction("stem", 8, client=4, server=4), prediction("layer2", 10, client=7, server=3)]}
    assert scheduler.select_splits(
        [{"client_id": "c"}], {"c": {}}, {"max_server_concurrency": 1}, options, round_id=3
    )["c"] == "layer2"
    assert scheduler.select_splits(
        [{"client_id": "c"}], {"c": {}}, {"max_server_concurrency": 1}, options, round_id=5
    )["c"] == "stem"


def test_cost_model_moves_late_when_measured_uplink_deteriorates() -> None:
    records = []
    for split_key, client_ms, server_ms, wire_bytes in (
        ("stem", 2.0, 4.0, 1_000_000),
        ("layer4", 8.0, 1.0, 10_000),
    ):
        records.append(
            {
                "success": True,
                "device_profile": "weak",
                "split_key": split_key,
                "batch_size": 2,
                "client_forward_ms": client_ms / 2,
                "client_backward_ms": client_ms / 2,
                "server_forward_ms": server_ms / 2,
                "server_backward_ms": server_ms / 2,
                "network_upload_ms": 1.0,
                "network_download_ms": 1.0,
                "server_queue_ms": 0.0,
                "client_peak_memory_mb": 100.0,
                "server_peak_memory_mb": 100.0,
                "boundary_forward_bytes": wire_bytes,
                "boundary_gradient_bytes": wire_bytes,
                "runtime_prepare_ms": 0.0,
            }
        )
    model = SplitCostModel().fit(records)
    server = {"max_server_concurrency": 2, "server_available_memory_mb": 10_000}
    poor = {"client_id": "c", "device_profile": "weak", "batch_size": 2, "uplink_mbps": 1, "downlink_mbps": 1, "rtt_ms": 10, "client_available_memory_mb": 1000}
    candidates = {
        key: {"split_key": key, "boundary_forward_bytes": size, "boundary_gradient_bytes": size}
        for key, size in (("stem", 1_000_000), ("layer4", 10_000))
    }
    predictions = [model.predict(poor, server, value) for value in candidates.values()]
    assert min(predictions, key=lambda item: item.predicted_round_ms).split_key == "layer4"
