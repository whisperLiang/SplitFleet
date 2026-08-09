"""Unit and attribution contracts for the RA-SplitFed cost and resource models."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from experiments.resource_adaptive_splitfed import experiment_runner
from experiments.resource_adaptive_splitfed.experiment_runner import (
    _batches_per_round,
    _oracle_probe,
    _predicted_memory_failure,
    _profile_assignment,
)
from experiments.resource_adaptive_splitfed.resource_emulator import ResourceEmulator
from experiments.resource_adaptive_splitfed.resource_monitor import PeakMemoryTracker
from experiments.resource_adaptive_splitfed.split_cost_model import (
    SplitCostModel,
    SplitCostPrediction,
)


def _record(**overrides):
    record = {
        "success": True,
        "device_profile": "weak",
        "split_key": "stem",
        "batch_size": 8,
        "client_forward_ms": 10.0,
        "client_backward_ms": 10.0,
        "server_forward_ms": 5.0,
        "server_backward_ms": 5.0,
        "network_upload_ms": 1.0,
        "network_download_ms": 1.0,
        "server_queue_ms": 0.0,
        "client_peak_memory_mb": 400.0,
        "server_peak_memory_mb": 200.0,
        "boundary_forward_bytes": 1_000,
        "boundary_gradient_bytes": 1_000,
        "runtime_prepare_ms": 250.0,
    }
    record.update(overrides)
    return record


def _client(**overrides):
    client = {
        "client_id": "c",
        "device_profile": "weak",
        "batch_size": 8,
        "client_available_memory_mb": 100_000.0,
    }
    client.update(overrides)
    return client


SERVER = {"max_server_concurrency": 1, "server_available_memory_mb": 100_000.0}


def test_prediction_scales_with_the_batches_a_round_executes() -> None:
    model = SplitCostModel().fit([_record()])

    single = model.predict(_client(batches_per_round=1), SERVER, {"split_key": "stem"})
    many = model.predict(_client(batches_per_round=10), SERVER, {"split_key": "stem"})

    assert many.predicted_round_ms == pytest.approx(single.predicted_round_ms * 10)
    assert many.predicted_client_compute_ms == pytest.approx(single.predicted_client_compute_ms * 10)
    # Peak memory is a peak, not a sum, so it must not scale with round length.
    assert many.predicted_client_peak_memory_mb == single.predicted_client_peak_memory_mb


def test_online_round_observation_is_comparable_with_the_profile_prediction() -> None:
    model = SplitCostModel(ema_alpha=1.0).fit([_record()])
    client = _client(batches_per_round=10)
    offline = model.predict(client, SERVER, {"split_key": "stem"})

    model.update_online(
        {"client_id": "c", "split_key": "stem", "success": True, "completion_ms": offline.predicted_round_ms}
    )
    online = model.predict(client, SERVER, {"split_key": "stem"})

    assert online.predicted_round_ms == pytest.approx(offline.predicted_round_ms)


def test_online_calibration_keeps_current_network_state_in_the_prediction() -> None:
    model = SplitCostModel(ema_alpha=1.0).fit(
        [_record(boundary_forward_bytes=1_000_000, boundary_gradient_bytes=1_000_000)]
    )
    fast_client = _client(uplink_mbps=1_000, downlink_mbps=1_000, rtt_ms=1)
    fast = model.predict(fast_client, SERVER, {"split_key": "stem"})
    model.update_online(
        {
            "client_id": "c",
            "split_key": "stem",
            "success": True,
            "completion_ms": fast.predicted_round_ms * 1.5,
            "predicted_completion_ms": fast.predicted_round_ms,
        }
    )

    slow_client = _client(uplink_mbps=1, downlink_mbps=1, rtt_ms=100)
    slow = model.predict(slow_client, SERVER, {"split_key": "stem"})
    uncalibrated_slow = SplitCostModel().fit(
        [_record(boundary_forward_bytes=1_000_000, boundary_gradient_bytes=1_000_000)]
    ).predict(slow_client, SERVER, {"split_key": "stem"})

    assert slow.predicted_round_ms == pytest.approx(
        uncalibrated_slow.predicted_round_ms * 1.5
    )
    assert slow.predicted_network_ms == pytest.approx(
        uncalibrated_slow.predicted_network_ms * 1.5
    )
    assert slow.predicted_round_ms > fast.predicted_round_ms * 100


def test_explicit_zero_switch_cost_is_not_replaced_by_the_profiled_cost() -> None:
    model = SplitCostModel().fit([_record()])
    client = _client(current_split_key="layer4")

    profiled = model.predict(client, SERVER, {"split_key": "stem"})
    ablated = model.predict(client, SERVER, {"split_key": "stem", "predicted_switch_ms": 0.0})

    assert profiled.predicted_switch_ms == pytest.approx(250.0)
    assert ablated.predicted_switch_ms == 0.0
    assert ablated.predicted_round_ms < profiled.predicted_round_ms


def test_peak_memory_is_extrapolated_from_the_profiled_batch_sizes() -> None:
    model = SplitCostModel().fit(
        [
            _record(batch_size=2, client_peak_memory_mb=150.0),
            _record(batch_size=8, client_peak_memory_mb=450.0),
        ]
    )

    exact = model.predict(_client(batch_size=8), SERVER, {"split_key": "stem"})
    extrapolated = model.predict(_client(batch_size=32), SERVER, {"split_key": "stem"})

    assert exact.predicted_client_peak_memory_mb == pytest.approx(450.0)
    # The two profiled points fit 50 MB fixed + 50 MB per sample, so batch 32
    # needs 1650 MB rather than batch 8's measured 450 MB.
    assert extrapolated.predicted_client_peak_memory_mb == pytest.approx(1650.0)


def test_single_profiled_batch_never_under_reports_memory() -> None:
    model = SplitCostModel().fit([_record(batch_size=8, client_peak_memory_mb=400.0)])

    prediction = model.predict(_client(batch_size=32), SERVER, {"split_key": "stem"})

    assert prediction.predicted_client_peak_memory_mb >= 400.0


def test_memory_infeasibility_is_detected_at_the_requested_batch_size() -> None:
    model = SplitCostModel().fit(
        [
            _record(batch_size=2, client_peak_memory_mb=150.0),
            _record(batch_size=8, client_peak_memory_mb=450.0),
        ]
    )

    small = model.predict(
        _client(batch_size=8, client_available_memory_mb=600.0), SERVER, {"split_key": "stem"}
    )
    large = model.predict(
        _client(batch_size=32, client_available_memory_mb=600.0), SERVER, {"split_key": "stem"}
    )

    assert small.feasible
    assert not large.feasible
    assert large.infeasible_reason == "client_memory"


def test_profile_assignment_rejects_over_and_under_specified_counts() -> None:
    assert _profile_assignment({"client_profiles": {"weak": 2, "strong": 1}}, 3) == {
        "0": "weak",
        "1": "weak",
        "2": "strong",
    }
    with pytest.raises(ValueError, match="must sum to num_clients"):
        _profile_assignment({"client_profiles": {"weak": 8, "medium": 6, "strong": 6}}, 10)
    with pytest.raises(ValueError, match="must sum to num_clients"):
        _profile_assignment({"client_profiles": {"weak": 2}}, 10)


def test_batches_per_round_counts_epochs_and_the_batch_cap() -> None:
    assert _batches_per_round(100, batch_size=32, local_epochs=1, max_batches_per_epoch=None) == 4
    assert _batches_per_round(100, batch_size=32, local_epochs=2, max_batches_per_epoch=None) == 8
    assert _batches_per_round(100, batch_size=32, local_epochs=1, max_batches_per_epoch=2) == 2
    assert _batches_per_round(0, batch_size=32, local_epochs=1, max_batches_per_epoch=None) == 1


def test_network_transfer_honors_a_cooperative_deadline() -> None:
    link = ResourceEmulator({"uplink_mbps": 0.001}).link
    started = time.perf_counter()

    with pytest.raises(TimeoutError, match="deadline"):
        link.transfer(
            bytes(1024),
            "uplink",
            deadline_ns=time.perf_counter_ns() + 5_000_000,
        )

    assert time.perf_counter() - started < 0.25


def test_configured_memory_limit_rejects_a_predicted_oom() -> None:
    prediction = SplitCostPrediction(
        split_key="stem",
        predicted_round_ms=1.0,
        predicted_client_compute_ms=1.0,
        predicted_network_ms=0.0,
        predicted_server_compute_ms=0.0,
        predicted_server_queue_ms=0.0,
        predicted_client_peak_memory_mb=512.0,
        predicted_server_gpu_time_ms=0.0,
        predicted_switch_ms=0.0,
        feasible=True,
        infeasible_reason=None,
    )

    assert "out of memory" in _predicted_memory_failure(
        prediction, {"memory_limit_mb": 256}
    )
    assert _predicted_memory_failure(prediction, {"memory_limit_mb": 1024}) is None


def test_oracle_probe_uses_the_current_network_controls(monkeypatch) -> None:
    class FakeRuntime:
        def __init__(self, client_id, *args, device="cpu", **kwargs):
            self.client_id = client_id
            self.device = torch.device(device)

        def activate(self, state, split_key):
            return None, None

        def train_batch(self, handle, inputs, targets, *, network, **kwargs):
            network.transfer(bytes(1_000), "uplink")
            return SimpleNamespace(client_peak_memory_mb=1.0)

    monkeypatch.setattr(experiment_runner, "LogicalClientRuntime", FakeRuntime)
    common = {
        "client_id": "c",
        "model_factory": lambda: None,
        "candidates": {"stem": object()},
        "global_state": {},
        "sample_inputs": torch.zeros(1, 1),
        "batch": (torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)),
        "device": "cpu",
        "learning_rate": 0.1,
        "server_controls": {},
        "server_concurrency": 1,
    }

    _, fast = _oracle_probe(**common, client_controls={"uplink_mbps": 1_000})
    _, slow = _oracle_probe(**common, client_controls={"uplink_mbps": 0.1})

    assert slow["stem"] > fast["stem"] + 50.0


def test_nested_same_thread_memory_windows_stay_attributable() -> None:
    outer = PeakMemoryTracker("cpu")
    inner = PeakMemoryTracker("cpu")

    def outer_body():
        inner.measure(lambda: bytearray(1024))
        return None

    outer.measure(outer_body)

    assert inner.peak_mb is not None
    assert outer.peak_mb is not None
    assert inner.nested and not outer.nested


def test_overlapping_threads_report_unattributable_peak_memory() -> None:
    first_started = threading.Event()
    second_started = threading.Event()
    trackers = {}

    def run(name: str, mine: threading.Event, theirs: threading.Event) -> None:
        tracker = PeakMemoryTracker("cpu")
        trackers[name] = tracker

        def body():
            mine.set()
            theirs.wait(timeout=5.0)
            return None

        tracker.measure(body)

    threads = [
        threading.Thread(target=run, args=("a", first_started, second_started)),
        threading.Thread(target=run, args=("b", second_started, first_started)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert trackers["a"].peak_mb is None
    assert trackers["b"].peak_mb is None
    assert not trackers["a"].attributable and not trackers["b"].attributable


def test_overlapping_host_load_is_reported_against_the_client_it_contends_with() -> None:
    loaded_started = threading.Event()
    quiet_started = threading.Event()
    emulators = {}

    def run(name: str, config: dict, mine: threading.Event, theirs: threading.Event) -> None:
        with ResourceEmulator(config) as emulator:
            emulators[name] = emulator
            mine.set()
            theirs.wait(timeout=10.0)

    threads = [
        threading.Thread(
            target=run,
            args=("loaded", {"cpu_background_load": 0.05}, loaded_started, quiet_started),
        ),
        threading.Thread(target=run, args=("quiet", {}, quiet_started, loaded_started)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20.0)

    # The quiet client is the one whose timing absorbed a foreign handicap.
    assert emulators["quiet"].overlapping_host_load_clients == 1
    assert emulators["loaded"].overlapping_host_load_clients == 0


def test_sequential_measurements_remain_attributable() -> None:
    first = PeakMemoryTracker("cpu")
    second = PeakMemoryTracker("cpu")

    first.measure(lambda: bytearray(1024))
    second.measure(lambda: bytearray(1024))

    assert first.peak_mb is not None
    assert second.peak_mb is not None
