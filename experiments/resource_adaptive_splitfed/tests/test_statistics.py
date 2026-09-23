from __future__ import annotations

import csv

import pytest

from experiments.resource_adaptive_splitfed.aggregate_results import _paired_comparisons, _protocol_hash
from experiments.resource_adaptive_splitfed.statistics import holm_adjust, paired_effect


def test_paired_effect_is_deterministic_and_preserves_pairing() -> None:
    effect = paired_effect([8.0, 9.0, 10.0, 11.0, 12.0], [10.0] * 5, seed=7)
    repeated = paired_effect([8.0, 9.0, 10.0, 11.0, 12.0], [10.0] * 5, seed=7)

    assert effect == repeated
    assert effect.paired_n == 5
    assert effect.mean_difference == 0.0
    assert effect.median_difference == 0.0
    assert effect.relative_difference == 0.0
    assert effect.confidence_interval_95[0] < 0 < effect.confidence_interval_95[1]
    assert effect.sign_flip_p_value == 1.0


def test_exact_sign_flip_detects_consistent_direction_but_not_with_five_pairs_at_point05() -> None:
    effect = paired_effect([1, 1, 1, 1, 1], [2, 2, 2, 2, 2])

    assert effect.mean_difference == -1.0
    assert effect.confidence_interval_95 == (-1.0, -1.0)
    assert effect.sign_flip_p_value == pytest.approx(2 / 32)
    assert effect.cohens_dz is None


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_paired_comparisons_keep_swept_protocols_separate_and_gate_noninferiority(tmp_path) -> None:
    rows = [
        {
            "experiment": "server_scaling", "protocol_hash": protocol,
            "method": method, "seed": seed, "round_id": 1,
            "round_time_ms": time_ms, "test_accuracy": 0.5,
        }
        for protocol, time_ms in (("four_clients", 10.0), ("eight_clients", 100.0))
        for method in ("resource_adaptive_splitfed", "best_global_fixed")
        for seed in (1, 2)
    ]
    destination = tmp_path / "paired.csv"
    _paired_comparisons(rows, destination)
    with destination.open(newline="", encoding="utf-8") as stream:
        comparisons = list(csv.DictReader(stream))
    accuracy = [row for row in comparisons if row["metric"] == "final_test_accuracy"]
    timing = [row for row in comparisons if row["metric"] == "mean_round_time_ms"]
    assert len(accuracy) == len(timing) == 2
    assert {row["protocol_hash"] for row in accuracy} == {"four_clients", "eight_clients"}
    assert all(row["paired_n"] == "2" and row["method_a_noninferior"] == ""
               for row in accuracy)
    assert all(float(row["mean_difference"]) == 0.0 for row in timing)


def test_protocol_hash_ignores_seed_batch_but_separates_resource_sweep() -> None:
    metadata = {"hostname": "test-host", "torch_version": "2.11", "cuda_version": None, "device_names": []}
    base = {"experiment": "server_scaling", "num_clients": 4, "server_concurrency": 2,
            "methods": ["fixed_early"], "seeds": [1, 2], "results_root": "first"}
    another_batch = {**base, "methods": ["resource_adaptive_splitfed"],
                     "seeds": [3, 4], "results_root": "second"}
    assert _protocol_hash(base, metadata) == _protocol_hash(another_batch, metadata)
    assert _protocol_hash(base, metadata) != _protocol_hash({**base, "num_clients": 8}, metadata)
    assert _protocol_hash(base, metadata) != _protocol_hash(base, {**metadata, "hostname": "other-host"})
