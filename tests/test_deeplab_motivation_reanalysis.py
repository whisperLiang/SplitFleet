"""Reject misleading motivation inputs; no performance claims or plotting."""
import hashlib
import json

import pytest

from experiments.motivation.analyze_deeplab import DEVICES, METHODS, analyse, synchronization_cost_model


@pytest.fixture
def cohort(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    write = lambda path, value: path.write_text(json.dumps(value))
    plan = {"source_identity": "frozen-test-source", "stages": [{
        "model_id": "deeplabv3_resnet50", "batch_size": 2, "rounds": 10,
        "seeds": [7401, 7402], "worker_ids": [row[0] for row in DEVICES.values()]}]}
    write(root / "frozen_plan.json", plan)
    write(root / "runtime_manifest.json", {"source_identity": "frozen-test-source"})
    registered = []
    for seed in (7401, 7402):
        for method in METHODS:
            directory = root / "physical" / f"deeplabv3_resnet50_timing_s{seed}" / f"semantic_segmentation_{method}"
            directory.mkdir(parents=True)
            cut_map = {f"{percent}%": f"admitted-cut-{percent}" for percent in (25, 50, 75)}
            canonical = "full_local" if method == "fedavg" else cut_map[method.removeprefix("splitfed_fixed") + "%"]
            partitions = {str(i): 4 + 2 * i for i in range(6)}
            fits = []
            for _, (identity, _, device, index) in DEVICES.items():
                for round_id in range(1, 11):
                    # Different epoch lengths, identical per-batch costs within
                    # seed; unequal seed lengths must not change seed weights.
                    batches = partitions[str(index)] // 2
                    seconds_per_batch = 1 if seed == 7401 else 3
                    metrics = {"logical_client_id": identity, "device": device,
                               "client_index": index, "boundary": canonical,
                               "num_examples": partitions[str(index)], "num_batches": batches,
                               "partition_hash": "paired-partition",
                               "fit_started_unix_ns": 1_000_000_000,
                               "fit_finished_unix_ns": 1_000_000_000 + batches * seconds_per_batch * 1_000_000_000,
                               "fit_duration_sec": batches * seconds_per_batch - .1}
                    if method == "fedavg":
                        metrics["round_id"] = round_id
                    fits.append({"round_id": round_id, "num_examples": partitions[str(index)], "metrics": metrics})
            result = {"model_id": "deeplabv3_resnet50", "task": "semantic_segmentation", "seed": seed,
                      "batch_size": 2, "rounds": 10, "optimizer": "adam", "learning_rate": .0001,
                      "local_epochs": 1, "worker_ids": plan["stages"][0]["worker_ids"], "server_device": "cuda:0",
                      "fit_failures": [], "fit_records": fits, "fixed_cut_resolution": cut_map,
                      "resolved_fixed_boundary": None if method == "fedavg" else canonical,
                      "partition_sizes": partitions, "initial_model_hash": f"initial-{seed}",
                      "data_content_hash": f"data-{seed}", "partition_hash": "paired-partition",
                      "pretrain_checkpoint_sha256": "checkpoint", "model_metadata": {}, "math_policy": {}}
            path = directory / "result.json"
            write(path, result)
            write(directory / "validation_report.json", {"valid": True})
            registered.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    registry = tmp_path / "registry.json"
    write(registry, {"sources": registered})
    return root, registry, tmp_path / "analysis"


def alter_result(cohort, mutate, *, rebind=False):
    root, registry, _ = cohort
    record = json.loads(registry.read_text())
    path = root / "physical/deeplabv3_resnet50_timing_s7401/semantic_segmentation_splitfed_fixed25/result.json"
    result = json.loads(path.read_text())
    mutate(result)
    path.write_text(json.dumps(result))
    if rebind:
        next(row for row in record["sources"] if row["path"] == str(path))["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        registry.write_text(json.dumps(record))


def test_uses_common_fit_window_and_normalizes_epoch_lengths(cohort):
    result = analyse(*cohort)
    assert result["client_round_records"] == 480
    assert all(row["mean_sec_per_batch"] == 2 for row in result["conditions"])
    assert all(row["seed_means_sec_per_batch"] == [1, 3] for row in result["conditions"])
    assert result["new_physical_training"] is False


def test_refuses_a_changed_historical_receipt(cohort):
    alter_result(cohort, lambda result: result["fit_records"].pop())
    with pytest.raises(ValueError, match="frozen input changed"):
        analyse(*cohort)
    assert not cohort[2].exists()


def test_refuses_missing_client_rounds_even_with_a_rebound_hash(cohort):
    alter_result(cohort, lambda result: result["fit_records"].pop(), rebind=True)
    with pytest.raises(ValueError, match="Missing client-round"):
        analyse(*cohort)


def test_refuses_a_different_partition_in_a_paired_condition(cohort):
    alter_result(cohort, lambda result: result.update(partition_hash="different-data-assignment"), rebind=True)
    with pytest.raises(ValueError, match="differ within a seed"):
        analyse(*cohort)


def test_refuses_a_cut_label_that_hides_another_executed_boundary(cohort):
    def change_cut(result):
        result["fit_records"][0]["metrics"]["boundary"] = "admitted-cut-75"
    alter_result(cohort, change_cut, rebind=True)
    with pytest.raises(ValueError, match="receipts disagree"):
        analyse(*cohort)


def alternating_cost_rows():
    """A changing slowest client exposes an incorrect max-after-mean shortcut."""
    rows = []
    for method, scale in zip(METHODS, (1, .5, .75, 1.25)):
        for seed in (7401, 7402):
            for round_id in range(1, 11):
                high = "cpu1" if round_id % 2 else "cpu2"
                for label in DEVICES:
                    rows.append({"method": method, "seed": seed, "round": round_id,
                                 "label": label, "sec_per_batch": (3 if label == high else 1) * scale,
                                 "requested_cut": "full_local" if method == "fedavg" else method.removeprefix("splitfed_fixed") + "%"})
    return rows


def test_takes_round_max_before_averaging_and_keeps_wait_distinct():
    model = synchronization_cost_model(alternating_cost_rows())
    local = next(row for row in model["conditions"] if row["method"] == "fedavg")
    assert local["mean_critical_cost_sec_per_batch"] == 3
    # Both alternating slow clients average 2; averaging first would miss 1s.
    cpu1 = next(row for row in model["per_client_condition"] if row["method"] == "fedavg" and row["label"] == "cpu1")
    assert cpu1["mean_cost_sec_per_batch"] == 2
    assert cpu1["mean_implied_wait_sec_per_batch"] == 1
    assert cpu1["highest_cost_fraction"] == .5
    assert model["actual_synchronization_wait_measured"] is False
    assert model["scope"] == "derived_equal_batch_comparison_not_observed_wait"


def test_relief_percentages_use_the_matching_seed_local_reference():
    model = synchronization_cost_model(alternating_cost_rows())
    changes = {row["method"]: row["mean_paired_critical_change_pct_vs_local"] for row in model["conditions"]}
    assert changes == dict(zip(METHODS, (0, -50, -25, 25)))
    for row in model["raw_client_slack"]:
        assert row["cost_sec_per_batch"] + row["implied_wait_sec_per_batch"] == row["critical_cost_sec_per_batch"]


def test_wait_model_refuses_an_incomplete_six_client_comparison():
    rows = alternating_cost_rows()
    rows.pop()
    with pytest.raises(ValueError, match="complete six-client round"):
        synchronization_cost_model(rows)
