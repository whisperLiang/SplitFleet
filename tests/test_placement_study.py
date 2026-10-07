"""Synthetic unit fixtures verify experiment accounting, never performance."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from experiments.placement_study import complete_simulation, simulate, validate_profiles
from tests.unit.test_cosplit_policy import _candidate


def test_offline_profile_executes_frozen_source_without_optimizer_updates(tmp_path):
    import json

    from experiments.placement_study import profile as measure_profile

    output = tmp_path / "profile"
    result = measure_profile(output, device="cpu", repeats=1)
    receipts = json.loads((output / "calibration_receipts.json").read_text())
    assert result["status"] == "completed"
    assert result["optimizer_steps"] == 0
    assert not result["distributed_round_latency_measured"]
    assert len(result["costs"]) == len(result["candidates"]) > 0
    assert len(receipts) == 1
    for receipt in receipts:
        assert receipt["optimizer_steps"] == receipt["persistent_optimizer_steps"] == 0
        assert receipt["model_hash_before"] == receipt["model_hash_after"]
        assert receipt["state_and_torch_rng_preserved"]
        assert all(row["optimizer_steps"] == 0 and row["optimizer_stage"] is None
                   for row in receipt["records"])
    assert all(row["activation_wire_bytes"] > 0 and row["gradient_wire_bytes"] > 0
               for row in result["costs"])
    validate_profiles([result])


def profile():
    candidates = [_candidate("early", .25), _candidate("middle", .5), _candidate("late", .75)]
    return {"status": "completed", "provenance": "measured_local", "device": "cpu",
        "scope": "synthetic UNIT TEST cost table; never exported as measured experiment evidence",
        "identity": {"model_id": "unit_fixture", "batch_size": 2},
        "candidates": [asdict(value) for value in candidates],
        "layer_domain": {"boundaries": ["early", "late"]},
        "costs": [{"boundary": value.boundary, "client_forward_ms": edge,
            "client_backward_ms": edge, "server_service_ms": service, "materialization_ms": 1,
            "activation_wire_bytes": byte_count, "gradient_wire_bytes": byte_count}
            for value, edge, service, byte_count in zip(candidates, (2, 5, 10), (20, 10, 2), (100000, 50000, 1000))]}


def config():
    return {"clients": [{"id": "a", "num_batches": 2}, {"id": "b", "num_batches": 1, "bandwidth_multiplier": .2}],
        "seeds": [1, 2], "rounds": 8, "adaptation_tolerance": .1, "sustained_rounds": 2,
        "scenario": {"name": "test_change", "phases": [
            {"starts_round": 1, "network": {"upload_mbps": 100, "download_mbps": 100}},
            {"starts_round": 5, "network": {"upload_mbps": 5, "download_mbps": 5},
             "resources": {"server_service_multiplier": 2}}]},
        "cosplit": {"target_scale_ms": 1, "min_residence_rounds": 0, "max_explorations_per_round": 0}}


def test_round_trace_keeps_cost_queue_conditions_oracle_scope_and_four_ablations():
    result = simulate(config(), [profile(), profile()])
    assert not result["physical_measurement"] and not result["oracle_is_deployable"]
    assert len(result["summaries"]) == 20
    assert len(result["placement_trace"]) == 20 * 8 * 2
    for row in result["placement_trace"]:
        assert row["oracle_gap"] >= -1e-12
        assert row["queue_ms"] >= 0
        assert row["provenance"] == "simulation_from_measured_local_profile"
        assert row["observed_cost_ms"] > 0
    oracle = [row for row in result["summaries"] if row["method"] == "Oracle"]
    assert all(abs(row["mean_oracle_gap"]) < 1e-12 for row in oracle)
    layer = [row for row in result["placement_trace"] if row["method"] == "Layer-only"]
    assert {row["boundary"] for row in layer} <= {"early", "late"}
    before = next(row for row in result["placement_trace"] if row["round_id"] == 4 and row["client_id"] == "a")
    after = next(row for row in result["placement_trace"] if row["round_id"] == 5 and row["client_id"] == "a")
    assert before["network"]["upload_mbps"] > after["network"]["upload_mbps"]
    assert after["resources"]["server_service_multiplier"] == 2


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["costs"].pop(), "complete"),
    (lambda p: p["costs"][0].update(client_forward_ms=float("nan")), "Invalid"),
    (lambda p: p["costs"][0].update(gradient_wire_bytes=1.5), "integer"),
    (lambda p: p["candidates"][0].update(feature_abi_id="changed"), "identities"),
    (lambda p: p["layer_domain"]["boundaries"].append("invented"), "invented"),
    (lambda p: p.update(provenance="synthetic"), "measured-local"),
])
def test_profiles_refuse_missing_costs_and_identity_drift(mutation, match):
    left, right = profile(), profile()
    mutation(right)
    with pytest.raises(ValueError, match=match): validate_profiles([left, right])


def test_exact_oracle_search_cap_does_not_silently_shrink_candidate_domain():
    settings = config()
    settings["oracle_max_assignments"] = 2
    with pytest.raises(ValueError, match="Explicitly restrict"):
        simulate(settings, [profile(), profile()])
    settings["candidate_boundaries"] = ["early"]
    assert simulate(settings, [profile(), profile()])["candidate_domain"] == ["early"]


def test_all_clients_use_one_common_server_profile():
    settings = config()
    settings["methods"] = ["Fixed-25"]
    fast, slow = profile(), profile()
    for row in slow["costs"]: row["server_service_ms"] *= 100
    result = simulate(settings, [fast, slow])
    settings["server_profile_index"] = 1
    changed = simulate(settings, [fast, slow])
    assert result["summaries"][0]["mean_round_makespan_ms"] < changed["summaries"][0]["mean_round_makespan_ms"]
    assert result["server_cost_scope"].startswith("one common")


def test_copied_profile_costs_cannot_change_after_config_freeze(tmp_path):
    import hashlib
    import json

    profiles = [profile(), profile()]
    settings = {**config(), "profile_contents_sha256": hashlib.sha256(
        json.dumps(profiles, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    profiles[0]["costs"][0]["client_forward_ms"] *= 10
    with pytest.raises(ValueError, match="profile contents changed"):
        complete_simulation(tmp_path, settings, profiles)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "failed"
    assert not (tmp_path / "placement_trace.jsonl").exists()
