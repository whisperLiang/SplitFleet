import pytest
import numpy as np
from dataclasses import replace

from experiments.analysis.oracle_gap import adaptation_rounds, oracle_gap
from experiments.heterogeneity.scenarios import bandwidth_degradation, server_load_change
from splitfleet.server.placement.cosplit_ucb.context import ContextEncoder
from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig, CoSplitUCBPlacementPolicy, PlacementFeedback, StaticCandidateProvider,
)
from tests.unit.test_cosplit_policy import _candidate


def test_declared_bandwidth_change_changes_simulated_observation_and_remains_labeled():
    scenario = bandwidth_degradation(change_round=10)
    old, new = (scenario.at(index).network.transfer_ms(100000, direction="upload") for index in (9, 10))
    assert new > old
    assert scenario.to_dict()["provenance"] == "simulation"
    assert scenario.to_dict()["physical_shaping_applied"] is False
    before = ContextEncoder().network_context(_candidate("early", .25), scenario.at(9).network.telemetry(), direction="upload")
    after = ContextEncoder().network_context(_candidate("early", .25), scenario.at(10).network.telemetry(), direction="upload")
    assert before[7] == after[7] == 0
    assert before[9] == after[9] == 0
    assert before[6] > after[6]
    np.testing.assert_array_equal(before[[0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12]],
                                  after[[0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12]])
    with pytest.raises(ValueError, match="known"):
        scenario.at(10).network.transfer_ms(None, direction="download")


def test_declared_server_change_keeps_other_conditions_identical():
    scenario = server_load_change(change_round=5)
    assert scenario.at(4).network == scenario.at(5).network
    assert scenario.at(5).resources.server_service_multiplier > scenario.at(4).resources.server_service_multiplier


def test_adaptation_metric_requires_sustained_recovery_and_preserves_censoring():
    rows = [dict(round_id=r, observed_ms=cost, oracle_ms=100) for r, cost in [(5, 130), (6, 105), (7, 105), (9, 102)]]
    assert adaptation_rounds(rows, change_round=5, tolerance=.1, sustained_rounds=2) == 1
    assert adaptation_rounds(rows, change_round=5, tolerance=.1, sustained_rounds=3) is None
    assert oracle_gap(None, 100) is None
    assert oracle_gap(120, 100) == .2


def test_online_feedback_moves_preference_after_declared_link_change():
    # This is a synthetic mechanism check, not a physical training result.
    # MiB-scale payloads exercise the canonical linear byte features. Scale
    # compute times equally to preserve the scenario's two optimal cuts.
    early = replace(_candidate("early", .25), boundary_forward_bytes=1000000, boundary_gradient_bytes=1000000,
                    metadata={"payload_batch_size": 1, "boundary_forward_bytes_by_batch_size": {1: 1000000}})
    late = replace(_candidate("late", .75), boundary_forward_bytes=100000, boundary_gradient_bytes=100000,
                   metadata={"payload_batch_size": 1, "boundary_forward_bytes_by_batch_size": {1: 100000}})
    scenario = bandwidth_degradation(change_round=41)
    telemetry = {"client": {"num_batches": 1, "batch_size": 1}}
    # Disable exploratory replacements to test the learned cost preference.
    # The physical Nano experiment retains production safe exploration.
    policy = CoSplitUCBPlacementPolicy(candidate_provider=StaticCandidateProvider([early, late]),
        telemetry_provider=telemetry, config=CoSplitUCBConfig(target_scale_ms=1, discount_gamma=.9,
            min_residence_rounds=0, max_explorations_per_round=0))
    choices = []
    candidates = {value.boundary: value for value in (early, late)}
    for round_id in range(1, 91):
        network = scenario.at(round_id).network
        telemetry["client"].update(network.telemetry())
        boundary = policy.plan_round(round_id=round_id, client_ids=["client"], training=True)["client"]
        choices.append(boundary)
        candidate = candidates[boundary]
        policy.observe_round(round_id=round_id, feedback=[PlacementFeedback(
            round_id=round_id, client_id="client", boundary=boundary,
            client_forward_ms=100 if boundary == "early" else 800, client_backward_ms=0,
            network_upload_ms=network.transfer_ms(candidate.boundary_forward_bytes, direction="upload"),
            network_download_ms=network.transfer_ms(candidate.boundary_gradient_bytes, direction="download"),
            network_roundtrip_ms=(network.transfer_ms(candidate.boundary_forward_bytes, direction="upload")
                                  + network.transfer_ms(candidate.boundary_gradient_bytes, direction="download")),
            server_service_ms=200 if boundary == "early" else 20, switch_ms=0, num_batches=1)])
    assert choices[20:40].count("early") > choices[20:40].count("late")
    assert choices[-20:].count("late") > choices[-20:].count("early")
    assert policy.learners.network.update_count("client") == 90
