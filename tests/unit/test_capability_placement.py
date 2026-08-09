import pytest

from splitfleet.server.placement import (
    CapabilityAwarePlacementPolicy,
    CapabilityPlacementConfig,
)


LADDER = ["after:fc1", "after:fc2", "after:fc3"]


def _policy(**overrides) -> CapabilityAwarePlacementPolicy:
    config = CapabilityPlacementConfig(
        warmup_rounds=0,
        min_rounds_between_switches=0,
        **overrides,
    )
    return CapabilityAwarePlacementPolicy(boundary_ladder=LADDER, config=config)


def test_ladder_requires_at_least_two_boundaries() -> None:
    with pytest.raises(ValueError, match="at least two boundaries"):
        CapabilityAwarePlacementPolicy(boundary_ladder=["after:fc1"])


def test_unknown_clients_start_at_the_configured_index() -> None:
    policy = CapabilityAwarePlacementPolicy(boundary_ladder=LADDER, start_index=0)
    assert policy(1, "fresh", True) == "after:fc1"
    assert CapabilityAwarePlacementPolicy(boundary_ladder=LADDER)(1, "fresh", True) == "after:fc2"


def test_slow_device_moves_to_a_lighter_prefix_and_fast_device_to_a_heavier_one() -> None:
    policy = _policy()
    for cid, duration in (("slow", 10.0), ("median", 5.0), ("fast", 1.0)):
        policy(1, cid, True)
        policy.observe_fit_metrics(
            round_id=1, cid=cid, num_examples=32, metrics={"fit_duration_sec": duration}
        )

    assert policy(2, "slow", True) == "after:fc1"
    assert policy(2, "median", True) == "after:fc2"
    assert policy(2, "fast", True) == "after:fc3"
    assert {decision.cid for decision in policy.decisions} == {"slow", "fast"}


def test_decisions_are_taken_once_per_round_so_fit_and_evaluate_agree() -> None:
    policy = _policy()
    policy(1, "slow", True)
    policy.observe_fit_metrics(
        round_id=1, cid="slow", num_examples=8, metrics={"fit_duration_sec": 10.0}
    )
    policy(1, "fast", True)
    policy.observe_fit_metrics(
        round_id=1, cid="fast", num_examples=8, metrics={"fit_duration_sec": 1.0}
    )

    fit_boundary = policy(2, "slow", True)
    assert policy(2, "slow", False) == fit_boundary
    # Late metrics for round 2 must not retroactively change round 2 placement.
    policy.observe_fit_metrics(
        round_id=2, cid="slow", num_examples=8, metrics={"fit_duration_sec": 0.1}
    )
    assert policy(2, "slow", True) == fit_boundary


def test_hysteresis_blocks_a_second_switch_until_the_window_elapses() -> None:
    policy = CapabilityAwarePlacementPolicy(
        boundary_ladder=LADDER,
        start_index=2,
        config=CapabilityPlacementConfig(warmup_rounds=0, min_rounds_between_switches=3),
    )
    for cid, duration in (("slow", 10.0), ("fast", 1.0)):
        policy(1, cid, True)
        policy.observe_fit_metrics(
            round_id=1, cid=cid, num_examples=8, metrics={"fit_duration_sec": duration}
        )

    assert policy(2, "slow", True) == "after:fc2"
    for round_id in (3, 4):
        policy.observe_fit_metrics(
            round_id=round_id - 1,
            cid="slow",
            num_examples=8,
            metrics={"fit_duration_sec": 10.0},
        )
        assert policy(round_id, "slow", True) == "after:fc2"
    policy.observe_fit_metrics(
        round_id=4, cid="slow", num_examples=8, metrics={"fit_duration_sec": 10.0}
    )
    assert policy(5, "slow", True) == "after:fc1"


def test_failed_device_backs_off_immediately_and_clamps_at_the_ladder_end() -> None:
    policy = _policy()
    assert policy(1, "flaky", True) == "after:fc2"

    policy.observe_failure(round_id=1, cid="flaky", reason="timeout")
    assert policy.boundary_for("flaky") == "after:fc1"

    policy.observe_failure(round_id=2, cid="flaky", reason="timeout")
    assert policy.boundary_for("flaky") == "after:fc1"
    assert policy.clients["flaky"].failures == 2


def test_target_round_seconds_overrides_the_observed_median() -> None:
    policy = _policy(target_round_seconds=20.0)
    for cid, duration in (("a", 10.0), ("b", 12.0)):
        policy(1, cid, True)
        policy.observe_fit_metrics(
            round_id=1, cid=cid, num_examples=8, metrics={"fit_duration_sec": duration}
        )

    # Both devices beat the absolute target, so both take on more local work.
    assert policy(2, "a", True) == "after:fc3"
    assert policy(2, "b", True) == "after:fc3"


def test_placement_state_reports_every_known_client() -> None:
    policy = _policy()
    policy(1, "a", True)
    policy(1, "b", True)
    assert policy.placement_state() == {"a": "after:fc2", "b": "after:fc2"}
