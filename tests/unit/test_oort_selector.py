from __future__ import annotations

import pytest

from splitfleet.server.client_selection import OortSelector, OortSelectorConfig


def test_register_client_tracks_state_and_unexplored() -> None:
    selector = OortSelector()

    selector.register_client("c1", num_examples=9, duration=2.0)

    assert "c1" in selector.clients
    assert "c1" in selector.unexplored
    assert selector.clients["c1"].duration == pytest.approx(2.0)


def test_select_returns_requested_number_of_clients() -> None:
    selector = OortSelector(OortSelectorConfig(seed=7))

    result = selector.select(
        round_id=1,
        candidate_cids=[f"c{i}" for i in range(5)],
        num_clients=3,
    )

    assert len(result.selected_cids) == 3
    assert len(set(result.selected_cids)) == 3


def test_high_reward_low_duration_client_gets_higher_selection_score() -> None:
    selector = OortSelector(
        OortSelectorConfig(
            exploration_factor=0.0,
            exploration_min=0.0,
            exploration_decay=1.0,
            round_threshold=100.0,
            cut_off_util=0.99,
            seed=11,
        )
    )
    selector.update_after_fit(
        round_id=1,
        cid="fast_high",
        num_examples=100,
        metrics={"loss": 5.0, "fit_duration_sec": 1.0},
    )
    selector.update_after_fit(
        round_id=1,
        cid="slow_low",
        num_examples=100,
        metrics={"loss": 0.2, "fit_duration_sec": 8.0},
    )
    selector.update_after_fit(
        round_id=1,
        cid="middle",
        num_examples=100,
        metrics={"loss": 0.5, "fit_duration_sec": 2.0},
    )

    result = selector.select(
        round_id=5,
        candidate_cids=["fast_high", "slow_low", "middle"],
        num_clients=1,
    )

    assert result.scores["fast_high"] > result.scores["slow_low"]
    assert result.selected_cids == ["fast_high"]


def test_duration_penalty_reduces_score_above_preferred_duration() -> None:
    selector = OortSelector(
        OortSelectorConfig(
            exploration_factor=0.0,
            exploration_min=0.0,
            exploration_decay=1.0,
            round_threshold=50.0,
            round_penalty=2.0,
            seed=5,
        )
    )
    selector.update_after_fit(
        round_id=1,
        cid="fast",
        num_examples=20,
        metrics={"loss": 1.0, "fit_duration_sec": 1.0},
    )
    selector.update_after_fit(
        round_id=1,
        cid="slow",
        num_examples=20,
        metrics={"loss": 1.0, "fit_duration_sec": 10.0},
    )

    result = selector.select(
        round_id=10,
        candidate_cids=["fast", "slow"],
        num_clients=1,
    )

    assert selector.round_prefer_duration < selector.clients["slow"].duration
    assert result.scores["slow"] < result.scores["fast"]


def test_exploration_decays_but_not_below_minimum() -> None:
    selector = OortSelector(
        OortSelectorConfig(
            exploration_factor=0.5,
            exploration_decay=0.5,
            exploration_min=0.2,
            seed=17,
        )
    )

    for round_id in range(1, 6):
        selector.select(
            round_id=round_id,
            candidate_cids=["c1", "c2", "c3"],
            num_clients=1,
        )

    assert selector.exploration == pytest.approx(0.2)


def test_blacklist_rounds_apply_without_blacklisting_all_clients() -> None:
    selector = OortSelector(
        OortSelectorConfig(
            blacklist_rounds=1,
            blacklist_max_len=1.0,
            seed=19,
        )
    )
    for cid in ("c1", "c2", "c3"):
        selector.register_client(cid)
    for cid in ("c1", "c2", "c3"):
        selector.update_after_fit(
            round_id=1,
            cid=cid,
            num_examples=4,
            metrics={"loss": 1.0, "duration": 1.0},
        )
        selector.update_after_fit(
            round_id=2,
            cid=cid,
            num_examples=4,
            metrics={"loss": 1.0, "duration": 1.0},
        )

    assert len(selector.blacklist) == 2
    assert len(selector.blacklist) < len(selector.clients)


def test_blacklist_relaxes_when_needed_to_satisfy_requested_count() -> None:
    selector = OortSelector(
        OortSelectorConfig(
            blacklist_rounds=0,
            blacklist_max_len=0.5,
            exploration_factor=0.0,
            exploration_decay=1.0,
            exploration_min=0.0,
            seed=21,
        )
    )
    cids = ["c1", "c2", "c3", "c4"]
    for cid in cids:
        selector.update_after_fit(
            round_id=1,
            cid=cid,
            num_examples=4,
            metrics={"loss": 1.0, "duration": 1.0},
        )

    result = selector.select(round_id=2, candidate_cids=cids, num_clients=4)

    assert len(selector.blacklist) == 2
    assert len(result.selected_cids) == 4
    assert set(result.metadata["relaxed_blacklist"]) == selector.blacklist


def test_update_after_fit_refreshes_client_statistics() -> None:
    selector = OortSelector()

    selector.update_after_fit(
        round_id=3,
        cid="c1",
        num_examples=12,
        metrics={
            "loss": 2.5,
            "accuracy": 0.7,
            "fit_duration_sec": 4.0,
            "num_examples": 12,
        },
    )

    state = selector.clients["c1"]
    assert state.reward > 0
    assert state.duration == pytest.approx(4.0)
    assert state.num_examples == 12
    assert state.last_selected_round == 3
    assert state.selected_count == 1
    assert state.last_loss == pytest.approx(2.5)
    assert state.last_accuracy == pytest.approx(0.7)
    assert "c1" not in selector.unexplored
    assert "c1" in selector.successful_clients


def test_update_after_failure_does_not_mark_successful() -> None:
    selector = OortSelector()

    selector.update_after_failure(round_id=4, cid="c1", reason=RuntimeError("boom"))

    assert "c1" in selector.clients
    assert "c1" not in selector.successful_clients
    assert selector.clients["c1"].metadata["failure_count"] == 1


def test_fixed_seed_selection_is_reproducible() -> None:
    config = OortSelectorConfig(seed=29)

    first = OortSelector(config).select(
        round_id=1,
        candidate_cids=[f"c{i}" for i in range(6)],
        num_clients=3,
    )
    second = OortSelector(config).select(
        round_id=1,
        candidate_cids=[f"c{i}" for i in range(6)],
        num_clients=3,
    )

    assert first.selected_cids == second.selected_cids


def test_missing_loss_and_duration_metrics_are_safe() -> None:
    selector = OortSelector()

    selector.update_after_fit(
        round_id=1,
        cid="c1",
        num_examples=5,
        metrics={},
    )

    state = selector.clients["c1"]
    assert state.duration == pytest.approx(1.0)
    assert state.last_loss == pytest.approx(0.0)
    assert state.reward >= 0.0
