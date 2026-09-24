from __future__ import annotations

from splitfleet.server.placement.cosplit_ucb import CandidateEstimate, GlobalPlacementSolver, SafeExplorationController


def _estimate(cid: str, boundary: str, mean: float, uncertainty: float) -> CandidateEstimate:
    zeros = dict(
        client_backward_mean_ms=0,
        network_upload_mean_ms=0,
        network_download_mean_ms=0,
        server_service_mean_ms=0,
        switch_mean_ms=0,
        client_backward_uncertainty_ms=0,
        network_upload_uncertainty_ms=0,
        network_download_uncertainty_ms=0,
        server_service_uncertainty_ms=0,
        switch_uncertainty_ms=0,
    )
    return CandidateEstimate(
        client_id=cid,
        boundary=boundary,
        client_forward_mean_ms=mean,
        client_forward_uncertainty_ms=uncertainty,
        **zeros,
    )


def test_safe_exploration_respects_global_epsilon_and_rejects_unsafe() -> None:
    solver = GlobalPlacementSolver()
    base = _estimate("a", "base", 100, 0)
    safe = _estimate("a", "safe", 90, 12)
    unsafe = _estimate("a", "unsafe", 80, 50)
    controller = SafeExplorationController(epsilon=0.05, max_explorations_per_round=1, forced_probe_interval=10)
    final, decisions = controller.apply(
        round_id=1,
        baseline={"a": base},
        estimates={"a": [base, safe, unsafe]},
        solver=solver,
    )
    assert final["a"].boundary == "safe"
    assert decisions[0].predicted_upper_makespan_ms <= 105


def test_forced_probe_is_still_required_to_be_safe_and_residence_can_lock() -> None:
    solver = GlobalPlacementSolver()
    base = _estimate("a", "base", 100, 0)
    unsafe = _estimate("a", "unsafe", 90, 30)
    controller = SafeExplorationController(epsilon=0.05, max_explorations_per_round=1, forced_probe_interval=2)
    final, decisions = controller.apply(
        round_id=20,
        baseline={"a": base},
        estimates={"a": [base, unsafe]},
        solver=solver,
    )
    assert final["a"].boundary == "base"
    assert decisions == []

    safe = _estimate("a", "safe", 95, 5)
    final, decisions = controller.apply(
        round_id=20,
        baseline={"a": base},
        estimates={"a": [base, safe]},
        solver=solver,
        residence_locked={"a"},
    )
    assert final["a"].boundary == "base"
    assert decisions == []


def test_candidate_level_exploration_history_round_trips() -> None:
    solver = GlobalPlacementSolver()
    base = _estimate("a", "base", 100, 0)
    safe = _estimate("a", "safe", 95, 5)
    controller = SafeExplorationController(
        epsilon=0.05, max_explorations_per_round=1, forced_probe_interval=2
    )
    final, decisions = controller.apply(
        round_id=20,
        baseline={"a": base},
        estimates={"a": [base, safe]},
        solver=solver,
    )
    assert final["a"].boundary == "safe"
    assert decisions[0].reason == "forced_safe_probe"
    controller.record_observation(
        round_id=20,
        client_id="a",
        boundary="safe",
        was_probe=True,
    )
    assert controller.candidate_last_explored_round[("a", "safe")] == 20

    restored = SafeExplorationController(
        epsilon=0.05, max_explorations_per_round=1, forced_probe_interval=2
    )
    restored.load_state_dict(controller.state_dict())
    assert restored.candidate_last_explored_round == {
        ("a", "safe"): 20
    }


def test_component_staleness_forces_only_a_safe_probe() -> None:
    solver = GlobalPlacementSolver()
    base = _estimate("a", "base", 100, 0)
    safe = _estimate("a", "safe", 95, 5)
    controller = SafeExplorationController(
        epsilon=0.05,
        max_explorations_per_round=1,
        forced_probe_interval=5,
        seed=9,
    )
    controller.candidate_last_explored_round[("a", "safe")] = 19
    _, decisions = controller.apply(
        round_id=20,
        baseline={"a": base},
        estimates={"a": [base, safe]},
        solver=solver,
        component_last_observation={
            ("a", "network"): 1,
            ("__global__", "server"): 19,
        },
    )
    assert decisions[0].reason == "forced_safe_probe"


def test_stale_last_probe_forces_probe_despite_fresh_component_observations() -> None:
    solver = GlobalPlacementSolver()
    base = _estimate("a", "base", 100, 0)
    safe = _estimate("a", "safe", 95, 5)
    controller = SafeExplorationController(
        epsilon=0.05,
        max_explorations_per_round=1,
        forced_probe_interval=5,
    )
    controller.last_probe_round["a"] = 1
    controller.candidate_last_explored_round[("a", "safe")] = 19
    _, decisions = controller.apply(
        round_id=20,
        baseline={"a": base},
        estimates={"a": [base, safe]},
        solver=solver,
        component_last_observation={
            ("a", "network"): 19,
            ("__global__", "server"): 19,
        },
    )
    assert decisions[0].reason == "forced_safe_probe"


def test_seed_and_rng_state_reproduce_equal_priority_exploration() -> None:
    solver = GlobalPlacementSolver()
    baseline = {
        "a": _estimate("a", "base", 100, 0),
        "b": _estimate("b", "base", 100, 0),
    }
    estimates = {
        cid: [base, _estimate(cid, "safe", 95, 5)]
        for cid, base in baseline.items()
    }
    original = SafeExplorationController(
        epsilon=0.05, max_explorations_per_round=1, seed=233
    )
    restored = SafeExplorationController(
        epsilon=0.05, max_explorations_per_round=1, seed=999
    )
    restored.load_state_dict(original.state_dict())
    left, _ = original.apply(
        round_id=1, baseline=baseline, estimates=estimates, solver=solver
    )
    right, _ = restored.apply(
        round_id=1, baseline=baseline, estimates=estimates, solver=solver
    )
    assert {cid: value.boundary for cid, value in left.items()} == {
        cid: value.boundary for cid, value in right.items()
    }
