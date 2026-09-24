from __future__ import annotations

import pytest
import numpy as np

from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig,
    CoSplitUCBPlacementPolicy,
    FeasibilityFilter,
    PlacementFeedback,
    SplitCandidateDescriptor,
    StaticCandidateProvider,
)


def _candidate(boundary: str, position: float) -> SplitCandidateDescriptor:
    prefix = int(position * 10)
    return SplitCandidateDescriptor(
        boundary=boundary,
        split_id=boundary,
        graph_position_ratio=position,
        prefix_node_count=prefix,
        suffix_node_count=10 - prefix,
        total_node_count=10,
        boundary_forward_bytes=1000,
        boundary_gradient_bytes=1000,
        boundary_tensor_count=1,
        prefix_parameter_bytes=1000,
        suffix_parameter_bytes=1000,
        client_memory_bytes=None,
        server_memory_bytes=None,
        trainable=True,
        feature_abi_id=f"abi-{boundary}",
        graph_signature="graph-v1",
        framework_backend="pytorch",
    )


def _policy(seed: int = 7, *, min_residence_rounds: int = 0) -> CoSplitUCBPlacementPolicy:
    config = CoSplitUCBConfig(
        alpha_edge=0,
        alpha_network=0,
        alpha_server=0,
        alpha_switch=0,
        max_explorations_per_round=0,
        min_residence_rounds=min_residence_rounds,
        target_scale_ms=1,
        seed=seed,
    )
    return CoSplitUCBPlacementPolicy(
        candidate_provider=StaticCandidateProvider(
            [_candidate("after:arbitrary_op_a", 0.25), _candidate("after:arbitrary_op_b", 0.75)]
        ),
        config=config,
    )


def test_policy_caches_one_assignment_and_learns_from_round_feedback() -> None:
    policy = _policy()
    first = policy.plan_round(round_id=1, client_ids=["b", "a"], training=True)
    again = policy.plan_round(round_id=1, client_ids=["a", "b"], training=True)
    assert first == again
    assert set(first) == {"a", "b"}

    policy.observe_round(
        round_id=1,
        feedback=[
            PlacementFeedback(
                round_id=1,
                client_id=cid,
                boundary=boundary,
                client_forward_ms=100,
                client_backward_ms=0,
                server_service_ms=0,
                switch_ms=0,
                num_examples=4,
                num_batches=2,
            )
            for cid, boundary in first.items()
        ],
    )
    second = policy.plan_round(round_id=2, client_ids=["a", "b"], training=True)
    assert set(second) == {"a", "b"}
    group_state = next(iter(policy.learners.edge.state_dict().values()))
    assert group_state["forward"]["num_updates"] == 2


def test_failure_does_not_create_fake_bandit_update_and_evaluation_does_not_explore() -> None:
    policy = _policy()
    training = policy.plan_round(round_id=1, client_ids=["a"], training=True)
    before = policy.learners.edge.state_dict()
    policy.observe_failure(
        round_id=1,
        client_id="a",
        boundary=training["a"],
        kind="oom",
        reason="out of memory",
    )
    assert policy.learners.edge.state_dict() == before
    evaluation = policy.plan_round(round_id=1, client_ids=["a"], training=False)
    assert policy.round_diagnostics[1]["exploration"] == []
    assert evaluation["a"] in {"after:arbitrary_op_a", "after:arbitrary_op_b"}


def test_same_seed_and_feedback_are_deterministic() -> None:
    left = _policy(seed=233)
    right = _policy(seed=233)
    assert left.plan_round(round_id=1, client_ids=["a", "b"], training=True) == right.plan_round(
        round_id=1, client_ids=["a", "b"], training=True
    )


def test_failed_resident_boundary_can_escape_to_a_feasible_candidate() -> None:
    policy = _policy(min_residence_rounds=2)
    first = policy.plan_round(round_id=1, client_ids=["a"], training=True)["a"]
    policy.observe_failure(
        round_id=1,
        client_id="a",
        boundary=first,
        kind="oom",
    )
    second = policy.plan_round(round_id=2, client_ids=["a"], training=True)["a"]
    assert second != first


def test_feedback_is_idempotent_and_must_match_the_planned_action() -> None:
    policy = _policy()
    boundary = policy.plan_round(round_id=1, client_ids=["a"], training=True)["a"]
    observation = PlacementFeedback(
        round_id=1,
        client_id="a",
        boundary=boundary,
        client_forward_ms=12,
        num_examples=4,
        num_batches=2,
    )
    policy.observe_round(round_id=1, feedback=[observation])
    first = policy.learners.edge.state_dict()
    policy.observe_round(round_id=1, feedback=[observation])
    assert policy.learners.edge.state_dict() == first

    other = "after:arbitrary_op_b" if boundary.endswith("_a") else "after:arbitrary_op_a"
    with pytest.raises(ValueError, match="does not match"):
        policy.observe_round(
            round_id=1,
            feedback=[PlacementFeedback(round_id=1, client_id="a", boundary=other, client_forward_ms=1)],
        )
    assert policy.learners.edge.state_dict() == first


def test_next_round_uses_observed_batch_count_and_evaluation_keeps_training_diagnostics() -> None:
    policy = _policy()
    boundary = policy.plan_round(round_id=1, client_ids=["a"], training=True)["a"]
    policy.observe_round(
        round_id=1,
        feedback=[PlacementFeedback(
            round_id=1,
            client_id="a",
            boundary=boundary,
            client_forward_ms=10,
            completion_ms=30,
            num_batches=3,
        )],
    )
    assert policy.round_diagnostics[1]["predicted_batch_counts"] == {"a": 1}
    assert policy.round_diagnostics[1]["exploration_suppressed_reason"] == "unknown_batch_count"
    assert policy.round_diagnostics[1]["prediction_residual_ms"] == {}
    policy.observe_round_wall_time(round_id=1, duration_ms=50)
    assert policy.round_diagnostics[1]["client_fit_duration_max_ms"] == 30
    assert policy.round_diagnostics[1]["actual_round_makespan_ms"] == 50

    policy.plan_round(round_id=2, client_ids=["a"], training=True)
    training_diagnostics = policy.round_diagnostics[2]
    assert training_diagnostics["predicted_batch_counts"] == {"a": 3}
    assert training_diagnostics["exploration_suppressed_reason"] is None
    policy.plan_round(round_id=2, client_ids=["a"], training=False)
    assert policy.round_diagnostics[2] is training_diagnostics
    assert policy.evaluation_diagnostics[2]["exploration"] == []


def test_client_reported_execution_profiles_separate_edge_updates() -> None:
    policy = _policy()
    assignment = policy.plan_round(round_id=1, client_ids=["a", "b"], training=True)
    profiles = {
        "a": "torch|torchlens_native|cpu|x86_64|fp32",
        "b": "torch|torchlens_native|cuda|orin|fp16",
    }
    policy.observe_round(
        round_id=1,
        feedback=[
            PlacementFeedback(
                round_id=1,
                client_id=cid,
                boundary=assignment[cid],
                client_forward_ms=10,
                num_batches=1,
                execution_profile=profiles[cid],
            )
            for cid in ("a", "b")
        ],
    )
    state = policy.learners.edge.state_dict()
    assert state[profiles["a"]]["forward"]["num_updates"] == 1
    assert state[profiles["b"]]["forward"]["num_updates"] == 1

    policy.plan_round(round_id=2, client_ids=["a", "b"], training=True)
    diagnostics = policy.round_diagnostics[2]["estimates"]
    assert {row["execution_profile"] for row in diagnostics["a"].values()} == {profiles["a"]}
    assert {row["execution_profile"] for row in diagnostics["b"].values()} == {profiles["b"]}


def test_evaluation_does_not_replace_training_feedback_context() -> None:
    telemetry = {"a": {"cpu_utilization": 10, "num_batches": 1}}
    policy = _policy()
    policy.telemetry_provider = telemetry
    boundary = policy.plan_round(round_id=1, client_ids=["a"], training=True)["a"]
    profile, training_context = policy._round_contexts[(1, True, "a", boundary)]
    telemetry["a"] = {"cpu_utilization": 90, "num_batches": 1}
    policy.plan_round(round_id=1, client_ids=["a"], training=False)

    policy.observe_round(
        round_id=1,
        feedback=[PlacementFeedback(
            round_id=1,
            client_id="a",
            boundary=boundary,
            client_forward_ms=12,
            num_batches=1,
        )],
    )
    actual_a = np.asarray(policy.learners.edge.state_dict()[profile.stable_id]["forward"]["A"])
    expected_a = np.eye(len(training_context.edge)) + np.outer(training_context.edge, training_context.edge)
    np.testing.assert_allclose(actual_a, expected_a)


def test_client_abi_failure_is_local_and_network_timeout_keeps_cut_feasible() -> None:
    candidate = _candidate("after:arbitrary_op_a", 0.25)
    feasibility = FeasibilityFilter()
    feasibility.observe_failure(round_id=1, client_id="a", boundary=candidate.boundary, kind="abi")
    assert not feasibility.check(candidate, client_id="a", round_id=2).feasible
    assert feasibility.check(candidate, client_id="b", round_id=2).feasible

    restored = FeasibilityFilter()
    restored.load_state_dict(feasibility.state_dict())
    assert not restored.check(candidate, client_id="a", round_id=2).feasible
    assert restored.check(candidate, client_id="b", round_id=2).feasible
    restored.observe_failure(round_id=2, client_id="b", boundary=candidate.boundary, kind="network")
    assert restored.check(candidate, client_id="b", round_id=3).feasible
    assert {tuple(row.values()) for row in restored.state_dict()["failure_counts"]} == {
        ("a", "abi", 1),
        ("b", "network", 1),
    }


def test_evaluation_uses_only_cuts_valid_in_both_graphs() -> None:
    early = _candidate("after:arbitrary_op_a", 0.25)
    late = _candidate("after:arbitrary_op_b", 0.75)

    class DifferentEvaluationGraph:
        def get_candidates(self, *, training: bool = True):
            return (early, late) if training else (late,)

    policy = CoSplitUCBPlacementPolicy(
        candidate_provider=DifferentEvaluationGraph(),
        config=CoSplitUCBConfig(max_explorations_per_round=0),
    )
    assert policy.plan_round(round_id=1, client_ids=["a"], training=True) == {
        "a": early.boundary
    }
    assert "a" not in policy._last_executed_boundary
    policy.observe_round(round_id=1, feedback=[PlacementFeedback(
        round_id=1, client_id="a", boundary=early.boundary,
        client_forward_ms=1, num_examples=1, num_batches=1,
    )])
    assert policy.plan_round(round_id=1, client_ids=["a"], training=False) == {
        "a": late.boundary
    }
    assert policy._last_boundary["a"] == early.boundary
    assert policy._last_executed_boundary["a"] == early.boundary
    policy.observe_evaluation(round_id=1, executions={"a": late.boundary})
    assert policy._last_executed_boundary["a"] == late.boundary
    restored = CoSplitUCBPlacementPolicy(
        candidate_provider=DifferentEvaluationGraph(),
        config=CoSplitUCBConfig(max_explorations_per_round=0),
    )
    restored.load_state_dict(policy.state_dict())
    assert restored._last_executed_boundary["a"] == late.boundary
    restored.plan_round(round_id=2, client_ids=["a"], training=True)
    assert restored._round_contexts[(2, True, "a", early.boundary)][1].switch[1] == 1
