from __future__ import annotations

import json

import pytest

from splitfleet.server.placement.cosplit_ucb import (
    BanditStateStore,
    CoSplitUCBConfig,
    CoSplitUCBPlacementPolicy,
    ContextEncoder,
    PlacementFeedback,
    SplitCandidateDescriptor,
    StaticCandidateProvider,
)


def _provider() -> StaticCandidateProvider:
    values = []
    for boundary, prefix in (("after:x", 2), ("after:y", 8)):
        values.append(
            SplitCandidateDescriptor(
                boundary=boundary,
                split_id=boundary,
                graph_position_ratio=prefix / 10,
                prefix_node_count=prefix,
                suffix_node_count=10 - prefix,
                total_node_count=10,
                boundary_forward_bytes=100,
                boundary_gradient_bytes=100,
                boundary_tensor_count=1,
                prefix_parameter_bytes=None,
                suffix_parameter_bytes=None,
                client_memory_bytes=None,
                server_memory_bytes=None,
                trainable=True,
                feature_abi_id=boundary,
                graph_signature="g",
                framework_backend="pytorch",
            )
        )
    return StaticCandidateProvider(values)


def _policy(store=None):
    return CoSplitUCBPlacementPolicy(
        candidate_provider=_provider(),
        config=CoSplitUCBConfig(
            max_explorations_per_round=0,
            min_residence_rounds=0,
            target_scale_ms=1,
        ),
        state_store=store,
    )


def test_warm_start_restore_preserves_predictions_uncertainty_and_assignment(tmp_path) -> None:
    path = tmp_path / "cosplit-state.json"
    original = _policy(BanditStateStore(path))
    assignment = original.plan_round(round_id=1, client_ids=["a"], training=True)
    original.observe_round(
        round_id=1,
        feedback=[
            PlacementFeedback(
                round_id=1,
                client_id="a",
                boundary=assignment["a"],
                client_forward_ms=11,
                client_backward_ms=7,
                network_upload_ms=3,
                network_download_ms=2,
                server_service_ms=13,
                switch_ms=1,
                num_examples=8,
                num_batches=2,
                execution_profile="torch|torchlens_native|cuda|orin|fp16",
            )
        ],
    )
    assert json.loads(path.read_text())["algorithm_version"] == "cosplit_ucb_v1"

    restored = _policy(BanditStateStore(path))
    expected = original.plan_round(round_id=2, client_ids=["a"], training=False)
    actual = restored.plan_round(round_id=2, client_ids=["a"], training=False)
    assert actual == expected
    left = original.evaluation_diagnostics[2]["estimates"]
    right = restored.evaluation_diagnostics[2]["estimates"]
    assert right == left


def test_failure_feasibility_is_persisted_immediately(tmp_path) -> None:
    path = tmp_path / "cosplit-state.json"
    policy = _policy(BanditStateStore(path))
    boundary = policy.plan_round(
        round_id=1, client_ids=["a"], training=True
    )["a"]
    policy.observe_failure(
        round_id=1,
        client_id="a",
        boundary=boundary,
        kind="oom",
    )
    state = json.loads(path.read_text())
    assert state["state"]["feasibility"]["blacklist"] == [
        {"boundary": boundary, "client_id": "a", "until": 4}
    ]


def test_state_uses_injected_context_feature_schema_version() -> None:
    class NextEncoder(ContextEncoder):
        feature_schema_version = "cosplit_context_v2_test"

    policy = CoSplitUCBPlacementPolicy(
        candidate_provider=_provider(),
        context_encoder=NextEncoder(),
    )
    policy.plan_round(round_id=1, client_ids=["a"], training=True)
    state = policy.state_dict()
    assert state["feature_schema_version"] == "cosplit_context_v2_test"
    with pytest.raises(ValueError, match="feature schema"):
        _policy().load_state_dict(state)
