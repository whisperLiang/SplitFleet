"""Round transfer measurements must not repeat with batches or delegated RPCs."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig, CoSplitUCBPlacementPolicy, GlobalPlacementSolver,
    PlacementFeedback, StaticCandidateProvider,
)
from splitfleet.server.placement.cosplit_ucb.rpc_timing import (
    TimedFitClientProxy, state_exchange_duration,
)
from tests.unit.test_cosplit_solver import _estimate
from tests.unit.test_cosplit_policy import _candidate


def test_fit_proxy_delegates_once_and_preserves_connection_identity(monkeypatch):
    calls = []
    result = SimpleNamespace(metrics={"original": 1})
    def fit(ins, **kwargs):
        calls.append((ins, kwargs))
        return result
    original = SimpleNamespace(cid="worker", node_id=42, properties={"kind": "gpu"}, fit=fit)
    proxy = TimedFitClientProxy(original)
    ticks = iter((1000000, 6000000))
    monkeypatch.setattr("splitfleet.server.placement.cosplit_ucb.rpc_timing.time.perf_counter_ns", lambda: next(ticks))
    ins = object()
    assert proxy.fit(ins, timeout=30, group_id=2) is result
    assert calls == [(ins, {"timeout": 30, "group_id": 2})]
    assert proxy.cid == "worker" and proxy.node_id == 42 and proxy.properties is original.properties
    assert result.metrics == {"original": 1, "coordinator_fit_rpc_ms": 5}


def test_exchange_target_subtracts_full_handler_and_keeps_state_work_without_switch_duplication():
    metrics = {"coordinator_fit_rpc_ms": 1200, "client_fit_handler_ms": 1000,
               "runtime_prepare_sec": .08, "state_export_sec": .02, "switch_ms": 50}
    assert state_exchange_duration(metrics) == pytest.approx(250)
    assert state_exchange_duration({**metrics, "switch_ms": 100}) == pytest.approx(220)


@pytest.mark.parametrize("metrics", [
    {}, {"coordinator_fit_rpc_ms": 2},
    {"coordinator_fit_rpc_ms": 2, "client_fit_handler_ms": 3},
    {"coordinator_fit_rpc_ms": float("nan"), "client_fit_handler_ms": 1},
    {"coordinator_fit_rpc_ms": 2, "client_fit_handler_ms": -1},
    {"coordinator_fit_rpc_ms": "invalid", "client_fit_handler_ms": 1},
    {"coordinator_fit_rpc_ms": 2, "client_fit_handler_ms": 1, "state_export_sec": float("inf")},
])
def test_invalid_exchange_metrics_do_not_become_learning_targets(metrics):
    assert state_exchange_duration(metrics) is None


def test_solver_counts_round_exchange_once_and_cache_includes_it():
    solver = GlobalPlacementSolver()
    base = _estimate("a", "early", arrival=2, service=3, tail=1)
    ten = {"a": 10}
    before = solver.simulate({"a": base}, batch_counts=ten)
    changed = replace(base, state_exchange_mean_ms=100, state_exchange_uncertainty_ms=20)
    after = solver.simulate({"a": changed}, batch_counts=ten)
    assert after.max_client_completion_ms - before.max_client_completion_ms == 100
    assert after.timelines["a"].queue_ms == before.timelines["a"].queue_ms
    assert solver.simulate({"a": changed}, use_upper=True, batch_counts=ten).max_client_completion_ms == 180
    assert solver.simulate({"a": base}, batch_counts=ten) == before
    late = replace(base, boundary="late", client_forward_mean_ms=0, state_exchange_mean_ms=200)
    assert solver.solve({"a": [changed, late]}, batch_counts=ten)["a"].boundary == "early"


def test_exchange_feedback_is_one_observation_per_round_not_per_batch_and_persists():
    candidate = replace(_candidate("early", .25), metadata={
        "optimizer_prefix_parameter_bytes": 1024, "optimizer_suffix_parameter_bytes": 3072,
        "boundary_forward_bytes_by_batch_size": {4: 1000}})
    config = CoSplitUCBConfig(max_explorations_per_round=0, min_residence_rounds=0)
    policy = CoSplitUCBPlacementPolicy(candidate_provider=StaticCandidateProvider([candidate]), config=config)
    policy.state_download_bytes = 4096
    boundary = policy.plan_round(round_id=1, client_ids=["a"], training=True)["a"]
    observation = PlacementFeedback(1, "a", boundary, state_exchange_ms=100, num_batches=10)
    policy.observe_round(round_id=1, feedback=[observation])
    model = policy.learners.network._models("a")["exchange"]
    x = policy.context_encoder.exchange_context(download_bytes=4096, upload_bytes=1024)
    np.testing.assert_allclose(model.A, np.eye(len(x)) * config.ridge_lambda + np.outer(x, x))
    assert model.num_updates == 1 and policy.learners.network.update_count("a") == 0
    before = policy.state_dict()
    policy.observe_round(round_id=1, feedback=[observation])
    assert policy.state_dict() == before
    restored = CoSplitUCBPlacementPolicy(candidate_provider=StaticCandidateProvider([candidate]), config=config)
    restored.load_state_dict(before)
    assert restored.learners.network.exchange_update_count("a") == 1
    assert restored.learners.network.predict_exchange("a", x) == model.predict(x)


def test_controlled_reply_probe_leaves_model_and_random_state_intact():
    import torch
    from splitfleet.common.model_state import tensor_state_hash
    from tests.test_cosplit_calibration import _native_cohort
    cohort = _native_cohort(config=CoSplitUCBConfig())
    client = cohort.clients["a"]
    before, rng = tensor_state_hash(client.model.state_dict()), torch.get_rng_state().clone()
    assert client.get_properties({"cosplit_probe_payload": b"small", "cosplit_probe_reply_bytes": 100}) == {"cosplit_probe_payload": bytes(100)}
    assert tensor_state_hash(client.model.state_dict()) == before and torch.equal(rng, torch.get_rng_state())
    for size in (0, -1, 2097153, True):
        with pytest.raises(ValueError, match="reply size"):
            client.get_properties({"cosplit_probe_payload": b"small", "cosplit_probe_reply_bytes": size})
