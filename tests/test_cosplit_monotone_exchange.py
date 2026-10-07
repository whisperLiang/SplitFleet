"""Regression tests for transfer extrapolation and absolute exploration cost."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig, CoSplitUCBPlacementPolicy, GlobalPlacementSolver,
    PlacementFeedback, SafeExplorationController, StaticCandidateProvider,
)
from splitfleet.server.placement.cosplit_ucb.bandit import MonotoneStateExchangeLinUCB
from splitfleet.server.placement.cosplit_ucb.context import ContextEncoder
from splitfleet.server.placement.cosplit_ucb.learners import CooperativeLearners
from splitfleet.server.placement.cosplit_ucb.residual import ResidualLinUCB
from tests.unit.test_cosplit_exploration import _estimate
from tests.unit.test_cosplit_policy import _candidate


def _model(dimension=5, **kwargs):
    return MonotoneStateExchangeLinUCB(
        dimension, target_scale=1, feature_schema="exchange-test", **kwargs
    )


def test_constrained_ridge_refits_active_coefficients_instead_of_clipping():
    model = _model(2, ridge_lambda=.1, discount_gamma=1)
    model.update(np.array([1, 1]), 3, round_id=0)
    model.update(np.array([1, 2]), 1, round_id=0)
    theta = model._fit_theta()
    np.testing.assert_allclose(theta, [4 / 2.1, 0])
    clipped = np.maximum(np.linalg.solve(model.A, model.b), 0)
    objective = lambda value: .5 * value @ model.A @ value - model.b @ value
    assert objective(theta) < objective(clipped) - .1
    assert model.predict(np.array([1, 100])).mean == pytest.approx(4 / 2.1)


def test_large_constant_download_does_not_hide_positive_intercept_gradient():
    model = _model(2, ridge_lambda=.1, discount_gamma=1)
    x = np.array([1., 420.])
    for _ in range(20):
        model.update(x, 20, round_id=0)
    expected = 400 * x / (.1 + 20 * float(x @ x))
    np.testing.assert_allclose(model._fit_theta(), expected, rtol=1e-6, atol=1e-10)
    assert model._fit_theta()[0] > .0001


@pytest.mark.parametrize("seed", [11, 37, 101])
def test_active_set_matches_independent_exhaustive_constraint_oracle(seed):
    rng = np.random.default_rng(seed)
    model = _model(ridge_lambda=.1, discount_gamma=1)
    for _ in range(20):
        # Constant download / changing upload tests the collinear byte regime.
        x = np.array([1., 420., rng.uniform(.1, 420), rng.integers(0, 2), rng.integers(0, 2)])
        model.update(x, rng.uniform(.1, 30), round_id=0)
    choices = []
    for bits in product((False, True), repeat=5):
        active = np.array(bits)
        theta = np.zeros(5)
        if active.any():
            theta[active] = np.linalg.solve(model.A[np.ix_(active, active)], model.b[active])
        if np.all(theta >= 0):
            choices.append((.5 * theta @ model.A @ theta - model.b @ theta, theta))
    oracle = min(choices, key=lambda item: item[0])[1]
    fitted = model._fit_theta()
    np.testing.assert_allclose(fitted, oracle, atol=1e-8, rtol=1e-7)
    gradient = model.A @ fitted - model.b
    assert np.all(fitted >= 0)
    assert np.all(gradient[fitted == 0] >= -1e-6)
    np.testing.assert_allclose(gradient[fitted > 0], 0, atol=1e-6)


@pytest.mark.parametrize("cache_size", [0, 32])
def test_larger_parameter_upload_never_predicts_cheaper_exchange(cache_size):
    model = _model(prediction_cache_size=cache_size)
    for round_id, (upload, cost) in enumerate([(1, 24), (150, 22), (400, 21)]):
        model.update(np.array([1, 420, upload, 0, 0]), cost, round_id=round_id)
    before = model.predict(np.array([1, 420, 1, 0, 0]))
    after = model.predict(np.array([1, 420, 600, 0, 0]))
    assert after.mean >= before.mean > 0
    assert after.uncertainty > before.uncertainty
    restored = _model(prediction_cache_size=cache_size)
    restored.load_state_dict(model.state_dict())
    assert restored.predict(np.array([1, 420, 600, 0, 0])) == after
    model.update(np.array([1, 420, 600, 0, 0]), 100, round_id=4)
    assert model.predict(np.array([1, 420, 600, 0, 0])).mean > after.mean


def test_weighted_exchange_statistics_discount_and_cache_round_trip():
    model = _model(2, discount_gamma=.5, ridge_lambda=.5)
    x = np.array([1., 3.])
    model.update(x, 8, round_id=1, sample_weight=.25)
    model.predict(x)
    model.advance_round(3)
    np.testing.assert_allclose(model.A, .5 * np.eye(2) + .0625 * np.outer(x, x))
    np.testing.assert_allclose(model.b, .5 * x)
    assert model.num_updates == 1
    before = model.predict(x)
    model.b *= 2  # Public sufficient-statistic changes must invalidate the cache.
    assert model.predict(x).mean == pytest.approx(before.mean * 2)


def test_signed_context_and_unconstrained_state_are_rejected_without_mutation():
    model = _model(2)
    before = model.state_dict()
    with pytest.raises(ValueError, match="non-negative"):
        model.update(np.array([1, -1]), 1, round_id=0)
    with pytest.raises(ValueError, match="non-negative"):
        model.predict(np.array([-1, 1]))
    state = dict(before)
    state.pop("coefficient_constraint")
    with pytest.raises(ValueError, match="constraint mismatch"):
        model.load_state_dict(state)
    assert model.state_dict() == before


def test_default_uses_constrained_exchange_and_residual_batch_costs():
    learners = CooperativeLearners(CoSplitUCBConfig(), ContextEncoder())
    models = learners.network._models("a")
    assert set(models) == {"roundtrip", "exchange"}
    assert type(models["exchange"]) is MonotoneStateExchangeLinUCB
    assert type(models["roundtrip"]) is ResidualLinUCB
    assert type(learners.server.model) is ResidualLinUCB


def test_uncertain_baseline_cannot_expand_online_exploration_allowance():
    solver = GlobalPlacementSolver()
    base = _estimate("a", "base", 100, 1000)
    probe = _estimate("a", "probe", 99, 100)
    args = dict(round_id=20, baseline={"a": base}, estimates={"a": [base, probe]}, solver=solver)
    online = SafeExplorationController(epsilon=.05)
    final, decisions = online.apply(**args)
    assert final["a"].boundary == "base" and decisions == []
    assert online.round_records[20]["safe_budget_ms"] == 105
    assert online.round_records[20]["upper_budget_basis"] == "baseline_mean"
    assert online.round_records[20]["rejected_by_upper_budget"] == 1
    safe = replace(probe, client_forward_uncertainty_ms=5)
    args["estimates"] = {"a": [base, safe]}
    final, decisions = online.apply(**args)
    assert final["a"].boundary == "probe"
    assert decisions[0].predicted_upper_makespan_ms == 104


def test_online_exploration_budget_mode_must_match_saved_state():
    controller = SafeExplorationController()
    controller.record_observation(round_id=3, client_id="a", boundary="b", was_probe=True)
    saved = controller.state_dict()
    restored = SafeExplorationController()
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    before = restored.state_dict()
    with pytest.raises(ValueError, match="budget basis mismatch"):
        restored.load_state_dict({**saved, "upper_budget_basis": "unknown"})
    assert restored.state_dict() == before


def test_policy_diagnostics_report_actual_online_exploration_budget():
    config = CoSplitUCBConfig(min_residence_rounds=0, max_explorations_per_round=1)
    policy = CoSplitUCBPlacementPolicy(candidate_provider=StaticCandidateProvider([_candidate("a", .25)]),
                                      config=config, telemetry_provider={"a": {"num_batches": 2}})
    policy.plan_round(round_id=1, client_ids=["a"], training=True)
    policy.observe_round(round_id=1, feedback=[PlacementFeedback(
        1, "a", "a", client_forward_ms=100, state_exchange_ms=100, num_batches=2)])
    policy.plan_round(round_id=2, client_ids=["a"], training=True)
    diagnostic = policy.round_diagnostics[2]
    assert diagnostic["baseline_makespan_ms"] > 0
    assert diagnostic["baseline_upper_makespan_ms"] > diagnostic["baseline_makespan_ms"]
    assert diagnostic["upper_budget_basis"] == "baseline_mean"
    assert diagnostic["safe_budget_ms"] == pytest.approx(diagnostic["baseline_makespan_ms"] * 1.05)
