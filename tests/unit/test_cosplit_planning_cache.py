from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from splitfleet.server.placement.cosplit_ucb import (
    CandidateEstimate, CoSplitUCBConfig, ContextEncoder, CooperativeLearners,
    DiscountedLinUCB, ExecutionProfileKey, GlobalPlacementSolver,
)


def _model(size=2):
    return DiscountedLinUCB(3, discount_gamma=0.5, feature_schema_version="cache-test",
                           prediction_cache_size=size)


def _raw_prediction(model, x):
    mean = max(float(x @ np.linalg.solve(model.A, model.b)) * model.target_scale, 0)
    radius = max(model.alpha * np.sqrt(max(float(x @ np.linalg.solve(model.A, x)), 0)) * model.target_scale, 0)
    return mean, radius


def test_prediction_reuse_avoids_solves_and_invalidates_on_feedback_discount_and_restore(monkeypatch):
    model = _model()
    x = np.array([1., 2., 3.])
    calls = []
    solve = model._solve
    monkeypatch.setattr(model, "_solve", lambda rhs: (calls.append(1), solve(rhs))[1])
    first = model.predict(x)
    assert len(calls) == 2
    assert model.predict(x.copy()) == first
    assert len(calls) == 2
    model.predict(x + 1)
    assert len(calls) == 3  # Same theta, one new confidence solve.
    saved = model.state_dict()
    for mutation in (
        lambda: model.update(x, 17, round_id=1),
        lambda: model.advance_round(3),
        lambda: model.load_state_dict(saved),
    ):
        mutation()
        before = len(calls)
        prediction = model.predict(x)
        assert (prediction.mean, prediction.uncertainty) == _raw_prediction(model, x)
        assert len(calls) == before + 2


def test_public_statistic_and_scale_edits_cannot_return_stale_predictions():
    model = _model()
    x = np.ones(3)
    model.predict(x)
    model.b[:] = 2
    model.A[0, 0] = 4
    model.alpha = 2
    model.target_scale = 1
    prediction = model.predict(x)
    assert (prediction.mean, prediction.uncertainty) == _raw_prediction(model, x)
    model.A[:] = 0
    with pytest.raises(np.linalg.LinAlgError):
        model.predict(x)


def test_cache_is_bounded_and_nonfinite_context_still_rejected():
    model = _model(2)
    for x in (np.ones(3), np.arange(3.), np.full(3, 2), np.ones(3)):
        prediction = model.predict(x)
        assert (prediction.mean, prediction.uncertainty) == _raw_prediction(model, x)
        assert len(model._prediction_cache) <= 2
    with pytest.raises(ValueError, match="non-finite"):
        model.predict(np.array([1., np.nan, 2.]))


def test_invalid_restore_is_atomic_even_when_prediction_is_cached():
    model = _model()
    x = np.ones(3)
    expected = model.predict(x)
    before = model.state_dict()
    invalid = {**before, "b": [100, 200, 300], "last_update_round": 4,
               "last_discount_round": 3}
    with pytest.raises(ValueError, match="discount round"):
        model.load_state_dict(invalid)
    assert model.state_dict() == before
    assert model.predict(x) == expected


def test_model_lookup_creates_only_one_pair_per_execution_group_and_link(monkeypatch):
    learners = CooperativeLearners(CoSplitUCBConfig(), ContextEncoder())
    factory = learners.edge._factory
    create = factory.make
    calls = []
    monkeypatch.setattr(factory, "make", lambda *args: (calls.append(args), create(*args))[1])
    profile = ExecutionProfileKey("pytorch", "native", "cuda", "orin", "fp32")
    for _ in range(5):
        learners.edge.predict(profile, np.ones(ContextEncoder.edge_dimension))
        learners.network.predict("a", np.ones(13), np.ones(13))
    assert len(calls) == 4
    learners.edge.predict(replace(profile, precision="fp16"), np.ones(ContextEncoder.edge_dimension))
    learners.network.predict("b", np.ones(13), np.ones(13))
    assert len(calls) == 8


def _estimate(cid, boundary="x", forward=1, service=5):
    return CandidateEstimate(cid, boundary, forward, 2, 3, 4, service, 6,
                             1, 2, 3, 4, 5, 6)


def test_simulation_key_includes_costs_counts_concurrency_and_upper_bound():
    solver = GlobalPlacementSolver(simulation_cache_size=2)
    assignment = {"a": _estimate("a")}
    first = solver.simulate(assignment)
    assert solver.simulate(dict(assignment)) == first
    assert solver.simulation_cache_hits == 1
    assert solver.simulate(assignment, use_upper=True).objective != first.objective
    assert solver.simulate(assignment, batch_counts={"a": 2}).objective != first.objective
    assert solver.simulate({"a": replace(assignment["a"], server_service_mean_ms=15)}).objective != first.objective
    solver.server_concurrency = 2
    solver.simulate(assignment)
    assert solver.simulation_cache_misses == 5
    assert len(solver._simulation_cache) == 2
    with pytest.raises(ValueError, match="positive"):
        solver.simulate(assignment, batch_counts={"a": 0})
    solver.clear_cache()
    assert not solver._simulation_cache
    assert solver.simulation_cache_hits == solver.simulation_cache_misses == 0


def test_cached_timeline_cannot_be_corrupted_and_assignment_order_is_irrelevant():
    solver = GlobalPlacementSolver()
    assignment = {"a": _estimate("a"), "b": _estimate("b")}
    first = solver.simulate(assignment)
    assert solver.simulate(dict(reversed(list(assignment.items())))) == first
    assert solver.simulation_cache_hits == 1
    with pytest.raises(TypeError):
        first.timelines["a"] = first.timelines["b"]


def test_equal_cost_cuts_share_simulation_without_returning_the_old_boundary():
    solver = GlobalPlacementSolver()
    first = solver.simulate({"a": _estimate("a", "first")})
    second = solver.simulate({"a": _estimate("a", "second")})
    assert solver.simulation_cache_hits == 1
    assert second.objective == first.objective
    assert first.timelines["a"].boundary == "first"
    assert second.timelines["a"].boundary == "second"


@pytest.mark.parametrize("lanes,use_upper", [(1, False), (1, True), (3, False), (3, True)])
def test_cached_solver_matches_uncached_on_queues_ties_and_unequal_batch_counts(lanes, use_upper):
    options = {
        cid: [_estimate(cid, "x", forward=0, service=5),
              _estimate(cid, "y", forward=6, service=1),
              replace(_estimate(cid, "z"), feasible=False)]
        for cid in ("a", "b", "c")
    }
    counts = {"a": 1, "b": 3, "c": 2}
    cached = GlobalPlacementSolver(server_concurrency=lanes)
    uncached = GlobalPlacementSolver(server_concurrency=lanes, simulation_cache_size=0)
    left = cached.solve(options, use_upper=use_upper, batch_counts=counts)
    right = uncached.solve(options, use_upper=use_upper, batch_counts=counts)
    assert left == right
    assert cached.simulate(left, use_upper=use_upper, batch_counts=counts) == uncached.simulate(
        right, use_upper=use_upper, batch_counts=counts)
    assert cached.solve(options, use_upper=use_upper, batch_counts=counts) == right
    assert cached.simulation_cache_hits > 0
