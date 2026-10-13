from dataclasses import replace

from experiments.baselines.independent_ucb import IndependentUCB, _IndependentSolver
from splitfleet.server.placement.cosplit_ucb import CoSplitUCBConfig, PlacementFeedback, StaticCandidateProvider
from splitfleet.server.placement.cosplit_ucb.solver import GlobalPlacementSolver
from splitfleet.server.placement.cosplit_ucb.types import CandidateEstimate
from tests.unit.test_cosplit_policy import _candidate


def independent():
    return IndependentUCB(candidate_provider=StaticCandidateProvider([_candidate("early", .25), _candidate("late", .75)]),
        config=CoSplitUCBConfig(target_scale_ms=1, min_residence_rounds=0, max_explorations_per_round=0))


def test_independent_deterministic_selection_and_original_descriptors():
    left, right = independent(), independent()
    assert left.plan_round(round_id=1, client_ids=["b", "a"], training=True) == right.plan_round(
        round_id=1, client_ids=["a", "b"], training=True)
    for policy in left.clients.values():
        assert policy.candidate_provider is left.candidate_provider
        assert set(policy._catalog()) == set(left.candidate_provider.get_candidates(training=True))


def test_independent_feedback_does_not_update_any_other_client():
    policy = independent()
    first = policy.plan_round(round_id=1, client_ids=["a", "b"], training=True)
    before = policy.clients["b"].learners.state_dict()
    value = PlacementFeedback(round_id=1, client_id="a", boundary=first["a"],
        client_forward_ms=100, client_backward_ms=20, server_service_ms=30, num_batches=1)
    policy.observe_round(round_id=1, feedback=[value])
    assert policy.clients["b"].learners.state_dict() == before
    assert policy.clients["a"].learners.server.model.num_updates == 1
    policy.observe_round(round_id=1, feedback=[value])
    assert policy.clients["a"].learners.server.model.num_updates == 1
    assert set(policy.plan_round(round_id=2, client_ids=["a", "b"], training=True)) == {"a", "b"}


def cost(cid, boundary, edge, server):
    return CandidateEstimate(cid, boundary, edge, 0, 0, 0, server, 0, 0, 0, 0, 0, 0, 0)






def test_independent_cost_score_matches_production_for_one_client():
    options = {"a": [replace(cost("a", "early", 6, 1), client_forward_uncertainty_ms=100),
                      replace(cost("a", "late", 1, 1), client_forward_uncertainty_ms=1)]}
    for use_upper in (False, True):
        for batches in (1, 3):
            independent = _IndependentSolver().solve(options, use_upper=use_upper, batch_counts={"a": batches})
            production = GlobalPlacementSolver().solve(options, use_upper=use_upper, batch_counts={"a": batches})
            assert independent == production
            assert independent["a"].boundary == "late"
