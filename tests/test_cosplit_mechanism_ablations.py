import numpy as np

from experiments.ablations.no_cooperation import NoCooperation
from experiments.ablations.no_joint_solver import NoQueueSolver
from experiments.ablations.no_nonstationary import no_nonstationary_updates
from splitfleet.server.placement.cosplit_ucb import CoSplitUCBConfig, PlacementFeedback, StaticCandidateProvider
from splitfleet.server.placement.cosplit_ucb.solver import GlobalPlacementSolver
from tests.test_experimental_placement_baselines import cost
from tests.unit.test_cosplit_policy import _candidate


def test_no_joint_solver_omits_cross_client_queue_effects():
    options = {cid: [cost(cid, "early", 1, 10), cost(cid, "late", 6, 1)] for cid in ["a", "b"]}
    solver = NoQueueSolver()
    assignment = solver.solve(options)
    assert {value.boundary for value in assignment.values()} == {"late"}
    alone = solver.simulate(assignment)
    queued = GlobalPlacementSolver().simulate(assignment)
    assert all(value.queue_ms == 0 for value in alone.timelines.values())
    assert queued.max_client_completion_ms > alone.max_client_completion_ms


def test_no_cooperation_preserves_joint_solver_but_isolates_every_learned_component():
    provider = StaticCandidateProvider([_candidate("early", .25), _candidate("late", .75)])
    config = CoSplitUCBConfig(target_scale_ms=1, min_residence_rounds=0, max_explorations_per_round=0)
    policy = NoCooperation(candidate_provider=provider, config=config)
    chosen = policy.plan_round(round_id=1, client_ids=["a", "b"], training=True)
    before = policy.client_learners["b"].state_dict()
    policy.observe_round(round_id=1, feedback=[PlacementFeedback(round_id=1, client_id="a", boundary=chosen["a"],
        client_forward_ms=2, client_backward_ms=3, network_upload_ms=4, network_download_ms=5,
        server_service_ms=6, switch_ms=7, num_batches=1)])
    assert policy.client_learners["b"].state_dict() == before
    assert policy.client_learners["a"].server.model.num_updates == 1
    assert policy.client_learners["a"].switch.model.num_updates == 1
    assert type(policy.solver) is GlobalPlacementSolver
    saved = policy.state_dict()
    restored = NoCooperation(candidate_provider=provider, config=config)
    restored.load_state_dict(saved)
    assert restored.client_learners["a"].state_dict() == policy.client_learners["a"].state_dict()
    assert restored.plan_round(round_id=2, client_ids=["a", "b"], training=True) == policy.plan_round(
        round_id=2, client_ids=["a", "b"], training=True)


def test_no_discount_learners_keep_observations_across_idle_rounds():
    policy = no_nonstationary_updates(candidate_provider=StaticCandidateProvider([_candidate("early", .25)]))
    chosen = policy.plan_round(round_id=1, client_ids=["a"], training=True)
    policy.observe_round(round_id=1, feedback=[PlacementFeedback(round_id=1, client_id="a", boundary=chosen["a"],
        client_forward_ms=20, server_service_ms=10, num_batches=1)])
    matrix, targets = policy.learners.server.model.A.copy(), policy.learners.server.model.b.copy()
    policy.learners.advance_round(20)
    np.testing.assert_array_equal(policy.learners.server.model.A, matrix)
    np.testing.assert_array_equal(policy.learners.server.model.b, targets)
    assert policy.config.discount_gamma == 1.0
