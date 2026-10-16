from __future__ import annotations

from dataclasses import replace

import pytest

from splitfleet.server.placement.cosplit_ucb import (
    CandidateEstimate,
    GlobalPlacementSolver,
)


def _estimate(cid: str, boundary: str, *, arrival: float, service: float, tail: float = 0.0, feasible: bool = True):
    return CandidateEstimate(
        client_id=cid,
        boundary=boundary,
        client_forward_mean_ms=arrival,
        client_backward_mean_ms=tail,
        network_upload_mean_ms=0,
        network_download_mean_ms=0,
        server_service_mean_ms=service,
        switch_mean_ms=0,
        client_forward_uncertainty_ms=0,
        client_backward_uncertainty_ms=0,
        network_upload_uncertainty_ms=0,
        network_download_uncertainty_ms=0,
        server_service_uncertainty_ms=0,
        switch_uncertainty_ms=0,
        feasible=feasible,
        infeasible_reason=None if feasible else "memory",
    )


def test_infeasible_candidate_never_enters_solver() -> None:
    solver = GlobalPlacementSolver()
    result = solver.solve({"a": [_estimate("a", "bad", arrival=0, service=0, feasible=False), _estimate("a", "ok", arrival=1, service=1)]})
    assert result["a"].boundary == "ok"


def test_global_solver_avoids_independent_contention() -> None:
    solver = GlobalPlacementSolver(server_concurrency=1, max_coordinate_passes=3)
    estimates = {
        cid: [
            _estimate(cid, "early", arrival=0, service=50),
            _estimate(cid, "late", arrival=60, service=0),
        ]
        for cid in ("a", "b")
    }
    independent = {cid: options[0] for cid, options in estimates.items()}
    assignment = solver.solve(estimates)
    assert {value.boundary for value in assignment.values()} == {"early", "late"}
    assert solver.simulate(assignment).max_client_completion_ms < solver.simulate(independent).max_client_completion_ms


def test_queue_is_derived_by_lane_simulation() -> None:
    solver = GlobalPlacementSolver(server_concurrency=1)
    simulation = solver.simulate({
        "a": _estimate("a", "x", arrival=10, service=20),
        "b": _estimate("b", "x", arrival=15, service=5),
    })
    assert simulation.timelines["a"].queue_ms == 0
    assert simulation.timelines["b"].queue_ms == 15


def test_multiple_batches_repeat_service_but_switch_once() -> None:
    solver = GlobalPlacementSolver(server_concurrency=1)
    estimate = CandidateEstimate(
        client_id="a",
        boundary="x",
        client_forward_mean_ms=5,
        client_backward_mean_ms=2,
        network_upload_mean_ms=0,
        network_download_mean_ms=0,
        server_service_mean_ms=10,
        switch_mean_ms=3,
        client_forward_uncertainty_ms=0,
        client_backward_uncertainty_ms=0,
        network_upload_uncertainty_ms=0,
        network_download_uncertainty_ms=0,
        server_service_uncertainty_ms=0,
        switch_uncertainty_ms=0,
    )
    timeline = solver.simulate({"a": estimate}, batch_counts={"a": 3}).timelines["a"]
    assert timeline.arrival_ms == 8
    assert timeline.completion_ms == 54
    assert timeline.queue_ms == 0


@pytest.mark.parametrize("service", [0., .125, 10.])
def test_equal_arrivals_repeated_batches_and_parallel_lane_ties(service):
    solver = GlobalPlacementSolver(server_concurrency=2, simulation_cache_size=0)
    assignment = {cid: _estimate(cid, "x", arrival=0, service=service)
                  for cid in ("a", "b", "c", "d")}
    simulation = solver.simulate(assignment, batch_counts={cid: 2 for cid in assignment})
    for cid in ("a", "b"):
        timeline = simulation.timelines[cid]
        assert timeline.server_start_ms == 0
        assert timeline.completion_ms == 3 * service
        assert timeline.queue_ms == service
    for cid in ("c", "d"):
        timeline = simulation.timelines[cid]
        assert timeline.server_start_ms == service
        assert timeline.completion_ms == 4 * service
        assert timeline.queue_ms == 2 * service


@pytest.mark.parametrize("lanes,workers", [(2, 2), (13, 13), (20, 3), (20, None)])
@pytest.mark.parametrize("upper", [False, True])
def test_sufficient_lanes_match_independent_batch_costs(lanes, workers, upper):
    solver = GlobalPlacementSolver(server_concurrency=lanes, client_concurrency=workers)
    assignment = {str(i): replace(_estimate(str(i), "x", arrival=2, service=5, tail=11),
        network_upload_mean_ms=3, network_download_mean_ms=7,
        network_roundtrip_mean_ms=13, switch_mean_ms=17, state_exchange_mean_ms=19,
        server_service_uncertainty_ms=2) for i in range(20)}
    counts = {cid: int(cid) % 4 + 1 for cid in assignment}
    simulation = solver.simulate(assignment, use_upper=upper, batch_counts=counts)
    cycle = 41 + (2 if upper else 0)
    for cid, row in simulation.timelines.items():
        assert row.queue_ms == 0
        assert row.arrival_ms == row.server_start_ms == row.worker_start_ms + 22
        assert row.completion_ms == row.worker_start_ms + 17 + counts[cid] * cycle + 19
        assert row.server_finish_ms == row.arrival_ms + (counts[cid] - 1) * cycle + 5 + (2 if upper else 0)


def test_independent_worker_releases_can_precede_another_clients_first_arrival():
    solver = GlobalPlacementSolver(server_concurrency=2, client_concurrency=2)
    assignment = {cid: _estimate(cid, "x", arrival=100 if cid == "a" else 0, service=1)
                  for cid in ("a", "b", "c", "d")}
    solver.dispatch_order = tuple(assignment)
    simulation = solver.simulate(assignment)
    assert list(simulation.timelines) == ["b", "c", "d", "a"]
    assert simulation.timelines["c"].worker_start_ms == 1
    assert simulation.timelines["d"].worker_start_ms == 2
    assert simulation.objective == (101, 107)


def test_zero_duration_workers_preserve_causal_dispatch_order():
    solver = GlobalPlacementSolver(server_concurrency=2, client_concurrency=1)
    solver.dispatch_order = ("z", "a", "b")
    assignment = {cid: _estimate(cid, "x", arrival=0, service=0) for cid in solver.dispatch_order}
    simulation = solver.simulate(assignment, batch_counts={cid: 3 for cid in assignment})
    assert list(simulation.timelines) == ["z", "a", "b"]
    assert simulation.objective == (0, 0)


@pytest.mark.parametrize("cost", [
    "network_download_mean_ms",
    "client_backward_mean_ms",
    "network_roundtrip_mean_ms",
    "state_exchange_mean_ms",
])
def test_worker_waves_follow_task_completion_instead_of_suffix_order(cost):
    assignment = {
        "a": replace(_estimate("a", "x", arrival=0, service=1), **{cost: 100}),
        "b": _estimate("b", "x", arrival=0, service=1),
        "c": _estimate("c", "x", arrival=0, service=100),
        "d": _estimate("d", "x", arrival=0, service=1),
    }
    solver = GlobalPlacementSolver(server_concurrency=1, client_concurrency=2)
    solver.dispatch_order = tuple(assignment)
    simulation = solver.simulate(assignment)
    assert simulation.timelines["c"].worker_start_ms == 2
    assert simulation.timelines["d"].worker_start_ms == 101
    assert simulation.timelines["d"].queue_ms == 1
    assert simulation.max_client_completion_ms == 103


def test_worker_release_events_interleave_with_later_batches_and_upper_bounds():
    assignment = {
        "a": replace(_estimate("a", "x", arrival=0, service=1),
                     client_backward_uncertainty_ms=100),
        "b": _estimate("b", "x", arrival=0, service=1),
        "c": _estimate("c", "x", arrival=0, service=100),
        "d": _estimate("d", "x", arrival=0, service=1),
    }
    solver = GlobalPlacementSolver(server_concurrency=1, client_concurrency=2)
    solver.dispatch_order = tuple(assignment)
    simulation = solver.simulate(assignment, use_upper=True, batch_counts={"a": 2})
    assert simulation.timelines["c"].worker_start_ms == 2
    assert simulation.timelines["a"].server_finish_ms == 103
    assert simulation.timelines["a"].completion_ms == 203
    assert simulation.timelines["d"].worker_start_ms == 102
    assert simulation.max_client_completion_ms == 203
