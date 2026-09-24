from __future__ import annotations

from splitfleet.server.placement.cosplit_ucb import CandidateEstimate, GlobalPlacementSolver


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
