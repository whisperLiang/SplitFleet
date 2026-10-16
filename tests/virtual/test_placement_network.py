import json

import pytest

from experiments.virtual_sfl.workloads import fixture_spec
from splitfleet.server.placement.cosplit_ucb import GlobalPlacementSolver
from splitfleet.virtual import VirtualConfig, run_virtual
from splitfleet.virtual.config import ClientPopulation, Network, Resources
from tests.unit.test_cosplit_solver import _estimate


def test_production_solver_models_worker_waves_separately_from_suffix_queue():
    assignment = {
        cid: _estimate(cid, "cut", arrival=2, service=5) for cid in ("a", "b", "c", "d")
    }
    solver = GlobalPlacementSolver(server_concurrency=1, client_concurrency=2)
    solver.dispatch_order = tuple(assignment)
    bounded = solver.simulate(assignment)
    assert (
        bounded.timelines["a"].worker_start_ms
        == bounded.timelines["b"].worker_start_ms
        == 0
    )
    assert bounded.timelines["c"].worker_start_ms == 7
    assert bounded.timelines["d"].worker_start_ms == 12
    assert bounded.timelines["b"].queue_ms == 5
    # Worker start is a separate admission condition, never a suffix service target.
    assert bounded.timelines["c"].arrival_ms == 9
    physical = GlobalPlacementSolver(server_concurrency=1).simulate(assignment)
    assert all(t.worker_start_ms == 0 for t in physical.timelines.values())


@pytest.mark.simulation
def test_native_cosplit_uses_measured_feedback_on_actual_paced_path(tmp_path):
    pytest.importorskip("ray")
    config = VirtualConfig(
        rounds=3,
        batch_size=2,
        boundary="after:head",
        placement="cosplit_ucb",
        virtual_clients=ClientPopulation(total=2),
        resources=Resources(max_concurrent_clients=2),
        networks={"0": Network(upload_mbps=1.0, download_mbps=2.0, rtt_ms=10.0)},
        network_by_round={
            2: {"0": Network(upload_mbps=10.0, download_mbps=10.0, rtt_ms=2.0)}
        },
    )
    output = tmp_path / "cosplit-network"
    result = run_virtual(config, fixture_spec(samples=8), output=output)
    assert result["training_tasks"] == 6
    rows = [
        json.loads(row)
        for row in (output / "client_metrics.jsonl").read_text().splitlines()
    ]
    for row in rows:
        if row["status"] != "success":
            continue
        metrics = row["metrics"]
        assert metrics["activation_upload_bytes"] > 0
        assert metrics["gradient_download_bytes"] > 0
        if row["partition_id"] == 0:
            assert metrics["virtual_state_via_rpc"]
            assert metrics["tcp_upload_bytes"] > metrics["activation_upload_bytes"]
            assert metrics["tcp_download_bytes"] > metrics["gradient_download_bytes"]
            assert metrics["tcp_upload_pacing_wait_sec"] > 0
            assert metrics["model_state_download_bytes"] > 0
        assert metrics["client_forward_ms"] > 0
        assert metrics["server_service_ms"] > 0
        assert metrics["suffix_server_queue_sec"] >= 0
    traces = [
        json.loads(row)
        for row in (output / "placement_trace.jsonl").read_text().splitlines()
    ]
    assert len(traces) == 3 and all(row["policy"] == "cosplit_ucb" for row in traces)
    assert all(row["diagnostics"].get("learner_update_counts") for row in traces)
    assert (output / "cosplit_state.json").is_file()
