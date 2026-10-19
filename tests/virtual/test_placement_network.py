import json

import pytest

from experiments.virtual_sfl.workloads import fixture_spec
from splitfleet.virtual import VirtualConfig, run_virtual
from splitfleet.virtual.config import ClientPopulation, Network, Resources




@pytest.mark.simulation
def test_native_greedy_uses_measured_feedback_on_actual_paced_path(tmp_path):
    pytest.importorskip("ray")
    config = VirtualConfig(
        rounds=3,
        batch_size=2,
        boundary="after:head",
        placement="greedy",
        virtual_clients=ClientPopulation(total=2),
        resources=Resources(max_concurrent_clients=2),
        networks={"0": Network(upload_mbps=128.0, download_mbps=128.0, rtt_ms=2.0)},
        network_by_round={
            2: {"0": Network(upload_mbps=16.0, download_mbps=32.0, rtt_ms=10.0)},
            3: {"0": Network(upload_mbps=128.0, download_mbps=128.0, rtt_ms=2.0)},
        },
    )
    output = tmp_path / "greedy-network"
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
        assert metrics["server_load_active_jobs"] >= 1
        assert metrics["server_load_execution_concurrency"] == config.resources.effective_server_concurrency
    traces = [
        json.loads(row)
        for row in (output / "placement_trace.jsonl").read_text().splitlines()
    ]
    assert len(traces) == 3 and all(row["policy"] == "greedy" for row in traces)
    assert all(len(row["diagnostics"]["feedback_clients"]) == 2 for row in traces)
    assert all(row["diagnostics"]["placement_objective"] == "predicted_solo_round_cost" for row in traces)
    assert all(row["diagnostics"]["latency_model_uses_live_network_context"] for row in traces)
    from splitfleet.common.network_probe import LARGE_BYTES, estimate_network
    manifest = json.loads((output / "client_manifest.json").read_text())
    paced_cid = next(row["node_id"] for row in manifest if row["virtual_partition_id"] == 0)
    contexts = [row["diagnostics"]["network_contexts"][paced_cid] for row in traces]
    assert contexts[1]["upload_mbps"] < min(contexts[0]["upload_mbps"], contexts[2]["upload_mbps"]) / 2
    assert contexts[1]["download_mbps"] < min(contexts[0]["download_mbps"], contexts[2]["download_mbps"]) / 1.5
    assert contexts[1]["rtt_ms"] > max(contexts[0]["rtt_ms"], contexts[2]["rtt_ms"])
    for row in traces:
        assert len(row["diagnostics"]["network_probes"]) == 2
        for cid, receipt in row["diagnostics"]["network_probes"].items():
            assert row["diagnostics"]["network_contexts"][cid] == estimate_network(receipt, round_id=row["round_id"])
            assert receipt["source"] == "measured_split_rpc"
            assert receipt["coordinator_elapsed_sec"] > 0
            if cid == paced_cid:
                assert receipt["tcp_counters"]["tcp_upload_bytes"] > 3 * LARGE_BYTES
                assert receipt["tcp_counters"]["tcp_download_bytes"] > 3 * LARGE_BYTES
    assert all(row["diagnostics"]["server_load"]["source"] == "measured_server_runtime" for row in traces)
    assert traces[1]["diagnostics"]["queue_wait_observed"]
    assert traces[1]["diagnostics"]["server_load"]["queue_observation_round"] == 1
    assert (output / "greedy_state.json").is_file()
