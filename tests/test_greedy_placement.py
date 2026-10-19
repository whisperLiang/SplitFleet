"""Round units, private costs and feedback of the default placement policy."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from splitfleet.server.placement import (
    GreedyConfig, GreedyPlacementPolicy, PlacementFeedback, StaticCandidateProvider,
)
from splitfleet.server.placement.greedy.calibration import ClientTelemetry
from splitfleet.common.network_probe import PROBE_CONFIG_KEY, PROBE_RESULT_KEY
from tests.unit.placement_fixtures import _candidate, _network_probe_receipt
import json


def make_policy(*, state_path=None):
    early = replace(_candidate("early", .25), prefix_parameter_bytes=100,
                    metadata={"optimizer_prefix_parameter_bytes": 100,
                              "boundary_forward_bytes_by_batch_size": {2: 2000}})
    late = replace(_candidate("late", .75), prefix_parameter_bytes=300,
                   metadata={"optimizer_prefix_parameter_bytes": 300,
                             "boundary_forward_bytes_by_batch_size": {2: 2000}})
    clients = {cid: {"logical_client_id": cid, "batch_size": 2, "num_batches": count}
               for cid, count in (("a", 3), ("b", 5))}
    receipt = {"server_calibration": {"records": [
        {"boundary": "early", "local_tail_service_ms": 500},
        {"boundary": "late", "local_tail_service_ms": 100}]},
        "client_samples": [{"logical_client_id": cid, "boundary": boundary,
            "client_forward_ms": front, "client_backward_ms": front}
            for cid in clients for boundary, front in (("early", 50), ("late", 100))],
        "transport_bootstrap": {"samples": [{"logical_client_id": cid,
            "payload_bytes": size, "roundtrip_ms": 10 + 2 * size / 1000}
            for cid in clients for size in (65536, 2097152)]},
        "state_exchange_bootstrap": {"samples": [{"logical_client_id": cid,
            "request_bytes": request, "reply_bytes": reply,
            "roundtrip_ms": 20 + (request + reply) / 1000}
            for cid in clients for request, reply in ((65536, 65536), (2097152, 65536))]}}
    telemetry = ClientTelemetry()
    telemetry.clients, telemetry.receipt = clients, receipt
    policy = GreedyPlacementPolicy(candidate_provider=StaticCandidateProvider([early, late]),
        config=GreedyConfig(state_path=state_path, warm_start=False), telemetry_provider=telemetry)
    telemetry.policy = policy
    policy.state_download_bytes = 400
    return policy


def test_round_cost_counts_batch_components_per_batch_and_state_once():
    policy = make_policy()
    assert policy.plan_round(round_id=1, client_ids=["b", "a"], training=True) == {"a": "late", "b": "late"}
    score = policy.round_diagnostics[1]["estimates"]["a"]["late"]["round_seconds"]
    # 3 * (front .2 + tail .1 + intercept .01 + 4000 bytes * 1e-6)
    # plus one state exchange: .02 + 700 bytes * 1e-6.
    assert score == pytest.approx(3 * (.2 + .1 + .01 + .004) + .02 + .0007)
    assert policy.cost_tables["a"] is not policy.cost_tables["b"]


def test_successful_feedback_changes_only_that_clients_predictions():
    policy = make_policy()
    policy.plan_round(round_id=1, client_ids=["a", "b"], training=True)
    observation = PlacementFeedback(round_id=1, client_id="a", boundary="late", num_batches=3,
        client_forward_ms=1500, client_backward_ms=1500, server_service_ms=100,
        network_roundtrip_ms=14, state_exchange_ms=20.7)
    policy.observe_round(round_id=1, feedback=[observation])
    assert policy.plan_round(round_id=2, client_ids=["a", "b"], training=True) == {"a": "early", "b": "late"}
    assert policy.cost_tables["b"]["observed_front"] == {}


def test_evaluation_does_not_bootstrap_or_update_training_state():
    policy = make_policy()
    policy.plan_round(round_id=1, client_ids=["a"], training=False)
    assert policy.profiles is None and not policy.cost_tables and not policy.decision_receipts
    assert policy.plan_round(round_id=1, client_ids=[], training=True) == {}


def test_failure_masks_candidate_without_making_a_latency_sample():
    policy = make_policy()
    policy.plan_round(round_id=1, client_ids=["a"], training=True)
    policy.observe_round(round_id=1, feedback=[PlacementFeedback(
        round_id=1, client_id="a", boundary="late", success=False)])
    assert policy.cost_tables["a"]["observed_front"] == {}
    assert policy.plan_round(round_id=2, client_ids=["a"], training=True) == {"a": "early"}


def test_round_plan_is_cached_and_cannot_change_selected_clients():
    policy = make_policy()
    first = policy.plan_round(round_id=1, client_ids=["a"], training=True)
    assert policy.plan_round(round_id=1, client_ids=["a"], training=True) == first
    assert len(policy.decision_receipts) == 1
    with pytest.raises(ValueError, match="client set"):
        policy.plan_round(round_id=1, client_ids=["a", "b"], training=True)


def test_live_network_context_reselects_after_recovery():
    policy = make_policy()
    early, late = policy.candidate_provider.get_candidates(training=True)
    # Earlier offloading saves compute but transfers a larger activation and
    # gradient. A slow link should move the cut later; recovery should undo it.
    early = replace(early, boundary_forward_bytes=1_000_000,
                    boundary_gradient_bytes=1_000_000,
                    metadata={**early.metadata,
                              "boundary_forward_bytes_by_batch_size": {2: 2_000_000}})
    policy.candidate_provider = StaticCandidateProvider([early, late])
    policy.telemetry_provider.receipt["server_calibration"]["records"][0][
        "local_tail_service_ms"
    ] = 100

    class MeasuredClient:
        cid = "a"

        def __init__(self, values):
            self.values = values
            self.probed_rounds = []

        def get_properties(self, ins, timeout, group_id):
            assert ins.config[PROBE_CONFIG_KEY] == group_id
            self.probed_rounds.append(group_id)
            return SimpleNamespace(properties={PROBE_RESULT_KEY: json.dumps(
                _network_probe_receipt(group_id, **self.values))})

    network = MeasuredClient({"upload_mbps": 1024, "download_mbps": 1024, "rtt_ms": 2})
    policy.bind_clients([network], round_id=1)
    assert policy.plan_round(round_id=1, client_ids=["a"], training=True) == {"a": "early"}
    assert policy.round_diagnostics[1]["network_contexts"]["a"] == pytest.approx({
        "upload_mbps": 1024.0, "download_mbps": 1024.0, "rtt_ms": 2.0,
    })
    assert policy.round_diagnostics[1]["latency_model_uses_live_network_context"]
    assert not policy.round_diagnostics[1]["latency_model_uses_live_load_features"]
    policy.observe_round(round_id=1, feedback=[PlacementFeedback(
        round_id=1, client_id="a", boundary="early", num_batches=3,
        client_forward_ms=50, client_backward_ms=50, server_service_ms=100,
        network_roundtrip_ms=43.25, state_exchange_ms=20.004)])

    network.values = {"upload_mbps": 128, "download_mbps": 256, "rtt_ms": 15}
    policy.bind_clients([network], round_id=2)
    assert policy.plan_round(round_id=2, client_ids=["a"], training=True) == {"a": "late"}
    assert policy.round_diagnostics[2]["network_contexts"]["a"] == pytest.approx({
        "upload_mbps": 128.0, "download_mbps": 256.0, "rtt_ms": 15.0,
    })
    policy.observe_round(round_id=2, feedback=[PlacementFeedback(
        round_id=2, client_id="a", boundary="late", num_batches=3,
        client_forward_ms=100, client_backward_ms=100, server_service_ms=100,
        network_roundtrip_ms=25.1875, state_exchange_ms=20.032)])

    network.values = {"upload_mbps": 1024, "download_mbps": 1024, "rtt_ms": 2}
    assert policy.plan_round(round_id=2, client_ids=["a"], training=True) == {"a": "late"}
    policy.bind_clients([network], round_id=3)
    assert policy.plan_round(round_id=3, client_ids=["a"], training=True) == {"a": "early"}
    assert network.probed_rounds == [1, 2, 3]
    assert policy.round_diagnostics[3]["network_contexts"] == policy.round_diagnostics[1]["network_contexts"]
    for boundary in ("early", "late"):
        assert policy.round_diagnostics[3]["estimates"]["a"][boundary]["round_seconds"] == pytest.approx(
            policy.round_diagnostics[1]["estimates"]["a"][boundary]["round_seconds"])


def test_live_cost_uses_measured_directional_rates_and_counts_state_once():
    policy = make_policy()
    policy.telemetry_provider.round_probes[1] = {"a": _network_probe_receipt(
        1, upload_mbps=2, download_mbps=8, rtt_ms=5)}
    policy.runtime_context_provider = SimpleNamespace(client_context=lambda cid: {
        "uplink_mbps": 100, "downlink_mbps": 100, "rtt_ms": 1,
    })
    policy.telemetry_provider.clients["a"].update({
        "upload_mbps": 100, "download_mbps": 100, "rtt_ms": 1,
    })
    policy.plan_round(round_id=1, client_ids=["a"], training=True)
    assert policy.round_diagnostics[1]["network_contexts"]["a"] == pytest.approx({
        "upload_mbps": 2.0, "download_mbps": 8.0, "rtt_ms": 5.0,
    })
    score = policy.round_diagnostics[1]["estimates"]["a"]["late"]["round_seconds"]
    # 2000 activation bytes up / 2000 gradient bytes down per batch;
    # 300 prefix-state bytes up / 400 model-state bytes down once per round.
    assert score == pytest.approx(3 * (.2 + .1 + .008 + .002 + .005)
                                 + .005 + .0012 + .0004)


@pytest.mark.parametrize("context", [
    {},
    {"upload_mbps": 100, "download_mbps": 100, "rtt_ms": 1},
    {"upload_mbps": 10},
    {"upload_mbps": 0, "download_mbps": 10},
    {"upload_mbps": float("nan"), "download_mbps": 10},
    {"upload_mbps": 10, "download_mbps": float("inf")},
    {"upload_mbps": 10, "download_mbps": 10, "rtt_ms": -1},
    {"upload_mbps": 10, "download_mbps": 10, "rtt_ms": None},
])
def test_configured_network_context_is_never_used_as_a_measurement(context):
    policy = make_policy()
    policy.plan_round(round_id=1, client_ids=["a"], training=True)
    policy.telemetry_provider.clients["a"].update(context)
    policy.runtime_context_provider = SimpleNamespace(client_context=lambda cid: context)
    assert policy.plan_round(round_id=2, client_ids=["a"], training=True) == {"a": "late"}
    assert policy.round_diagnostics[2]["estimates"] == policy.round_diagnostics[1]["estimates"]
    assert policy.round_diagnostics[2]["network_contexts"] == {}
    assert not policy.round_diagnostics[2]["latency_model_uses_live_network_context"]


def test_previous_round_probe_is_not_reused_as_current_network_context():
    policy = make_policy()
    policy.telemetry_provider.round_probes[1] = {"a": _network_probe_receipt(
        1, upload_mbps=2, download_mbps=8, rtt_ms=5)}
    policy.plan_round(round_id=2, client_ids=["a"], training=True)
    assert policy.round_diagnostics[2]["network_contexts"] == {}


def test_state_keeps_costs_and_rejects_other_algorithms():
    source = make_policy()
    source.plan_round(round_id=1, client_ids=["a"], training=True)
    state = source.state_dict()
    target = make_policy()
    target.load_state_dict(state)
    assert target.plan_round(round_id=2, client_ids=["a"], training=True) == {"a": "late"}
    with pytest.raises(ValueError, match="identity"):
        target.load_state_dict({**state, "algorithm": "cosplit_ucb"})


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_execution_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        GreedyConfig(server_concurrency=capacity)
