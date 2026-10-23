"""Fresh directional measurements, real RPC traffic and probe-only auth."""

import copy
import json
from threading import Lock
import time
from types import SimpleNamespace

import grpc
import pytest

from splitfleet.common import network_probe
from splitfleet.common.network_probe import (
    LARGE_BYTES, SMALL_BYTES, PROBE_CONFIG_KEY, PROBE_METHOD, PROBE_RESULT_KEY,
    PROBE_ATTEMPTS, PROBE_REPEATS, UnresolvedNetworkProbe,
    estimate_network, measure_network, probe_response,
)
from splitfleet.common import BatchData, ControlCode
from splitfleet.server.grpc.servicer import ServerModelServicer
from splitfleet.server.placement.greedy.calibration import ClientTelemetry
from splitfleet.virtual.config import Network
from splitfleet.virtual.network import TaskNetwork
from splitfleet.virtual.server_adapter import RuntimeConditions
from splitfleet.virtual.suffix_service import LeasedSuffixServicer, SuffixService, suffix_connection
from tests.unit.placement_fixtures import _network_probe_receipt


def test_directional_estimates_use_measured_rpc_durations():
    receipt = _network_probe_receipt(4, upload_mbps=20, download_mbps=80, rtt_ms=7)
    assert estimate_network(receipt, round_id=4) == pytest.approx({
        "upload_mbps": 20, "download_mbps": 80, "rtt_ms": 7})


@pytest.mark.parametrize("fault", ["stale", "configured_source", "missing_sample", "nan", "unresolved"])
def test_invalid_or_stale_samples_cannot_become_current_context(fault):
    receipt = _network_probe_receipt(1, upload_mbps=20, download_mbps=80, rtt_ms=7)
    if fault == "stale":
        receipt["round_id"] = 0
    elif fault == "configured_source":
        receipt["source"] = "configured_pacing_rates"
    elif fault == "missing_sample":
        receipt["samples"].pop()
    elif fault == "nan":
        receipt["samples"][0]["roundtrip_ms"] = float("nan")
    else:
        for sample in receipt["samples"]:
            sample["roundtrip_ms"] = 7
    with pytest.raises(ValueError, match="Active network probe"):
        estimate_network(receipt, round_id=1)


def test_probe_calls_split_rpc_and_uses_one_monotonic_clock(monkeypatch):
    now_ns, requests = [0], []

    class Proxy:
        def _blocking_request(self, method, request, *, _streams_, _timeout_):
            assert method == PROBE_METHOD and not _streams_
            request_bytes = len(request.data["payload"])
            reply_bytes = int(request.metadata["reply_bytes"])
            elapsed_ms = 7 + (request_bytes - SMALL_BYTES) * 8 / 20_000 \
                           + (reply_bytes - SMALL_BYTES) * 8 / 80_000
            now_ns[0] += round(elapsed_ms * 1e6)
            requests.append((request_bytes, reply_bytes))
            return probe_response(request.data, request.metadata)

    monkeypatch.setattr(network_probe.time, "perf_counter_ns", lambda: now_ns[0])
    receipt = measure_network(Proxy(), round_id=3)
    shapes = {(SMALL_BYTES, SMALL_BYTES), (LARGE_BYTES, SMALL_BYTES),
              (SMALL_BYTES, LARGE_BYTES)}
    assert {(row['request_bytes'], row['reply_bytes'])
            for row in receipt['warmup_samples']} == shapes
    assert len(requests) == 3 + 3 * PROBE_REPEATS
    assert all(requests.count(shape) == 1 + PROBE_REPEATS for shape in shapes)
    assert estimate_network(receipt, round_id=3) == pytest.approx({
        "upload_mbps": 20, "download_mbps": 80, "rtt_ms": 7})


def test_probe_retries_unresolved_samples_and_retains_rejected_observations(monkeypatch):
    now_ns, calls = [0], []

    class Proxy:
        def _blocking_request(self, method, request, **kwargs):
            request_bytes = len(request.data["payload"])
            reply_bytes = int(request.metadata["reply_bytes"])
            # The first set has small-RPC queuing delay exceeding the large
            # RPC duration. The next set observes an uncongested channel.
            if 3 <= len(calls) < 3 + 3 * PROBE_REPEATS:
                elapsed_ms = 30 if request_bytes == reply_bytes == SMALL_BYTES else 10
            else:
                elapsed_ms = 7 + (request_bytes - SMALL_BYTES) * 8 / 20_000 \
                               + (reply_bytes - SMALL_BYTES) * 8 / 80_000
            now_ns[0] += round(elapsed_ms * 1e6)
            calls.append(elapsed_ms)
            return probe_response(request.data, request.metadata)

    monkeypatch.setattr(network_probe.time, "perf_counter_ns", lambda: now_ns[0])
    monkeypatch.setattr(network_probe.time, "perf_counter", lambda: now_ns[0] / 1e9)
    receipt = measure_network(Proxy(), round_id=1)
    assert len(calls) == 3 + 2 * 3 * PROBE_REPEATS
    assert len(receipt["rejected_attempts"]) == 1
    assert [row["roundtrip_ms"] for row in receipt["rejected_attempts"][0]["samples"]] == [30, 10, 10] * PROBE_REPEATS
    assert receipt["elapsed_sec"] == pytest.approx(sum(calls) / 1000)
    assert estimate_network(receipt, round_id=1) == pytest.approx({
        "upload_mbps": 20, "download_mbps": 80, "rtt_ms": 7})


def test_permanently_unresolved_channel_still_fails_after_bounded_retries(monkeypatch):
    now_ns, calls = [0], []

    class Proxy:
        def _blocking_request(self, method, request, **kwargs):
            now_ns[0] += 7_000_000
            calls.append(request)
            return probe_response(request.data, request.metadata)

    monkeypatch.setattr(network_probe.time, "perf_counter_ns", lambda: now_ns[0])
    with pytest.raises(UnresolvedNetworkProbe, match="could not resolve"):
        measure_network(Proxy(), round_id=1)
    assert len(calls) == 3 + PROBE_ATTEMPTS * 3 * PROBE_REPEATS


def test_invalid_reply_is_not_retried(monkeypatch):
    now_ns, calls = [0], []

    class Proxy:
        def _blocking_request(self, method, request, **kwargs):
            now_ns[0] += 7_000_000
            calls.append(request)
            response = probe_response(request.data, request.metadata)
            response.metadata["round_id"] = "0"
            return response

    monkeypatch.setattr(network_probe.time, "perf_counter_ns", lambda: now_ns[0])
    with pytest.raises(ValueError, match="reply failed validation"):
        measure_network(Proxy(), round_id=1)
    assert len(calls) == 1


def test_clients_probe_without_competing_for_the_rpc_service():
    telemetry = ClientTelemetry(policy=SimpleNamespace(
        uses_active_network_probes=True, client_concurrency=4))
    channel = Lock()

    def properties(ins, timeout, group_id):
        assert channel.acquire(blocking=False), "Network probes overlap on the shared channel"
        try:
            time.sleep(.01)
            return SimpleNamespace(properties={PROBE_RESULT_KEY: json.dumps(_network_probe_receipt(
                group_id, upload_mbps=20, download_mbps=80, rtt_ms=7))})
        finally:
            channel.release()

    clients = [SimpleNamespace(cid=str(i), get_properties=properties) for i in range(4)]
    telemetry.refresh_transport(clients, 1)
    assert set(telemetry.round_probes[1]) == {client.cid for client in clients}


def test_telemetry_refreshes_once_per_selected_client_per_round():
    telemetry = ClientTelemetry(policy=SimpleNamespace(uses_active_network_probes=True))
    seen = []

    def properties(ins, timeout, group_id):
        assert ins.config[PROBE_CONFIG_KEY] == group_id
        seen.append(group_id)
        return SimpleNamespace(properties={PROBE_RESULT_KEY: json.dumps(_network_probe_receipt(
            group_id, upload_mbps=20 * group_id, download_mbps=80, rtt_ms=7))})

    client = SimpleNamespace(cid="a", get_properties=properties)
    telemetry.refresh_transport([client], 1)
    telemetry.refresh_transport([client], 1)
    telemetry.refresh_transport([client], 2)
    assert seen == [1, 2]
    assert telemetry.network_context("a", 1)["upload_mbps"] == pytest.approx(20)
    assert telemetry.network_context("a", 2)["upload_mbps"] == pytest.approx(40)
    assert telemetry.network_context("a", 3) is None


def test_failed_cohort_does_not_publish_partial_or_old_probes():
    telemetry = ClientTelemetry(policy=SimpleNamespace(uses_active_network_probes=True))
    fresh = _network_probe_receipt(2, upload_mbps=20, download_mbps=80, rtt_ms=7)
    stale = copy.deepcopy(fresh)
    stale["round_id"] = 1
    clients = [SimpleNamespace(cid=cid, get_properties=lambda *args, _receipt=receipt, **kwargs:
        SimpleNamespace(properties={PROBE_RESULT_KEY: json.dumps(_receipt)}))
        for cid, receipt in (("a", fresh), ("b", stale))]
    with pytest.raises(ValueError, match="current round"):
        telemetry.refresh_transport(clients, 2)
    assert telemetry.round_probes == {}


def test_runtime_context_does_not_access_emulator_network_settings():
    assert RuntimeConditions(SimpleNamespace()).client_context("a") == {}


def test_physical_property_handler_measures_actual_paced_split_channel():
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient

    # No strategy/model exists: network measurements must finish before a cut
    # or suffix replica is constructed.
    manager = SimpleNamespace()
    servicer = ServerModelServicer(manager)
    service = SuffixService(manager, servicer=servicer).start()
    try:
        with TaskNetwork(service.address, Network(upload_mbps=64, download_mbps=128, rtt_ms=5),
                         round_id=1) as path:
            with suffix_connection(path.address, "a") as proxy:
                client = AutoSplitSplitLearningClient.__new__(AutoSplitSplitLearningClient)
                client.server_model_proxy = proxy
                receipt = json.loads(client.get_properties({PROBE_CONFIG_KEY: 1})[PROBE_RESULT_KEY])
                context = estimate_network(receipt, round_id=1)
                assert 0 < context["upload_mbps"] < 64 * 1.3
                assert context["download_mbps"] > context["upload_mbps"] * 1.2
                assert context["rtt_ms"] >= 5
                counters = path.metrics()
                assert counters["tcp_upload_bytes"] > 3 * LARGE_BYTES
                assert counters["tcp_download_bytes"] > 3 * LARGE_BYTES
                assert servicer.active_executions == 0
    finally:
        service.stop()


def test_virtual_probe_tokens_cannot_start_training_and_expire_after_selection():
    manager = SimpleNamespace(_delegate=SimpleNamespace(live={}))
    servicer = LeasedSuffixServicer(manager, capacity=1, concurrency=1, state_store=None, seed=3)
    servicer.prepare_network_probes(["a"], 2)
    service = SuffixService(manager, servicer=servicer).start()
    request = BatchData(data={"payload": bytes(SMALL_BYTES)}, control_code=ControlCode.OK,
                        metadata={"reply_bytes": str(SMALL_BYTES), "round_id": "2"})
    try:
        token = servicer.network_probe_tokens["a"]
        with suffix_connection(service.address, "a", metadata=(("authorization", f"Bearer {token}"),)) as proxy:
            response = proxy._blocking_request(PROBE_METHOD, request, _streams_=False, _timeout_=2)
            assert len(response.data["payload"]) == SMALL_BYTES
            with pytest.raises(grpc.RpcError) as denied:
                proxy.virtual_acquire(BatchData(data={}, control_code=ControlCode.OK), _streams_=False)
            assert denied.value.code() == grpc.StatusCode.UNAUTHENTICATED
            servicer.network_probe_tokens.clear()
            with pytest.raises(grpc.RpcError) as expired:
                proxy._blocking_request(PROBE_METHOD, request, _streams_=False, _timeout_=2)
            assert expired.value.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        service.stop()
