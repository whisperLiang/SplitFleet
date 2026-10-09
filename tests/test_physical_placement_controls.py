from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np

from experiments.physical_placement import PhysicalIndependentUCB
from experiments.tcp_pacing import PacedProxy
from splitfleet.server.placement.cosplit_ucb.calibration import CalibratedTelemetry
from tests.test_cosplit_calibration import _bootstrap_policy, _receipt


def test_physical_independent_bootstrap_and_updates_remain_private(tmp_path):
    base, worker, props = _bootstrap_policy(tmp_path)
    provider = base.candidate_provider
    policy = PhysicalIndependentUCB(candidate_provider=provider, config=base.config)
    policy.telemetry_provider = CalibratedTelemetry(initial_model_hash="initial",
        server_receipt=_receipt(provider.get_candidates(training=True), server=True), policy=policy.bootstrap)
    other_props = {**props, "logical_client_id": "b"}
    def properties(ins, **kwargs):
        if "cosplit_probe_payload" in ins.config:
            reply = ins.config.get("cosplit_probe_reply_bytes")
            return SimpleNamespace(properties={"cosplit_probe_payload":
                ins.config["cosplit_probe_payload"] if reply is None else bytes(reply)})
        return SimpleNamespace(properties=other_props)
    other = SimpleNamespace(cid="b", get_properties=properties)
    policy.bind_clients([worker, other], 1)
    assert set(policy.telemetry_provider.clients) == {"a", "b"}
    assert len(policy.telemetry_provider.receipt["client_samples"]) == 6
    left, right = policy.clients["a"], policy.clients["b"]
    assert left.learners is not right.learners
    assert set(left.telemetry_provider.clients) == {"a"}
    assert set(right.telemetry_provider.clients) == {"b"}
    before = right.learners.server.model.b.copy()
    left.learners.server.model.update(np.ones(left.context_encoder.server_dimension), 100, round_id=1)
    np.testing.assert_array_equal(before, right.learners.server.model.b)
    policy.bind_clients([worker, other], 2)
    assert left.learners.server.model.num_updates == 4
    assert right.learners.server.model.num_updates == 3
    policy.state_download_bytes = 1024
    assert set(policy.plan_round(round_id=2, client_ids=["a", "b"], training=True)) == {"a", "b"}
    assert left.state_download_bytes == right.state_download_bytes == 1024


def test_tcp_pacing_shared_rate_applies_to_actual_concurrent_streams(tmp_path):
    async def execute():
        import time
        control = tmp_path / "control.json"
        control.write_text(json.dumps({"round_id": 6, "cap_mbps": 4}))
        async def echo(reader, writer):
            while block := await reader.read(16384):
                writer.write(block)
                await writer.drain()
            writer.close()
            await writer.wait_closed()
        target = await asyncio.start_server(echo, "127.0.0.1", 0)
        proxy = PacedProxy(target=target.sockets[0].getsockname(), control=control, output=tmp_path / "receipt.json")
        server = await asyncio.start_server(proxy.handle, "127.0.0.1", 0)
        payload = bytes(range(256)) * 256
        async def transfer():
            reader, writer = await asyncio.open_connection(*server.sockets[0].getsockname())
            writer.write(payload)
            await writer.drain()
            value = await reader.readexactly(len(payload))
            writer.close()
            await writer.wait_closed()
            assert value == payload
        started = time.perf_counter()
        await asyncio.gather(transfer(), transfer())
        elapsed = time.perf_counter() - started
        assert elapsed >= 2 * len(payload) * 8 / 4_000_000 * .9
        await asyncio.sleep(.02)
        server.close(); target.close()
        await server.wait_closed(); await target.wait_closed()
        receipt = proxy.receipt()
        assert receipt["round_counters"]["6"]["upload_bytes"] == 2 * len(payload)
        assert receipt["round_counters"]["6"]["download_bytes"] == 2 * len(payload)
        assert not receipt["errors"]
    asyncio.run(execute())
