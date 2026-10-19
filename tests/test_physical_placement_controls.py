from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np

from experiments.tcp_pacing import PacedProxy
from splitfleet.server.placement.greedy.calibration import CalibratedTelemetry




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
