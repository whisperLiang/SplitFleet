"""Per-task TCP pacing of the actual suffix and model-state RPC channel."""

import asyncio
import tempfile
from dataclasses import asdict
from pathlib import Path
from threading import Thread

from experiments.tcp_pacing import PacedProxy


class TaskNetwork:
    def __init__(self, address, network, *, round_id):
        host, port = address.rsplit(":", 1)
        self.target = (host.strip("[]"), int(port))
        self.network = network
        self.round_id = round_id
        self.loop = self.thread = self.server = None
        self.directory = None
        self.proxy = None
        self.address = address

    async def _start(self):
        state = {**asdict(self.network), "round_id": self.round_id}
        self.proxy = PacedProxy(
            target=self.target,
            control=lambda: state,
            output=Path(self.directory.name) / "receipt.json",
        )
        self.server = await asyncio.start_server(self.proxy.handle, "127.0.0.1", 0)
        self.address = f"127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    def __enter__(self):
        if self.network.constrained:
            self.directory = tempfile.TemporaryDirectory(prefix="splitfleet-network-")
            self.loop = asyncio.new_event_loop()
            self.thread = Thread(
                target=self.loop.run_forever,
                name="splitfleet-client-pacer",
                daemon=True,
            )
            self.thread.start()
            try:
                asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(
                    timeout=30
                )
            except BaseException:
                self.__exit__(None, None, None)
                raise
        return self

    async def _stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        if self.proxy:
            for task in tuple(self.proxy.tasks):
                task.cancel()
            await asyncio.gather(*tuple(self.proxy.tasks), return_exceptions=True)

    def __exit__(self, *_):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._stop(), self.loop).result(timeout=30)
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=30)
            self.loop.close()
            self.directory.cleanup()

    def metrics(self):
        row = (
            {} if self.proxy is None else self.proxy.counts.get(str(self.round_id), {})
        )
        return {
            "network_constraint_source": "emulated"
            if self.network.constrained
            else "measured",
            "tcp_upload_bytes": int(row.get("upload_bytes", 0)),
            "tcp_download_bytes": int(row.get("download_bytes", 0)),
            "tcp_upload_pacing_wait_sec": float(row.get("upload_pacing_wait_sec", 0)),
            "tcp_download_pacing_wait_sec": float(
                row.get("download_pacing_wait_sec", 0)
            ),
        }
