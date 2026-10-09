"""Actual TCP stream pacing for isolated physical experiment endpoints.

The two directions share one rate schedule per physical host across all its
connections. Counters are forwarded TCP application bytes, not NIC traffic.
No interface, route, firewall, or SSH management connection is modified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import signal
import time


class DirectionPacer:
    def __init__(self, control):
        self.control = Path(control)
        self.lock = asyncio.Lock()
        self.next_due = 0.
        self.last_rate = None

    async def wait(self, count):
        state = json.loads(self.control.read_text())
        rate = state.get("cap_mbps")
        if rate is not None and (isinstance(rate, bool) or rate <= 0):
            raise ValueError("TCP cap must be positive or null")
        async with self.lock:
            now = time.perf_counter()
            if rate != self.last_rate:
                self.next_due = now
                self.last_rate = rate
            if rate is None:
                due = now
            else:
                due = max(now, self.next_due) + count * 8 / (rate * 1_000_000)
            self.next_due = due
        delay = max(0., due - time.perf_counter())
        if delay:
            await asyncio.sleep(delay)
        return state, delay


class PacedProxy:
    def __init__(self, *, target, control, output):
        self.target = target
        self.output = Path(output)
        self.pacers = {name: DirectionPacer(control) for name in ("upload", "download")}
        self.counts = {}
        self.errors = []
        self.connections = 0
        self.started = time.perf_counter_ns()
        self.tasks = set()

    def receipt(self):
        return {"schema": "splitfleet.measured-tcp-pacing.v1", "physical_transfer": True,
                "counter_domain": "forwarded TCP application stream including gRPC framing; excludes SSH/TCP/IP/NIC overhead",
                "direction_rate_sharing": "one aggregate pacer per direction and physical host",
                "started_monotonic_ns": self.started, "checked_monotonic_ns": time.perf_counter_ns(),
                "connections": self.connections, "round_counters": self.counts, "errors": self.errors}

    def save(self):
        tmp = self.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.receipt(), indent=2, allow_nan=False) + "\n")
        tmp.replace(self.output)

    async def pump(self, reader, writer, direction):
        while True:
            block = await reader.read(16 * 1024)
            if not block:
                break
            state, delay = await self.pacers[direction].wait(len(block))
            writer.write(block)
            await writer.drain()
            key = str(state.get("round_id", 0))
            row = self.counts.setdefault(key, {"cap_mbps": state.get("cap_mbps"),
                                              "upload_bytes": 0, "download_bytes": 0,
                                              "upload_pacing_wait_sec": 0., "download_pacing_wait_sec": 0.})
            row[direction + "_bytes"] += len(block)
            row[direction + "_pacing_wait_sec"] += delay
        try:
            writer.write_eof()
        except (OSError, AttributeError):
            pass

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        remote_writer = None
        self.connections += 1
        try:
            remote_reader, remote_writer = await asyncio.open_connection(*self.target)
            await asyncio.gather(self.pump(reader, remote_writer, "upload"),
                                 self.pump(remote_reader, writer, "download"))
        except (ConnectionError, OSError) as exc:
            self.errors.append({"type": type(exc).__name__, "reason": str(exc)})
        finally:
            for stream in (writer, remote_writer):
                if stream is not None:
                    stream.close()
                    try:
                        await stream.wait_closed()
                    except (ConnectionError, OSError):
                        pass
            self.tasks.discard(task)
            self.save()

    async def run(self, bind):
        server = await asyncio.start_server(self.handle, *bind)
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stopped.set)
        self.save()
        print(json.dumps({"event": "proxy_listening", "bind": bind, "target": self.target}), flush=True)
        async def periodic():
            while not stopped.is_set():
                self.save()
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
        reporting = asyncio.create_task(periodic())
        await stopped.wait()
        server.close()
        await server.wait_closed()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
        await reporting
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    address = lambda value: (value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1]))
    asyncio.run(PacedProxy(target=address(args.target), control=args.control, output=args.output).run(address(args.bind)))


if __name__ == "__main__":
    main()
