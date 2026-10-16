"""Actual TCP stream pacing for isolated physical experiment endpoints.

The two directions share one rate schedule per physical host across all its
connections. Counters are forwarded TCP application bytes, not NIC traffic.
No interface, route, firewall, or SSH management connection is modified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from pathlib import Path


class DirectionPacer:
    def __init__(self, control, direction=None):
        self.control = control if callable(control) else Path(control)
        self.direction = direction
        self.lock = asyncio.Lock()
        self.next_due = 0.
        self.last_rate = None

    async def wait(self, count):
        state, due = await self.reserve(count)
        delay = max(0., due - time.perf_counter())
        if delay:
            await asyncio.sleep(delay)
        return state, delay

    async def reserve(self, count):
        state = self.control() if callable(self.control) else json.loads(self.control.read_text())
        rate = state.get(f"{self.direction}_mbps", state.get("cap_mbps"))
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
        latency = float(state.get("rtt_ms", 0.)) / 2000.
        if latency < 0:
            raise ValueError("TCP RTT must be nonnegative")
        return state, due + latency


class PacedProxy:
    def __init__(self, *, target, control, output):
        self.target = target
        self.output = Path(output)
        self.pacers = {name: DirectionPacer(control, name) for name in ("upload", "download")}
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
        if callable(self.pacers[direction].control):
            return await self._pipeline_pump(reader, writer, direction)
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

    async def _pipeline_pump(self, reader, writer, direction):
        """Pipeline propagation delay without imposing another bandwidth cap.

        Rate-limited streams retain a bandwidth-delay product plus four blocks
        in flight. Uncapped streams have no artificial queue-size limit: their
        in-flight bytes reflect the actual source rate and propagation delay.
        """
        block_size = 16 * 1024
        queue = asyncio.Queue()
        space = asyncio.Event()
        writable = asyncio.Event()
        writable.set()
        buffered_bytes = 0

        async def producer():
            nonlocal buffered_bytes
            while True:
                # Propagation can retain a flight of uncapped traffic, but a
                # blocked destination must still apply TCP backpressure.
                await writable.wait()
                block = await reader.read(block_size)
                if not block:
                    break
                state, due = await self.pacers[direction].reserve(len(block))
                rate = state.get(f"{direction}_mbps", state.get("cap_mbps"))
                if rate is not None:
                    latency = float(state.get("rtt_ms", 0.)) / 2000.
                    window_bytes = rate * 1_000_000 / 8 * latency + 4 * block_size
                    while buffered_bytes + len(block) > window_bytes:
                        space.clear()
                        await space.wait()
                buffered_bytes += len(block)
                queue.put_nowait((block, state, due))
                # In an uncapped direction read()/put_nowait() may both finish
                # immediately; let the consumer deliver bytes whose delay ended.
                await asyncio.sleep(0)
            queue.put_nowait(None)

        async def consumer():
            nonlocal buffered_bytes
            while (item := await queue.get()) is not None:
                block, state, due = item
                delay = max(0., due - time.perf_counter())
                if delay:
                    await asyncio.sleep(delay)
                writer.write(block)
                writable.clear()
                await writer.drain()
                writable.set()
                row = self.counts.setdefault(str(state.get("round_id", 0)), {
                    "upload_mbps": state.get("upload_mbps"), "download_mbps": state.get("download_mbps"),
                    "rtt_ms": state.get("rtt_ms", 0.), "upload_bytes": 0, "download_bytes": 0,
                    "upload_pacing_wait_sec": 0., "download_pacing_wait_sec": 0.})
                row[direction + "_bytes"] += len(block)
                row[direction + "_pacing_wait_sec"] += delay
                buffered_bytes -= len(block)
                space.set()

        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())
        try:
            await asyncio.gather(producer_task, consumer_task)
        finally:
            producer_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(producer_task, consumer_task, return_exceptions=True)
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
                except TimeoutError:
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
