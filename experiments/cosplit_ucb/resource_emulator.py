"""Safe, explicitly-labelled controlled resource limits for unprivileged hosts."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from typing import Any


# CPU background load is imposed with host-wide busy processes, so it cannot be
# confined to one client thread. Overlapping windows are counted rather than
# hidden: a timing measured while another client's load was running is not
# attributable to this client's device profile alone.
_LOAD_LOCK = threading.Lock()
_ACTIVE_WINDOWS: list["ResourceEmulator"] = []
_ACTIVE_LOAD_EMULATORS: list["ResourceEmulator"] = []


def _cpu_load(stop: Any, duty_cycle: float) -> None:
    period = 0.05
    busy = period * duty_cycle
    while not stop.is_set():
        start = time.perf_counter()
        while time.perf_counter() - start < busy:
            _ = sum(index * index for index in range(256))
        remaining = period - (time.perf_counter() - start)
        if remaining > 0:
            stop.wait(remaining)


@dataclass
class TransferMeasurement:
    elapsed_ms: float
    achieved_mbps: float | None


class ControlledNetworkLink:
    """Pace actual encoded bytes when tc/netem is unavailable.

    The elapsed duration is measured; configured bandwidth and RTT are controls,
    never copied into metric fields as if they were observations.
    """

    def __init__(
        self,
        *,
        uplink_mbps: float | None = None,
        downlink_mbps: float | None = None,
        rtt_ms: float | None = None,
    ) -> None:
        self.uplink_mbps = _positive_or_none(uplink_mbps)
        self.downlink_mbps = _positive_or_none(downlink_mbps)
        self.rtt_ms = None if rtt_ms is None else max(0.0, float(rtt_ms))

    @property
    def emulated(self) -> bool:
        return any(value is not None for value in (self.uplink_mbps, self.downlink_mbps, self.rtt_ms))

    def transfer(
        self,
        payload: bytes,
        direction: str,
        *,
        deadline_ns: int | None = None,
    ) -> TransferMeasurement:
        if direction not in {"uplink", "downlink"}:
            raise ValueError("direction must be 'uplink' or 'downlink'")
        bandwidth = self.uplink_mbps if direction == "uplink" else self.downlink_mbps
        started = time.perf_counter_ns()
        # Touch every byte to retain a real serialization/copy cost.
        memoryview(payload).tobytes()
        delay_s = 0.0
        if bandwidth is not None:
            delay_s += len(payload) * 8.0 / (bandwidth * 1_000_000.0)
        if self.rtt_ms is not None:
            delay_s += self.rtt_ms / 2000.0
        if delay_s > 0:
            if deadline_ns is None:
                time.sleep(delay_s)
            else:
                remaining_s = (int(deadline_ns) - time.perf_counter_ns()) / 1_000_000_000.0
                if remaining_s <= 0:
                    raise TimeoutError("client round deadline exceeded before network transfer")
                time.sleep(min(delay_s, remaining_s))
                if delay_s > remaining_s:
                    raise TimeoutError("client round deadline exceeded during network transfer")
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        achieved = None
        if payload and elapsed_ms > 0:
            achieved = len(payload) * 8.0 / (elapsed_ms * 1000.0)
        return TransferMeasurement(elapsed_ms=elapsed_ms, achieved_mbps=achieved)

    def probe(self, payload_bytes: int = 64 * 1024) -> tuple[float | None, float | None, float]:
        payload = bytes(payload_bytes)
        uplink = self.transfer(payload, "uplink")
        downlink = self.transfer(payload, "downlink")
        started = time.perf_counter_ns()
        if self.rtt_ms is not None:
            time.sleep(self.rtt_ms / 1000.0)
        else:
            memoryview(b"ping").tobytes()
        measured_rtt_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return uplink.achieved_mbps, downlink.achieved_mbps, measured_rtt_ms


class ResourceEmulator:
    """Lifecycle-managed unprivileged CPU contention and network pacing."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._stop: Any | None = None
        self._processes: list[mp.Process] = []
        self._original_affinity: set[int] | None = None
        self.overlapping_host_load_clients = 0
        self.link = ControlledNetworkLink(
            uplink_mbps=self.config.get("uplink_mbps"),
            downlink_mbps=self.config.get("downlink_mbps"),
            rtt_ms=self.config.get("rtt_ms"),
        )

    @property
    def mode(self) -> str:
        if self._processes or self.link.emulated or self.config.get("cpu_background_load"):
            return "controlled_in_process"
        return "none"

    @property
    def imposes_host_load(self) -> bool:
        """Whether this emulator contends for CPU outside its own thread."""

        return float(self.config.get("cpu_background_load", 0.0) or 0.0) > 0

    def _register_window(self) -> None:
        with _LOAD_LOCK:
            _ACTIVE_WINDOWS.append(self)
            if self.imposes_host_load:
                _ACTIVE_LOAD_EMULATORS.append(self)
            self._refresh_overlap()

    def _unregister_window(self) -> None:
        with _LOAD_LOCK:
            for registry in (_ACTIVE_WINDOWS, _ACTIVE_LOAD_EMULATORS):
                if self in registry:
                    registry.remove(self)

    @staticmethod
    def _refresh_overlap() -> None:
        """Record, for every open window, how many *other* clients load the host."""

        for emulator in _ACTIVE_WINDOWS:
            foreign = sum(
                1 for other in _ACTIVE_LOAD_EMULATORS if other is not emulator
            )
            emulator.overlapping_host_load_clients = max(
                emulator.overlapping_host_load_clients, foreign
            )

    def __enter__(self) -> "ResourceEmulator":
        affinity = self.config.get("cpu_affinity")
        if affinity is not None and hasattr(os, "sched_getaffinity"):
            self._original_affinity = set(os.sched_getaffinity(0))
            requested = {int(index) for index in affinity}
            allowed = requested & self._original_affinity
            if not allowed:
                raise ValueError("Requested CPU affinity has no CPUs allowed by the host.")
            os.sched_setaffinity(0, allowed)
        load = float(self.config.get("cpu_background_load", 0.0) or 0.0)
        if load > 0:
            if not 0 < load < 1:
                raise ValueError("cpu_background_load must be in (0, 1).")
            self._stop = mp.Event()
            workers = max(1, int(self.config.get("cpu_load_workers", 1)))
            for _ in range(workers):
                process = mp.Process(target=_cpu_load, args=(self._stop, load), daemon=True)
                process.start()
                self._processes.append(process)
        self._register_window()
        return self

    def close(self) -> None:
        self._unregister_window()
        if self._stop is not None:
            self._stop.set()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self._processes.clear()
        if self._original_affinity is not None:
            os.sched_setaffinity(0, self._original_affinity)
            self._original_affinity = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _positive_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number <= 0:
        raise ValueError("Bandwidth controls must be positive.")
    return number
