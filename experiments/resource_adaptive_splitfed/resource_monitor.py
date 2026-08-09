"""Best-effort real resource measurement with explicit missing values."""

from __future__ import annotations

import os
import subprocess
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import psutil
import torch

from .metrics import timed_call

T = TypeVar("T")
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _WARNED.add(key)


@dataclass
class ClientResourceState:
    client_id: str
    device_profile: str
    client_compute_score: float
    client_available_memory_mb: float
    client_peak_memory_mb: float | None
    client_cpu_utilization: float
    client_gpu_utilization: float | None
    client_gpu_memory_free_mb: float | None
    uplink_mbps: float | None
    downlink_mbps: float | None
    rtt_ms: float | None


@dataclass
class ServerResourceState:
    server_queue_length: int
    server_gpu_utilization: float | None
    server_gpu_memory_free_mb: float | None
    server_active_jobs: int
    max_server_concurrency: int
    server_available_memory_mb: float


class EnergyMeter:
    """Read Linux RAPL counters when the host exposes a reliable energy source."""

    def __init__(self) -> None:
        self._paths = sorted(Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj"))
        self._attributable = os.environ.get("SPLITFLEET_RAPL_ATTRIBUTABLE") == "1"
        if not self._paths or not self._attributable:
            _warn_once(
                "energy",
                "Attributable energy measurement is unavailable; energy fields will be null. "
                "Use separate readable Intel RAPL/hardware meters and set "
                "SPLITFLEET_RAPL_ATTRIBUTABLE=1 only on a dedicated measured host.",
            )

    @property
    def available(self) -> bool:
        return bool(self._paths) and self._attributable

    def read_joules(self) -> float | None:
        if not self.available:
            return None
        try:
            return sum(float(path.read_text().strip()) for path in self._paths) / 1_000_000.0
        except (OSError, ValueError):
            _warn_once("energy-read", "RAPL energy read failed; energy fields will be null.")
            return None


_MEASUREMENT_LOCK = threading.Lock()
_ACTIVE_MEASUREMENTS: dict[int, list["PeakMemoryTracker"]] = {}


class PeakMemoryTracker:
    """Track process RSS and CUDA peak allocation over a measured operation.

    Process RSS and ``torch.cuda`` peak statistics are process- and
    device-global, so a window that overlaps another thread's window measures
    both threads. Such a measurement is reported as ``None`` (an explicit
    missing value) rather than attributed to one client. Windows nested inside
    one thread stay attributable: only the outermost window resets the CUDA
    peak counter, so a nested window no longer clears its parent's peak.
    """

    def __init__(self, device: str | torch.device = "cpu", sample_interval_s: float = 0.002):
        self.device = torch.device(device)
        self.sample_interval_s = sample_interval_s
        self.peak_mb: float | None = None
        self.attributable = True
        self.nested = False

    def _enter(self) -> None:
        thread_id = threading.get_ident()
        with _MEASUREMENT_LOCK:
            stack = _ACTIVE_MEASUREMENTS.setdefault(thread_id, [])
            self.nested = bool(stack)
            self.attributable = True
            if any(
                other_id != thread_id and trackers
                for other_id, trackers in _ACTIVE_MEASUREMENTS.items()
            ):
                for trackers in _ACTIVE_MEASUREMENTS.values():
                    for tracker in trackers:
                        tracker.attributable = False
                self.attributable = False
            stack.append(self)

    def _exit(self) -> None:
        thread_id = threading.get_ident()
        with _MEASUREMENT_LOCK:
            stack = _ACTIVE_MEASUREMENTS.get(thread_id, [])
            if self in stack:
                stack.remove(self)
            if not stack:
                _ACTIVE_MEASUREMENTS.pop(thread_id, None)

    def measure(self, fn: Callable[[], T]) -> tuple[T, float | None]:
        process = psutil.Process(os.getpid())
        self._enter()
        try:
            baseline = process.memory_info().rss
            peak = baseline
            stop = threading.Event()

            def sample() -> None:
                nonlocal peak
                while not stop.wait(self.sample_interval_s):
                    try:
                        peak = max(peak, process.memory_info().rss)
                    except psutil.Error:
                        return

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            cuda = self.device.type == "cuda" and torch.cuda.is_available()
            if cuda and not self.nested:
                torch.cuda.reset_peak_memory_stats(self.device)
            try:
                value = fn()
            finally:
                stop.set()
                sampler.join(timeout=1.0)
        finally:
            self._exit()
        rss_delta = max(0, peak - baseline) / (1024.0**2)
        if not self.attributable:
            self.peak_mb = None
        elif cuda and not self.nested:
            cuda_peak = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
            self.peak_mb = max(rss_delta, cuda_peak)
        else:
            # A nested window cannot read a scoped CUDA peak without resetting
            # the counter its parent is still using, so it reports RSS only.
            self.peak_mb = rss_delta
        return value, self.peak_mb


class ResourceMonitor:
    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.process = psutil.Process(os.getpid())
        self.energy = EnergyMeter()
        psutil.cpu_percent(interval=None)

    def compute_score(self) -> float:
        """Measured operations/second for a fixed real matrix multiplication."""
        size = 256
        left = torch.ones((size, size), device=self.device)
        right = torch.full((size, size), 0.5, device=self.device)
        _, elapsed_ms = timed_call(lambda: torch.mm(left, right), self.device)
        operations = 2.0 * size**3
        return operations / max(elapsed_ms / 1000.0, 1e-9)

    def gpu_state(self) -> tuple[float | None, float | None]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            _warn_once(
                "gpu",
                "The measured workload is not running on CUDA; GPU utilization and GPU memory "
                "fields will be null rather than reporting an unrelated installed GPU.",
            )
            return None, None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.free",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device.index or 0),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            utilization, free = result.stdout.strip().splitlines()[0].split(",")
            return float(utilization.strip()), float(free.strip())
        except Exception:
            _warn_once(
                "nvidia-smi",
                "nvidia-smi metrics failed; GPU utilization and GPU memory fields will be null.",
            )
            return None, None

    def client_state(
        self,
        client_id: str,
        device_profile: str,
        *,
        peak_memory_mb: float | None = None,
        uplink_mbps: float | None = None,
        downlink_mbps: float | None = None,
        rtt_ms: float | None = None,
    ) -> ClientResourceState:
        vm = psutil.virtual_memory()
        gpu_utilization, gpu_free = self.gpu_state()
        return ClientResourceState(
            client_id=str(client_id),
            device_profile=str(device_profile),
            client_compute_score=self.compute_score(),
            client_available_memory_mb=vm.available / (1024.0**2),
            client_peak_memory_mb=peak_memory_mb,
            client_cpu_utilization=psutil.cpu_percent(interval=None),
            client_gpu_utilization=gpu_utilization,
            client_gpu_memory_free_mb=gpu_free,
            uplink_mbps=uplink_mbps,
            downlink_mbps=downlink_mbps,
            rtt_ms=rtt_ms,
        )

    def server_state(self, pool: "ServerJobPool") -> ServerResourceState:
        vm = psutil.virtual_memory()
        gpu_utilization, gpu_free = self.gpu_state()
        return ServerResourceState(
            server_queue_length=pool.queue_length,
            server_gpu_utilization=gpu_utilization,
            server_gpu_memory_free_mb=gpu_free,
            server_active_jobs=pool.active_jobs,
            max_server_concurrency=pool.max_concurrency,
            server_available_memory_mb=vm.available / (1024.0**2),
        )


class ServerJobPool:
    """Bound real concurrent suffix execution and measure semaphore queue time."""

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.max_concurrency = int(max_concurrency)
        self._semaphore = threading.Semaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._active_jobs = 0
        self._queue_length = 0

    @property
    def active_jobs(self) -> int:
        with self._lock:
            return self._active_jobs

    @property
    def queue_length(self) -> int:
        with self._lock:
            return self._queue_length

    def execute(
        self,
        fn: Callable[[], T],
        *,
        deadline_ns: int | None = None,
    ) -> tuple[T, float]:
        queued_at = time.perf_counter_ns()
        with self._lock:
            self._queue_length += 1
        if deadline_ns is None:
            acquired = self._semaphore.acquire()
        else:
            remaining_s = (int(deadline_ns) - time.perf_counter_ns()) / 1_000_000_000.0
            acquired = remaining_s > 0 and self._semaphore.acquire(timeout=remaining_s)
        if not acquired:
            with self._lock:
                self._queue_length -= 1
            raise TimeoutError("client round deadline exceeded in the server queue")
        started_at = time.perf_counter_ns()
        with self._lock:
            self._queue_length -= 1
            self._active_jobs += 1
        try:
            return fn(), (started_at - queued_at) / 1_000_000.0
        finally:
            with self._lock:
                self._active_jobs -= 1
            self._semaphore.release()
