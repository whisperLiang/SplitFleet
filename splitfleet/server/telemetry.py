"""Host measurements and actual suffix-executor load, shared across RPC threads."""

from __future__ import annotations

from collections import deque
import math
import subprocess
from threading import Event, RLock, Thread
import time


SERVER_LOAD_FIELDS = (
    "gpu_utilization", "memory_free_ratio", "active_jobs", "queue_length",
    "execution_concurrency",
)


class NvidiaDeviceSampler:
    """Query the model's CUDA UUID; logical CUDA indices are not NVML indices."""

    def __init__(self, device):
        self.device = str(device)
        self.uuid = None

    def __call__(self):
        if "cuda" not in self.device.lower() and "gpu" not in self.device.lower():
            return {}
        if self.uuid is None:
            import torch

            index = int(self.device.rsplit(":", 1)[-1]) if ":" in self.device else torch.cuda.current_device()
            uuid = str(torch.cuda.get_device_properties(index).uuid)
            self.uuid = uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"
        result = subprocess.run(
            ["nvidia-smi", f"--id={self.uuid}",
             "--query-gpu=utilization.gpu,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=1,
        )
        utilization, free, total = (float(item.strip()) for item in result.stdout.strip().split(","))
        if not (all(math.isfinite(v) for v in (utilization, free, total))
                and 0 <= utilization <= 100 and 0 <= free <= total and total > 0):
            raise ValueError("Invalid NVIDIA device measurements")
        # Telemetry consumers accept ratios as well as percentages. Normalize here
        # so a measured 1 percent cannot be mistaken for a ratio of 1 (100%).
        return {"gpu_utilization": utilization / 100., "memory_free_ratio": free / total,
                "gpu_uuid": self.uuid}


class LiveServerTelemetry:
    """No generated load: queue counts and waits come from actual RPC execution.

    GPU queries run outside the RPC loop. Missing/expired hardware readings stay
    absent. The last training round's measured waits survive the idle interval
    between rounds and carry an explicit observation age and round identity.
    """

    def __init__(self, *, device="cpu", concurrency=1, sampler=None,
                 interval_sec=1., clock=time.monotonic):
        if concurrency < 1 or interval_sec <= 0:
            raise ValueError("Invalid telemetry concurrency/sampling interval")
        self.concurrency = int(concurrency)
        self.interval_sec = float(interval_sec)
        self._sampler = sampler or NvidiaDeviceSampler(device)
        self._clock = clock
        self._lock = RLock()
        self._stop = Event()
        self._thread = None
        self._hardware = {}
        self._hardware_at = None
        self._hardware_error = None
        self._active = 0
        self._queued = 0
        self._round_id = None
        self._waits = deque(maxlen=8192)
        self._last_waits = ()
        self._last_round_id = None

    def sample_hardware(self):
        try:
            values = dict(self._sampler())
        except (OSError, RuntimeError, ValueError, AttributeError, subprocess.SubprocessError) as exc:
            with self._lock:
                self._hardware = {}
                self._hardware_at = None
                self._hardware_error = type(exc).__name__
            return
        with self._lock:
            self._hardware = values
            self._hardware_at = self._clock()
            self._hardware_error = None

    def start(self):
        if self._thread is not None:
            return
        self.sample_hardware()
        self._stop.clear()

        def sample():
            while not self._stop.wait(self.interval_sec):
                self.sample_hardware()

        self._thread = Thread(target=sample, name="splitfleet-server-telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.)
            self._thread = None

    def begin_round(self, round_id):
        with self._lock:
            if self._round_id == int(round_id):
                return
            # An empty/failed round must not silently carry older load forward
            # as though it came from the immediately preceding training round.
            self._last_waits = tuple(self._waits)
            self._last_round_id = self._round_id
            self._waits.clear()
            self._round_id = int(round_id)

    def enqueued(self):
        with self._lock:
            self._queued += 1

    def cancel_queued(self):
        with self._lock:
            self._queued -= 1

    def started(self, queue_sec):
        with self._lock:
            self._queued -= 1
            self._active += 1
            self._waits.append((self._clock(), float(queue_sec) * 1000.))

    def finished(self):
        with self._lock:
            self._active -= 1

    def execution_context(self):
        """Constant-size per-batch snapshot; do not sort round history here."""
        with self._lock:
            now = self._clock()
            values = {
                "source": "measured_server_runtime",
                "active_jobs": self._active, "queue_length": self._queued,
                "execution_concurrency": self.concurrency,
                "hardware_error": self._hardware_error,
            }
            if self._hardware_at is not None:
                age = max(now - self._hardware_at, 0.)
                values["hardware_sample_age_sec"] = age
                if age <= max(3 * self.interval_sec, 3.):
                    values.update(self._hardware)
            return values

    def server_context(self):
        with self._lock:
            values = self.execution_context()
            observations = tuple(self._waits) or self._last_waits
            observed_round = self._round_id if self._waits else self._last_round_id
        if observations:
            waits = sorted(value for _, value in observations)
            values.update(
                queue_wait_mean_ms=sum(waits) / len(waits),
                queue_wait_p95_ms=waits[math.ceil(.95 * len(waits)) - 1],
                queue_wait_samples=len(waits),
                queue_observation_age_sec=max(self._clock() - observations[-1][0], 0.),
                queue_observation_round=observed_round,
            )
        return values
