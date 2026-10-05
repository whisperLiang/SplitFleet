"""Serializable conditions for controlled simulation and experiment planning."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NetworkProfile:
    upload_mbps: float
    download_mbps: float
    rtt_ms: float = 0

    def __post_init__(self):
        for field in ("upload_mbps", "download_mbps", "rtt_ms"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0 or (field != "rtt_ms" and value == 0):
                raise ValueError("Bandwidth must be positive and RTT nonnegative; all values must be finite")

    def transfer_ms(self, payload_bytes, *, direction):
        if direction not in ("upload", "download") or payload_bytes is None or payload_bytes < 0:
            raise ValueError("An explicit direction and known nonnegative byte count are required")
        bandwidth = self.upload_mbps if direction == "upload" else self.download_mbps
        return payload_bytes * 8 / (bandwidth * 1000) + self.rtt_ms / 2

    def telemetry(self):
        return {"uplink_mbps": self.upload_mbps,
                "downlink_mbps": self.download_mbps, "rtt_ms": self.rtt_ms}


@dataclass(frozen=True)
class ResourceProfile:
    client_compute_multiplier: float = 1
    server_service_multiplier: float = 1
    server_concurrency: int = 1

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in
               (self.client_compute_multiplier, self.server_service_multiplier)) or self.server_concurrency < 1:
            raise ValueError("Resource multipliers and server concurrency must be positive")
