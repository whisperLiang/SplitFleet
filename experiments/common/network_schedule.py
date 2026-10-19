"""Round-indexed controls for real TCP pacing in matched physical experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path


class RoundNetworkSchedule:
    def __init__(self, phases, control):
        if control is None:
            raise ValueError("A network schedule requires a control output")
        self.phases = []
        previous = 0
        for row in phases:
            start = row["start_round"]
            if isinstance(start, bool) or not isinstance(start, int) or start <= previous:
                raise ValueError("Network phase starts must be increasing positive integers")
            phase = {"start_round": start}
            for name in ("upload_mbps", "download_mbps", "rtt_ms"):
                value = row.get(name, 0. if name == "rtt_ms" else None)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or (value < 0 if name == "rtt_ms" else value <= 0)):
                    raise ValueError(f"Invalid network {name}")
                if name == "rtt_ms" and value is None:
                    raise ValueError("Network RTT must be finite and nonnegative")
                phase[name] = value
            self.phases.append(phase)
            previous = start
        if not self.phases or self.phases[0]["start_round"] != 1:
            raise ValueError("A network schedule must start at round 1")
        self.control = Path(control)
        self.applied = []

    @classmethod
    def from_file(cls, path, control):
        return cls(json.loads(Path(path).read_text())["phases"], control)

    def publish(self, round_id):
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 1:
            raise ValueError("Network round must be a positive integer")
        phase = next(row for row in reversed(self.phases) if row["start_round"] <= round_id)
        self._publish(round_id, phase)

    def publish_initialization(self):
        """Keep calibration/probe traffic separate from the first fit round."""
        self._publish(0, self.phases[0])

    def _publish(self, round_id, phase):
        state = {"round_id": round_id, **{key: value for key, value in phase.items() if key != "start_round"}}
        self.control.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.control.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, allow_nan=False) + "\n")
        temporary.replace(self.control)
        self.applied.append(state)

    def receipt(self):
        return {"schema": "splitfleet.round-network-schedule.v1",
                "bandwidth_shaping": any(row["upload_mbps"] is not None or row["download_mbps"] is not None
                                         for row in self.phases),
                "latency_shaping": any(row["rtt_ms"] > 0 for row in self.phases),
                "tcp_proxy": True,
                "scope": "actual TCP application streams; common aggregate link in each direction",
                "phases": self.phases, "applied": self.applied,
                "future_phases_visible_to_controller": False}
