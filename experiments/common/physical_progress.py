"""Crash-surviving round receipts and phase markers for physical experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
import time


class PhysicalRoundRecorder:
    def __init__(self, output, *, metadata, measurement_started_ns):
        self.directory = Path(output).with_suffix(".progress")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.measurement_started_ns = int(measurement_started_ns)
        self.lock = Lock()

    def event(self, event, *, round_id=None, **details):
        record = {"event": event, "round_id": round_id, "pid": os.getpid(),
                  "monotonic_ns": time.monotonic_ns(), **details}
        payload = json.dumps(record, allow_nan=False) + "\n"
        with self.lock:
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(payload)

    def _atomic(self, name, value):
        # A crash while writing the next file must leave the prior round intact.
        payload = json.dumps(value, allow_nan=False) + "\n"
        destination = self.directory / name
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)

    def save_round(self, strategy, round_id):
        round_id = int(round_id)
        diagnostics = getattr(strategy.placement_policy, "round_diagnostics", {})
        schedule = getattr(strategy, "network_phase_provider", None)
        receipt = {"schema": "splitfleet.physical-round-receipt.v1", "complete_job": False,
                   "round_id": round_id, "metadata": self.metadata,
                   "measurement_started_monotonic_ns": self.measurement_started_ns,
                   "saved_monotonic_ns": time.monotonic_ns(),
                   "fit_records": [row for row in strategy.fit_records if row["round_id"] == round_id],
                   "fit_failures": [row for row in strategy.fit_failures if row["round_id"] == round_id],
                   "server_fit_records": [row for row in strategy.server_fit_records if row["round_id"] == round_id],
                   "evaluation_records": [row for row in strategy.evaluation_records if row["round_id"] == round_id],
                   "state_download_records": [row for row in strategy.state_download_records if row["round_id"] == round_id],
                   "round_diagnostics": diagnostics.get(round_id, diagnostics.get(str(round_id), {})),
                   "network_schedule": schedule.receipt() if schedule is not None else None}
        name = f"round_{round_id:03d}.json"
        self._atomic(name, receipt)
        self._atomic("latest.json", {"last_completed_round": round_id, "receipt": name, "complete_job": False})
        self.event("round_saved", round_id=round_id, receipt=name)


def enable_native_diagnostics():
    import faulthandler

    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=True)
