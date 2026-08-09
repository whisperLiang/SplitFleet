"""Strict JSONL result records and run-directory management."""

from __future__ import annotations

import json
import math
import platform
import socket
import sys
import threading
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from .config_utils import git_commit, save_config


RESULT_FILES = (
    "round_metrics.jsonl",
    "client_metrics.jsonl",
    "split_decisions.jsonl",
    "resource_metrics.jsonl",
    "failures.jsonl",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN and Inf are forbidden in experiment records.")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return str(value)


@dataclass
class ClientMetricRecord:
    run_id: str
    method: str
    seed: int
    round_id: int
    client_id: str
    split_key: str
    resource_phase: str
    num_examples: int
    completion_ms: float
    client_forward_ms: float
    client_backward_ms: float
    server_forward_ms: float
    server_backward_ms: float
    network_upload_ms: float
    network_download_ms: float
    server_queue_ms: float
    boundary_forward_bytes: int
    boundary_gradient_bytes: int
    client_peak_memory_mb: float | None
    server_peak_memory_mb: float | None
    client_energy_j: float | None
    server_energy_j: float | None
    loss: float
    success: bool
    failure_reason: str | None = None
    # Number of *other* clients whose host-global CPU load emulation overlapped
    # this client's measured window. Anything above 0 means the timing carries
    # another device profile's handicap and is not attributable to this one.
    overlapping_host_load_clients: int = 0


@dataclass
class SplitDecisionRecord:
    run_id: str
    method: str
    seed: int
    round_id: int
    client_id: str
    split_key: str
    resource_phase: str
    old_split_key: str
    new_split_key: str
    predicted_improvement_ratio: float
    actual_improvement_ratio: float | None
    switch_reason: str
    runtime_prepare_ms: float
    model_transfer_ms: float
    optimizer_state_transfer_ms: float
    total_switch_ms: float


@dataclass
class ProfileRecord:
    run_id: str
    device_profile: str
    network_profile: str
    split_key: str
    batch_size: int
    repetition: int
    client_forward_ms: float | None
    client_backward_ms: float | None
    server_forward_ms: float | None
    server_backward_ms: float | None
    boundary_forward_bytes: int | None
    boundary_gradient_bytes: int | None
    client_peak_memory_mb: float | None
    server_peak_memory_mb: float | None
    client_energy_j: float | None
    server_energy_j: float | None
    network_upload_ms: float | None
    network_download_ms: float | None
    server_queue_ms: float | None
    end_to_end_batch_ms: float | None
    success: bool
    failure_reason: str | None
    emulation_mode: str
    runtime_prepare_ms: float | None = None
    client_compute_score: float | None = None
    measured_uplink_mbps: float | None = None
    measured_downlink_mbps: float | None = None


class ResultWriter:
    """Thread-safe append-only writer for one run directory."""

    def __init__(
        self,
        root: str | Path,
        run_id: str,
        *,
        config: Mapping[str, Any],
        method: str,
        seed: int,
        emulation_mode: str,
    ) -> None:
        self.run_id = run_id
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "figures").mkdir()
        self._lock = threading.Lock()
        self._handles = {
            name: (self.path / name).open("a", encoding="utf-8") for name in RESULT_FILES
        }
        commit = git_commit()
        self.metadata = {
            "run_id": run_id,
            "method": method,
            "dataset": config.get("dataset"),
            "model": config.get("model"),
            "normalization": config.get("normalization", "groupnorm"),
            "seed": seed,
            "git_commit": commit,
            "start_time": utc_now(),
            "end_time": None,
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchlens_version": _package_version("torchlens"),
            "cuda_version": torch.version.cuda,
            "device_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
            "emulation_mode": emulation_mode,
        }
        self._write_json("metadata.json", self.metadata)
        save_config(self.path / "resolved_config.yaml", config)
        (self.path / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        self._write_json(
            "environment.json",
            {
                "platform": sys.platform,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchlens": _package_version("torchlens"),
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            },
        )

    def _write_json(self, name: str, value: Any) -> None:
        (self.path / name).write_text(
            json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_json(self, name: str, value: Any) -> None:
        self._write_json(name, value)

    def append(self, name: str, record: Any) -> None:
        if name not in self._handles:
            raise KeyError(f"Unknown result stream {name!r}")
        payload = json.dumps(_json_safe(record), sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._handles[name].write(payload + "\n")
            self._handles[name].flush()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self.metadata["end_time"] = utc_now()
        self._write_json("metadata.json", self.metadata)

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None
