"""Append-only experiment evidence and explicit measurement provenance."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import RLock

import numpy as np
import torch


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not a JSON evidence value: {type(value).__name__}")


class Evidence:
    def __init__(self, directory, config):
        self.directory = Path(directory).resolve()
        workspace = Path(__file__).resolve().parents[2]
        prepared = (
            self.directory.is_dir()
            and (self.directory / "runtime_snapshot").resolve() == workspace
        )
        if prepared:
            from experiments.common.evidence import claim_frozen_run

            frozen = claim_frozen_run(self.directory)
            frozen.pop("config_sha256", None)
            # JSON stringifies integer schedule keys; normalize through JSON.
            if json.loads(json.dumps(config.as_dict())) != frozen:
                raise RuntimeError(
                    "Execution config differs from the frozen run configuration"
                )
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
            from experiments.www2027_study import freeze_runtime

            freeze_runtime(workspace, self.directory)
        self.lock = RLock()
        self.started = time.perf_counter()
        self.config = config
        self.expected_capacity_observation = None
        capacity_receipt = self.directory / "capacity_observation.json"
        if capacity_receipt.exists():
            self.expected_capacity_observation = json.loads(
                capacity_receipt.read_text(encoding="utf-8")
            )
        if not prepared:
            self.write("config.json", config.as_dict())
        versions = {}
        for name in ("flwr", "ray", "torch", "torchlens", "grpcio"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        self.write(
            "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "host": socket.gethostname(),
                "versions": versions,
                "cpu_count": os.cpu_count(),
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "cuda_available": torch.cuda.is_available(),
                "git_revision": revision.stdout.strip()
                if revision.returncode == 0
                else None,
            },
        )
        for name in (
            "round_metrics.jsonl",
            "client_metrics.jsonl",
            "placement_trace.jsonl",
        ):
            (self.directory / name).touch(exist_ok=False)

    def write(self, name, value):
        encoded = (
            json.dumps(
                value, ensure_ascii=False, indent=2, allow_nan=False, default=json_value
            )
            + "\n"
        )
        with self.lock, (self.directory / name).open("x", encoding="utf-8") as stream:
            stream.write(encoded)

    def append(self, name, value):
        encoded = (
            json.dumps(value, ensure_ascii=False, allow_nan=False, default=json_value)
            + "\n"
        )
        with self.lock, (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(encoded)

    def read_rows(self, name):
        return [
            json.loads(line)
            for line in (self.directory / name).read_text().splitlines()
        ]

    def record_capacity_observation(self, resources, *, native_pool_size):
        """Persist and, on replay, verify the native Worker admission receipt."""
        observation = resources.capacity_observation
        if observation is None:
            raise RuntimeError("Native Worker admission did not produce a capacity receipt")
        receipt = {
            "worker_target_count": resources.worker_target_count,
            "predicted_worker_capacity": observation.get("predicted"),
            "native_worker_pool_size": int(native_pool_size),
            "effective_worker_capacity": resources.effective_worker_capacity,
            "capacity_observation": observation,
        }
        expected = self.expected_capacity_observation
        if expected is not None:
            expected_pool = expected.get("native_worker_pool_size")
            expected_effective = expected.get("effective_worker_capacity")
            if (
                expected_pool is not None
                and int(expected_pool) != int(native_pool_size)
            ) or (
                expected_effective is not None
                and int(expected_effective)
                != int(resources.effective_worker_capacity)
            ):
                raise RuntimeError(
                    "Native Worker admission differs from the frozen capacity receipt: "
                    f"expected pool={expected_pool}, effective={expected_effective}; "
                    f"observed pool={native_pool_size}, "
                    f"effective={resources.effective_worker_capacity}"
                )
            return receipt
        self.write("capacity_observation.json", receipt)
        return receipt
