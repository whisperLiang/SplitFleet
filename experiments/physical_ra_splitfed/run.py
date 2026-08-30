"""Run auditable CIFAR-10/ResNet-18 SplitFed and FedAvg on physical hosts.

This runner is intentionally separate from the controlled in-process
RA-SplitFed runner for fixed-cut and confirmatory physical experiments.  It
preserves the same model, partition, and state-hash helpers across methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, start_server as start_flower_server
from flwr.server.strategy import FedAvg
from torch import nn
from torch.utils.data import DataLoader, Subset

from experiments.resource_adaptive_splitfed.config_utils import (
    git_commit,
    set_reproducible_seed,
    stable_hash,
    tensor_state_hash,
)
from experiments.resource_adaptive_splitfed.model_data import (
    FLOWER_CIFAR10_PIPELINE,
    build_model,
    cifar10_datasets,
    dataset_targets,
    dirichlet_partition,
    make_loader,
    partition_manifest,
)
from experiments.resource_adaptive_splitfed.split_candidates import (
    discover_split_candidates,
)
from experiments.resource_adaptive_splitfed.split_cost_model import SplitCostPrediction
from experiments.resource_adaptive_splitfed.split_scheduler import (
    ResourceAdaptiveSplitScheduler,
    select_edge_local,
)
from experiments.physical_ra_splitfed.native_prefix import (
    NativePrefixSplitClient,
    NativePrefixSplitFedStrategy,
    PREFIX_EXECUTORS,
)
from splitfleet.client.app import start_client
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.client.numpy_client import NumPyClient
from splitfleet.common.constants import AUTOSPLIT_MODEL_VERSION_CONFIG_KEY
from splitfleet.server.app import start_server
from splitfleet.server.server_model.autosplit_tail_server_model import (
    AutoSplitTailServerModel,
)
from splitfleet.server.strategy import AutoSplitStrategy


DEFAULT_CANDIDATES = ("stem", "layer2", "layer4")
FEDAVG_METHOD = "fedavg_full_local"
SOURCE_PATHS = (
    "experiments/physical_ra_splitfed/run.py",
    "experiments/physical_ra_splitfed/orchestrate.py",
    "experiments/physical_ra_splitfed/validate_run.py",
    "experiments/physical_ra_splitfed/protocol.yaml",
    "experiments/physical_ra_splitfed/amendment_A7.md",
    "experiments/physical_ra_splitfed/amendment_A8.md",
    "experiments/physical_ra_splitfed/amendment_A9.md",
    "experiments/physical_ra_splitfed/amendment_A10.md",
    "experiments/physical_ra_splitfed/protocol_large_model.yaml",
    "experiments/physical_ra_splitfed/protocol_system_optimized.yaml",
    "experiments/physical_ra_splitfed/protocol_latest_rerun.yaml",
    "experiments/physical_ra_splitfed/native_prefix.py",
    "experiments/resource_adaptive_splitfed/model_data.py",
    "experiments/resource_adaptive_splitfed/split_candidates.py",
    "splitfleet/client/autosplit_split_client.py",
    "splitfleet/client/app.py",
    "splitfleet/client/grpc/connection.py",
    "splitfleet/client/grpc/message_handler.py",
    "splitfleet/common/serde.py",
    "splitfleet/common/address.py",
    "splitfleet/server/app.py",
    "splitfleet/server/server_model/manager/grpc_manager.py",
    "splitfleet/server/stage_runtime/manager.py",
    "splitfleet/server/strategy/plain_strategy.py",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable")
    return requested


def _source_manifest() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    manifest: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment source is missing: {path}")
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _environment(device: str) -> dict[str, Any]:
    source_manifest = _source_manifest()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "torchlens": _package_version("torchlens"),
        "flower": _package_version("flwr"),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_hash": stable_hash(source_manifest),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _make_model(seed: int, device: str, model_name: str = "resnet18") -> nn.Module:
    set_reproducible_seed(seed)
    return build_model(model_name, normalization="groupnorm").to(device)


def _load_partition(
    *,
    data_root: str,
    seed: int,
    num_clients: int,
    download: bool,
) -> tuple[Any, Any, dict[str, list[int]], dict[str, Any]]:
    train, test = cifar10_datasets(
        data_root,
        download=download,
    )
    targets = dataset_targets(train)
    assignments = dirichlet_partition(
        targets,
        num_clients,
        alpha=0.5,
        seed=seed,
    )
    return train, test, assignments, partition_manifest(assignments, targets)


class LocalEpochIterable:
    """Create fresh shuffled iterators for complete Flower-style local epochs."""

    def __init__(self, source: Iterable[Any], local_epochs: int) -> None:
        if int(local_epochs) < 1:
            raise ValueError("local_epochs must be positive")
        self.source = source
        self.local_epochs = int(local_epochs)

    def __iter__(self) -> Iterator[Any]:
        for _ in range(self.local_epochs):
            yield from iter(self.source)


def _model_to_ndarrays(model: nn.Module) -> list[np.ndarray]:
    """Export the complete named PyTorch state in deterministic state-dict order."""
    return [
        value.detach().cpu().numpy().copy()
        for value in model.state_dict().values()
    ]


def _load_model_ndarrays(model: nn.Module, arrays: Sequence[np.ndarray]) -> None:
    """Load a complete Flower parameter vector without changing state ownership."""
    current = model.state_dict()
    if len(arrays) != len(current):
        raise ValueError(
            f"Full-model parameter count mismatch: received {len(arrays)}, "
            f"expected {len(current)}"
        )
    loaded = {}
    for (name, reference), array in zip(current.items(), arrays):
        tensor = torch.as_tensor(np.asarray(array), device=reference.device)
        if tuple(tensor.shape) != tuple(reference.shape):
            raise ValueError(
                f"Full-model shape mismatch for {name}: received {tuple(tensor.shape)}, "
                f"expected {tuple(reference.shape)}"
            )
        loaded[name] = tensor.to(dtype=reference.dtype)
    model.load_state_dict(loaded, strict=True)


def _ndarray_bytes(arrays: Sequence[np.ndarray]) -> int:
    return sum(int(np.asarray(array).nbytes) for array in arrays)


def _sync_device(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class TaggedFedAvgClient(NumPyClient):
    """Train the complete model locally and report physical FedAvg telemetry."""

    def __init__(
        self,
        *,
        logical_client_id: str,
        partition_hash: str,
        client_index_hash: str,
        environment: dict[str, Any],
        model: nn.Module,
        train_data: Iterable[Any],
        device: str,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        paced_client_id: str | None = None,
        paced_round_start: int | None = None,
        paced_round_end: int | None = None,
        paced_uplink_mbps: float = 10.0,
        paced_downlink_mbps: float = 50.0,
        nominal_uplink_mbps: float = 1000.0,
        nominal_downlink_mbps: float = 1000.0,
    ) -> None:
        self.logical_client_id = str(logical_client_id)
        self.partition_hash = str(partition_hash)
        self.client_index_hash = str(client_index_hash)
        self.environment = dict(environment)
        self.model = model
        self.train_data = train_data
        self.device = str(device)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.paced_client_id = str(paced_client_id or "")
        self.paced_round_start = paced_round_start
        self.paced_round_end = paced_round_end
        self.paced_uplink_mbps = float(paced_uplink_mbps)
        self.paced_downlink_mbps = float(paced_downlink_mbps)
        self.nominal_uplink_mbps = float(nominal_uplink_mbps)
        self.nominal_downlink_mbps = float(nominal_downlink_mbps)

    def get_parameters(self, config):
        _ = config
        return _model_to_ndarrays(self.model)

    def fit(self, parameters, config):
        fit_started = time.perf_counter()
        round_id = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        constrained_window = (
            self.logical_client_id == self.paced_client_id
            and self.paced_round_start is not None
            and self.paced_round_end is not None
            and int(self.paced_round_start) <= round_id <= int(self.paced_round_end)
        )
        active_uplink = (
            self.paced_uplink_mbps if constrained_window else self.nominal_uplink_mbps
        )
        active_downlink = (
            self.paced_downlink_mbps if constrained_window else self.nominal_downlink_mbps
        )
        download_bytes = _ndarray_bytes(parameters)
        prepare_started = time.perf_counter()
        _load_model_ndarrays(self.model, parameters)
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        _sync_device(self.device)
        runtime_prepare_sec = time.perf_counter() - prepare_started
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        loss_fn = nn.CrossEntropyLoss()
        num_examples = 0
        num_batches = 0
        weighted_loss = 0.0
        min_batch_size = 0
        max_batch_size = 0
        compute_started = time.perf_counter()
        for inputs, targets in self.train_data:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = int(targets.shape[0])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.model(inputs), targets)
            loss.backward()
            optimizer.step()
            num_examples += batch_size
            num_batches += 1
            weighted_loss += float(loss.detach().cpu()) * batch_size
            min_batch_size = (
                batch_size if min_batch_size == 0 else min(min_batch_size, batch_size)
            )
            max_batch_size = max(max_batch_size, batch_size)
        _sync_device(self.device)
        full_model_compute_sec = time.perf_counter() - compute_started
        if num_examples <= 0:
            raise RuntimeError("FedAvg client executed no training examples")
        updated = _model_to_ndarrays(self.model)
        upload_bytes = _ndarray_bytes(updated)
        metrics: dict[str, Any] = {
            "logical_client_id": self.logical_client_id,
            "partition_hash": self.partition_hash,
            "client_index_hash": self.client_index_hash,
            "hostname": self.environment["hostname"],
            "torch_version": self.environment["torch"],
            "runner_sha256": self.environment["runner_sha256"],
            "source_manifest_hash": self.environment["source_manifest_hash"],
            "device": self.device,
            "loss": weighted_loss / num_examples,
            "fit_duration_sec": time.perf_counter() - fit_started,
            "runtime_prepare_sec": runtime_prepare_sec,
            "full_model_compute_sec": full_model_compute_sec,
            "num_examples": num_examples,
            "num_batches": num_batches,
            "local_epochs": int(getattr(self.train_data, "local_epochs", 1)),
            "optimizer_momentum": self.momentum,
            "optimizer_weight_decay": self.weight_decay,
            "skipped_batches": 0,
            "skipped_examples": 0,
            "min_batch_size": min_batch_size,
            "max_batch_size": max_batch_size,
            "model_download_bytes": download_bytes,
            "model_upload_bytes": upload_bytes,
            "boundary_upload_bytes": 0,
            "boundary_download_bytes": 0,
            "app_pacing_delay_sec": 0.0,
            "pacing_scope": "split_boundary_only_not_applicable_to_fedavg",
            "resource_phase": "paced_link" if constrained_window else "normal_link",
            "configured_uplink_mbps": active_uplink,
            "configured_downlink_mbps": active_downlink,
        }
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            metrics["peak_cuda_memory_mb"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
            )
        return updated, num_examples, metrics

    def evaluate(self, parameters, config):
        _ = (parameters, config)
        return 0.0, 0, {"central_evaluation_only": True}


class TaggedPhysicalClient(AutoSplitSplitLearningClient):
    """Attach immutable device and data identities to every fit result."""

    def __init__(
        self,
        *,
        logical_client_id: str,
        partition_hash: str,
        client_index_hash: str,
        environment: dict[str, Any],
        paced_client_id: str | None = None,
        paced_round_start: int | None = None,
        paced_round_end: int | None = None,
        paced_uplink_mbps: float = 10.0,
        paced_downlink_mbps: float = 50.0,
        nominal_uplink_mbps: float = 1000.0,
        nominal_downlink_mbps: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.logical_client_id = str(logical_client_id)
        self.partition_hash = str(partition_hash)
        self.client_index_hash = str(client_index_hash)
        self.environment = environment
        self.paced_client_id = str(paced_client_id or "")
        self.paced_round_start = paced_round_start
        self.paced_round_end = paced_round_end
        self.paced_uplink_mbps = float(paced_uplink_mbps)
        self.paced_downlink_mbps = float(paced_downlink_mbps)
        self.nominal_uplink_mbps = float(nominal_uplink_mbps)
        self.nominal_downlink_mbps = float(nominal_downlink_mbps)
        self._active_uplink_mbps = self.nominal_uplink_mbps
        self._active_downlink_mbps = self.nominal_downlink_mbps
        self._pacing_delay_sec = 0.0

    def fit(self, parameters, config):
        round_id = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        paced = (
            self.logical_client_id == self.paced_client_id
            and self.paced_round_start is not None
            and self.paced_round_end is not None
            and int(self.paced_round_start) <= round_id <= int(self.paced_round_end)
        )
        self._active_uplink_mbps = (
            self.paced_uplink_mbps if paced else self.nominal_uplink_mbps
        )
        self._active_downlink_mbps = (
            self.paced_downlink_mbps if paced else self.nominal_downlink_mbps
        )
        self._pacing_delay_sec = 0.0
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        updated, examples, metrics = super().fit(parameters, config)
        metrics = dict(metrics)
        metrics.update(
            {
                "logical_client_id": self.logical_client_id,
                "partition_hash": self.partition_hash,
                "client_index_hash": self.client_index_hash,
                "hostname": self.environment["hostname"],
                "torch_version": self.environment["torch"],
                "runner_sha256": self.environment["runner_sha256"],
                "source_manifest_hash": self.environment["source_manifest_hash"],
                "resource_phase": "paced_link" if paced else "normal_link",
                "configured_uplink_mbps": self._active_uplink_mbps,
                "configured_downlink_mbps": self._active_downlink_mbps,
                "app_pacing_delay_sec": self._pacing_delay_sec,
            }
        )
        # Flower ConfigRecord deliberately rejects None.  Absence means the
        # process has no attributable CUDA counter (for example Windows CPU),
        # while a present numeric value is a real measurement.
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            metrics["peak_cuda_memory_mb"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
            )
        return updated, examples, metrics

    def _call_tail(self, **kwargs):
        telemetry = kwargs.get("telemetry")
        before_upload = int(getattr(telemetry, "upload_bytes", 0))
        before_download = int(getattr(telemetry, "download_bytes", 0))
        result = super()._call_tail(**kwargs)
        if telemetry is None:
            return result
        upload = int(telemetry.upload_bytes) - before_upload
        download = int(telemetry.download_bytes) - before_download
        delay = (
            upload * 8.0 / (max(self._active_uplink_mbps, 1e-9) * 1_000_000.0)
            + download
            * 8.0
            / (max(self._active_downlink_mbps, 1e-9) * 1_000_000.0)
        )
        if delay > 0:
            time.sleep(delay)
            telemetry.tail_wait_sec += delay
            self._pacing_delay_sec += delay
        return result


class TimedTailServerModel(AutoSplitTailServerModel):
    """Record suffix service time without changing the training operation."""

    def configure_fit(self, ins) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        super().configure_fit(ins)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.runtime_prepare_sec = time.perf_counter() - started
        self.server_compute_sec = 0.0
        self.server_request_count = 0

    def train_tail(self, batches):
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = super().train_tail(batches)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.server_compute_sec += time.perf_counter() - started
        self.server_request_count += 1
        return result

    def get_fit_result(self):
        result = super().get_fit_result()
        metrics = {
            "server_compute_sec": float(self.server_compute_sec),
            "server_request_count": int(self.server_request_count),
            "runtime_prepare_sec": float(self.runtime_prepare_sec),
        }
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
            metrics.update(
                {
                    "server_cuda_allocated_mb": float(
                        torch.cuda.memory_allocated(self.device) / (1024.0**2)
                    ),
                    "server_cuda_reserved_mb": float(
                        torch.cuda.memory_reserved(self.device) / (1024.0**2)
                    ),
                    "server_cuda_peak_allocated_mb": float(
                        torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
                    ),
                }
            )
        result.config.update(metrics)
        return result


class ProfileGuidedPlacementPolicy:
    """Use validated physical profiles with the reference global scheduler."""

    METHODS = (
        "best_global_fixed",
        "static_heterogeneous",
        "edge_local_adaptive",
        "resource_adaptive_splitfed",
    )

    def __init__(
        self,
        *,
        boundaries: dict[str, str],
        profile_report: dict[str, Any],
        method: str,
        batches_per_round: int,
        calibration_rounds: int = 3,
        server_concurrency: int = 4,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unsupported profile-guided method {method!r}")
        if not profile_report.get("valid"):
            raise ValueError("Adaptive placement requires a valid physical profile report")
        self.boundaries = dict(boundaries)
        self.boundary_to_key = {value: key for key, value in self.boundaries.items()}
        self.profile = profile_report
        self.method = str(method)
        self.batches_per_round = max(1, int(batches_per_round))
        self.calibration_rounds = int(calibration_rounds)
        self.server_concurrency = max(1, int(server_concurrency))
        self.calibration_schedule = tuple(DEFAULT_CANDIDATES)
        if self.calibration_rounds != len(self.calibration_schedule):
            raise ValueError("The physical protocol requires exactly three calibration rounds")
        self.required_split_keys = tuple(DEFAULT_CANDIDATES)
        self.cid_to_logical_id: dict[str, str] = {}
        self.current_keys: dict[str, str] = {}
        self.online_scales: dict[tuple[str, str], float] = {}
        self.links: dict[str, tuple[float, float]] = {}
        self.round_assignments: dict[int, dict[str, str]] = {}
        self.decision_records: list[dict[str, Any]] = []
        self.scheduler = ResourceAdaptiveSplitScheduler(
            min_relative_improvement_to_switch=0.10,
            min_rounds_between_switches=3,
            max_switches_per_round=None,
            use_memory_constraint=True,
            use_hysteresis=True,
            use_server_state=True,
        )

    def __call__(self, server_round: int, cid: str, training: bool) -> str:
        _ = training
        round_id = int(server_round)
        cid = str(cid)
        if round_id <= self.calibration_rounds:
            key = self.calibration_schedule[round_id - 1]
            self.current_keys[cid] = key
            return self.boundaries[key]
        if round_id not in self.round_assignments:
            self._decide_round(round_id)
        key = self.round_assignments[round_id].get(cid, "layer2")
        self.current_keys[cid] = key
        return self.boundaries[key]

    def observe_fit_metrics(
        self,
        *,
        round_id: int,
        cid: str,
        num_examples: int,
        metrics: dict[str, Any],
    ) -> None:
        _ = num_examples
        cid = str(cid)
        logical_id = str(metrics.get("logical_client_id", ""))
        if logical_id:
            stale_cids = [
                known_cid
                for known_cid, known_logical_id in self.cid_to_logical_id.items()
                if known_logical_id == logical_id and known_cid != cid
            ]
            for stale_cid in stale_cids:
                self.cid_to_logical_id.pop(stale_cid, None)
                self.current_keys.pop(stale_cid, None)
            self.cid_to_logical_id[cid] = logical_id
            self.links[logical_id] = (
                float(metrics.get("configured_uplink_mbps", 1000.0)),
                float(metrics.get("configured_downlink_mbps", 1000.0)),
            )
        split_key = self.boundary_to_key.get(str(metrics.get("boundary", "")))
        if not logical_id or split_key is None:
            return
        self.current_keys[cid] = split_key
        measured_service_ms = max(
            1e-6,
            (
                float(metrics.get("fit_duration_sec", 0.0))
                - float(metrics.get("runtime_prepare_sec", 0.0))
            )
            * 1000.0,
        )
        unscaled = self._prediction(
            cid,
            logical_id,
            split_key,
            apply_online_scale=False,
        ).predicted_round_ms
        target = measured_service_ms / max(unscaled, 1e-6)
        key = (logical_id, split_key)
        previous = self.online_scales.get(key)
        self.online_scales[key] = (
            target if previous is None else 0.25 * target + 0.75 * previous
        )

    def observe_failure(self, *, round_id: int, cid: str, reason: Any = None) -> None:
        self.decision_records.append(
            {
                "round_id": int(round_id),
                "flower_cid": str(cid),
                "event": "failure",
                "reason": str(reason),
            }
        )

    def _decide_round(self, round_id: int) -> None:
        cids = sorted(self.cid_to_logical_id)
        if not cids:
            self.round_assignments[round_id] = {}
            return
        predictions = {
            cid: {
                split_key: self._prediction(
                    cid,
                    self.cid_to_logical_id[cid],
                    split_key,
                    apply_online_scale=True,
                )
                for split_key in DEFAULT_CANDIDATES
            }
            for cid in cids
        }
        if self.method == "best_global_fixed":
            selected_key = str(self.profile["selection"]["best_global_fixed"])
            selected = {cid: selected_key for cid in cids}
        elif self.method == "static_heterogeneous":
            static = self.profile["selection"]["static_heterogeneous"]
            selected = {
                cid: str(static[self.cid_to_logical_id[cid]]) for cid in cids
            }
        elif self.method == "edge_local_adaptive":
            selected = select_edge_local(predictions)
        else:
            selected = self.scheduler.select_splits(
                cids,
                {cid: {"client_id": cid} for cid in cids},
                {"max_server_concurrency": self.server_concurrency},
                predictions,
                round_id=round_id,
            )
        self.round_assignments[round_id] = dict(selected)
        for cid, split_key in selected.items():
            prediction = predictions[cid][split_key]
            self.decision_records.append(
                {
                    "round_id": int(round_id),
                    "flower_cid": cid,
                    "logical_client_id": self.cid_to_logical_id[cid],
                    "old_split_key": self.current_keys.get(cid, "layer4"),
                    "new_split_key": split_key,
                    "predicted_round_ms": prediction.predicted_round_ms,
                    "method": self.method,
                }
            )

    def _prediction(
        self,
        cid: str,
        logical_id: str,
        split_key: str,
        *,
        apply_online_scale: bool,
    ) -> SplitCostPrediction:
        values = self.profile["per_client"][logical_id]["splits"][split_key]
        server_values = self.profile["server_summary"][split_key]
        is_native_prefix = self.profile.get("split_runtime") == "native_prefix"
        count_scale = 1 if is_native_prefix else self.batches_per_round
        prefix_ms = (
            float(values["steady_prefix_compute_mean_sec"])
            * count_scale
            * 1000.0
        )
        server_ms = (
            float(server_values["mean_compute_sec_per_client_batch"])
            * count_scale
            * 1000.0
        )
        uplink_mbps, downlink_mbps = self.links.get(
            logical_id, (1000.0, 1000.0)
        )
        network_ms = count_scale * (
            int(values["upload_bytes_per_batch"])
            * 8.0
            / (max(uplink_mbps, 1e-9) * 1000.0)
            + int(values["download_bytes_per_batch"])
            * 8.0
            / (max(downlink_mbps, 1e-9) * 1000.0)
        )
        prepare_ms = float(values["steady_runtime_prepare_mean_sec"]) * 1000.0
        scale = (
            float(self.online_scales.get((logical_id, split_key), 1.0))
            if apply_online_scale
            else 1.0
        )
        prefix_ms *= scale
        server_ms *= scale
        network_ms *= scale
        coordinator_ms = 0.0
        if is_native_prefix:
            # Native profile rows store per-round totals.  The difference
            # between physical round time and the slowest client fit captures
            # Flower prefix serialization, transfer, reassembly, and global
            # aggregation.  This is especially important for the layer4
            # prefix, which is almost the complete Wide-ResNet state.
            slowest_fit_sec = max(
                float(
                    client_values["splits"][split_key]["steady_fit_mean_sec"]
                )
                for client_values in self.profile["per_client"].values()
            )
            coordinator_ms = max(
                0.0,
                float(
                    self.profile["round_summary"][split_key]["steady_mean_sec"]
                )
                - slowest_fit_sec,
            ) * 1000.0
        total = prefix_ms + server_ms + network_ms + prepare_ms + coordinator_ms
        memory = values.get("steady_peak_cuda_memory_mb")
        return SplitCostPrediction(
            split_key=split_key,
            predicted_round_ms=total,
            predicted_client_compute_ms=prefix_ms,
            predicted_network_ms=network_ms,
            predicted_server_compute_ms=server_ms,
            predicted_server_queue_ms=0.0,
            predicted_client_peak_memory_mb=float(memory or 0.0),
            predicted_server_gpu_time_ms=server_ms,
            predicted_switch_ms=0.0,
            feasible=True,
            infeasible_reason=None,
            predicted_server_peak_memory_mb=None,
        )


class PhysicalRecordingStrategy(AutoSplitStrategy):
    """Append raw physical records during execution and evaluate centrally."""

    def __init__(
        self,
        *args: Any,
        output_dir: Path,
        run_id: str,
        method: str,
        model_seed: int,
        model_name: str,
        test_loader: DataLoader | None,
        evaluation_device: str,
        schedule_policy: Any | None = None,
        boundary_keys: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.output_dir = output_dir
        self.run_id = str(run_id)
        self.method = str(method)
        self.model_seed = int(model_seed)
        self.model_name = str(model_name)
        self.test_loader = test_loader
        self.evaluation_device = evaluation_device
        self.schedule_policy = schedule_policy
        self.boundary_to_key = {
            boundary: key for key, boundary in dict(boundary_keys or {}).items()
        }
        self.fit_records: list[dict[str, Any]] = []
        self.server_fit_records: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.evaluation_records: list[dict[str, Any]] = []
        self.round_started: dict[int, float] = {}
        self.cid_to_logical_id: dict[str, str] = {}
        self._evaluation_model = _make_model(
            model_seed, evaluation_device, self.model_name
        )
        super().__init__(*args, **kwargs)

    def _make_server_model(self):
        if self._runtime_manager is None:
            raise RuntimeError("AutoSplitStrategy is not bound to a stage runtime manager")
        return TimedTailServerModel(
            runtime_manager=self._runtime_manager,
            model=self.model,
            optimizer_fn=self.optimizer_fn,
            loss_fn=self.loss_fn,
            boundary=self.boundary,
            mode=self.mode,
            device=self.runtime_device,
        )

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_started[int(server_round)] = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        for proxy, fit_res in results:
            metrics = _json_safe(dict(fit_res.metrics or {}))
            split_key = self.boundary_to_key.get(
                str(metrics.get("boundary", "")), self.method
            )
            logical_id = str(metrics.get("logical_client_id", ""))
            if logical_id:
                self.cid_to_logical_id[str(proxy.cid)] = logical_id
            record = {
                "schema": "splitfleet.physical-client-metric.v1",
                "run_id": self.run_id,
                "method": self.method,
                "round_id": int(server_round),
                "split_key": split_key,
                "flower_cid": str(proxy.cid),
                "logical_client_id": logical_id,
                "num_examples": int(fit_res.num_examples),
                "success": True,
                "metrics": metrics,
            }
            self.fit_records.append(record)
            _append_jsonl(self.output_dir / "client_metrics.jsonl", record)
        for failure in failures:
            record = {
                "schema": "splitfleet.physical-failure.v1",
                "run_id": self.run_id,
                "method": self.method,
                "round_id": int(server_round),
                "stage": "client_fit",
                "failure_reason": str(failure),
                "failure_type": type(failure).__name__,
                "failure_repr": repr(failure),
            }
            self.failures.append(record)
            _append_jsonl(self.output_dir / "failures.jsonl", record)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_server_fit(self, server_round, results):
        for result in results:
            placement = self._client_placement_cache.get(
                (int(server_round), str(result.sid), True)
            )
            split_key = self.boundary_to_key.get(
                str(getattr(placement, "boundary", "")), self.method
            )
            record = {
                "schema": "splitfleet.physical-server-metric.v1",
                "run_id": self.run_id,
                "method": self.method,
                "round_id": int(server_round),
                "split_key": split_key,
                "sid": str(result.sid),
                "logical_client_id": self.cid_to_logical_id.get(str(result.sid), ""),
                "config": _json_safe(dict(result.config)),
            }
            self.server_fit_records.append(record)
            _append_jsonl(self.output_dir / "server_metrics.jsonl", record)
        aggregated = super().aggregate_server_fit(server_round, results)
        round_client_records = [
            item for item in self.fit_records if item["round_id"] == int(server_round)
        ]
        placements = {
            str(item["logical_client_id"]): str(item["split_key"])
            for item in round_client_records
        }
        distinct_splits = sorted(set(placements.values()))
        record = {
            "schema": "splitfleet.physical-round-metric.v1",
            "run_id": self.run_id,
            "method": self.method,
            "round_id": int(server_round),
            "split_key": distinct_splits[0] if len(distinct_splits) == 1 else "mixed",
            "placements": placements,
            "round_time_sec": time.perf_counter()
            - self.round_started.get(int(server_round), time.perf_counter()),
            "successful_clients": sum(
                item["round_id"] == int(server_round) for item in self.fit_records
            ),
            "failed_clients": sum(
                item["round_id"] == int(server_round) for item in self.failures
            ),
        }
        _append_jsonl(self.output_dir / "round_metrics.jsonl", record)
        policy_decisions = getattr(self.schedule_policy, "decision_records", None)
        if policy_decisions is not None:
            for decision in [
                item for item in policy_decisions if item["round_id"] == int(server_round)
            ]:
                _append_jsonl(self.output_dir / "split_decisions.jsonl", decision)
        return aggregated

    def evaluate(self, server_round, client_parameters, server_parameters):
        _ = server_parameters
        if self.test_loader is None:
            return None
        arrays = parameters_to_ndarrays(client_parameters)
        self.backend_adapter.load_ndarrays(self._evaluation_model, arrays)
        self._evaluation_model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        correct = 0
        examples = 0
        with torch.inference_mode():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.evaluation_device)
                targets = targets.to(self.evaluation_device)
                logits = self._evaluation_model(inputs)
                total_loss += float(loss_fn(logits, targets).item())
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                examples += int(targets.shape[0])
        record = {
            "schema": "splitfleet.physical-evaluation.v1",
            "run_id": self.run_id,
            "method": self.method,
            "round_id": int(server_round),
            "test_loss": total_loss / max(examples, 1),
            "test_accuracy": correct / max(examples, 1),
            "num_examples": examples,
            "source": "cifar10_test_set",
        }
        self.evaluation_records.append(record)
        _append_jsonl(self.output_dir / "evaluation_metrics.jsonl", record)
        return float(record["test_loss"]), {"test_accuracy": float(record["test_accuracy"])}


class PhysicalFedAvgStrategy(FedAvg):
    """Standard synchronous FedAvg with the physical experiment record schema."""

    def __init__(
        self,
        *,
        model: nn.Module,
        output_dir: Path,
        run_id: str,
        test_loader: DataLoader | None,
        evaluation_device: str,
        num_clients: int,
    ) -> None:
        self.model = model
        self.output_dir = output_dir
        self.run_id = str(run_id)
        self.test_loader = test_loader
        self.evaluation_device = str(evaluation_device)
        self.fit_records: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.evaluation_records: list[dict[str, Any]] = []
        self.round_started: dict[int, float] = {}
        self.cid_to_logical_id: dict[str, str] = {}
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=int(num_clients),
            min_evaluate_clients=0,
            min_available_clients=int(num_clients),
            evaluate_fn=self._evaluate_parameters,
            on_fit_config_fn=lambda server_round: {
                AUTOSPLIT_MODEL_VERSION_CONFIG_KEY: int(server_round)
            },
            accept_failures=True,
            initial_parameters=ndarrays_to_parameters(_model_to_ndarrays(model)),
        )

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_started[int(server_round)] = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        for proxy, fit_res in results:
            metrics = _json_safe(dict(fit_res.metrics or {}))
            logical_id = str(metrics.get("logical_client_id", ""))
            if logical_id:
                stale = [
                    cid
                    for cid, known in self.cid_to_logical_id.items()
                    if known == logical_id and cid != str(proxy.cid)
                ]
                for cid in stale:
                    self.cid_to_logical_id.pop(cid, None)
                self.cid_to_logical_id[str(proxy.cid)] = logical_id
            record = {
                "schema": "splitfleet.physical-client-metric.v1",
                "run_id": self.run_id,
                "method": FEDAVG_METHOD,
                "round_id": int(server_round),
                "split_key": "full_local",
                "flower_cid": str(proxy.cid),
                "logical_client_id": logical_id,
                "num_examples": int(fit_res.num_examples),
                "success": True,
                "metrics": metrics,
            }
            self.fit_records.append(record)
            _append_jsonl(self.output_dir / "client_metrics.jsonl", record)
        for failure in failures:
            record = {
                "schema": "splitfleet.physical-failure.v1",
                "run_id": self.run_id,
                "method": FEDAVG_METHOD,
                "round_id": int(server_round),
                "stage": "client_fit",
                "failure_reason": str(failure),
                "failure_type": type(failure).__name__,
                "failure_repr": repr(failure),
            }
            self.failures.append(record)
            _append_jsonl(self.output_dir / "failures.jsonl", record)
        ordered_results = sorted(
            results,
            key=lambda item: (
                str((item[1].metrics or {}).get("logical_client_id", "")),
                str(item[0].cid),
            ),
        )
        aggregated = super().aggregate_fit(server_round, ordered_results, failures)
        round_records = [
            row for row in self.fit_records if row["round_id"] == int(server_round)
        ]
        placements = {
            str(row["logical_client_id"]): "full_local" for row in round_records
        }
        round_record = {
            "schema": "splitfleet.physical-round-metric.v1",
            "run_id": self.run_id,
            "method": FEDAVG_METHOD,
            "round_id": int(server_round),
            "split_key": "full_local",
            "placements": placements,
            "round_time_sec": time.perf_counter()
            - self.round_started.get(int(server_round), time.perf_counter()),
            "successful_clients": len(round_records),
            "failed_clients": len(failures),
        }
        _append_jsonl(self.output_dir / "round_metrics.jsonl", round_record)
        return aggregated

    def _evaluate_parameters(self, server_round, arrays, config):
        _ = config
        # The existing physical SplitFed runner starts centralized evaluation
        # after round 1, so suppress Flower's additional round-0 evaluation.
        if int(server_round) == 0 or self.test_loader is None:
            return None
        _load_model_ndarrays(self.model, arrays)
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        correct = 0
        examples = 0
        with torch.inference_mode():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.evaluation_device)
                targets = targets.to(self.evaluation_device)
                logits = self.model(inputs)
                total_loss += float(loss_fn(logits, targets).item())
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                examples += int(targets.shape[0])
        record = {
            "schema": "splitfleet.physical-evaluation.v1",
            "run_id": self.run_id,
            "method": FEDAVG_METHOD,
            "round_id": int(server_round),
            "test_loss": total_loss / max(examples, 1),
            "test_accuracy": correct / max(examples, 1),
            "num_examples": examples,
            "source": "cifar10_test_set",
        }
        self.evaluation_records.append(record)
        _append_jsonl(self.output_dir / "evaluation_metrics.jsonl", record)
        return float(record["test_loss"]), {
            "test_accuracy": float(record["test_accuracy"])
        }


def _candidate_boundaries(model: nn.Module, sample_inputs: torch.Tensor) -> dict[str, str]:
    candidates = discover_split_candidates(model, sample_inputs)
    mapping = {item.split_key: str(item.boundary) for item in candidates}
    missing = sorted(set(DEFAULT_CANDIDATES) - set(mapping))
    if missing:
        raise RuntimeError(f"ResNet-18 candidate discovery missed required cuts: {missing}")
    return mapping


def run_server(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = _make_model(args.seed, device, args.model)
    sample_inputs = torch.zeros(2, 3, 32, 32, device=device)
    if args.method == FEDAVG_METHOD:
        boundaries = {"full_local": None}
    elif args.split_runtime == "native_prefix":
        # Explicit torchvision module cuts need neither graph discovery nor
        # TorchLens replay.
        boundaries = {key: key for key in DEFAULT_CANDIDATES}
    else:
        boundaries = _candidate_boundaries(model, sample_inputs)
    train, test, assignments, manifest = _load_partition(
        data_root=args.data_root,
        seed=args.seed,
        num_clients=args.num_clients,
        download=args.download,
    )
    estimated_batches_per_round = math.ceil(
        len(train) / args.num_clients / args.batch_size
    ) * int(args.local_epochs)
    schedule_policy = None
    placement_fn = None
    if args.method == FEDAVG_METHOD:
        boundary = None
    elif args.method in ProfileGuidedPlacementPolicy.METHODS:
        if not args.profile_report:
            raise ValueError(f"Method {args.method!r} requires --profile-report")
        profile_report = json.loads(
            Path(args.profile_report).read_text(encoding="utf-8")
        )
        schedule_policy = ProfileGuidedPlacementPolicy(
            boundaries=boundaries,
            profile_report=profile_report,
            method=args.method,
            batches_per_round=estimated_batches_per_round,
            calibration_rounds=args.calibration_rounds,
            server_concurrency=args.server_concurrency,
        )
        placement_fn = schedule_policy
        boundary = boundaries["stem"]
    else:
        if args.boundary_key not in boundaries:
            raise ValueError(f"Unknown boundary key {args.boundary_key!r}")
        boundary = boundaries[args.boundary_key]

    test_loader = None
    if not args.no_evaluate:
        test_loader = make_loader(
            test,
            None,
            batch_size=args.evaluation_batch_size,
            shuffle=False,
            seed=args.seed,
        )
    initial_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    source_manifest = _source_manifest()
    repo = Path(__file__).resolve().parents[2]
    protocol_path = Path(args.protocol_file)
    if not protocol_path.is_absolute():
        protocol_path = repo / protocol_path
    protocol_relative = protocol_path.resolve().relative_to(repo).as_posix()
    if protocol_relative not in source_manifest:
        raise ValueError(
            f"Protocol {protocol_relative!r} is not part of the frozen source manifest"
        )
    metadata = {
        "schema": "splitfleet.physical-run-metadata.v1",
        "run_id": args.run_id,
        "method": args.method,
        "seed": int(args.seed),
        "evidence_mode": "physical_multi_host",
        "git_commit": git_commit(),
        "environment": _environment(device),
        "source_manifest": source_manifest,
        "source_manifest_hash": stable_hash(source_manifest),
        "protocol_path": protocol_relative,
        "protocol_sha256": source_manifest[protocol_relative],
        "model": args.model,
        "initial_model_hash": tensor_state_hash(initial_state),
        "partition_hash": manifest["partition_hash"],
        "partition_manifest": manifest,
        "logical_client_partitions": {
            "win136": "0",
            "orin140": "1",
            "orin118": "2",
            "orin238": "3",
        },
        "candidate_boundaries": boundaries,
        "split_runtime": args.split_runtime,
        "prefix_executor": args.prefix_executor,
        "compile_backend": args.compile_backend,
        "parameter_sync_mode": (
            "full_model"
            if args.method == FEDAVG_METHOD
            else (
                "prefix_parameters"
                if args.split_runtime == "native_prefix"
                else "full_model"
            )
        ),
        "persistent_server_suffix": (
            args.method != FEDAVG_METHOD and args.split_runtime == "native_prefix"
        ),
        "torchlens_in_timed_path": (
            args.method != FEDAVG_METHOD and args.split_runtime != "native_prefix"
        ),
        "rounds": int(args.rounds),
        "batch_size": int(args.batch_size),
        "local_epochs": int(args.local_epochs),
        "train_samples": len(train),
        "test_samples": len(test),
        "data_pipeline": FLOWER_CIFAR10_PIPELINE,
        "partitioner": "flower_dirichlet",
        "optimizer": {
            "name": "sgd",
            "lr": float(args.learning_rate),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
        },
        "calibration_rounds": int(args.calibration_rounds),
        "profile_report": str(args.profile_report or ""),
        "profile_report_hash": (
            stable_hash(profile_report) if args.method in ProfileGuidedPlacementPolicy.METHODS else None
        ),
        "benchmark_classification": (
            "post_primary_secondary_traditional_fl_baseline"
            if args.method == FEDAVG_METHOD
            else (
                "post_hoc_systems_engineering_optimization"
                if args.split_runtime == "native_prefix"
                else (
                    "post_hoc_exploratory_large_model_stress"
                    if args.model != "resnet18"
                    else "original_splitfed_matrix"
                )
            )
        ),
        "server_role": (
            "flower_aggregation_and_central_evaluation"
            if args.method == FEDAVG_METHOD
            else "flower_control_suffix_and_central_evaluation"
        ),
        "resource_scenario": {
            "mode": "controlled_application_pacing_on_physical_link",
            "pacing_scope": "split_boundary_messages_only",
            "fedavg_disclosure": (
                "FedAvg has no split-boundary messages; full-model Flower traffic "
                "uses the physical LAN without added pacing, as it does in all methods."
                if args.method == FEDAVG_METHOD
                else None
            ),
            "paced_client_id": args.paced_client_id,
            "round_start": args.paced_round_start,
            "round_end": args.paced_round_end,
            "paced_uplink_mbps": args.paced_uplink_mbps,
            "paced_downlink_mbps": args.paced_downlink_mbps,
            "nominal_uplink_mbps": args.nominal_uplink_mbps,
            "nominal_downlink_mbps": args.nominal_downlink_mbps,
        },
        "started_unix": time.time(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.method == FEDAVG_METHOD:
        strategy = PhysicalFedAvgStrategy(
            model=model,
            output_dir=output,
            run_id=args.run_id,
            test_loader=test_loader,
            evaluation_device=device,
            num_clients=args.num_clients,
        )
        history = start_flower_server(
            server_address=args.bind,
            config=ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        metadata.update(
            {
                "finished_unix": time.time(),
                "observed_logical_clients": sorted(
                    set(strategy.cid_to_logical_id.values())
                ),
                "fit_failure_count": len(strategy.failures),
                "history": _json_safe(getattr(history, "__dict__", history)),
            }
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    method_name = (
        f"fixed_{args.boundary_key}" if args.method == "fixed" else args.method
    )
    if args.split_runtime == "native_prefix":
        split_selector = placement_fn or (
            lambda server_round, cid, training: args.boundary_key
        )
        strategy = NativePrefixSplitFedStrategy(
            model=model,
            model_factory=lambda: _make_model(args.seed, device, args.model),
            split_selector=split_selector,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            runtime_device=device,
            output_dir=output,
            run_id=args.run_id,
            method=method_name,
            test_loader=test_loader,
            evaluation_device=device,
            schedule_policy=schedule_policy,
            prefix_executor=args.prefix_executor,
            fraction_fit=1.0,
            min_fit_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )
        history = start_server(
            server_address=args.bind,
            config=ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        metadata.update(
            {
                "finished_unix": time.time(),
                "observed_logical_clients": sorted(
                    set(strategy.cid_to_logical_id.values())
                ),
                "fit_failure_count": len(strategy.failures),
                "history": _json_safe(getattr(history, "__dict__", history)),
            }
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    strategy = PhysicalRecordingStrategy(
        model=model,
        sample_inputs=sample_inputs,
        boundary=boundary,
        client_placement_fn=placement_fn,
        aggregation_policy="splitfed",
        dynamic_batch=(1, args.batch_size),
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_fn=lambda module: torch.optim.SGD(
            module.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        ),
        runtime_device=device,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        output_dir=output,
        run_id=args.run_id,
        method=method_name,
        model_seed=args.seed,
        model_name=args.model,
        test_loader=test_loader,
        evaluation_device=device,
        schedule_policy=schedule_policy,
        boundary_keys=boundaries,
    )
    # Build every required graph/runtime contract before the gRPC server starts
    # worker threads.  This keeps TorchLens capture inside its supported
    # single-owner-thread proof domain and moves preparation out of measured
    # round time.  Client-side preparation remains measured on each device.
    warmup_keys = getattr(schedule_policy, "required_split_keys", (args.boundary_key,))
    for split_key in warmup_keys:
        placement = strategy.get_or_create_placement_plan(boundaries[split_key])
        strategy._autosplit_config(0, training=True, placement=placement)
    history = start_server(
        server_address=args.bind,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    metadata.update(
        {
            "finished_unix": time.time(),
            "observed_logical_clients": sorted(set(strategy.cid_to_logical_id.values())),
            "fit_failure_count": len(strategy.failures),
            "history": _json_safe(getattr(history, "__dict__", history)),
        }
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_client(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    train, _test, assignments, manifest = _load_partition(
        data_root=args.data_root,
        seed=args.seed,
        num_clients=args.num_clients,
        download=args.download,
    )
    client_id = str(args.client_id)
    indices = assignments[str(args.client_index)]
    loader = make_loader(
        train,
        indices,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed * 1_000_003 + args.client_index,
    )
    batches = LocalEpochIterable(loader, args.local_epochs)
    model = _make_model(args.seed, device, args.model)
    environment = _environment(device)
    common_client = {
        "logical_client_id": client_id,
        "partition_hash": manifest["partition_hash"],
        "client_index_hash": stable_hash(indices),
        "environment": environment,
        "paced_client_id": args.paced_client_id,
        "paced_round_start": args.paced_round_start,
        "paced_round_end": args.paced_round_end,
        "paced_uplink_mbps": args.paced_uplink_mbps,
        "paced_downlink_mbps": args.paced_downlink_mbps,
        "nominal_uplink_mbps": args.nominal_uplink_mbps,
        "nominal_downlink_mbps": args.nominal_downlink_mbps,
    }
    if args.method == FEDAVG_METHOD:
        client = TaggedFedAvgClient(
            **common_client,
            model=model,
            train_data=batches,
            device=device,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.split_runtime == "native_prefix":
        client = NativePrefixSplitClient(
            **common_client,
            model=model,
            train_data=batches,
            sample_inputs=torch.zeros(2, 3, 32, 32, device=device),
            device=device,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            prefix_executor=args.prefix_executor,
            compile_backend=args.compile_backend,
        )
    else:
        client = TaggedPhysicalClient(
            **common_client,
            model=model,
            train_data=batches,
            evaluate_data=batches,
            sample_inputs=torch.zeros(2, 3, 32, 32, device=device),
            optimizer_fn=lambda module: torch.optim.SGD(
                module.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            ),
            device=device,
        )
    print(
        json.dumps(
            {
                "event": "client_start",
                "client_id": client_id,
                "client_index": int(args.client_index),
                "num_partition_examples": len(indices),
                "partition_hash": manifest["partition_hash"],
                "server": args.server,
                "method": args.method,
                "model": args.model,
                "split_runtime": args.split_runtime,
                "prefix_executor": args.prefix_executor,
                "compile_backend": args.compile_backend,
                "environment": environment,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    start_client(
        server_address=args.server,
        client=client.to_client(),
        max_retries=args.max_retries,
        max_wait_time=args.max_wait_time,
    )
    print(json.dumps({"event": "client_stop", "client_id": client_id}), flush=True)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet50", "resnet101", "wide_resnet50_2"),
        default="resnet18",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--paced-client-id")
    parser.add_argument("--paced-round-start", type=int)
    parser.add_argument("--paced-round-end", type=int)
    parser.add_argument("--paced-uplink-mbps", type=float, default=10.0)
    parser.add_argument("--paced-downlink-mbps", type=float, default=50.0)
    parser.add_argument("--nominal-uplink-mbps", type=float, default=1000.0)
    parser.add_argument("--nominal-downlink-mbps", type=float, default=1000.0)
    parser.add_argument(
        "--split-runtime",
        choices=("torchlens", "native_prefix"),
        default="torchlens",
    )
    parser.add_argument(
        "--prefix-executor",
        choices=PREFIX_EXECUTORS,
        default="persistent_eager",
    )
    parser.add_argument("--compile-backend", default="aot_eager")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)
    server = subparsers.add_parser("server")
    _common(server)
    server.add_argument("--bind", default="0.0.0.0:18097")
    server.add_argument("--run-id", required=True)
    server.add_argument(
        "--protocol-file",
        default="experiments/physical_ra_splitfed/protocol.yaml",
    )
    server.add_argument("--rounds", type=int, default=3)
    server.add_argument(
        "--method",
        choices=(
            "fixed",
            FEDAVG_METHOD,
            *ProfileGuidedPlacementPolicy.METHODS,
        ),
        default="fixed",
    )
    server.add_argument("--boundary-key", choices=DEFAULT_CANDIDATES, default="layer2")
    server.add_argument("--profile-report")
    server.add_argument("--calibration-rounds", type=int, default=3)
    server.add_argument("--server-concurrency", type=int, default=4)
    server.add_argument("--evaluation-batch-size", type=int, default=256)
    server.add_argument("--no-evaluate", action="store_true")
    server.add_argument("--device", default="auto")
    server.add_argument("--output", required=True)
    server.set_defaults(run=run_server)

    client = subparsers.add_parser("client")
    _common(client)
    client.add_argument("--server", required=True)
    client.add_argument(
        "--method",
        choices=(FEDAVG_METHOD, *ProfileGuidedPlacementPolicy.METHODS),
        default="best_global_fixed",
    )
    client.add_argument("--client-id", required=True)
    client.add_argument("--client-index", type=int, required=True)
    client.add_argument("--device", default="auto")
    client.add_argument("--max-retries", type=int, default=60)
    client.add_argument("--max-wait-time", type=float, default=900.0)
    client.set_defaults(run=run_client)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
