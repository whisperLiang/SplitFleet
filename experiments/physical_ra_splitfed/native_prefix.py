"""Native, state-partitioned SplitFed path for the physical experiment.

This module deliberately avoids TorchLens.  A client keeps one persistent
ResNet prefix executor and exchanges only the state entries owned by the
selected prefix.  A persistent per-client server replica owns and executes the
complementary suffix.  The strategy reassembles one logical model update per
client before applying sample-weighted SplitFed aggregation.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from flwr.common import (
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from torch import nn

from splitfleet.client.numpy_client import NumPyClient
from splitfleet.common import (
    BatchData,
    ControlCode,
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)
from splitfleet.common.constants import (
    AUTOSPLIT_MODEL_VERSION_CONFIG_KEY,
    CLIENT_ID_CONFIG_KEY,
)
from splitfleet.server.server_model.server_model import ServerModel
from splitfleet.server.strategy.plain_strategy import PlainSlStrategy
from splitfleet.transport import decode_bundle_wire, encode_bundle_wire


NATIVE_SPLIT_CONFIG_KEY = "splitfleet.native_split_key"
NATIVE_SYNC_MODE_CONFIG_KEY = "splitfleet.native_sync_mode"
NATIVE_EXECUTOR_CONFIG_KEY = "splitfleet.native_prefix_executor"
NATIVE_PROTOCOL_VERSION_CONFIG_KEY = "splitfleet.native_protocol_version"
NATIVE_PROTOCOL_VERSION = "native-prefix-v1"
PREFIX_PARAMETER_SYNC = "prefix_parameters"
SPLIT_KEYS = ("stem", "layer2", "layer4")
PREFIX_EXECUTORS = ("persistent_eager", "torch_compile")


def _state_scope(name: str) -> str:
    return str(name).split(".", 1)[0]


def prefix_state_names(model: nn.Module, split_key: str) -> tuple[str, ...]:
    """Return the deterministic state-dict subset owned by one prefix."""

    allowed = {
        "stem": {"conv1", "bn1"},
        "layer2": {"conv1", "bn1", "layer1", "layer2"},
        "layer4": {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"},
    }
    try:
        scopes = allowed[str(split_key)]
    except KeyError as exc:
        raise ValueError(f"Unsupported native ResNet split {split_key!r}") from exc
    return tuple(name for name in model.state_dict() if _state_scope(name) in scopes)


def suffix_state_names(model: nn.Module, split_key: str) -> tuple[str, ...]:
    prefix = set(prefix_state_names(model, split_key))
    return tuple(name for name in model.state_dict() if name not in prefix)


def validate_state_partition(model: nn.Module, split_key: str) -> None:
    all_names = tuple(model.state_dict())
    prefix = prefix_state_names(model, split_key)
    suffix = suffix_state_names(model, split_key)
    if set(prefix) & set(suffix):
        raise RuntimeError(f"Native split {split_key!r} has overlapping state ownership")
    if set(prefix) | set(suffix) != set(all_names):
        raise RuntimeError(f"Native split {split_key!r} does not cover the model state")
    if tuple(name for name in all_names if name in set(prefix)) != prefix:
        raise RuntimeError("Prefix state order differs from state_dict order")
    if tuple(name for name in all_names if name in set(suffix)) != suffix:
        raise RuntimeError("Suffix state order differs from state_dict order")


def export_named_state(model: nn.Module, names: Sequence[str]) -> list[np.ndarray]:
    state = model.state_dict()
    missing = [name for name in names if name not in state]
    if missing:
        raise ValueError(f"Unknown model state entries: {missing[:3]}")
    return [state[name].detach().cpu().numpy().copy() for name in names]


def load_named_state(
    model: nn.Module,
    names: Sequence[str],
    arrays: Sequence[np.ndarray],
) -> None:
    if len(names) != len(arrays):
        raise ValueError(
            f"Named state count mismatch: {len(arrays)} arrays for {len(names)} names"
        )
    state = model.state_dict()
    with torch.no_grad():
        for name, array in zip(names, arrays):
            if name not in state:
                raise ValueError(f"Unknown model state entry {name!r}")
            destination = state[name]
            source = torch.as_tensor(np.asarray(array), device=destination.device)
            if tuple(source.shape) != tuple(destination.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: received {tuple(source.shape)}, "
                    f"expected {tuple(destination.shape)}"
                )
            destination.copy_(source.to(dtype=destination.dtype))


def ndarray_bytes(arrays: Sequence[np.ndarray]) -> int:
    return sum(int(np.asarray(value).nbytes) for value in arrays)


def _sync_device(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class _PrefixCut(nn.Module):
    """A fixed-cut view over a persistent torchvision ResNet."""

    def __init__(self, model: nn.Module, split_key: str) -> None:
        super().__init__()
        if split_key not in SPLIT_KEYS:
            raise ValueError(f"Unsupported native ResNet split {split_key!r}")
        self.model = model
        self.split_key = str(split_key)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        model = self.model
        value = model.conv1(inputs)
        value = model.bn1(value)
        value = model.relu(value)
        if self.split_key == "stem":
            return value
        value = model.maxpool(value)
        value = model.layer1(value)
        value = model.layer2(value)
        if self.split_key == "layer2":
            return value
        value = model.layer3(value)
        return model.layer4(value)


class PersistentResNetPrefixExecutor:
    """Prebuild and optionally compile all fixed prefix graphs once per client."""

    def __init__(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        *,
        mode: str = "persistent_eager",
        compile_backend: str = "aot_eager",
        prewarm: bool = True,
    ) -> None:
        if mode not in PREFIX_EXECUTORS:
            raise ValueError(f"Unsupported prefix executor {mode!r}")
        self.model = model
        self.mode = str(mode)
        self.compile_backend = str(compile_backend)
        self.runners: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        started = time.perf_counter()
        for split_key in SPLIT_KEYS:
            view = _PrefixCut(self.model, split_key)
            if self.mode == "torch_compile":
                if not hasattr(torch, "compile"):
                    raise RuntimeError("torch_compile was requested but torch.compile is unavailable")
                runner = torch.compile(view, backend=self.compile_backend, dynamic=True)
            else:
                runner = view
            self.runners[split_key] = runner
        if prewarm:
            self._prewarm(sample_inputs)
        _sync_device(str(sample_inputs.device))
        self.prepare_duration_sec = time.perf_counter() - started

    def _prewarm(self, sample_inputs: torch.Tensor) -> None:
        previous_training = self.model.training
        self.model.train()
        for split_key in SPLIT_KEYS:
            self.model.zero_grad(set_to_none=True)
            sample = sample_inputs.detach().clone().requires_grad_(False)
            boundary = self.runners[split_key](sample)
            boundary.sum().backward()
        self.model.zero_grad(set_to_none=True)
        self.model.train(previous_training)

    def __call__(self, inputs: torch.Tensor, split_key: str) -> torch.Tensor:
        try:
            runner = self.runners[str(split_key)]
        except KeyError as exc:
            raise ValueError(f"Unsupported native ResNet split {split_key!r}") from exc
        return runner(inputs)


def forward_resnet_suffix(
    model: nn.Module,
    boundary: torch.Tensor,
    split_key: str,
) -> torch.Tensor:
    value = boundary
    if split_key == "stem":
        value = model.maxpool(value)
        value = model.layer1(value)
        value = model.layer2(value)
        value = model.layer3(value)
        value = model.layer4(value)
    elif split_key == "layer2":
        value = model.layer3(value)
        value = model.layer4(value)
    elif split_key != "layer4":
        raise ValueError(f"Unsupported native ResNet split {split_key!r}")
    value = model.avgpool(value)
    value = torch.flatten(value, 1)
    return model.fc(value)


class NativePrefixSplitClient(NumPyClient):
    """Train only the selected persistent native prefix on a physical client."""

    def __init__(
        self,
        *,
        logical_client_id: str,
        partition_hash: str,
        client_index_hash: str,
        environment: Mapping[str, Any],
        model: nn.Module,
        train_data: Iterable[Any],
        sample_inputs: torch.Tensor,
        device: str,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        prefix_executor: str = "persistent_eager",
        compile_backend: str = "aot_eager",
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
        for split_key in SPLIT_KEYS:
            validate_state_partition(self.model, split_key)
        self.executor = PersistentResNetPrefixExecutor(
            self.model,
            sample_inputs,
            mode=prefix_executor,
            compile_backend=compile_backend,
            prewarm=True,
        )
        self.paced_client_id = str(paced_client_id or "")
        self.paced_round_start = paced_round_start
        self.paced_round_end = paced_round_end
        self.paced_uplink_mbps = float(paced_uplink_mbps)
        self.paced_downlink_mbps = float(paced_downlink_mbps)
        self.nominal_uplink_mbps = float(nominal_uplink_mbps)
        self.nominal_downlink_mbps = float(nominal_downlink_mbps)

    def get_parameters(self, config):
        _ = config
        # The strategy owns initialization and sends a cut-specific prefix.
        return []

    def fit(self, parameters, config):
        fit_started = time.perf_counter()
        split_key = str(config[NATIVE_SPLIT_CONFIG_KEY])
        if str(config.get(NATIVE_PROTOCOL_VERSION_CONFIG_KEY, "")) != NATIVE_PROTOCOL_VERSION:
            raise ValueError("Native prefix protocol version mismatch")
        if str(config.get(NATIVE_SYNC_MODE_CONFIG_KEY, "")) != PREFIX_PARAMETER_SYNC:
            raise ValueError("Native prefix client requires prefix-parameter synchronization")
        if str(config.get(NATIVE_EXECUTOR_CONFIG_KEY, "")) != self.executor.mode:
            raise ValueError("Native prefix executor differs from the server configuration")
        names = prefix_state_names(self.model, split_key)
        download_bytes = ndarray_bytes(parameters)
        prepare_started = time.perf_counter()
        load_named_state(self.model, names, parameters)
        self.model.train()
        parameter_names = set(names)
        trainable = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name in parameter_names
        ]
        if not trainable:
            raise RuntimeError(f"Native prefix {split_key!r} has no trainable parameters")
        optimizer = torch.optim.SGD(
            trainable,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        _sync_device(self.device)
        runtime_prepare_sec = time.perf_counter() - prepare_started
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        round_id = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        flower_cid = str(config.get(CLIENT_ID_CONFIG_KEY, ""))
        paced = (
            self.logical_client_id == self.paced_client_id
            and self.paced_round_start is not None
            and self.paced_round_end is not None
            and int(self.paced_round_start) <= round_id <= int(self.paced_round_end)
        )
        active_uplink = self.paced_uplink_mbps if paced else self.nominal_uplink_mbps
        active_downlink = self.paced_downlink_mbps if paced else self.nominal_downlink_mbps

        num_examples = 0
        num_batches = 0
        weighted_loss = 0.0
        prefix_compute_sec = 0.0
        tail_wait_sec = 0.0
        boundary_upload_bytes = 0
        boundary_download_bytes = 0
        pacing_delay_sec = 0.0
        min_batch_size = 0
        max_batch_size = 0
        for inputs, targets in self.train_data:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = int(targets.shape[0])
            optimizer.zero_grad(set_to_none=True)
            _sync_device(self.device)
            prefix_started = time.perf_counter()
            boundary = self.executor(inputs, split_key)
            _sync_device(self.device)
            prefix_compute_sec += time.perf_counter() - prefix_started

            boundary_payload = encode_bundle_wire(boundary.detach(), backend="torch")
            target_payload = encode_bundle_wire(targets, backend="torch")
            request = BatchData(
                data={
                    "boundary": boundary_payload,
                    "targets": target_payload,
                    "metadata": json.dumps(
                        {
                            "num_examples": batch_size,
                            "round_id": round_id,
                            "client_id": flower_cid,
                            "split_key": split_key,
                        },
                        sort_keys=True,
                    ).encode("utf-8"),
                },
                control_code=ControlCode.OK,
            )
            tail_started = time.perf_counter()
            response = self._require_server_model_proxy().train_native_tail(
                request, _streams_=False
            )
            response_bytes = sum(
                len(value) for value in response.data.values() if isinstance(value, bytes)
            )
            delay = (
                (len(boundary_payload) + len(target_payload))
                * 8.0
                / (max(active_uplink, 1e-9) * 1_000_000.0)
                + response_bytes
                * 8.0
                / (max(active_downlink, 1e-9) * 1_000_000.0)
            )
            if delay > 0:
                time.sleep(delay)
                pacing_delay_sec += delay
            tail_wait_sec += time.perf_counter() - tail_started
            boundary_upload_bytes += len(boundary_payload) + len(target_payload)
            boundary_download_bytes += response_bytes

            response_metadata = json.loads(response.data["metadata"].decode("utf-8"))
            if int(response_metadata["round_id"]) != round_id:
                raise RuntimeError("Native suffix returned a stale model version")
            gradient = decode_bundle_wire(response.data["gradient"], self.device)
            _sync_device(self.device)
            backward_started = time.perf_counter()
            torch.autograd.backward(boundary, gradient)
            optimizer.step()
            _sync_device(self.device)
            prefix_compute_sec += time.perf_counter() - backward_started

            examples = int(response_metadata["num_examples"])
            loss_value = float(response_metadata["loss"])
            num_examples += examples
            num_batches += 1
            weighted_loss += loss_value * examples
            min_batch_size = batch_size if min_batch_size == 0 else min(min_batch_size, batch_size)
            max_batch_size = max(max_batch_size, batch_size)

        if num_examples <= 0:
            raise RuntimeError("Native prefix client executed no training examples")
        updated = export_named_state(self.model, names)
        upload_bytes = ndarray_bytes(updated)
        metrics: dict[str, Scalar] = {
            "logical_client_id": self.logical_client_id,
            "partition_hash": self.partition_hash,
            "client_index_hash": self.client_index_hash,
            "hostname": str(self.environment["hostname"]),
            "torch_version": str(self.environment["torch"]),
            "runner_sha256": str(self.environment["runner_sha256"]),
            "source_manifest_hash": str(self.environment["source_manifest_hash"]),
            "device": self.device,
            "loss": weighted_loss / num_examples,
            "fit_duration_sec": time.perf_counter() - fit_started,
            "runtime_prepare_sec": runtime_prepare_sec,
            "prefix_compute_sec": prefix_compute_sec,
            "tail_wait_sec": tail_wait_sec,
            "num_examples": num_examples,
            "num_batches": num_batches,
            "local_epochs": int(getattr(self.train_data, "local_epochs", 1)),
            "optimizer_momentum": self.momentum,
            "optimizer_weight_decay": self.weight_decay,
            "skipped_batches": 0,
            "skipped_examples": 0,
            "min_batch_size": min_batch_size,
            "max_batch_size": max_batch_size,
            "boundary": split_key,
            "split_key": split_key,
            "model_download_bytes": download_bytes,
            "model_upload_bytes": upload_bytes,
            "full_model_parameter_bytes": int(config["full_model_bytes"]),
            "prefix_parameter_download_bytes": download_bytes,
            "prefix_parameter_upload_bytes": upload_bytes,
            "suffix_parameter_download_bytes": 0,
            "suffix_parameter_upload_bytes": 0,
            "upload_bytes": boundary_upload_bytes,
            "download_bytes": boundary_download_bytes,
            "boundary_upload_bytes": boundary_upload_bytes,
            "boundary_download_bytes": boundary_download_bytes,
            "app_pacing_delay_sec": pacing_delay_sec,
            "pacing_scope": "split_boundary_messages_only",
            "resource_phase": "paced_link" if paced else "normal_link",
            "configured_uplink_mbps": active_uplink,
            "configured_downlink_mbps": active_downlink,
            "parameter_sync_mode": PREFIX_PARAMETER_SYNC,
            "prefix_executor": self.executor.mode,
            "prefix_executor_prepare_sec_process_start": self.executor.prepare_duration_sec,
            "torchlens_in_timed_path": False,
        }
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            metrics["peak_cuda_memory_mb"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
            )
        return updated, num_examples, metrics

    def evaluate(self, parameters, config):
        _ = (parameters, config)
        return 0.0, 0, {"central_evaluation_only": True}

    def _require_server_model_proxy(self):
        proxy = getattr(self, "server_model_proxy", None)
        if proxy is None:
            raise RuntimeError("NativePrefixSplitClient requires a server_model_proxy")
        return proxy


class PersistentNativeResNetTailServerModel(ServerModel):
    """Keep one client-specific ResNet suffix resident across all rounds."""

    def __init__(
        self,
        *,
        model: nn.Module,
        device: str,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = str(device)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Optimizer | None = None
        self.split_key = ""
        self.sid = ""
        self.round_id = 0
        self.num_examples = 0
        self.loss_total = 0.0
        self.server_compute_sec = 0.0
        self.server_request_count = 0
        self.runtime_prepare_sec = 0.0
        self.configure_count = 0
        self._lock = RLock()
        for split_key in SPLIT_KEYS:
            validate_state_partition(self.model, split_key)

    def get_parameters(self):
        if not self.split_key:
            return []
        return export_named_state(self.model, suffix_state_names(self.model, self.split_key))

    def configure_fit(self, ins: ServerModelFitIns) -> None:
        started = time.perf_counter()
        split_key = str(ins.config[NATIVE_SPLIT_CONFIG_KEY])
        if str(ins.config.get(NATIVE_PROTOCOL_VERSION_CONFIG_KEY, "")) != NATIVE_PROTOCOL_VERSION:
            raise ValueError("Native suffix protocol version mismatch")
        names = suffix_state_names(self.model, split_key)
        load_named_state(self.model, names, ins.parameters)
        self.model.train()
        trainable_names = set(names)
        trainable = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name in trainable_names
        ]
        if not trainable:
            raise RuntimeError(f"Native suffix {split_key!r} has no trainable parameters")
        self.optimizer = torch.optim.SGD(
            trainable,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.split_key = split_key
        self.sid = str(ins.sid)
        self.round_id = int(ins.config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        self.num_examples = 0
        self.loss_total = 0.0
        self.server_compute_sec = 0.0
        self.server_request_count = 0
        _sync_device(self.device)
        self.runtime_prepare_sec = time.perf_counter() - started
        self.configure_count += 1
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def train_native_tail(self, batches: list[BatchData]) -> list[BatchData]:
        responses = []
        with self._lock:
            for batch in batches:
                metadata = json.loads(batch.data["metadata"].decode("utf-8"))
                if int(metadata["round_id"]) != self.round_id:
                    raise RuntimeError("Native suffix received a stale model version")
                if str(metadata["split_key"]) != self.split_key:
                    raise RuntimeError("Native suffix received the wrong split key")
                boundary = decode_bundle_wire(batch.data["boundary"], self.device)
                targets = decode_bundle_wire(batch.data["targets"], self.device)
                boundary = boundary.detach().requires_grad_(True)
                if self.optimizer is None:
                    raise RuntimeError("Native suffix optimizer is not configured")
                self.optimizer.zero_grad(set_to_none=True)
                _sync_device(self.device)
                started = time.perf_counter()
                logits = forward_resnet_suffix(self.model, boundary, self.split_key)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                gradient = boundary.grad.detach().clone()
                self.optimizer.step()
                _sync_device(self.device)
                self.server_compute_sec += time.perf_counter() - started
                self.server_request_count += 1
                examples = int(metadata["num_examples"])
                loss_value = float(loss.detach().cpu())
                self.num_examples += examples
                self.loss_total += loss_value * examples
                responses.append(
                    BatchData(
                        data={
                            "gradient": encode_bundle_wire(gradient, backend="torch"),
                            "metadata": json.dumps(
                                {
                                    "loss": loss_value,
                                    "num_examples": examples,
                                    "round_id": self.round_id,
                                    "split_key": self.split_key,
                                },
                                sort_keys=True,
                            ).encode("utf-8"),
                        },
                        control_code=ControlCode.OK,
                        metadata={"sid": self.sid},
                    )
                )
        return responses

    def get_fit_result(self) -> ServerModelFitRes:
        names = suffix_state_names(self.model, self.split_key)
        config: dict[str, Scalar] = {
            "num_examples": self.num_examples,
            "avg_loss": self.loss_total / max(self.num_examples, 1),
            "server_compute_sec": self.server_compute_sec,
            "server_request_count": self.server_request_count,
            "runtime_prepare_sec": self.runtime_prepare_sec,
            "split_key": self.split_key,
            "suffix_parameter_bytes": ndarray_bytes(export_named_state(self.model, names)),
            "persistent_server_model": True,
            "persistent_reused": self.configure_count > 1,
            "configure_count": self.configure_count,
            "optimizer_momentum": self.momentum,
            "optimizer_weight_decay": self.weight_decay,
            "torchlens_in_timed_path": False,
        }
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            _sync_device(self.device)
            config.update(
                {
                    "server_cuda_allocated_mb": torch.cuda.memory_allocated(self.device)
                    / (1024.0**2),
                    "server_cuda_reserved_mb": torch.cuda.memory_reserved(self.device)
                    / (1024.0**2),
                    "server_cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(
                        self.device
                    )
                    / (1024.0**2),
                }
            )
        return ServerModelFitRes(
            parameters=export_named_state(self.model, names),
            config=config,
        )

    def configure_evaluate(self, ins: ServerModelEvaluateIns) -> None:
        _ = ins


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _append_jsonl(path: Path, value: Any) -> None:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class NativePrefixSplitFedStrategy(PlainSlStrategy):
    """Aggregate cut-specific prefix/suffix updates into full logical models."""

    persistent_server_models = True

    def __init__(
        self,
        *,
        model: nn.Module,
        model_factory: Callable[[], nn.Module],
        split_selector: Callable[[int, str, bool], str],
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        runtime_device: str,
        output_dir: Path,
        run_id: str,
        method: str,
        test_loader: Iterable[Any] | None,
        evaluation_device: str,
        schedule_policy: Any | None,
        prefix_executor: str,
        fraction_fit: float,
        min_fit_clients: int,
        min_available_clients: int,
    ) -> None:
        self.model = model
        self.model_factory = model_factory
        self.split_selector = split_selector
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.runtime_device = str(runtime_device)
        self.output_dir = Path(output_dir)
        self.run_id = str(run_id)
        self.method = str(method)
        self.test_loader = test_loader
        self.evaluation_device = str(evaluation_device)
        self.schedule_policy = schedule_policy
        self.prefix_executor = str(prefix_executor)
        self.state_names = tuple(model.state_dict())
        self._initial_arrays = export_named_state(model, self.state_names)
        self._evaluation_model = model_factory().to(evaluation_device)
        self._round_splits: dict[tuple[int, str], str] = {}
        self._round_client_updates: dict[
            int, dict[str, tuple[list[np.ndarray], int]]
        ] = {}
        self._round_suffix_updates: dict[int, dict[str, list[np.ndarray]]] = {}
        self._round_initial: dict[int, list[np.ndarray]] = {}
        self.round_started: dict[int, float] = {}
        self.cid_to_logical_id: dict[str, str] = {}
        self.fit_records: list[dict[str, Any]] = []
        self.server_fit_records: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.evaluation_records: list[dict[str, Any]] = []
        for split_key in SPLIT_KEYS:
            validate_state_partition(model, split_key)
        super().__init__(
            init_server_model_fn=self._make_server_model,
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
            common_server_model=False,
            process_clients_as_batch=False,
        )

    def _make_server_model(self) -> ServerModel:
        return PersistentNativeResNetTailServerModel(
            model=self.model_factory(),
            device=self.runtime_device,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            loss_fn=nn.CrossEntropyLoss(),
        )

    def initialize_parameters(self, client_manager) -> Parameters:
        _ = client_manager
        return ndarrays_to_parameters([array.copy() for array in self._initial_arrays])

    def initialize_server_parameters(self):
        return [array.copy() for array in self._initial_arrays]

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        self.round_started[int(server_round)] = time.perf_counter()
        full_arrays = parameters_to_ndarrays(parameters)
        if len(full_arrays) != len(self.state_names):
            raise RuntimeError("Native strategy lost the full server-side global state")
        self._round_initial[int(server_round)] = [array.copy() for array in full_arrays]
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = self.select_fit_clients(
            server_round=server_round,
            client_manager=client_manager,
            sample_size=sample_size,
            min_num_clients=min_num_clients,
        )
        self._round_active_clients = [client.cid for client in clients]
        full = dict(zip(self.state_names, full_arrays))
        instructions = []
        for client in clients:
            split_key = str(self.split_selector(server_round, client.cid, True))
            if split_key not in SPLIT_KEYS:
                raise ValueError(f"Split selector returned unsupported key {split_key!r}")
            self._round_splits[(int(server_round), str(client.cid))] = split_key
            names = prefix_state_names(self.model, split_key)
            config: dict[str, Scalar] = {
                AUTOSPLIT_MODEL_VERSION_CONFIG_KEY: int(server_round),
                NATIVE_SPLIT_CONFIG_KEY: split_key,
                NATIVE_SYNC_MODE_CONFIG_KEY: PREFIX_PARAMETER_SYNC,
                NATIVE_EXECUTOR_CONFIG_KEY: self.prefix_executor,
                NATIVE_PROTOCOL_VERSION_CONFIG_KEY: NATIVE_PROTOCOL_VERSION,
                "full_model_bytes": ndarray_bytes(full_arrays),
                "prefix_parameter_bytes": ndarray_bytes([full[name] for name in names]),
            }
            instructions.append(
                (
                    client,
                    FitIns(
                        ndarrays_to_parameters([full[name] for name in names]), config
                    ),
                )
            )
        return instructions

    def configure_server_fit(
        self,
        server_round: int,
        parameters: Sequence[np.ndarray],
        cids: list[str],
    ) -> list[ServerModelFitIns]:
        if len(parameters) != len(self.state_names):
            raise RuntimeError("Native suffix configuration requires the full global state")
        full = dict(zip(self.state_names, parameters))
        self._cid_to_sid_mapping = {str(cid): str(cid) for cid in cids}
        instructions = []
        for cid in cids:
            split_key = self._round_splits[(int(server_round), str(cid))]
            names = suffix_state_names(self.model, split_key)
            config: dict[str, Scalar] = {
                AUTOSPLIT_MODEL_VERSION_CONFIG_KEY: int(server_round),
                NATIVE_SPLIT_CONFIG_KEY: split_key,
                NATIVE_SYNC_MODE_CONFIG_KEY: PREFIX_PARAMETER_SYNC,
                NATIVE_PROTOCOL_VERSION_CONFIG_KEY: NATIVE_PROTOCOL_VERSION,
                "suffix_parameter_bytes": ndarray_bytes([full[name] for name in names]),
            }
            instructions.append(
                ServerModelFitIns(
                    parameters=[full[name] for name in names],
                    config=config,
                    sid=str(cid),
                )
            )
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        pending: dict[str, tuple[list[np.ndarray], int]] = {}
        for proxy, fit_res in results:
            cid = str(proxy.cid)
            split_key = self._round_splits[(int(server_round), cid)]
            arrays = parameters_to_ndarrays(fit_res.parameters)
            expected = prefix_state_names(self.model, split_key)
            if len(arrays) != len(expected):
                raise RuntimeError(
                    f"Client {cid} returned {len(arrays)} prefix arrays, expected {len(expected)}"
                )
            pending[cid] = (arrays, int(fit_res.num_examples))
            metrics = _json_safe(dict(fit_res.metrics or {}))
            logical_id = str(metrics.get("logical_client_id", ""))
            if logical_id:
                for stale in [
                    known_cid
                    for known_cid, known_id in self.cid_to_logical_id.items()
                    if known_id == logical_id and known_cid != cid
                ]:
                    self.cid_to_logical_id.pop(stale, None)
                self.cid_to_logical_id[cid] = logical_id
            if self.schedule_policy is not None and hasattr(
                self.schedule_policy, "observe_fit_metrics"
            ):
                self.schedule_policy.observe_fit_metrics(
                    round_id=int(server_round),
                    cid=cid,
                    num_examples=int(fit_res.num_examples),
                    metrics=metrics,
                )
            record = {
                "schema": "splitfleet.physical-client-metric.v1",
                "run_id": self.run_id,
                "method": self.method,
                "round_id": int(server_round),
                "split_key": split_key,
                "flower_cid": cid,
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
        self._round_client_updates[int(server_round)] = pending
        self.requests_state = {}
        self._round_active_clients = []
        total_examples = sum(examples for _, examples in pending.values())
        weighted_loss = sum(
            float(fit_res.metrics.get("loss", 0.0)) * int(fit_res.num_examples)
            for _, fit_res in results
        )
        metrics = {
            "loss": weighted_loss / max(total_examples, 1),
            "successful_clients": len(results),
        }
        return None, metrics

    def aggregate_server_fit(self, server_round, results):
        suffix_updates: dict[str, list[np.ndarray]] = {}
        for result in results:
            cid = str(result.sid)
            split_key = self._round_splits[(int(server_round), cid)]
            expected = suffix_state_names(self.model, split_key)
            if len(result.parameters) != len(expected):
                raise RuntimeError(
                    f"Suffix {cid} returned {len(result.parameters)} arrays, expected {len(expected)}"
                )
            suffix_updates[cid] = list(result.parameters)
            record = {
                "schema": "splitfleet.physical-server-metric.v1",
                "run_id": self.run_id,
                "method": self.method,
                "round_id": int(server_round),
                "split_key": split_key,
                "sid": cid,
                "logical_client_id": self.cid_to_logical_id.get(cid, ""),
                "config": _json_safe(dict(result.config)),
            }
            self.server_fit_records.append(record)
            _append_jsonl(self.output_dir / "server_metrics.jsonl", record)
        self._round_suffix_updates[int(server_round)] = suffix_updates
        round_clients = [
            row for row in self.fit_records if row["round_id"] == int(server_round)
        ]
        placements = {
            str(row["logical_client_id"]): str(row["split_key"])
            for row in round_clients
        }
        distinct = sorted(set(placements.values()))
        round_record = {
            "schema": "splitfleet.physical-round-metric.v1",
            "run_id": self.run_id,
            "method": self.method,
            "round_id": int(server_round),
            "split_key": distinct[0] if len(distinct) == 1 else "mixed",
            "placements": placements,
            "round_time_sec": time.perf_counter()
            - self.round_started.get(int(server_round), time.perf_counter()),
            "successful_clients": len(round_clients),
            "failed_clients": sum(
                row["round_id"] == int(server_round) for row in self.failures
            ),
        }
        _append_jsonl(self.output_dir / "round_metrics.jsonl", round_record)
        decisions = getattr(self.schedule_policy, "decision_records", None)
        if decisions is not None:
            for decision in [
                row for row in decisions if row["round_id"] == int(server_round)
            ]:
                _append_jsonl(self.output_dir / "split_decisions.jsonl", decision)
        return None

    def finalize_round(self, server_round, client_parameters, server_parameters):
        _ = (client_parameters, server_parameters)
        round_id = int(server_round)
        client_updates = self._round_client_updates.get(round_id, {})
        suffix_updates = self._round_suffix_updates.get(round_id, {})
        initial = dict(zip(self.state_names, self._round_initial[round_id]))
        weighted_states: list[tuple[list[np.ndarray], int]] = []
        ordered_cids = sorted(
            client_updates,
            key=lambda cid: (self.cid_to_logical_id.get(cid, cid), cid),
        )
        for cid in ordered_cids:
            prefix_arrays, examples = client_updates[cid]
            if cid not in suffix_updates:
                continue
            split_key = self._round_splits[(round_id, cid)]
            prefix_names = prefix_state_names(self.model, split_key)
            suffix_names = suffix_state_names(self.model, split_key)
            assembled = dict(initial)
            assembled.update(zip(prefix_names, prefix_arrays))
            assembled.update(zip(suffix_names, suffix_updates[cid]))
            if set(assembled) != set(self.state_names):
                raise RuntimeError(f"Reassembled update for {cid} is incomplete")
            weighted_states.append(
                ([np.asarray(assembled[name]) for name in self.state_names], examples)
            )
        if not weighted_states:
            self._cleanup_round(round_id)
            return None, None
        aggregated = aggregate(weighted_states)
        full_parameters = ndarrays_to_parameters(aggregated)
        self._cleanup_round(round_id)
        return full_parameters, aggregated

    def _cleanup_round(self, round_id: int) -> None:
        self._round_client_updates.pop(round_id, None)
        self._round_suffix_updates.pop(round_id, None)
        self._round_initial.pop(round_id, None)
        for key in [key for key in self._round_splits if key[0] <= round_id]:
            self._round_splits.pop(key, None)

    def configure_evaluate(self, server_round, parameters, client_manager):
        _ = (server_round, parameters, client_manager)
        return []

    def configure_server_evaluate(self, server_round, parameters, cids):
        _ = (server_round, parameters, cids)
        return []

    def evaluate(self, server_round, client_parameters, server_parameters):
        _ = server_parameters
        if self.test_loader is None:
            return None
        arrays = parameters_to_ndarrays(client_parameters)
        load_named_state(self._evaluation_model, self.state_names, arrays)
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
        return float(record["test_loss"]), {
            "test_accuracy": float(record["test_accuracy"])
        }
