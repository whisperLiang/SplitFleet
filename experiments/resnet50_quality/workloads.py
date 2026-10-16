"""Frozen CIFAR inputs and complete native ResNet50 virtual training hooks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch

from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBPlacementPolicy,
)
from splitfleet.server.placement.cosplit_ucb.candidate_provider import (
    TorchLensCandidateProvider,
)
from splitfleet.tasks import TASK_SPECS
from splitfleet.tasks.base import TaskBatch
from splitfleet.virtual.backend_ops import FrameworkOps
from splitfleet.virtual.runtime import TrainingSpec

from .data import load_cifar
from .model_contract import NAMES, TRAINABLE_PARAMETERS, state_hash
from .native import (
    BACKENDS,
    build_model,
    configure_backend,
    native_parameters,
    train_step,
)
from .protocol import hash_file, learning_rate, validate_main_protocol, validate_worker_protocol


@lru_cache(maxsize=1)
def cifar_data(root):
    return load_cifar(root)


@lru_cache(maxsize=1)
def frozen_initial(path):
    with np.load(path) as record:
        return {name: record[name].copy() for name in NAMES}


@dataclass(frozen=True)
class QualityFactory:
    data_root: str
    initial_path: str
    backend: str = "pytorch"
    device: str = "cpu"

    def model(self, device="cpu"):
        configure_backend(self.backend)
        model = build_model(self.backend, frozen_initial(self.initial_path), device)
        FrameworkOps(self.backend).set_training(model, True)
        if sum(int(np.prod(value.shape)) for value in native_parameters(self.backend, model).values()) != TRAINABLE_PARAMETERS:
            raise RuntimeError("Quality workload requires the complete native ResNet50")
        return model

    def dataset(self):
        return QualityDataset(cifar_data(self.data_root), backend=self.backend, device=self.device)


class QualityDataset:
    def __init__(self, data, *, backend="pytorch", device="cpu"):
        self.data, self.backend, self.device = data, backend, device

    def __len__(self):
        return len(self.data.train_images)

    def __getitem__(self, sample_id):
        return (torch.from_numpy(self.data.images([sample_id])[0]),
                torch.tensor(self.data.train_labels[sample_id], dtype=torch.int64))


class QualityBatches:
    """The shared sample/client/round keyed augmentation and ordering protocol."""

    def __init__(self, dataset, indices, *, batch_size, seed, partition_id, round_id=1,
                 local_epochs=1, collate_fn=None, deadline=None, fail_after=None):
        if local_epochs != 1:
            raise ValueError("The frozen quality protocol requires exactly one local epoch")
        self.dataset, self.indices = dataset, list(indices)
        self.batch_size, self.seed, self.partition_id, self.round_id = batch_size, seed, partition_id, round_id
        self.deadline, self.fail_after = deadline, fail_after
        self.executed, self.samples_seen, self.batch_hash = 0, 0, ""

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        digest = hashlib.sha256()
        self.executed, self.samples_seen = 0, 0
        for ids, images, labels in self.dataset.data.batches(
            self.indices, seed=self.seed, client=self.partition_id, round_id=self.round_id,
            batch_size=self.batch_size,
        ):
            if self.deadline is not None and time.perf_counter() >= self.deadline:
                raise TimeoutError("Quality client deadline exceeded")
            if self.fail_after is not None and self.executed >= self.fail_after:
                raise ConnectionError("Injected virtual client dropout")
            digest.update(np.asarray(ids, dtype="<i8").tobytes())
            digest.update(images.tobytes())
            digest.update(labels.tobytes())
            yield torch.from_numpy(images), torch.from_numpy(labels)
            self.executed += 1
            self.samples_seen += len(ids)
            self.batch_hash = digest.hexdigest()


def full_model_trainer(model, batch, optimizer, device, lr, *, backend="pytorch"):
    if isinstance(batch, TaskBatch):
        batch = FrameworkOps(backend).prepare_batch(model, batch, device)
        images, labels = batch.inputs.args[-1], batch.targets
    else:
        images, labels = (value.to(device) for value in batch)
    result = train_step(backend, model, images, labels, native_optimizer=optimizer, learning_rate=lr)
    return len(labels), result["loss"]


class StageCandidateProvider(TorchLensCandidateProvider):
    """Preserve the complete native catalog and separately describe stage eligibility."""

    def __init__(self, *, backend="pytorch", **kwargs):
        self.quality_backend = backend
        kwargs["require_trainable_prefix"] = True
        super().__init__(**kwargs)
        self.stage_endpoints = None

    def stage_boundaries(self):
        self.get_candidates(training=True)
        if self.stage_endpoints is None:
            handle = self._backends[True].make_handle()
            from .gate_torch import _stage_boundaries
            self.stage_endpoints = _stage_boundaries(handle)
        result = tuple(row["boundary"] for row in self.stage_endpoints.values())
        if not set(result).issubset(self._candidates[True]):
            raise RuntimeError("Native full catalog excludes a required residual stage endpoint")
        return result


@dataclass(frozen=True)
class QualityEvaluation:
    data_root: str
    test: bool = False
    backend: str = "pytorch"

    def __call__(self, model, device):
        data = cifar_data(self.data_root)
        indices = np.arange(len(data.test_images)) if self.test else data.validation_indices
        loss_sum, correct = 0.0, 0
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(indices), 64):
                ids = indices[start:start + 64]
                inputs = torch.from_numpy(data.images(ids, test=self.test)).to(device)
                labels = torch.from_numpy((data.test_labels if self.test else data.train_labels)[ids]).to(device)
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                if not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
                    raise FloatingPointError("Quality evaluation contains nonfinite values")
                loss_sum += float(loss)
                correct += int((logits.argmax(1) == labels).sum())
        return {"loss": loss_sum / len(indices), "top1": correct / len(indices), "samples": len(indices)}


@dataclass
class RoundCheckpoints:
    backend: str = "pytorch"
    started: float = field(default_factory=time.perf_counter, repr=False)

    def start(self, started):
        self.started = started

    def __call__(self, round_id, arrays, metrics, output, *, phase):
        output = Path(output)
        state = dict(zip(NAMES, arrays, strict=True))
        state_hash(state)
        if round_id > 0:
            path = output / "round_metrics.csv"
            if path.exists():
                with path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
            else:
                rows = []
            if phase == "training":
                if any(int(row["round"]) == round_id for row in rows):
                    raise RuntimeError("Quality round metrics cannot replay a completed training round")
                cumulative = int(rows[-1]["examples_seen"]) if rows else 0
                rows.append({"round": round_id, "train_online_ce": metrics["train_loss"],
                             "val_ce": None, "val_top1": None, "elapsed_wall_sec": None,
                             "examples_seen": cumulative + metrics["num_examples"],
                             "successful_clients": metrics["successful_clients"],
                             "optimization_steps": metrics["optimization_steps"],
                             "learning_rate": float(learning_rate(round_id))})
            elif phase == "evaluation":
                row = next((row for row in rows if int(row["round"]) == round_id), None)
                if row is None:
                    raise RuntimeError("Quality evaluation metrics require a completed training round")
                row.update(val_ce=metrics["loss"], val_top1=metrics["top1"],
                           elapsed_wall_sec=time.perf_counter() - self.started)
            temporary = output / "round_metrics.csv.tmp"
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
        if phase == "evaluation" and round_id > 0:
            path = output / "best_validation.json"
            previous = json.loads(path.read_text()) if path.exists() else None
            if previous is None or metrics["top1"] > previous["metrics"]["top1"]:
                np.savez(output / "best_model.npz", **state)
                path.write_text(json.dumps({"round": round_id, "metrics": metrics}, indent=2) + "\n")


def quality_spec(config):
    options = config.workload
    backend = options["backend"]
    if backend not in BACKENDS:
        raise ValueError(f"Unknown quality backend {backend!r}")
    protocol = Path(options["protocol"])
    frozen = json.loads((protocol / "protocol.json").read_text())
    validate_worker_protocol(frozen)
    rounds, _ = validate_main_protocol(frozen)
    if config.rounds != rounds or config.batch_size != 16 or config.local_epochs != 1:
        raise ValueError("Runtime differs from the frozen training budget")
    pair = next((row for row in frozen["pairs"]
                 if row["backend"] == backend and row["seed"] == config.seed), None)
    if pair is None:
        raise ValueError("The frozen protocol has no matching native backend/seed pair")
    with np.load(protocol / f"initial_seed_{config.seed}.npz") as initial:
        expected_hash = state_hash({name: initial[name] for name in NAMES})
    if expected_hash != pair["initial_hash"]:
        raise ValueError("Canonical initialization differs from the frozen protocol hash")
    partition_path = protocol / pair["partition_manifest"]
    if hash_file(partition_path) != pair["partition_sha256"]:
        raise ValueError("Partition manifest differs from the frozen protocol hash")
    factory = QualityFactory(options["data_root"], str(protocol / f"initial_seed_{config.seed}.npz"), backend)
    data = cifar_data(factory.data_root)
    if config.virtual_clients.total != BACKENDS[backend]:
        raise ValueError("Complete training requires the declared backend client population")
    manifest = json.loads(partition_path.read_text())
    partitions = {cid: row["indices"] for cid, row in manifest["clients"].items()}
    indices = [index for ids in partitions.values() for index in ids]
    if (set(partitions) != {str(cid) for cid in range(BACKENDS[backend])}
            or len(indices) != len(data.train_indices) or len(set(indices)) != len(indices)
            or set(indices) != set(data.train_indices)):
        raise ValueError("Partitions must cover every training image exactly once without validation/test leakage")
    return TrainingSpec(
        task=TASK_SPECS.get("image_classification"), model_factory=factory.model,
        train_dataset_factory=factory.dataset, batch_axes={"/args/0": 0},
        frozen_partitions=partitions, batch_factory=QualityBatches, learning_rate_fn=learning_rate,
        trainer=partial(full_model_trainer, backend=backend),
        candidate_provider_factory=partial(StageCandidateProvider, backend=backend),
        placement_policy_factory=CoSplitUCBPlacementPolicy,
        evaluate_fn=QualityEvaluation(factory.data_root, backend=backend),
        round_callback=RoundCheckpoints(backend), integer_aggregation_policy="maximum", source="real",
        framework_backend="torch",
        round_timeout_sec=options.get("round_timeout_sec"),
        data_manifest={"dataset": "CIFAR-10", "model": "ResNet50-V1.5", "initial_hash": expected_hash,
                       "canonical_initial_hash": expected_hash, "backend": backend,
                       "wire_state_layout": "native_adapter_schema",
                       "protocol": str(protocol), "phase": options["stage"],
                       "trainable_parameters": TRAINABLE_PARAMETERS, "mean": data.mean.tolist(),
                       "std": data.std.tolist(), "augmentation": "sample/client/round/seed keyed; shared data.py"},
    )
