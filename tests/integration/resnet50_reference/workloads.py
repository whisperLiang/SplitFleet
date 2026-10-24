"""Model, input and training fixtures for framework regression tests."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from splitfleet.server.placement.greedy.candidate_provider import TorchLensCandidateProvider
from splitfleet.tasks.base import ModelInputs, TaskBatch
from splitfleet.virtual.backend_ops import FrameworkOps

from experiments.resnet50_quality.data import load_cifar
from experiments.resnet50_quality.model_contract import NAMES, TRAINABLE_PARAMETERS
from .native import build_model, configure_backend, native_parameters, train_step


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
        if self.backend != "pytorch":
            return self.data.images([sample_id])[0], np.asarray(self.data.train_labels[sample_id], np.int64)
        return (torch.from_numpy(self.data.images([sample_id])[0]),
                torch.tensor(self.data.train_labels[sample_id], dtype=torch.int64))


@dataclass(frozen=True)
class QualityCollate:
    backend: str
    device: str

    def from_arrays(self, images, labels):
        # A dataset is shared by logical CPU/GPU clients. Keep inputs on the
        # host until prepare_batch knows the actual client's model and device.
        images, labels = np.asarray(images, np.float32), np.asarray(labels, np.int64)
        if self.backend == "pytorch":
            return torch.from_numpy(images), torch.from_numpy(labels)
        if self.backend == "tensorflow":
            images = images.transpose(0, 2, 3, 1)
        return TaskBatch(ModelInputs(args=(images,)), labels, num_examples=int(labels.shape[0]))

    def __call__(self, items):
        images, labels = zip(*items, strict=True)
        return self.from_arrays(np.stack(images), np.asarray(labels, np.int64))


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
            backend = getattr(self.dataset, "backend", "pytorch")
            if backend == "pytorch":
                yield torch.from_numpy(images), torch.from_numpy(labels)
            else:
                yield QualityCollate(backend, self.dataset.device).from_arrays(images, labels)
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
            if self.quality_backend == "pytorch":
                from experiments.resnet50_quality.gate_torch import _stage_boundaries
                self.stage_endpoints = _stage_boundaries(handle)
            else:
                from .array_helpers import stage_boundaries
                self.stage_endpoints = stage_boundaries(handle, self.quality_backend)
        result = tuple(row["boundary"] for row in self.stage_endpoints.values())
        if not set(result).issubset(self._candidates[True]):
            raise RuntimeError("Native full catalog excludes a required residual stage endpoint")
        return result
