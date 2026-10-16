"""Seeded, disjoint data ownership and deterministic per-client minibatches."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.common.partition import dirichlet_partition

from .state import client_seed


def partition_hash(indices):
    return hashlib.sha256(
        json.dumps(list(indices), separators=(",", ":")).encode()
    ).hexdigest()


def partition_dataset(
    size, total, *, config, seed, labels=None, family="image_classification"
):
    if total < 1 or config.min_samples < 1 or size < total * config.min_samples:
        raise ValueError("Dataset is too small for the declared clients/min_samples")
    if config.method == "iid":
        indices = np.random.default_rng(seed).permutation(size)
        if config.sample_counts is None:
            chunks = np.array_split(indices, total)
        else:
            counts = config.sample_counts
            if (
                len(counts) != total
                or sum(counts) != size
                or min(counts) < config.min_samples
            ):
                raise ValueError(
                    "sample_counts must cover the dataset exactly and respect min_samples"
                )
            chunks = np.split(indices, np.cumsum(counts)[:-1])
        result = {str(i): chunk.tolist() for i, chunk in enumerate(chunks)}
    elif config.method == "dirichlet":
        if family not in {"image_classification", "text_classification"}:
            raise ValueError(
                "Single-label Dirichlet is unsupported for detection/masks; use IID or a task-specific partitioner"
            )
        if labels is None or len(labels) != size or config.sample_counts is not None:
            raise ValueError(
                "Dirichlet requires one scalar class label per sample; sample_counts applies to IID"
            )
        if total == 1:
            result = {"0": np.random.default_rng(seed).permutation(size).tolist()}
        else:
            result = dirichlet_partition(
                labels,
                total,
                alpha=config.alpha,
                seed=seed,
                min_partition_size=config.min_samples,
            )
    else:
        raise ValueError("partition.method must be iid or dirichlet")
    flat = [i for part in result.values() for i in part]
    if (
        len(flat) != size
        or set(flat) != set(range(size))
        or min(map(len, result.values())) < config.min_samples
    ):
        raise RuntimeError(
            "Data partitioner lost, duplicated or underallocated samples"
        )
    return result


class ClientBatches:
    """Only indices in this partition are visible; sampling ignores worker order."""

    def __init__(
        self,
        dataset,
        indices,
        *,
        batch_size,
        seed,
        partition_id,
        round_id=1,
        local_epochs=1,
        collate_fn=None,
        deadline=None,
        fail_after=None,
    ):
        self.dataset = Subset(dataset, list(indices))
        self.batch_size = batch_size
        self.seed, self.partition_id, self.round_id = seed, partition_id, round_id
        self.local_epochs = local_epochs
        self.collate_fn = collate_fn
        self.deadline = deadline
        self.fail_after = fail_after
        self.executed = 0

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size) * self.local_epochs

    def __iter__(self):
        import time

        self.executed = 0
        for epoch in range(self.local_epochs):
            generator = torch.Generator().manual_seed(
                client_seed(
                    self.seed, self.partition_id, f"data:{self.round_id}:{epoch}"
                )
            )
            loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
                collate_fn=self.collate_fn,
                num_workers=0,
            )
            for raw in loader:
                if self.deadline is not None and time.perf_counter() >= self.deadline:
                    raise TimeoutError("Virtual client training deadline exceeded")
                if self.fail_after is not None and self.executed >= self.fail_after:
                    raise ConnectionError("Injected virtual client dropout")
                yield raw
                self.executed += 1
