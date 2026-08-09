"""Real CIFAR-10/ResNet-18 construction and deterministic client partitions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from .config_utils import stable_hash


def build_model(
    name: str,
    *,
    num_classes: int = 10,
    normalization: str = "groupnorm",
) -> torch.nn.Module:
    normalized = name.lower().replace("-", "")
    if normalized != "resnet18":
        raise ValueError("The reference RA-SplitFed experiment currently supports resnet18.")
    norm = normalization.lower().replace("_", "")
    if norm == "groupnorm":
        norm_layer = lambda channels: torch.nn.GroupNorm(min(32, channels), channels)
    elif norm == "batchnorm":
        norm_layer = torch.nn.BatchNorm2d
    else:
        raise ValueError("normalization must be 'groupnorm' or 'batchnorm'.")
    # GroupNorm keeps the actual ResNet-18 graph and trainable normalization,
    # while making a genuine batch-size-one training execution well-defined.
    return models.resnet18(
        weights=None,
        num_classes=num_classes,
        norm_layer=norm_layer,
    )


def cifar10_datasets(
    root: str | Path,
    *,
    download: bool,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train: Dataset = datasets.CIFAR10(root=str(root), train=True, download=download, transform=transform)
    test: Dataset = datasets.CIFAR10(root=str(root), train=False, download=download, transform=transform)
    if max_train_samples is not None:
        train = Subset(train, range(min(int(max_train_samples), len(train))))
    if max_test_samples is not None:
        test = Subset(test, range(min(int(max_test_samples), len(test))))
    return train, test


def dataset_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base = dataset_targets(dataset.dataset)
        return base[np.asarray(dataset.indices, dtype=np.int64)]
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return np.asarray([int(dataset[index][1]) for index in range(len(dataset))])
    return np.asarray(targets, dtype=np.int64)


def dirichlet_partition(
    targets: Sequence[int],
    num_clients: int,
    *,
    alpha: float,
    seed: int,
    client_profiles: Mapping[str, str] | None = None,
    rare_classes: Sequence[int] = (),
    weak_rare_multiplier: float = 1.0,
) -> dict[str, list[int]]:
    if num_clients < 2 or alpha <= 0:
        raise ValueError("Dirichlet partition requires num_clients >= 2 and alpha > 0.")
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets, dtype=np.int64)
    assignments = {str(index): [] for index in range(num_clients)}
    profiles = client_profiles or {}
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        weights = rng.dirichlet(np.full(num_clients, alpha))
        if label in rare_classes and weak_rare_multiplier != 1.0:
            for index in range(num_clients):
                if profiles.get(str(index)) == "weak":
                    weights[index] *= weak_rare_multiplier
            weights /= weights.sum()
        counts = rng.multinomial(len(indices), weights)
        cursor = 0
        for client_index, count in enumerate(counts):
            assignments[str(client_index)].extend(indices[cursor : cursor + count].tolist())
            cursor += count
    # Empty clients make synchronized comparisons ill-defined. Move one sample
    # from the largest partition deterministically; this is partition repair,
    # not a generated observation.
    for client_id, indices in assignments.items():
        if indices:
            continue
        donor = max(assignments, key=lambda cid: (len(assignments[cid]), cid))
        if len(assignments[donor]) <= 1:
            raise RuntimeError("Unable to construct non-empty client partitions.")
        indices.append(assignments[donor].pop())
    # Guarantee that no class is exclusive to a single client. This keeps the
    # fairness experiment robust without creating or duplicating samples.
    for label in sorted(set(labels.tolist())):
        holders = [
            client_id
            for client_id, indices in assignments.items()
            if any(labels[index] == label for index in indices)
        ]
        if len(holders) >= 2:
            continue
        donor = holders[0]
        sample = next(index for index in assignments[donor] if labels[index] == label)
        recipient = max(
            (cid for cid in assignments if cid != donor),
            key=lambda cid: (len(assignments[cid]), cid),
        )
        assignments[donor].remove(sample)
        assignments[recipient].append(sample)
        # Moving a donor's only sample would undo the empty-client repair above,
        # so trade one of the recipient's samples back instead of emptying it.
        if not assignments[donor]:
            recipient_counts = Counter(int(labels[index]) for index in assignments[recipient])
            # Only trade back a sample the recipient holds more than once, so the
            # swap cannot make some other class exclusive in its turn.
            returned = next(
                (
                    index
                    for index in assignments[recipient]
                    if labels[index] != label and recipient_counts[int(labels[index])] > 1
                ),
                None,
            )
            if returned is None:
                raise RuntimeError(
                    "Unable to keep every client non-empty while removing class exclusivity."
                )
            assignments[recipient].remove(returned)
            assignments[donor].append(returned)
    empty = sorted(client_id for client_id, indices in assignments.items() if not indices)
    if empty:
        raise RuntimeError(f"Partition repair left clients without samples: {empty}.")
    for indices in assignments.values():
        indices.sort()
    return assignments


def stratified_client_holdout(
    assignments: Mapping[str, Sequence[int]],
    targets: Sequence[int],
    *,
    fraction: float,
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Create deterministic, disjoint per-client train/evaluation partitions."""

    if not 0.0 <= fraction < 1.0:
        raise ValueError("Client holdout fraction must be in [0, 1).")
    labels = np.asarray(targets, dtype=np.int64)
    train: dict[str, list[int]] = {}
    holdout: dict[str, list[int]] = {}
    for ordinal, client_id in enumerate(sorted(assignments, key=lambda value: int(value))):
        by_class: dict[int, list[int]] = {}
        for index in assignments[client_id]:
            by_class.setdefault(int(labels[index]), []).append(int(index))
        client_train: list[int] = []
        client_holdout: list[int] = []
        rng = np.random.default_rng(seed * 1_000_003 + ordinal)
        for label in sorted(by_class):
            indices = np.asarray(sorted(by_class[label]), dtype=np.int64)
            rng.shuffle(indices)
            if fraction == 0.0 or len(indices) < 2:
                holdout_count = 0
            else:
                holdout_count = min(
                    len(indices) - 1,
                    max(1, int(round(len(indices) * fraction))),
                )
            client_holdout.extend(indices[:holdout_count].tolist())
            client_train.extend(indices[holdout_count:].tolist())
        if not client_train:
            raise RuntimeError(f"Client {client_id!r} has no training sample after holdout.")
        train[str(client_id)] = sorted(client_train)
        holdout[str(client_id)] = sorted(client_holdout)

    original = sorted(index for values in assignments.values() for index in values)
    reconstructed = sorted(
        index for values in (*train.values(), *holdout.values()) for index in values
    )
    if reconstructed != original or set(index for values in train.values() for index in values) & set(
        index for values in holdout.values() for index in values
    ):
        raise RuntimeError("Client holdout split must be disjoint and lossless.")
    return train, holdout


def partition_manifest(
    assignments: Mapping[str, Sequence[int]],
    targets: Sequence[int],
    client_holdouts: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(targets, dtype=np.int64)
    clients = {}
    for client_id, indices in sorted(assignments.items()):
        class_counts = Counter(int(labels[index]) for index in indices)
        clients[str(client_id)] = {
            "num_examples": len(indices),
            "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
            "index_hash": stable_hash(list(indices)),
        }
        if client_holdouts is not None:
            heldout = list(client_holdouts.get(str(client_id), ()))
            heldout_counts = Counter(int(labels[index]) for index in heldout)
            clients[str(client_id)].update(
                {
                    "heldout_num_examples": len(heldout),
                    "heldout_class_counts": {
                        str(key): value for key, value in sorted(heldout_counts.items())
                    },
                    "heldout_index_hash": stable_hash(heldout),
                }
            )
    partition_identity: Any = assignments
    if client_holdouts is not None:
        partition_identity = {"train": assignments, "client_holdout": client_holdouts}
    return {"partition_hash": stable_hash(partition_identity), "clients": clients}


def make_loader(
    dataset: Dataset,
    indices: Sequence[int] | None,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    selected = dataset if indices is None else Subset(dataset, list(indices))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
