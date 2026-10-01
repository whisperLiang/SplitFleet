"""Lossless deterministic Dirichlet partitions for paired client workloads."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def dirichlet_partition(
    targets: Sequence[int],
    num_clients: int,
    *,
    alpha: float,
    seed: int,
    min_partition_size: int = 10,
) -> dict[str, list[int]]:
    """Sample a lossless Dirichlet partition under a minimum-size constraint."""

    if num_clients < 2 or alpha <= 0:
        raise ValueError("Dirichlet partition requires num_clients >= 2 and alpha > 0.")
    if min_partition_size < 1:
        raise ValueError("min_partition_size must be positive.")
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets, dtype=np.int64)
    if len(labels) < num_clients:
        raise ValueError("The dataset must contain at least one sample per client.")
    required_minimum = min(int(min_partition_size), len(labels) // num_clients)
    assignments = {str(index): [] for index in range(num_clients)}
    for label in sorted(set(labels.tolist())):
        label_indices = np.flatnonzero(labels == label)
        weights = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
        weights /= weights.sum()
        split_points = (np.cumsum(weights) * len(label_indices)).astype(int)[:-1]
        for client_index, split in enumerate(np.split(label_indices, split_points)):
            assignments[str(client_index)].extend(split.tolist())
    if min(len(indices) for indices in assignments.values()) < required_minimum:
        _enforce_minimum_partition_size(assignments, labels, required_minimum)
    for indices in assignments.values():
        rng.shuffle(indices)
    return assignments


def _enforce_minimum_partition_size(
    assignments: dict[str, list[int]], labels: np.ndarray, minimum: int,
) -> None:
    """Move existing samples without duplication or loss to enforce the minimum."""
    for recipient, indices in assignments.items():
        while len(indices) < minimum:
            donor = max(assignments, key=lambda cid: len(assignments[cid]))
            source = assignments[donor]
            counts = Counter(int(labels[index]) for index in source)
            present = {int(labels[index]) for index in indices}
            chosen = max(source, key=lambda index: (
                int(labels[index]) not in present, counts[int(labels[index])], -index,
            ))
            source.remove(chosen)
            indices.append(chosen)

    # Share classes with at least two samples across two clients where possible.
    # A singleton class cannot be shared without fabricating a sample.
    for label in sorted(set(labels.tolist())):
        holders = [cid for cid, values in assignments.items()
                   if any(int(labels[index]) == label for index in values)]
        if len(holders) != 1 or np.count_nonzero(labels == label) < 2:
            continue
        donor = holders[0]
        source = assignments[donor]
        chosen = next(index for index in source if int(labels[index]) == label)
        for recipient, destination in assignments.items():
            if recipient == donor:
                continue
            if len(source) > minimum:
                source.remove(chosen)
                destination.append(chosen)
                break
            destination_counts = Counter(int(labels[index]) for index in destination)
            donor_labels = {int(labels[index]) for index in source}
            swap = next((index for index in destination
                         if destination_counts[int(labels[index])] > 1
                         or int(labels[index]) not in donor_labels), None)
            if swap is not None:
                source.remove(chosen)
                destination.remove(swap)
                source.append(swap)
                destination.append(chosen)
                break
