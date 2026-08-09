"""Partition repair invariants and split-candidate guards."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.resource_adaptive_splitfed.model_data import (
    dirichlet_partition,
    stratified_client_holdout,
)
from experiments.resource_adaptive_splitfed.split_candidates import discover_split_candidates


TARGETS = [index % 4 for index in range(24)]


def _holders(assignments, targets, label):
    return [
        client_id
        for client_id, indices in assignments.items()
        if any(targets[index] == label for index in indices)
    ]


@pytest.mark.parametrize("seed", range(40))
def test_repaired_partitions_are_non_empty_lossless_and_never_class_exclusive(seed: int) -> None:
    assignments = dirichlet_partition(TARGETS, 6, alpha=0.05, seed=seed)

    assert all(assignments.values()), "partition repair left a client empty"
    assert sorted(index for indices in assignments.values() for index in indices) == list(
        range(len(TARGETS))
    )
    for label in sorted(set(TARGETS)):
        assert len(_holders(assignments, TARGETS, label)) >= 2

    # The repaired partition must still survive the downstream holdout split.
    train, _ = stratified_client_holdout(assignments, TARGETS, fraction=0.1, seed=seed)
    assert all(train.values())


@pytest.mark.parametrize(
    ("common", "rare", "num_clients", "alpha", "seed"),
    [(10, 1, 4, 0.1, 0), (10, 2, 5, 0.1, 0), (20, 1, 6, 0.05, 0), (12, 2, 3, 0.3, 15)],
)
def test_a_rare_class_never_empties_the_client_that_holds_it(
    common: int, rare: int, num_clients: int, alpha: float, seed: int
) -> None:
    """Removing class exclusivity must not undo the empty-client repair.

    A class held by a single client that owns exactly one sample cannot be
    shared by moving that sample away: doing so empties the client, and the
    round then dies in ``stratified_client_holdout`` before training starts.
    """

    targets = [0] * common + [1] * common + list(range(2, 2 + rare))

    assignments = dirichlet_partition(targets, num_clients, alpha=alpha, seed=seed)

    assert all(assignments.values()), "class-exclusivity repair emptied a client"
    assert sorted(index for indices in assignments.values() for index in indices) == list(
        range(len(targets))
    )
    train, _ = stratified_client_holdout(assignments, targets, fraction=0.1, seed=seed)
    assert all(train.values())


def test_candidate_discovery_names_the_required_cut_count() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    with pytest.raises(RuntimeError, match="distinct cuts are required"):
        discover_split_candidates(model, torch.zeros(2, 4))
