from __future__ import annotations

import torch

from experiments.resource_adaptive_splitfed.model_data import (
    build_model,
    stratified_client_holdout,
)


def test_groupnorm_resnet18_trains_a_genuine_singleton_batch() -> None:
    model = build_model("resnet18", normalization="groupnorm").train()
    inputs = torch.randn(1, 3, 32, 32)
    targets = torch.tensor([3])

    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_client_holdouts_are_stratified_disjoint_lossless_and_deterministic() -> None:
    targets = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    assignments = {"0": [0, 1, 2, 3, 4, 5], "1": [6, 7, 8, 9]}

    train, holdout = stratified_client_holdout(
        assignments, targets, fraction=0.34, seed=17
    )
    repeated = stratified_client_holdout(assignments, targets, fraction=0.34, seed=17)

    assert (train, holdout) == repeated
    assert all(set(train[cid]).isdisjoint(holdout[cid]) for cid in assignments)
    assert sorted(index for values in (*train.values(), *holdout.values()) for index in values) == list(range(10))
    for cid in assignments:
        assert {targets[index] for index in train[cid]} == {0, 1}
        assert {targets[index] for index in holdout[cid]} == {0, 1}
