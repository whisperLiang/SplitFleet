from __future__ import annotations

import torch

from experiments.unified_multitask.models import build_model


def test_groupnorm_resnet18_trains_a_genuine_singleton_batch() -> None:
    model = build_model("resnet18", normalization="groupnorm").train()
    inputs = torch.randn(1, 3, 32, 32)
    targets = torch.tensor([3])

    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
