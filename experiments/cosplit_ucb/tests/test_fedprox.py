from __future__ import annotations

import torch
from torch import nn

from experiments.cosplit_ucb.training_runtime import fedprox_penalty


def test_fedprox_penalty_uses_the_round_global_reference() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    reference = {"weight": model.weight.detach().clone()}
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))

    assert torch.equal(fedprox_penalty(model, reference), torch.tensor(2.5))
