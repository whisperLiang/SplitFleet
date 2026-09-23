from __future__ import annotations

import torch
from torch import nn

from experiments.resource_adaptive_splitfed.experiment_runner import _select_splits
from experiments.resource_adaptive_splitfed.training_runtime import fedprox_penalty


def test_fedprox_penalty_uses_the_round_global_reference() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    reference = {"weight": model.weight.detach().clone()}
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))

    assert torch.equal(fedprox_penalty(model, reference), torch.tensor(2.5))


def test_fedprox_is_routed_to_the_full_local_fl_path() -> None:
    selected = _select_splits(
        "fedprox",
        ["a", "b"],
        {"a": "weak", "b": "strong"},
        {},
        scheduler=None,  # not used for a full-local baseline
        resources={},
        server_state=None,
        round_id=1,
        best_global_fixed="layer2",
        oracle_choices={},
    )

    assert selected == {"a": "full_local", "b": "full_local"}
