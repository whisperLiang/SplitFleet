from __future__ import annotations

import torch
import pytest
from torch import nn

from experiments.common.training import aggregate_named_states, fedprox_penalty


def test_fedprox_penalty_uses_the_round_global_reference() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    reference = {"weight": model.weight.detach().clone()}
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))

    assert torch.equal(fedprox_penalty(model, reference), torch.tensor(2.5))


def test_name_based_full_model_aggregation_preserves_every_stage() -> None:
    base = {"stem.weight": torch.zeros(2, 2), "tail.weight": torch.zeros(1, 2)}
    states = [{name: tensor + offset for name, tensor in base.items()}
              for offset in (1, 3, 5)]

    aggregated = aggregate_named_states(states, [1, 2, 1])

    assert list(aggregated) == list(base)
    for name in base:
        assert torch.equal(aggregated[name], torch.full_like(base[name], 3.0))


def test_aggregation_rejects_partial_states_and_invalid_sample_counts() -> None:
    full = {"stem.weight": torch.ones(2, 2), "tail.weight": torch.ones(1, 2)}
    with pytest.raises(ValueError, match="full model schema"):
        aggregate_named_states([full, {"stem.weight": full["stem.weight"]}], [1, 1])
    with pytest.raises(ValueError, match="positive"):
        aggregate_named_states([full], [0])
