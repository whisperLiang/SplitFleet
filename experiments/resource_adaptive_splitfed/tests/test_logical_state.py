from __future__ import annotations

import torch
from torch import nn

from experiments.resource_adaptive_splitfed.logical_state import (
    LogicalClientModelState,
    aggregate_named_states,
)


class NamedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(2, 2, bias=False)
        self.tail = nn.Linear(2, 1, bias=False)

    def forward(self, value):
        return self.tail(self.stem(value))


def test_name_based_full_model_aggregation_across_different_cuts() -> None:
    model = NamedModel()
    names = list(model.state_dict())
    base = {name: torch.zeros_like(value) for name, value in model.state_dict().items()}
    early = LogicalClientModelState("early", {name: value + 1 for name, value in base.items()}, split_key="stem")
    middle = LogicalClientModelState("middle", {name: value + 3 for name, value in base.items()}, split_key="layer2")
    local = LogicalClientModelState("local", {name: value + 5 for name, value in base.items()}, split_key="full_local")

    aggregated = aggregate_named_states(
        [early.full_state_dict, middle.full_state_dict, local.full_state_dict],
        [1, 2, 1],
    )

    assert list(aggregated) == names
    assert len(aggregated) == len(names)
    for name in names:
        assert torch.equal(aggregated[name], torch.full_like(aggregated[name], 3.0))


def test_state_schema_and_optimizer_state_are_never_silently_dropped() -> None:
    model = NamedModel()
    state = LogicalClientModelState.from_model("client", model)
    state.load_model(NamedModel())
    state.optimizer_state_by_parameter["stem.weight"] = {"momentum_buffer": torch.ones(1)}
    try:
        state.validate_stateless_sgd()
    except ValueError as exc:
        assert "must not be silently discarded" in str(exc)
    else:
        raise AssertionError("Stateful optimizers must be rejected by the reference runner")
