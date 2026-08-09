"""Full logical client state and split-independent, name-based FedAvg."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from flwr.server.strategy.aggregate import aggregate


TensorState = Mapping[str, torch.Tensor]


@dataclass
class LogicalClientModelState:
    """A client's complete model, independent of where each parameter executed."""

    client_id: str
    full_state_dict: dict[str, torch.Tensor]
    optimizer_state_by_parameter: dict[str, Any] = field(default_factory=dict)
    split_key: str = "full_local"
    last_switch_round: int = 0

    @classmethod
    def from_model(
        cls, client_id: str, model: torch.nn.Module, *, split_key: str = "full_local"
    ) -> "LogicalClientModelState":
        return cls(
            client_id=str(client_id),
            full_state_dict={
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            split_key=split_key,
        )

    def load_model(self, model: torch.nn.Module) -> None:
        expected = set(model.state_dict())
        actual = set(self.full_state_dict)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Full state schema mismatch; missing={missing}, extra={extra}")
        model.load_state_dict(self.full_state_dict, strict=True)

    def capture_model(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        if set(current) != set(self.full_state_dict):
            raise ValueError("Cannot capture a model with a different full-state schema.")
        self.full_state_dict = {
            name: tensor.detach().cpu().clone() for name, tensor in current.items()
        }

    def validate_stateless_sgd(self) -> None:
        if self.optimizer_state_by_parameter:
            raise ValueError(
                "The reference RA-SplitFed runner supports only stateless SGD "
                "(no momentum). Optimizer state must not be silently discarded."
            )


def _weighted_tensor(values: Sequence[torch.Tensor], weights: Sequence[int]) -> torch.Tensor:
    reference = values[0]
    if reference.is_floating_point() or reference.is_complex():
        arrays = [value.detach().cpu().numpy() for value in values]
        averaged = aggregate([(array, int(weight)) for array, weight in zip(arrays, weights)])
        return torch.from_numpy(np.asarray(averaged)).to(dtype=reference.dtype)
    total = float(sum(weights))
    accumulator = torch.zeros_like(reference, dtype=torch.float64)
    for value, weight in zip(values, weights):
        accumulator.add_(value.detach().cpu().to(torch.float64), alpha=float(weight))
    return (accumulator / total).round().to(dtype=reference.dtype)


def aggregate_named_states(
    client_states: Sequence[TensorState], sample_counts: Sequence[int]
) -> "OrderedDict[str, torch.Tensor]":
    """FedAvg complete states by original parameter/buffer name.

    Flower's weighted aggregation primitive is applied independently to every
    floating-point named entry.  This prevents different cut points from
    changing aggregation keys or silently dropping prefix/suffix parameters.
    """

    if not client_states or len(client_states) != len(sample_counts):
        raise ValueError("client_states and sample_counts must be non-empty and aligned.")
    if any(int(count) <= 0 for count in sample_counts):
        raise ValueError("FedAvg sample counts must be positive.")
    keys = list(client_states[0].keys())
    expected = set(keys)
    for index, state in enumerate(client_states):
        if set(state) != expected:
            raise ValueError(f"Client state {index} does not match the full model schema.")
    return OrderedDict(
        (name, _weighted_tensor([state[name] for state in client_states], sample_counts))
        for name in keys
    )
