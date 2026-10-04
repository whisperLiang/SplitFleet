"""Named full-state aggregation and the FedProx training penalty."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import numpy as np
import torch
from flwr.server.strategy.aggregate import aggregate


TensorState = Mapping[str, torch.Tensor]


def _weighted_tensor(values: Sequence[torch.Tensor], weights: Sequence[int]) -> torch.Tensor:
    reference = values[0]
    if reference.is_floating_point() or reference.is_complex():
        arrays = [value.detach().cpu().numpy() for value in values]
        # Flower aggregates a list of model tensors. Keep a named entry as a
        # one-tensor model so scalar parameters and empty leading axes retain
        # their shape instead of being iterated as model layers.
        averaged = aggregate([([array], int(weight)) for array, weight in zip(arrays, weights)])[0]
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
        for name in keys:
            if state[name].shape != client_states[0][name].shape or state[name].dtype != client_states[0][name].dtype:
                raise ValueError(f"Client state {index} has an incompatible tensor schema for {name!r}.")
    return OrderedDict(
        (name, _weighted_tensor([state[name] for state in client_states], sample_counts))
        for name in keys
    )


def fedprox_penalty(
    model: torch.nn.Module,
    reference_parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return ``0.5 * ||w - w_global||^2`` over trainable parameters.

    The caller applies FedProx's ``mu`` coefficient.  Keeping the coefficient
    outside makes the primitive directly testable and prevents an experiment
    configuration value from being hidden in model state.
    """

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("FedProx requires at least one trainable model parameter.")
    missing = [name for name, _ in trainable if name not in reference_parameters]
    if missing:
        raise ValueError(f"FedProx reference is missing trainable parameters: {missing}")
    penalty = trainable[0][1].new_zeros(())
    for name, parameter in trainable:
        reference = reference_parameters[name].to(device=parameter.device, dtype=parameter.dtype)
        if reference.shape != parameter.shape:
            raise ValueError(
                f"FedProx reference shape mismatch for {name!r}: "
                f"{tuple(reference.shape)} != {tuple(parameter.shape)}"
            )
        penalty = penalty + 0.5 * torch.sum((parameter - reference) ** 2)
    return penalty
