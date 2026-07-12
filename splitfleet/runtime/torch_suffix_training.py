"""Torch suffix training compatible with in-place residual operations."""

from __future__ import annotations

from typing import Any

import torch
from torchlens.split import ReplayBoundary


def train_torch_suffix(runtime, boundary: ReplayBoundary, targets: Any, *, loss_fn=None, optimizer=None):
    """Train suffix while replaying non-leaf views of leaf gradient roots.

    TorchLens 2.31 replays its leaf roots directly. Models such as timm ResNet
    use in-place residual adds, which PyTorch correctly rejects on leaf tensors.
    A zero-add view remains connected to the root while being safe for replay.
    """
    runtime.validate_boundary(boundary)
    roots: dict[str, torch.Tensor] = {}
    replay: dict[str, Any] = {}
    for key, value in boundary.tensors.items():
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            root = value.detach().clone().requires_grad_(True)
            roots[key] = root
            replay[key] = root + torch.zeros((), dtype=root.dtype, device=root.device)
        else:
            replay[key] = value
    replay_boundary = ReplayBoundary(
        backend=boundary.backend,
        tensors=replay,
        spec=boundary.spec,
        metadata={**boundary.metadata, "suffix_training_roots": tuple(roots)},
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = runtime.run_suffix(replay_boundary)
    if loss_fn is not None:
        loss = loss_fn(output, targets)
    elif targets is not None and isinstance(output, torch.Tensor) and isinstance(targets, torch.Tensor):
        loss = torch.nn.functional.mse_loss(output, targets)
    else:
        raise ValueError("Torch suffix training requires loss_fn for non-tensor outputs")
    loss.backward()
    gradients = {
        key: root.grad.detach().clone()
        for key, root in roots.items()
        if root.grad is not None
    }
    if optimizer is not None:
        optimizer.step()
    return loss, gradients
