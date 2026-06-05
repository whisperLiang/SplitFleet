"""Minimal TorchLens split training demo for SplitFleet."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18

from splitfleet.autosplit import prepare_torchlens_runtime


def main() -> None:
    torch.manual_seed(7)
    model = resnet18(weights=None)
    runtime = prepare_torchlens_runtime(
        model,
        torch.randn(2, 3, 96, 96),
        boundary="50%",
        trainable=True,
        dynamic_batch=(2, 8),
    )
    x = torch.randn(3, 3, 96, 96)
    labels = torch.randint(0, 1000, (3,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    boundary = runtime.backend.run_prefix(x)
    output = runtime.backend.run_suffix(boundary)

    training_boundary = runtime.backend.run_prefix(x, training=True)
    loss, boundary_grads = runtime.backend.train_suffix(
        training_boundary,
        labels,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
    )
    runtime.backend.backward_prefix(
        training_boundary,
        boundary_grads=boundary_grads,
        optimizer=optimizer,
    )

    print(f"split_id={runtime.plan.split_id}")
    print(f"graph_signature={runtime.plan.graph_signature}")
    print(f"boundary_labels={list(boundary.tensors)}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"loss={float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
