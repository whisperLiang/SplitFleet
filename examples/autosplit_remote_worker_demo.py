"""Ariadne autosplit demo with coordinator-local suffix execution.

The old node-level remote stage replay demo was removed with the Ariadne backend.
This example keeps the filename for discoverability but now demonstrates the
supported client-prefix/coordinator-suffix path.
"""

from __future__ import annotations

import torch
from torch import nn

from splitfleet.autosplit import AutoSplitSession


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def main() -> None:
    torch.manual_seed(7)
    model = DemoNet()
    session = AutoSplitSession()
    placement = session.plan(
        model,
        torch.randn(2, 4),
        boundary="50%",
        dynamic_batch=(2, 8),
    )
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    output = session.run_eval(placement, inputs)
    result = session.run_train(
        placement,
        inputs,
        targets,
        loss_fn=nn.MSELoss(),
        prefix_optimizer=optimizer,
        suffix_optimizer=optimizer,
    )

    print(f"plan_id={placement.plan_id}")
    print(f"split_id={placement.split_id}")
    print(f"graph_signature={placement.graph_signature}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"loss={float(result['loss'].detach()):.6f}")


if __name__ == "__main__":
    main()
