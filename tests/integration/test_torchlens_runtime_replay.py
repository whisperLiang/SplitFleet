from __future__ import annotations

import pytest
import torch
from torch import nn

from splitfleet.autosplit import prepare_torchlens_runtime


class TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


class TinyResNetLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x) + residual)
        return self.head(x)


class TinyTransformerLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Linear(4, 4)
        self.ff = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.token(x))
        x = self.norm(hidden + torch.relu(self.ff(hidden)))
        return self.head(x.mean(dim=1))


@pytest.mark.parametrize(
    ("model", "trace_inputs", "make_inputs"),
    [
        (TinyMlp(), torch.randn(2, 4), lambda batch: torch.randn(batch, 4)),
        (TinyCnn(), torch.randn(2, 3, 8, 8), lambda batch: torch.randn(batch, 3, 8, 8)),
        (TinyResNetLike(), torch.randn(2, 4), lambda batch: torch.randn(batch, 4)),
        (TinyTransformerLike(), torch.randn(2, 3, 4), lambda batch: torch.randn(batch, 3, 4)),
    ],
)
def test_torchlens_runtime_replay_matches_full_model(model, trace_inputs, make_inputs) -> None:
    torch.manual_seed(101)
    model.eval()
    handle = prepare_torchlens_runtime(
        model,
        trace_inputs,
        boundary="50%",
        dynamic_batch=(1, 4),
        trace_batch_mode="batch_gt1",
    )

    for batch_size in (1, 2, 4):
        inputs = make_inputs(batch_size)
        with torch.no_grad():
            expected = model(inputs)
            split = handle.backend.run_suffix(handle.backend.run_prefix(inputs))
            replay = handle.runtime.replay(inputs)
        assert torch.allclose(split, expected, atol=1e-5, rtol=1e-4)
        assert torch.allclose(replay, expected, atol=1e-5, rtol=1e-4)


def test_torchlens_runtime_training_prefix_suffix_backward() -> None:
    torch.manual_seed(103)
    model = TinyMlp().train()
    handle = prepare_torchlens_runtime(
        model,
        torch.randn(2, 4),
        boundary="50%",
        dynamic_batch=(1, 4),
        trace_batch_mode="batch_gt1",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)

    boundary = handle.backend.run_prefix(inputs, training=True)
    loss, boundary_grads = handle.backend.train_suffix(
        boundary,
        targets,
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
    )
    handle.backend.backward_prefix(boundary, boundary_grads=boundary_grads, optimizer=optimizer)

    assert torch.isfinite(loss)
    assert boundary_grads
    assert any(
        not torch.allclose(before[name], param.detach())
        for name, param in model.named_parameters()
    )
