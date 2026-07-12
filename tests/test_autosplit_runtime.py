from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from splitfleet.autosplit import AutoSplitSession, PlacementConstraint, WorkerSpec
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient, _model_to_ndarrays
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
    AUTOSPLIT_BOUNDARY_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
)


class BranchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = torch.relu(self.left(x))
        right = torch.sigmoid(self.right(x))
        return self.head(torch.cat([left, right], dim=-1))


class BatchNormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.bn = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.bn(self.fc1(x)))


def test_torchlens_autosplit_eval_matches_direct_forward() -> None:
    torch.manual_seed(7)
    model = BranchNet().eval()
    trace_inputs = torch.randn(2, 4)
    runtime_inputs = torch.randn(3, 4)
    session = AutoSplitSession(device="cpu")
    placement = session.plan(
        model,
        trace_inputs,
        worker_specs=[WorkerSpec(worker_id="coordinator", device="cpu")],
        constraints=PlacementConstraint(),
        preferred_stage_count=2,
        client_stage_count=1,
        dynamic_batch=(2, 8),
    )

    replayed = session.run_eval(placement, runtime_inputs)
    expected = model(runtime_inputs)

    assert placement.stage_count == 2
    assert placement.split_id
    assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)


def test_torchlens_autosplit_train_runs_suffix_and_prefix_backward() -> None:
    torch.manual_seed(13)
    model = BranchNet().train()
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)
    session = AutoSplitSession(device="cpu")
    placement = session.plan(
        model,
        torch.randn(2, 4),
        preferred_stage_count=2,
        client_stage_count=1,
        dynamic_batch=(2, 8),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    result = session.run_train(
        placement,
        inputs,
        targets,
        loss_fn=nn.MSELoss(),
        prefix_optimizer=optimizer,
        suffix_optimizer=optimizer,
    )

    assert torch.isfinite(result["loss"])
    assert result["split_id"] == placement.split_id
    assert any(
        not torch.allclose(before[name], param.detach())
        for name, param in model.named_parameters()
    )


def test_client_prepares_mode_specific_torchlens_runtimes() -> None:
    torch.manual_seed(31)
    client = AutoSplitSplitLearningClient(
        model=BatchNormNet(),
        train_data=[],
        sample_inputs=torch.randn(2, 4),
    )
    config = {
        AUTOSPLIT_BACKEND_CONFIG_KEY: AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
        AUTOSPLIT_PLAN_ID_CONFIG_KEY: "mode-sensitive-plan",
        AUTOSPLIT_BOUNDARY_CONFIG_KEY: "after:fc1",
        AUTOSPLIT_MODE_CONFIG_KEY: "generated_eager",
    }

    train_handle = client._prepare_round(_model_to_ndarrays(client.model), config, training=True)
    train_inputs = torch.randn(3, 4)
    expected_train = copy.deepcopy(client.model).train()(train_inputs)
    split_train = train_handle.backend.run_suffix(train_handle.backend.run_prefix(train_inputs))

    assert torch.allclose(split_train, expected_train, atol=1e-5, rtol=1e-5)

    eval_handle = client._prepare_round(_model_to_ndarrays(client.model), config, training=False)
    eval_inputs = torch.randn(3, 4)
    with torch.no_grad():
        expected_eval = copy.deepcopy(client.model).eval()(eval_inputs)
        split_eval = eval_handle.backend.run_suffix(eval_handle.backend.run_prefix(eval_inputs))

    assert train_handle is not eval_handle
    assert torch.allclose(split_eval, expected_eval, atol=1e-5, rtol=1e-5)


def test_client_rejects_missing_backend_instead_of_falling_back() -> None:
    client = AutoSplitSplitLearningClient(
        model=BatchNormNet(), train_data=[], sample_inputs=torch.randn(2, 4)
    )
    with pytest.raises(ValueError, match="explicitly declare"):
        client._prepare_round(
            _model_to_ndarrays(client.model),
            {AUTOSPLIT_PLAN_ID_CONFIG_KEY: "missing-backend"},
            training=False,
        )


def test_torchlens_planner_rejects_sample_kwargs() -> None:
    with pytest.raises(ValueError, match="positional model inputs only"):
        AutoSplitSession(device="cpu").plan(
            BranchNet(),
            torch.randn(2, 4),
            sample_kwargs={"x": torch.randn(2, 4)},
        )
