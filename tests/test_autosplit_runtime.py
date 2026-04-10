from __future__ import annotations

import copy

import torch
from torch import nn

from splitfleet.autosplit import AutoSplitSession, PlacementConstraint, WorkerSpec


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


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(prediction, target)


def _build_workers(count: int) -> list[WorkerSpec]:
    return [
        WorkerSpec(worker_id=f"worker-{index}", device="cpu", bandwidth_mbps=500.0)
        for index in range(count)
    ]


def test_autosplit_eval_matches_direct_forward() -> None:
    torch.manual_seed(7)
    model = BranchNet().eval()
    inputs = torch.randn(3, 4)
    session = AutoSplitSession(device="cpu")
    placement = session.plan(
        model,
        inputs,
        worker_specs=_build_workers(3),
        constraints=PlacementConstraint(max_stages=3, max_candidates=8, max_frontier_size=2),
        preferred_stage_count=3,
    )

    replayed = session.run_eval(placement, inputs)
    expected = model(inputs)

    assert placement.partition_plan.stage_count == 3
    assert torch.allclose(replayed, expected, atol=1e-5, rtol=1e-5)


def _assert_parameter_grads_match(lhs: nn.Module, rhs: nn.Module) -> None:
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(
        lhs.named_parameters(),
        rhs.named_parameters(),
    ):
        assert lhs_name == rhs_name
        assert lhs_param.grad is not None
        assert rhs_param.grad is not None
        assert torch.allclose(lhs_param.grad, rhs_param.grad, atol=1e-5, rtol=1e-5)


def test_autosplit_train_matches_direct_backward_for_three_stages() -> None:
    torch.manual_seed(13)
    base_model = BranchNet().train()
    direct_model = copy.deepcopy(base_model)
    split_model = copy.deepcopy(base_model)
    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 2)

    direct_output = direct_model(inputs)
    direct_loss = mse_loss(direct_output, targets)
    direct_loss.backward()

    session = AutoSplitSession(device="cpu")
    placement = session.plan(
        split_model,
        inputs,
        worker_specs=_build_workers(3),
        constraints=PlacementConstraint(max_stages=3, max_candidates=8, max_frontier_size=2),
        preferred_stage_count=3,
    )
    result = session.run_train(
        placement,
        inputs,
        targets=targets,
        loss_fn=mse_loss,
        step_optimizer=False,
    )

    assert torch.allclose(result["output"], direct_output, atol=1e-5, rtol=1e-5)
    _assert_parameter_grads_match(direct_model, split_model)
