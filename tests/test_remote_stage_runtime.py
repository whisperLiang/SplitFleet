import copy

import torch
from torch import nn

from splitfleet.autosplit.planner import build_partition_plan
from splitfleet.autosplit.runtime import AutoSplitSession
from splitfleet.autosplit.types import PlacementPlan
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.worker import start_worker


class DeepNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def _build_remote_placement(model, sample, worker_specs) -> PlacementPlan:
    tracer = AutoSplitSession().planner.tracer
    traced = tracer.trace(model, (sample,))
    cutoff = max(0, min(len(traced.execution_plan.nodes) - 2, len(traced.execution_plan.nodes) // 2 - 1))
    partition_plan = build_partition_plan(
        traced.execution_plan,
        [cutoff],
        model_name=model.__class__.__name__,
    )
    return PlacementPlan(
        partition_plan=partition_plan,
        stage_to_worker={
            partition_plan.stages[0].stage_id: worker_specs[0].worker_id,
            partition_plan.stages[1].stage_id: worker_specs[1].worker_id,
        },
        worker_specs={worker.worker_id: worker for worker in worker_specs},
        score=0.0,
    )


def test_remote_stage_runtime_eval_matches_direct_forward() -> None:
    torch.manual_seed(7)
    model = DeepNet()
    x = torch.randn(5, 4)

    handle1 = start_worker(
        worker_id="remote-a",
        model=model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )
    handle2 = start_worker(
        worker_id="remote-b",
        model=model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )
    try:
        manager = StageRuntimeManager()
        manager.register_worker(handle1.worker_spec)
        manager.register_worker(handle2.worker_spec)
        placement = _build_remote_placement(model, x, [handle1.worker_spec, handle2.worker_spec])
        manager.set_placement_plan(placement)

        expected = model(x)
        actual = manager.run_eval(x)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    finally:
        handle1.stop()
        handle2.stop()


def test_remote_stage_runtime_train_matches_direct_update() -> None:
    torch.manual_seed(11)
    base = DeepNet()
    distributed_model = copy.deepcopy(base)
    reference_model = copy.deepcopy(base)
    x = torch.randn(6, 4)
    targets = torch.randn(6, 2)

    handle1 = start_worker(
        worker_id="remote-a",
        model=distributed_model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )
    handle2 = start_worker(
        worker_id="remote-b",
        model=distributed_model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )
    try:
        manager = StageRuntimeManager()
        manager.register_worker(handle1.worker_spec)
        manager.register_worker(handle2.worker_spec)
        placement = _build_remote_placement(
            distributed_model,
            x,
            [handle1.worker_spec, handle2.worker_spec],
        )
        manager.set_placement_plan(placement)

        distributed_optimizer = torch.optim.SGD(distributed_model.parameters(), lr=0.1)
        manager.run_train(
            x,
            targets=targets,
            loss_fn=nn.MSELoss(),
            optimizer=distributed_optimizer,
        )

        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = nn.MSELoss()(reference_model(x), targets)
        reference_loss.backward()
        reference_optimizer.step()

        for distributed_param, reference_param in zip(
            distributed_model.parameters(),
            reference_model.parameters(),
        ):
            assert torch.allclose(
                distributed_param,
                reference_param,
                atol=1e-6,
                rtol=1e-6,
            )
    finally:
        handle1.stop()
        handle2.stop()
