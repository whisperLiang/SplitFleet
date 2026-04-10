import torch
from torch import nn

from splitfleet.autosplit.planner import build_partition_plan
from splitfleet.autosplit.runtime import AutoSplitSession
from splitfleet.autosplit.types import PlacementPlan
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.worker import start_worker


class DemoNet(nn.Module):
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


def build_demo_placement(model: nn.Module, sample: torch.Tensor, worker_specs) -> PlacementPlan:
    traced = AutoSplitSession().planner.tracer.trace(model, (sample,))
    cutoff = max(0, min(len(traced.execution_plan.nodes) - 2, len(traced.execution_plan.nodes) // 2 - 1))
    partition_plan = build_partition_plan(traced.execution_plan, [cutoff], model_name="DemoNet")
    return PlacementPlan(
        partition_plan=partition_plan,
        stage_to_worker={
            partition_plan.stages[0].stage_id: worker_specs[0].worker_id,
            partition_plan.stages[1].stage_id: worker_specs[1].worker_id,
        },
        worker_specs={worker.worker_id: worker for worker in worker_specs},
        score=0.0,
    )


def main() -> None:
    torch.manual_seed(42)
    model = DemoNet()
    x = torch.randn(8, 4)
    targets = torch.randn(8, 2)

    worker_a = start_worker(
        worker_id="demo-worker-a",
        model=model,
        sample_inputs=(x,),
        server_address="127.0.0.1:50071",
        register_with_registry=False,
    )
    worker_b = start_worker(
        worker_id="demo-worker-b",
        model=model,
        sample_inputs=(x,),
        server_address="127.0.0.1:50072",
        register_with_registry=False,
    )

    try:
        placement = build_demo_placement(model, x, [worker_a.worker_spec, worker_b.worker_spec])
        manager = StageRuntimeManager()
        manager.register_worker(worker_a.worker_spec)
        manager.register_worker(worker_b.worker_spec)
        manager.set_placement_plan(placement)

        print("eval output shape:", tuple(manager.run_eval(x).shape))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        result = manager.run_train(
            x,
            targets=targets,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
        )
        print("loss:", float(result["loss"].detach()))
        print("stage_to_worker:", result["stage_to_worker"])
    finally:
        worker_a.stop()
        worker_b.stop()


if __name__ == "__main__":
    main()
