import pytest
import torch
from torch import nn

from splitfleet.server.stage_runtime.manager import REMOTE_STAGE_ERROR, StageRuntimeManager
from splitfleet.worker import start_worker


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_old_remote_stage_execution_api_is_removed() -> None:
    manager = StageRuntimeManager()

    with pytest.raises(NotImplementedError, match="old node-level remote stage execution"):
        manager.run_stage_forward()

    assert "coordinator-local suffix execution" in REMOTE_STAGE_ERROR


def test_remote_worker_registration_still_exposes_clear_error() -> None:
    model = TinyNet()
    x = torch.randn(2, 4)
    handle = start_worker(
        worker_id="remote-a",
        model=model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )
    try:
        with pytest.raises(NotImplementedError, match="old node-level remote stage execution"):
            handle.runtime.execute_stage()
    finally:
        handle.stop()
