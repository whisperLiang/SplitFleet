from __future__ import annotations

import torch
from torch import nn

from splitfleet.autosplit import BoundaryPayload, prepare_torchlens_runtime
from splitfleet.autosplit.serde import dumps_torch_object, loads_torch_object


class ToyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_torchlens_boundary_payload_round_trips_through_torch_serde() -> None:
    torch.manual_seed(11)
    model = ToyMlp().eval()
    handle = prepare_torchlens_runtime(
        model,
        torch.randn(2, 4),
        boundary="after:linear_1_1",
        trainable=True,
        dynamic_batch=(2, 8),
    )
    x = torch.randn(3, 4)

    boundary = handle.backend.run_prefix(x)
    blob = dumps_torch_object(boundary)
    restored = loads_torch_object(blob)

    output1 = handle.backend.run_suffix(boundary)
    output2 = handle.backend.run_suffix(restored)

    assert isinstance(restored, BoundaryPayload)
    assert restored.tensors
    assert restored.batch_size == 3
    assert restored.spec is not None
    assert restored.spec.boundary_tensor_labels
    assert restored.passthrough_inputs == ()
    for label, tensor in boundary.tensors.items():
        assert label in restored.tensors
        assert tuple(restored.tensors[label].shape) == tuple(tensor.shape)
        assert restored.tensors[label].dtype == tensor.dtype
    assert output1.shape == output2.shape
    assert torch.allclose(output1, output2, atol=1e-5, rtol=1e-4)
