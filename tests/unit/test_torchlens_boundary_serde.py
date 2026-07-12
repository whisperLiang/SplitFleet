from __future__ import annotations

import torch
from torch import nn

from splitfleet.autosplit import BoundaryPayload, prepare_torchlens_runtime
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.transport import decode_boundary, encode_boundary
from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary


class ToyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_torchlens_boundary_payload_round_trips_without_pickle() -> None:
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
    contract = graph_contract_for_runtime_handle(handle)
    envelope = boundary_to_envelope(
        boundary,
        round_id=1,
        client_id="client",
        step_id="step",
        plan_id=handle.plan.plan_id,
        split_id=contract.split_id,
        canonical_graph_hash=contract.canonical_graph_hash,
        boundary_schema_hash=contract.boundary_schema_hash,
        model_version=1,
    )
    blob = encode_boundary(envelope)
    restored = envelope_to_boundary(decode_boundary(blob), handle.runtime, "cpu")

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
