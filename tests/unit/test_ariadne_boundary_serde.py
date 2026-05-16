from __future__ import annotations

import pytest
import torch

from splitfleet.autosplit import prepare_ariadne_runtime
from splitfleet.autosplit.serde import dumps_torch_object, loads_torch_object


def test_ariadne_boundary_payload_round_trips_through_torch_serde() -> None:
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None).eval()
    runtime = prepare_ariadne_runtime(
        model,
        torch.randn(2, 3, 96, 96),
        boundary="after:layer3",
        trainable=True,
        dynamic_batch=(2, 3),
    )
    x = torch.randn(3, 3, 96, 96)

    boundary = runtime.runtime.run_prefix(x)
    blob = dumps_torch_object(boundary)
    restored = loads_torch_object(blob)

    output1 = runtime.runtime.run_suffix(boundary)
    output2 = runtime.runtime.run_suffix(restored)

    assert restored.tensors
    assert hasattr(restored, "passthrough_inputs")
    assert output1.shape == output2.shape
    assert torch.allclose(output1, output2, atol=1e-5, rtol=1e-4)

