from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from tests.integration.torchlens_real_model_helpers import (
    clone_trainable_state,
    make_runtime,
    nested_tensor_loss,
    parameter_delta_nonzero,
    run_split_inference_equivalence,
    run_real_model_test_isolated,
    skip_if_missing_dependency,
)


RUN_HEAVY = os.environ.get("SPLITFLEET_RUN_HEAVY_REAL_MODELS", "0") == "1"


class YOLOTensorWrapper(nn.Module):
    def __init__(self, yolo):
        super().__init__()
        self.model = yolo.model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output


class RFDETRTensorWrapper(nn.Module):
    def __init__(self, rfdetr):
        super().__init__()
        self.model = rfdetr.model.model

    def forward(self, x):
        from rfdetr.utilities.tensors import NestedTensor

        mask = torch.zeros(
            (x.shape[0], x.shape[2], x.shape[3]),
            dtype=torch.bool,
            device=x.device,
        )
        return self.model(NestedTensor(x, mask))


@pytest.mark.skipif(not RUN_HEAVY, reason="Set SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 to run YOLO/RF-DETR tests.")
def test_yolov8n_optional_split_smoke(request) -> None:
    if run_real_model_test_isolated(request):
        return
    skip_if_missing_dependency("ultralytics")
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.yaml")
    model = YOLOTensorWrapper(yolo).eval()
    trace_inputs = torch.randn(2, 3, 160, 160)
    runtime_inputs = torch.randn(2, 3, 160, 160)
    # Exercise a frontier before YOLO's late multi-output branch using a
    # fixed batch size; this smoke test does not claim dynamic-batch support.
    with torch.no_grad():
        model(trace_inputs)
    # YOLO caches these tensors as plain attributes. Declare the warmed,
    # fixed-shape inference state as buffers so native replay can resolve it.
    head = model.model.model[-1]
    model.register_buffer("cached_anchors", head.anchors, persistent=False)
    model.register_buffer("cached_strides", head.strides, persistent=False)
    run_split_inference_equivalence(
        model, trace_inputs, runtime_inputs, "35%", dynamic_batch=(2, 2), batch_axes={},
    )


@pytest.mark.skipif(not RUN_HEAVY, reason="Set SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 to run YOLO/RF-DETR tests.")
def test_rfdetr_nano_optional_split_smoke(request) -> None:
    if run_real_model_test_isolated(request):
        return
    skip_if_missing_dependency("rfdetr")
    from rfdetr import RFDETRNano

    model = RFDETRTensorWrapper(RFDETRNano(pretrain_weights=None)).eval()
    trace_inputs = torch.randn(2, 3, 224, 224)
    runtime_inputs = torch.randn(2, 3, 224, 224)
    # The native dynamic-batch replay probe cannot reconstruct RF-DETR's
    # batch-dependent query containers. Validate the declared fixed B=2 call.
    run_split_inference_equivalence(
        model,
        trace_inputs,
        runtime_inputs,
        "50%",
        dynamic_batch=(2, 2),
        batch_axes={},
    )
    # RF-DETR's train() path builds batch-dependent Python containers before
    # the chosen boundary. Keep the heavy smoke in eval mode while still
    # exercising TorchLens's training prefix/suffix/backward APIs.
    runtime = make_runtime(
        model,
        trace_inputs,
        "50%",
        dynamic_batch=(2, 2),
        batch_axes={},
    )
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    before = clone_trainable_state(model)
    boundary = runtime.backend.run_prefix(runtime_inputs, training=True)
    loss, boundary_grads = runtime.backend.train_suffix(
        boundary,
        None,
        loss_fn=lambda output, _targets: nested_tensor_loss(output),
        optimizer=optimizer,
    )
    runtime.backend.backward_prefix(boundary, boundary_grads=boundary_grads, optimizer=optimizer)
    after = clone_trainable_state(model)
    assert torch.isfinite(loss)
    assert boundary.tensors
    assert boundary_grads
    assert parameter_delta_nonzero(before, after)
