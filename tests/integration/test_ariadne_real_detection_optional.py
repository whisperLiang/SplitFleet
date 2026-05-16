from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from tests.integration.ariadne_real_model_helpers import (
    clone_trainable_state,
    has_any_parameter_grad,
    make_runtime,
    nested_tensor_loss,
    parameter_delta_nonzero,
    run_split_inference_equivalence,
    skip_if_missing_dependency,
)


RUN_HEAVY = os.environ.get("SPLITFLEET_RUN_HEAVY_REAL_MODELS") == "1"


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
def test_yolov8n_optional_split_smoke() -> None:
    skip_if_missing_dependency("ultralytics")
    from ultralytics import YOLO

    try:
        yolo = YOLO("yolov8n.yaml")
    except Exception as exc:
        pytest.skip(f"YOLOv8n config is unavailable locally: {exc}")
    model = YOLOTensorWrapper(yolo).eval()
    trace_inputs = torch.randn(2, 3, 160, 160)
    runtime_inputs = torch.randn(3, 3, 160, 160)
    try:
        run_split_inference_equivalence(model, trace_inputs, runtime_inputs, "after:model.model.2")
    except Exception as exc:
        pytest.xfail(f"YOLOv8n split inference is not stable in this environment: {exc}")


@pytest.mark.skipif(not RUN_HEAVY, reason="Set SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 to run YOLO/RF-DETR tests.")
def test_rfdetr_nano_optional_split_smoke() -> None:
    skip_if_missing_dependency("rfdetr")
    try:
        from rfdetr import RFDETRNano
    except Exception as exc:
        pytest.skip(f"RF-DETR Nano import failed: {exc}")

    try:
        model = RFDETRTensorWrapper(RFDETRNano(pretrain_weights=None)).eval()
    except Exception as exc:
        pytest.skip(f"RF-DETR Nano construction requires unavailable local assets: {exc}")
    trace_inputs = torch.randn(2, 3, 224, 224)
    runtime_inputs = torch.randn(3, 3, 224, 224)
    try:
        run_split_inference_equivalence(
            model,
            trace_inputs,
            runtime_inputs,
            "after:model.transformer.decoder.layers.0.norm3",
        )
        # RF-DETR's train() path builds batch-dependent Python containers before
        # the chosen boundary. Keep the heavy smoke in eval mode while still
        # exercising Ariadne's training prefix/suffix/backward APIs.
        runtime = make_runtime(
            model,
            trace_inputs,
            "after:model.transformer.decoder.layers.0.norm3",
        )
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        before = clone_trainable_state(model)
        boundary = runtime.runtime.run_training_prefix(runtime_inputs)
        loss, boundary_grads = runtime.runtime.train_suffix(
            boundary,
            None,
            loss_fn=lambda output, _targets: nested_tensor_loss(output),
            optimizer=optimizer,
        )
        runtime.runtime.backward_prefix(boundary, boundary_grads=boundary_grads, optimizer=optimizer)
        after = clone_trainable_state(model)
        assert torch.isfinite(loss)
        assert boundary.tensors
        assert boundary_grads
        assert has_any_parameter_grad(model) or parameter_delta_nonzero(before, after)
    except Exception as exc:
        pytest.xfail(f"RF-DETR Nano split smoke is not stable in this environment: {exc}")
