from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from splitfleet.autosplit import prepare_torchlens_runtime
from tests.integration.torchlens_real_model_helpers import (
    nested_tensor_loss,
    run_split_inference_equivalence,
    run_split_training_smoke,
    run_real_model_test_isolated,
    skip_if_missing_dependency,
)


class HFSequenceClassifierWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class SegmentationOutWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


class DetectionBackboneHeadWrapper(nn.Module):
    def __init__(self, model, kind: str):
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, x):
        features = self.model.backbone(x)
        feature_list = list(features.values()) if isinstance(features, dict) else [features]
        if self.kind == "fasterrcnn":
            objectness, box_regression = self.model.rpn.head(feature_list)
            tensors = [*objectness, *box_regression]
        else:
            head_out = self.model.head(feature_list)
            tensors = list(head_out.values()) if isinstance(head_out, dict) else list(head_out)
        return torch.cat([tensor.flatten(1).mean(dim=1, keepdim=True) for tensor in tensors], dim=1)


def build_torchvision_resnet18():
    skip_if_missing_dependency("torchvision")
    from torchvision.models import resnet18

    return resnet18(weights=None), 1000


def build_mobilenet_v3_large():
    skip_if_missing_dependency("torchvision")
    from torchvision.models import mobilenet_v3_large

    return mobilenet_v3_large(weights=None), 1000


def build_timm_resnet50():
    skip_if_missing_dependency("timm")
    import timm

    return timm.create_model("resnet50", pretrained=False), 1000


def build_timm_swin_tiny():
    skip_if_missing_dependency("timm")
    import timm

    return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False), 1000


def build_distilbert_sequence_classifier():
    skip_if_missing_dependency("transformers")
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    config = DistilBertConfig(
        vocab_size=30522,
        n_layers=2,
        dim=128,
        hidden_dim=256,
        n_heads=4,
        num_labels=2,
    )
    return HFSequenceClassifierWrapper(DistilBertForSequenceClassification(config)), 2


def build_bert_sequence_classifier():
    skip_if_missing_dependency("transformers")
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        num_labels=2,
    )
    return HFSequenceClassifierWrapper(BertForSequenceClassification(config)), 2


def build_roberta_sequence_classifier():
    skip_if_missing_dependency("transformers")
    from transformers import RobertaConfig, RobertaForSequenceClassification

    config = RobertaConfig(
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        num_labels=2,
    )
    return HFSequenceClassifierWrapper(RobertaForSequenceClassification(config)), 2


def make_text_inputs(batch_size: int, vocab_size: int = 30522):
    input_ids = torch.randint(0, vocab_size, (batch_size, 16))
    attention_mask = torch.ones(batch_size, 16, dtype=torch.long)
    return input_ids, attention_mask


def build_deeplabv3_resnet50_wrapper():
    skip_if_missing_dependency("torchvision")
    from torchvision.models.segmentation import deeplabv3_resnet50

    return SegmentationOutWrapper(deeplabv3_resnet50(weights=None, weights_backbone=None)), 21


def build_fcn_resnet50_wrapper():
    skip_if_missing_dependency("torchvision")
    from torchvision.models.segmentation import fcn_resnet50

    return SegmentationOutWrapper(fcn_resnet50(weights=None, weights_backbone=None)), 21


def build_lraspp_wrapper():
    skip_if_missing_dependency("torchvision")
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    return SegmentationOutWrapper(lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)), 21


def build_fasterrcnn_head_wrapper():
    skip_if_missing_dependency("torchvision")
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    return DetectionBackboneHeadWrapper(model, "fasterrcnn"), 1


def build_retinanet_head_wrapper():
    skip_if_missing_dependency("torchvision")
    from torchvision.models.detection import retinanet_resnet50_fpn

    model = retinanet_resnet50_fpn(weights=None, weights_backbone=None)
    return DetectionBackboneHeadWrapper(model, "retinanet"), 1


IMAGE_CLASSIFICATION_MODELS = [
    ("torchvision_resnet18", build_torchvision_resnet18, (3, 96, 96), "50%"),
    ("torchvision_mobilenet_v3_large", build_mobilenet_v3_large, (3, 96, 96), "50%"),
    ("timm_resnet50", build_timm_resnet50, (3, 96, 96), "50%"),
    ("timm_swin_tiny", build_timm_swin_tiny, (3, 224, 224), "50%"),
]

TEXT_CLASSIFICATION_MODELS = [
    ("distilbert_sequence_cls", build_distilbert_sequence_classifier, "50%"),
    ("bert_sequence_cls", build_bert_sequence_classifier, "50%"),
    ("roberta_sequence_cls", build_roberta_sequence_classifier, "50%"),
]

SEGMENTATION_MODELS = [
    ("deeplabv3_resnet50", build_deeplabv3_resnet50_wrapper, (3, 96, 96), "50%"),
    ("fcn_resnet50", build_fcn_resnet50_wrapper, (3, 96, 96), "50%"),
    ("lraspp_mobilenet_v3_large", build_lraspp_wrapper, (3, 96, 96), "50%"),
]

DETECTION_MODELS = [
    ("fasterrcnn_mobilenet_v3_fpn_head", build_fasterrcnn_head_wrapper, (3, 128, 128), "50%"),
    ("retinanet_resnet50_fpn_head", build_retinanet_head_wrapper, (3, 128, 128), "50%"),
]


@pytest.mark.parametrize("name,builder,input_shape,boundary", IMAGE_CLASSIFICATION_MODELS)
def test_real_image_classification_models(request, name, builder, input_shape, boundary) -> None:
    if run_real_model_test_isolated(request):
        return
    torch.manual_seed(100)
    model, num_classes = builder()
    trace_inputs = torch.randn(2, *input_shape)
    # TorchLens 2.31 cannot yet replay Swin window reshapes across a changed
    # concrete batch; other models in this matrix cover dynamic batch.
    runtime_batch = 2 if name == "timm_swin_tiny" else 3
    runtime_inputs = torch.randn(runtime_batch, *input_shape)
    labels = torch.randint(0, num_classes, (runtime_batch,))
    run_split_inference_equivalence(model, trace_inputs, runtime_inputs, boundary)
    run_split_training_smoke(
        model,
        trace_inputs,
        runtime_inputs,
        labels,
        nn.CrossEntropyLoss(),
        boundary=boundary,
    )


def test_torchvision_resnet18_split_training_step_matches_full_model_state() -> None:
    torch.manual_seed(918)
    model, num_classes = build_torchvision_resnet18()
    model.train()
    initial_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    full_model = copy.deepcopy(model).train()
    split_model = copy.deepcopy(model).train()
    trace_inputs = torch.randn(2, 3, 96, 96)
    runtime_inputs = torch.randn(3, 3, 96, 96)
    labels = torch.randint(0, num_classes, (3,))
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-4

    handle = prepare_torchlens_runtime(
        split_model,
        trace_inputs,
        boundary="50%",
        trainable=True,
        dynamic_batch=(2, 3),
    )
    # Runtime preparation traces in train mode, so reset both models before
    # comparing the actual single training step.
    full_model.load_state_dict(initial_state)
    split_model.load_state_dict(initial_state)
    full_model.train()
    split_model.train()

    full_optimizer = torch.optim.SGD(
        [param for param in full_model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    split_optimizer = torch.optim.SGD(
        [param for param in split_model.parameters() if param.requires_grad],
        lr=learning_rate,
    )

    full_optimizer.zero_grad(set_to_none=True)
    full_loss = loss_fn(full_model(runtime_inputs), labels)
    full_loss.backward()
    full_optimizer.step()

    boundary_payload = handle.backend.run_prefix(runtime_inputs, training=True)
    split_loss, boundary_grads = handle.backend.train_suffix(
        boundary_payload,
        labels,
        loss_fn=loss_fn,
        optimizer=split_optimizer,
    )
    handle.backend.backward_prefix(
        boundary_payload,
        boundary_grads=boundary_grads,
        optimizer=split_optimizer,
    )

    assert torch.allclose(full_loss.detach(), split_loss.detach(), rtol=1e-5, atol=1e-7)
    assert boundary_grads
    split_parameters = dict(split_model.named_parameters())
    for key, full_tensor in full_model.named_parameters():
        assert torch.allclose(
            full_tensor,
            split_parameters[key],
            rtol=1e-5,
            atol=1e-7,
        ), key


@pytest.mark.parametrize("name,builder,boundary", TEXT_CLASSIFICATION_MODELS)
def test_real_text_classification_models(request, name, builder, boundary) -> None:
    if run_real_model_test_isolated(request):
        return
    torch.manual_seed(200)
    model, num_labels = builder()
    trace_inputs = make_text_inputs(2)
    runtime_inputs = make_text_inputs(3)
    labels = torch.randint(0, num_labels, (3,))
    run_split_inference_equivalence(model, trace_inputs, runtime_inputs, boundary)
    run_split_training_smoke(
        model,
        trace_inputs,
        runtime_inputs,
        labels,
        nn.CrossEntropyLoss(),
        boundary=boundary,
    )


@pytest.mark.parametrize("name,builder,input_shape,boundary", SEGMENTATION_MODELS)
def test_real_segmentation_models(request, name, builder, input_shape, boundary) -> None:
    if run_real_model_test_isolated(request):
        return
    torch.manual_seed(300)
    model, num_classes = builder()
    trace_inputs = torch.randn(2, *input_shape)
    runtime_inputs = torch.randn(3, *input_shape)
    mask = torch.randint(0, num_classes, (3, input_shape[1], input_shape[2]))
    runtime = run_split_inference_equivalence(model, trace_inputs, runtime_inputs, boundary)
    split_output = runtime.backend.run_suffix(runtime.backend.run_prefix(runtime_inputs))
    assert split_output.shape[:2] == (3, num_classes)
    run_split_training_smoke(
        model,
        trace_inputs,
        runtime_inputs,
        mask,
        nn.CrossEntropyLoss(),
        boundary=boundary,
    )


@pytest.mark.parametrize("name,builder,input_shape,boundary", DETECTION_MODELS)
def test_real_detection_head_models(request, name, builder, input_shape, boundary) -> None:
    if run_real_model_test_isolated(request):
        return
    torch.manual_seed(400)
    model, _ = builder()
    trace_inputs = torch.randn(2, *input_shape)
    runtime_inputs = torch.randn(3, *input_shape)
    run_split_inference_equivalence(model, trace_inputs, runtime_inputs, boundary)
    run_split_training_smoke(
        model,
        trace_inputs,
        runtime_inputs,
        None,
        lambda output, _targets: nested_tensor_loss(output),
        boundary=boundary,
    )
