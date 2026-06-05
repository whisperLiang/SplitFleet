from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.integration.torchlens_real_model_helpers import (
    nested_tensor_loss,
    run_split_inference_equivalence,
    run_split_training_smoke,
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
def test_real_image_classification_models(name, builder, input_shape, boundary) -> None:
    torch.manual_seed(100)
    model, num_classes = builder()
    trace_inputs = torch.randn(2, *input_shape)
    runtime_inputs = torch.randn(3, *input_shape)
    labels = torch.randint(0, num_classes, (3,))
    run_split_inference_equivalence(model, trace_inputs, runtime_inputs, boundary)
    run_split_training_smoke(
        model,
        trace_inputs,
        runtime_inputs,
        labels,
        nn.CrossEntropyLoss(),
        boundary=boundary,
    )


@pytest.mark.parametrize("name,builder,boundary", TEXT_CLASSIFICATION_MODELS)
def test_real_text_classification_models(name, builder, boundary) -> None:
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
def test_real_segmentation_models(name, builder, input_shape, boundary) -> None:
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
def test_real_detection_head_models(name, builder, input_shape, boundary) -> None:
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
