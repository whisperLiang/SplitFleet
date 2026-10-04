"""Standard pretrained architectures for the primary edge-device study.

Workers reconstruct architectures from the bundle and receive the frozen full
state; checkpoint and tokenizer downloads are coordinator preparation steps.
BatchNorm statistics are explicitly frozen for batch-one edge fine-tuning,
while affine parameters, all backbone weights, and native dropout remain live.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from .data import PetMasks, VOCBoxes, _build_workload, _detection_collate, _subset


EDGE_MODELS = {
    "resnet50_pretrained": {"task": "image_classification", "image_size": 224},
    "bert_base": {"task": "text_classification", "sequence_length": 128},
    "rfdetr_nano": {"task": "object_detection", "image_size": 384},
    "deeplabv3_resnet50": {"task": "semantic_segmentation", "image_size": 320},
}
EDGE_TASK_MODELS = {
    "image_classification": "resnet50_pretrained",
    "text_classification": "bert_base",
    "object_detection": "rfdetr_nano",
    "semantic_segmentation": "deeplabv3_resnet50",
}
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenStatisticsVisionModel(nn.Module):
    """Retain torchvision graphs and train affine BN using fixed statistics."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images):
        return self.model(images)


class BertTextClassifier(nn.Module):
    def __init__(self, config: dict[str, Any], *, pretrained_directory: str | None = None):
        super().__init__()
        from transformers import BertConfig, BertForSequenceClassification

        configuration = BertConfig.from_dict({**config, "num_labels": 4})
        if (configuration.hidden_size, configuration.num_hidden_layers,
                configuration.num_attention_heads, configuration.intermediate_size,
                configuration.vocab_size) != (768, 12, 12, 3072, 30522):
            raise ValueError("The primary text model must be the full BERT-base architecture")
        # Capture the unfused attention graph with its normal training dropout.
        configuration._attn_implementation = "eager"
        if pretrained_directory:
            self.model = BertForSequenceClassification.from_pretrained(
                pretrained_directory, config=configuration, local_files_only=True,
                attn_implementation="eager",
            )
        else:
            self.model = BertForSequenceClassification(configuration)
        # from_pretrained returns eval mode; capture and worker reconstruction
        # must start from the same training graph, including native dropout.
        self.train()

    def forward(self, input_ids, attention_mask):
        # The Transformers mask factory can choose a different graph when all
        # tokens are unmasked. This mathematically equivalent additive mask
        # makes padded and unpadded minibatches use the same captured path.
        additive_mask = (1.0 - attention_mask[:, None, None, :].to(self.model.dtype)) \
            * torch.finfo(self.model.dtype).min
        return {"logits": self.model(input_ids=input_ids, attention_mask=additive_mask,
                                      return_dict=False)[0]}


def make_edge_model(name: str, *, config: dict | None = None,
                    pretrain_weights: str | None = None, tokenizer_path: str | None = None):
    if name not in EDGE_MODELS:
        raise ValueError(f"Unknown primary edge model {name!r}")
    if pretrain_weights and not Path(pretrain_weights).is_file():
        raise FileNotFoundError(pretrain_weights)
    if name == "rfdetr_nano":
        from experiments.rfdetr_nano_physical import RFDETRNanoDetector

        return RFDETRNanoDetector(pretrain_weights=pretrain_weights)
    if name == "bert_base":
        if config is None:
            if not tokenizer_path:
                raise ValueError("BERT requires its pinned configuration/tokenizer directory")
            config = json.loads((Path(tokenizer_path) / "config.json").read_text())
        if pretrain_weights:
            if Path(pretrain_weights).parent != Path(tokenizer_path):
                raise ValueError("BERT weights and tokenizer must belong to one pinned snapshot")
        return BertTextClassifier(config, pretrained_directory=tokenizer_path if pretrain_weights else None)
    from torchvision import models

    if name == "resnet50_pretrained":
        model = models.resnet50(weights=None)
        if pretrain_weights:
            model.load_state_dict(torch.load(pretrain_weights, map_location="cpu", weights_only=True), strict=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        model = models.segmentation.deeplabv3_resnet50(
            weights=None, weights_backbone=None, num_classes=21, aux_loss=True,
        )
        if pretrain_weights:
            model.load_state_dict(torch.load(pretrain_weights, map_location="cpu", weights_only=True), strict=True)
        model.classifier[4] = nn.Conv2d(256, 3, 1)
        model.aux_classifier[4] = nn.Conv2d(256, 3, 1)
    return FrozenStatisticsVisionModel(model)


def model_configuration(model: nn.Module, name: str) -> dict:
    return model.model.config.to_dict() if name == "bert_base" else {}


class TokenizedAGNews(Dataset):
    def __init__(self, rows: list[list[str]], tokenizer, *, source_indices: list[int]):
        self.source_indices = source_indices
        self.targets = [int(row[0]) - 1 for row in rows]
        if any(not 0 <= target < 4 for target in self.targets):
            raise ValueError("AG News labels must be in 1..4")
        self.tokens = tokenizer([row[1] + " " + row[2] for row in rows],
                                max_length=128, truncation=True, padding="max_length",
                                return_tensors="pt", return_token_type_ids=False)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {"input_ids": self.tokens["input_ids"][index],
                "attention_mask": self.tokens["attention_mask"][index],
                "labels": torch.tensor(self.targets[index])}


class NormalizedImages(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from torchvision.transforms.functional import normalize

        image, target = self.dataset[index]
        return normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), target


def load_edge_workload(task: str, *, name: str, data_root: str,
                       max_train_samples: int | None, max_test_samples: int | None,
                       tokenizer_path: str | None = None):
    if name not in EDGE_MODELS or EDGE_MODELS[name]["task"] != task:
        raise ValueError(f"Model {name!r} does not implement task {task!r}")
    if any(maximum is not None and maximum < 1 for maximum in (max_train_samples, max_test_samples)):
        raise ValueError("Edge workload sample budgets must be positive")
    root = Path(data_root)
    metadata: dict[str, Any] = {"model_id": name, **EDGE_MODELS[name],
        "normalization_policy": "pretrained BatchNorm statistics frozen; affine parameters trainable"
                                if name in ("resnet50_pretrained", "deeplabv3_resnet50") else "native",
        "dropout_policy": "native probabilities; advancing per-process PyTorch RNG"}
    configuration = None
    if task == "image_classification":
        from torchvision.datasets import CIFAR10
        from torchvision.models import ResNet50_Weights

        transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
        train = CIFAR10(str(root), train=True, download=False, transform=transform)
        test = CIFAR10(str(root), train=False, download=False, transform=transform)
        metadata["preprocessing"] = "resize shorter side 256, center crop 224, ImageNet mean/std"
    elif task == "text_classification":
        if not tokenizer_path:
            raise ValueError("Tokenized AG News requires the pinned BERT tokenizer")
        from transformers import AutoTokenizer

        directory = Path(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        configuration = json.loads((directory / "config.json").read_text())
        rows = {}
        for split in ("train", "test"):
            with (root / "ag_news_csv" / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
                rows[split] = list(csv.reader(handle))
            if any(len(row) < 3 for row in rows[split]):
                raise ValueError("AG News CSV must contain label, title and description")
        content = lambda row: " ".join((row[1] + " " + row[2]).lower().split())
        heldout_contents = {content(row) for row in rows["test"]}
        kept = [index for index, row in enumerate(rows["train"]) if content(row) not in heldout_contents]
        removed = sorted(set(range(len(rows["train"]))) - set(kept))
        # Remove normalized train/evaluation exact duplicates before sampling.
        train_rows = [rows["train"][index] for index in kept]
        import numpy as np
        selected = lambda length, maximum: np.linspace(0, length - 1, min(maximum or length, length), dtype=int).tolist()
        chosen_train = selected(len(train_rows), max_train_samples)
        chosen_test = selected(len(rows["test"]), max_test_samples)
        train = TokenizedAGNews([train_rows[index] for index in chosen_train], tokenizer,
                                source_indices=[kept[index] for index in chosen_train])
        test = TokenizedAGNews([rows["test"][index] for index in chosen_test], tokenizer,
                               source_indices=chosen_test)
        max_train_samples = max_test_samples = None
        metadata.update(tokenizer_files={filename: file_sha256(directory / filename)
            for filename in ("config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json")
            if (directory / filename).is_file()},
            requested_pretrained_revision=BERT_REVISION, removed_training_source_indices=removed,
            attention_mask_policy="equivalent additive 4D mask; one graph for padded/unpadded inputs",
            text_deduplication="remove train/test lowercase, collapsed-whitespace exact article overlap")
    elif task == "object_detection":
        size = EDGE_MODELS[name]["image_size"]
        train = VOCBoxes(root, image_set="trainval", download=False, image_size=size)
        test = VOCBoxes(root, image_set="test", download=False, image_size=size)
        from experiments.rfdetr_nano_physical import _config

        metadata["preprocessing"] = "resize 384x384, ToTensor [0,1]; normalized xyxy targets"
        metadata["rfdetr_recipe"] = _config().model_dump()
    else:
        train = NormalizedImages(PetMasks(root, split="trainval", download=False, image_size=320))
        test = NormalizedImages(PetMasks(root, split="test", download=False, image_size=320))
        metadata["preprocessing"] = "resize 320x320, ImageNet mean/std, nearest-neighbor trimaps"
    workload = _build_workload(task, lambda: make_edge_model(name, config=configuration),
        _subset(train, max_train_samples), _subset(test, max_test_samples),
        _detection_collate if task == "object_detection" else None, source="real")
    return workload, metadata, configuration
