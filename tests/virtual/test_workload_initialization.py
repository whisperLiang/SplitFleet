"""Explicit initialization reaches the native model and frozen run metadata."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.virtual_sfl.__main__ import resolve_paths
from experiments.virtual_sfl.workloads import EdgeFactory, build_spec
from splitfleet.tasks import TASK_SPECS
from splitfleet.virtual import VirtualConfig


class SmallVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 1000)

    def forward(self, images):
        return self.fc(self.bn1(self.conv1(images)).mean(dim=(2, 3)))


def test_configured_checkpoint_initializes_backbone_and_records_its_hash(tmp_path, monkeypatch):
    from torchvision import models
    from experiments.unified_multitask.edge_models import make_edge_model

    monkeypatch.setattr(models, "resnet50", lambda **kwargs: SmallVisionModel())
    pretrained = SmallVisionModel()
    with torch.no_grad():
        pretrained.conv1.weight.fill_(.125)
        pretrained.bn1.running_mean.fill_(.75)
    checkpoint = tmp_path / "initial.pt"
    torch.save(pretrained.state_dict(), checkpoint)
    workload = SimpleNamespace(
        task=TASK_SPECS.get("image_classification"),
        partition_labels=(0, 1), collate_fn=None,
        model_factory=lambda: make_edge_model("resnet50_pretrained"),
    )
    monkeypatch.setattr(EdgeFactory, "workload", lambda self: (workload, {}, None))
    monkeypatch.chdir(tmp_path)
    config = resolve_paths(VirtualConfig(workload={
        "source": "real", "task": "image_classification",
        "model": "resnet50_pretrained", "pretrain_weights": "initial.pt",
    }))
    assert config.workload["pretrain_weights"] == str(checkpoint)
    spec = build_spec(config)
    model = spec.model_factory()
    torch.testing.assert_close(model.model.conv1.weight, pretrained.conv1.weight)
    torch.testing.assert_close(model.model.bn1.running_mean, pretrained.bn1.running_mean)
    assert model.model.fc.out_features == 10
    assert not model.model.bn1.training
    assert spec.data_manifest["initialization"] == {
        "source": "checkpoint", "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def test_missing_explicit_checkpoint_does_not_fall_back_to_random_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(EdgeFactory, "workload", lambda self: (None, {}, None))
    with pytest.raises(FileNotFoundError):
        build_spec(VirtualConfig(workload={
            "model": "resnet50_pretrained", "pretrain_weights": str(tmp_path / "missing.pt"),
        }))
