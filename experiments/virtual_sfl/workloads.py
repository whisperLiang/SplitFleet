"""Dataset/model factories reusing the physical task integrations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache, partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from experiments.unified_multitask.data import _subset
from splitfleet.tasks import TASK_SPECS
from splitfleet.virtual.runtime import TrainingSpec


class TinyClassifier(nn.Module):
    """Small correctness fixture; never labelled a real dataset experiment."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 8)
        self.middle = nn.Linear(8, 6)
        self.tail = nn.Linear(6, 2)

    def forward(self, x):
        return self.tail(self.middle(self.head(x).relu()).relu())


@lru_cache(maxsize=1)
def fixture_dataset(samples=400, seed=233):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, 4, generator=generator)
    return TensorDataset(x, (x.sum(dim=1) > 0).long())


def fixture_spec(*, samples=400, seed=233):
    return TrainingSpec(
        task=TASK_SPECS.get("image_classification"),
        model_factory=TinyClassifier,
        train_dataset_factory=partial(fixture_dataset, samples, seed),
        batch_axes={"/args/0": 0},
        partition_labels=tuple(
            int(y) for y in fixture_dataset(samples, seed).tensors[1]
        ),
        source="fixture",
        data_manifest={
            "name": "synthetic differentiable correctness fixture",
            "seed": seed,
        },
    )


@lru_cache(maxsize=1)
def cifar_dataset(root, samples, train=True):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    return _subset(
        CIFAR10(root=root, train=train, download=False, transform=ToTensor()), samples
    )


def resnet18():
    from torchvision.models import resnet18 as factory

    from experiments.unified_multitask.edge_models import FrozenStatisticsVisionModel

    return FrozenStatisticsVisionModel(factory(weights=None, num_classes=10))


def evaluate_classification(
    model, device, *, dataset_factory, task_name, collate_fn=None
):
    import numpy as np
    from torch.utils.data import DataLoader

    from experiments.common.workload_training import _move

    task = TASK_SPECS.get(task_name)
    adapter = task.make_adapter()
    targets, predictions = [], []
    weighted_loss, samples = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for raw in DataLoader(dataset_factory(), batch_size=2, collate_fn=collate_fn):
            batch = adapter.prepare_batch(raw, training=False)
            call = batch.inputs.map_values(
                lambda value: _move(value, torch.device(device))
            )
            expected = _move(batch.targets, torch.device(device))
            outputs = model(*call.args, **call.kwargs)
            loss = adapter.loss(outputs, expected)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Native virtual evaluation loss is nonfinite")
            weighted_loss += float(loss) * batch.num_examples
            samples += batch.num_examples
            if isinstance(outputs, dict):
                outputs = outputs[
                    "out" if task_name == "semantic_segmentation" else "logits"
                ]
            if not torch.isfinite(outputs).all():
                raise FloatingPointError(
                    "Native virtual evaluation predictions are nonfinite"
                )
            targets.extend(expected.cpu().numpy())
            predictions.extend(outputs.argmax(1).cpu().numpy())
    if not samples:
        raise ValueError("Evaluation dataset is empty")
    return {
        "loss": weighted_loss / samples,
        **task.evaluate(np.asarray(targets), np.asarray(predictions)),
    }


@dataclass(frozen=True)
class EdgeFactory:
    task: str
    name: str
    root: str
    samples: int
    evaluation_samples: int
    tokenizer_path: str | None = None
    pretrain_weights: str | None = None

    def workload(self):
        return edge_workload(self)

    def train_dataset(self):
        return self.workload()[0].train_dataset

    def model(self):
        workload, _, configuration = self.workload()
        if self.pretrain_weights is not None:
            from experiments.unified_multitask.edge_models import make_edge_model

            return make_edge_model(
                self.name, config=configuration, pretrain_weights=self.pretrain_weights,
                tokenizer_path=self.tokenizer_path,
            )
        return workload.model_factory()

    def evaluate(self, model, device):
        workload, _, _ = self.workload()
        if self.task == "object_detection":
            from experiments.common.workload_training import _evaluate

            return _evaluate(workload, model, batch_size=1, device=torch.device(device))
        return evaluate_classification(
            model,
            device,
            dataset_factory=lambda: workload.test_dataset,
            task_name=self.task,
            collate_fn=workload.collate_fn,
        )


@lru_cache(maxsize=1)
def edge_workload(factory):
    from experiments.unified_multitask.edge_models import load_edge_workload

    return load_edge_workload(
        factory.task,
        name=factory.name,
        data_root=factory.root,
        max_train_samples=factory.samples,
        max_test_samples=factory.evaluation_samples,
        tokenizer_path=factory.tokenizer_path,
    )


def build_spec(config):
    value = config.workload
    source = value.get("source", "real")
    if source == "fixture":
        if value.get("model", "tiny") != "tiny":
            raise ValueError(
                "Only the tiny differentiable fixture is supported for source=fixture"
            )
        return fixture_spec(samples=int(value.get("samples", 400)), seed=config.seed)
    if source != "real":
        raise ValueError("workload.source must be real or fixture")
    root = str(Path(value.get("data_root", "data")).resolve())
    name = value.get("model", "resnet18")
    task_name = value.get("task", "image_classification")
    samples = int(value.get("samples", 400))
    evaluation_samples = int(value.get("evaluation_samples", 16))
    if name == "resnet18":
        if task_name != "image_classification":
            raise ValueError("resnet18 preset requires image_classification")
        train = partial(cifar_dataset, root, samples)
        test = partial(cifar_dataset, root, evaluation_samples, False)
        dataset = train()
        labels = tuple(int(dataset.dataset.targets[i]) for i in dataset.indices)
        return TrainingSpec(
            task=TASK_SPECS.get(task_name),
            model_factory=resnet18,
            train_dataset_factory=train,
            partition_labels=labels,
            batch_axes={"/args/0": 0},
            source="real",
            evaluate_fn=partial(
                evaluate_classification, dataset_factory=test, task_name=task_name
            ),
            data_manifest={
                "dataset": "CIFAR-10",
                "model": name,
                "root": root,
                "source_indices": list(dataset.indices),
                "samples": len(dataset),
            },
        )
    factory = EdgeFactory(
        task=task_name,
        name=name,
        root=root,
        samples=samples,
        evaluation_samples=evaluation_samples,
        tokenizer_path=value.get("tokenizer_path"),
        pretrain_weights=value.get("pretrain_weights"),
    )
    if task_name == "text_classification" and factory.tokenizer_path is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        from experiments.unified_multitask.edge_models import BERT_REVISION

        directory = (
            Path(HF_HUB_CACHE)
            / "models--bert-base-uncased"
            / "snapshots"
            / BERT_REVISION
        )
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(
                "Set workload.tokenizer_path to the pinned BERT tokenizer/config directory"
            )
        factory = replace(factory, tokenizer_path=str(directory))
    workload, metadata, _ = factory.workload()
    if factory.pretrain_weights is not None:
        from experiments.unified_multitask.edge_models import file_sha256

        metadata = {**metadata, "initialization": {
            "source": "checkpoint", "path": factory.pretrain_weights,
            "sha256": file_sha256(factory.pretrain_weights),
        }}
    else:
        metadata = {**metadata, "initialization": {"source": "random"}}
    provider = None
    if task_name == "object_detection":
        from experiments.rfdetr_nano_physical import RFDETRCandidateProvider

        provider = RFDETRCandidateProvider
    axes = (
        {"/kwargs/input_ids": 0, "/kwargs/attention_mask": 0}
        if task_name == "text_classification"
        else {"/args/0": 0}
    )
    return TrainingSpec(
        task=workload.task,
        model_factory=factory.model,
        train_dataset_factory=factory.train_dataset,
        partition_labels=workload.partition_labels
        if task_name in {"image_classification", "text_classification"}
        else None,
        collate_fn=workload.collate_fn,
        batch_axes=axes,
        evaluate_fn=factory.evaluate,
        candidate_provider_factory=provider,
        source="real",
        data_manifest=metadata,
    )
