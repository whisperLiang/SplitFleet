"""Real and deterministic fixture datasets for the four-task benchmark."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from splitfleet.tasks import MetricSpec, TASK_SPECS, TaskSpec, detection_map, macro_f1, mean_dice, mean_iou

from .models import GridDetector, ImageClassifier, Segmenter, TextClassifier, detection_loss


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class Workload:
    task: TaskSpec
    model_factory: Callable[[], torch.nn.Module]
    train_dataset: Dataset
    test_dataset: Dataset
    partition_labels: tuple[int, ...]
    data_content_hash: str
    collate_fn: Callable[[list[Any]], Any] | None = None
    source: str = "fixture"


class AGNewsDataset(Dataset):
    """Read the canonical local AG News CSV layout with training-only vocabulary."""

    def __init__(self, path: str | Path, vocabulary: dict[str, int], *, max_length: int = 64) -> None:
        self.records: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    raise ValueError(f"AG News row in {path} needs label,title,description.")
                label = int(row[0]) - 1
                if not 0 <= label < 4:
                    raise ValueError(f"AG News label must be in 1..4, got {row[0]!r}.")
                tokens = TOKEN_PATTERN.findall((row[1] + " " + row[2]).lower())[:max_length]
                ids = [vocabulary.get(token, 1) for token in tokens]
                mask = [1] * len(ids)
                ids.extend([0] * (max_length - len(ids)))
                mask.extend([0] * (max_length - len(mask)))
                self.records.append(
                    (torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), label)
                )
        if not self.records:
            raise ValueError(f"AG News CSV is empty: {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids, mask, label = self.records[index]
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor(label)}

    @property
    def targets(self) -> list[int]:
        return [label for _ids, _mask, label in self.records]


def _ag_news_vocabulary(path: Path, *, max_terms: int = 20_000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                raise ValueError(f"AG News row in {path} needs label,title,description.")
            counts.update(TOKEN_PATTERN.findall((row[1] + " " + row[2]).lower()))
    return {token: index + 2 for index, (token, _count) in enumerate(
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_terms]
    )}


class VOCBoxes(Dataset):
    def __init__(self, root: str | Path, *, image_set: str, download: bool, image_size: int = 96) -> None:
        from torchvision.datasets import VOCDetection

        self.base = VOCDetection(root=str(root), year="2007", image_set=image_set, download=download)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from torchvision.transforms import functional as TF

        image, annotation = self.base[index]
        original_width, original_height = image.size
        image = TF.to_tensor(TF.resize(image, (self.image_size, self.image_size)))
        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        boxes: list[list[float]] = []
        labels: list[int] = []
        difficult: list[bool] = []
        for item in objects:
            if item["name"] not in VOC_CLASSES:
                continue
            bounds = item["bndbox"]
            boxes.append([
                float(bounds["xmin"]) / original_width,
                float(bounds["ymin"]) / original_height,
                float(bounds["xmax"]) / original_width,
                float(bounds["ymax"]) / original_height,
            ])
            labels.append(VOC_CLASSES.index(item["name"]))
            difficult.append(bool(int(item.get("difficult", 0))))
        return image, {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp(0, 1),
            "labels": torch.tensor(labels, dtype=torch.long),
            "difficult": torch.tensor(difficult, dtype=torch.bool),
        }


class PetMasks(Dataset):
    def __init__(self, root: str | Path, *, split: str, download: bool, image_size: int = 64) -> None:
        from torchvision.datasets import OxfordIIITPet

        self.base = OxfordIIITPet(
            root=str(root), split=split, target_types="segmentation", download=download
        )
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        image, mask = self.base[index]
        image = TF.to_tensor(TF.resize(image, (self.image_size, self.image_size)))
        mask = np.asarray(
            TF.resize(mask, (self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
            dtype=np.int64,
        ).copy()
        # Oxford-IIIT Pet trimaps use 1, 2, 3; 0 marks corrupt/void pixels.
        mask = np.where(mask == 0, 255, mask - 1)
        return image, torch.from_numpy(mask.astype(np.int64))


def _detection_collate(batch: list[Any]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return torch.stack(images), list(targets)


def _subset(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None:
        return dataset
    if maximum < 1:
        raise ValueError("max_train_samples and max_test_samples must be positive.")
    count = min(maximum, len(dataset))
    # A CSV may be ordered by class (AG News is); taking its first N rows
    # silently changes a four-class task into a one-class pilot.  Spanning the
    # complete source index range is deterministic and paired across methods.
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    return Subset(dataset, indices)


def _partition_labels(dataset: Dataset, task: str) -> tuple[int, ...]:
    if task == "image_classification":
        if isinstance(dataset, Subset) and hasattr(dataset.dataset, "targets"):
            return tuple(int(dataset.dataset.targets[index]) for index in dataset.indices)
        if hasattr(dataset, "targets"):
            return tuple(int(value) for value in dataset.targets)
    result: list[int] = []
    for index in range(len(dataset)):
        item = dataset[index]
        if task == "text_classification":
            result.append(int(item["labels"]))
        elif task == "object_detection":
            labels = item[1]["labels"]
            result.append(int(labels[0]) if len(labels) else 0)
        else:
            pixels = item[1].reshape(-1)
            pixels = pixels[pixels != 255]
            result.append(int(torch.bincount(pixels, minlength=3).argmax()) if len(pixels) else 0)
    return tuple(result)


def _fixture(task: str, *, seed: int, train_size: int, test_size: int) -> Workload:
    generator = torch.Generator().manual_seed(seed)
    count = train_size + test_size
    if task == "image_classification":
        images = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.arange(count) % 10
        all_data: Dataset = TensorDataset(images, labels)
        model_factory = lambda: ImageClassifier(10)
        collate = None
    elif task == "text_classification":
        ids = torch.randint(2, 100, (count, 12), generator=generator)
        labels = torch.arange(count) % 4

        class FixtureText(Dataset):
            def __len__(self) -> int:
                return count

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                return {"input_ids": ids[index], "attention_mask": torch.ones(12), "labels": labels[index]}

        all_data = FixtureText()
        model_factory = lambda: TextClassifier(100)
        collate = None
    elif task == "object_detection":
        images = torch.randn(count, 3, 32, 32, generator=generator)

        class FixtureDetection(Dataset):
            def __len__(self) -> int:
                return count

            def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                return images[index], {
                    "boxes": torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
                    "labels": torch.tensor([index % 3]),
                }

        all_data = FixtureDetection()
        model_factory = lambda: GridDetector(3)
        collate = _detection_collate
    else:
        images = torch.randn(count, 3, 32, 32, generator=generator)
        masks = torch.randint(0, 3, (count, 32, 32), generator=generator)
        all_data = TensorDataset(images, masks)
        model_factory = lambda: Segmenter(3)
        collate = None
    train = Subset(all_data, range(train_size))
    test = Subset(all_data, range(train_size, count))
    return _build_workload(task, model_factory, train, test, collate, source="fixture")


def _build_workload(
    task: str,
    model_factory: Callable[[], torch.nn.Module],
    train: Dataset,
    test: Dataset,
    collate: Callable[[list[Any]], Any] | None,
    *,
    source: str,
) -> Workload:
    base = TASK_SPECS.get(task)
    if task == "object_detection":
        def adapter_factory():
            return base.adapter_factory(model_loss=False, loss_fn=detection_loss)
    else:
        adapter_factory = base.adapter_factory
    metrics = dict(base.metrics)
    if task == "image_classification":
        metrics["macro_f1"] = MetricSpec(
            "macro_f1", lambda target, pred: macro_f1(target, pred, labels=range(10))
        )
    elif task == "text_classification":
        metrics["macro_f1"] = MetricSpec(
            "macro_f1", lambda target, pred: macro_f1(target, pred, labels=range(4))
        )
    elif task == "object_detection" and source == "real":
        metrics["map50"] = MetricSpec(
            "map50", lambda target, pred: detection_map(target, pred, ap_mode="voc07_11point")
        )
        metrics["map50_allpoint"] = MetricSpec(
            "map50_allpoint", lambda target, pred: detection_map(target, pred, ap_mode="all_point")
        )
    if task == "semantic_segmentation":
        metrics = {
            "miou": MetricSpec("miou", lambda target, pred: mean_iou(target, pred, num_classes=3)),
            "dice": MetricSpec("dice", lambda target, pred: mean_dice(target, pred, num_classes=3)),
        }
    spec = replace(
        base,
        adapter_factory=adapter_factory,
        model_factory=model_factory,
        dataset_factory=lambda: (train, test),
        candidate_cuts=("25%", "50%", "75%"),
        metrics=metrics,
        metadata={"source": source},
    )
    return Workload(
        task=spec,
        model_factory=model_factory,
        train_dataset=train,
        test_dataset=test,
        partition_labels=_partition_labels(train, task),
        data_content_hash=_dataset_pair_hash(train, test),
        collate_fn=collate,
        source=source,
    )


def _dataset_pair_hash(train: Dataset, test: Dataset) -> str:
    """Hash exactly the transformed samples consumed by training and testing."""

    digest = hashlib.sha256()
    for split_name, dataset in (("train", train), ("test", test)):
        digest.update(split_name.encode("ascii"))
        digest.update(len(dataset).to_bytes(8, "little"))
        for index in range(len(dataset)):
            digest.update(index.to_bytes(8, "little"))
            _hash_value(digest, dataset[index])
    return digest.hexdigest()


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(b"tensor")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _hash_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence")
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(repr(value).encode("utf-8"))


def load_workload(
    task: str,
    *,
    data_root: str | Path = "data",
    source: str = "real",
    download: bool = False,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    seed: int = 2026,
) -> Workload:
    """Load one task without implicit downloads or changing its evaluation set."""

    task = TASK_SPECS.get(task).name
    if task == "instance_segmentation":
        raise ValueError("The four-task benchmark excludes instance segmentation; use the correctness matrix.")
    if source == "fixture":
        return _fixture(
            task,
            seed=seed,
            train_size=max_train_samples or 8,
            test_size=max_test_samples or 4,
        )
    if source != "real":
        raise ValueError("source must be 'real' or 'fixture'.")
    root = Path(data_root)
    if task == "image_classification":
        from torchvision import datasets, transforms

        transform = transforms.ToTensor()
        train = datasets.CIFAR10(str(root), train=True, download=download, transform=transform)
        test = datasets.CIFAR10(str(root), train=False, download=download, transform=transform)
        model_factory = lambda: ImageClassifier(10)
        collate = None
    elif task == "text_classification":
        train_path = root / "ag_news_csv" / "train.csv"
        test_path = root / "ag_news_csv" / "test.csv"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(
                "AG News needs data/ag_news_csv/train.csv and test.csv "
                "(CSV columns: label 1..4, title, description)."
            )
        vocabulary = _ag_news_vocabulary(train_path)
        train = AGNewsDataset(train_path, vocabulary)
        test = AGNewsDataset(test_path, vocabulary)
        model_factory = lambda: TextClassifier(len(vocabulary) + 2)
        collate = None
    elif task == "object_detection":
        train = VOCBoxes(root, image_set="trainval", download=download)
        test = VOCBoxes(root, image_set="test", download=download)
        model_factory = lambda: GridDetector(len(VOC_CLASSES))
        collate = _detection_collate
    else:
        train = PetMasks(root, split="trainval", download=download)
        test = PetMasks(root, split="test", download=download)
        model_factory = lambda: Segmenter(3)
        collate = None
    return _build_workload(
        task,
        model_factory,
        _subset(train, max_train_samples),
        _subset(test, max_test_samples),
        collate,
        source="real",
    )


__all__ = ["AGNewsDataset", "VOCBoxes", "PetMasks", "Workload", "load_workload"]
