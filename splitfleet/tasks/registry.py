"""Task-level experiment contracts and an explicit registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from splitfleet.tasks.adapters import (
    DetectionTask,
    ImageClassificationTask,
    InstanceSegmentationTask,
    SemanticSegmentationTask,
    TextClassificationTask,
)
from splitfleet.tasks.base import TaskAdapter
from splitfleet.tasks.metrics import (
    _array,
    classification_accuracy,
    detection_map50,
    macro_f1,
    mean_dice,
    mean_iou,
)


class TaskFamily(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_CLASSIFICATION = "text_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    evaluate: Callable[[Any, Any], float]
    higher_is_better: bool = True
    unit: str = "ratio"

    def __post_init__(self) -> None:
        normalized = str(self.name).strip().lower()
        if not normalized:
            raise ValueError("MetricSpec.name must not be empty.")
        if not callable(self.evaluate):
            raise TypeError("MetricSpec.evaluate must be callable.")
        object.__setattr__(self, "name", normalized)


@dataclass(frozen=True)
class TaskSpec:
    """One task's model/data/objective/metric/candidate-cut integration point.

    Only the adapter and metrics are mandatory.  Dataset and model factories
    stay optional so the core package does not download data or import a heavy
    framework merely to discover a task.
    """

    name: str
    family: TaskFamily | str
    adapter_factory: Callable[..., TaskAdapter]
    metrics: Mapping[str, MetricSpec]
    primary_metric: str
    model_factory: Callable[..., Any] | None = None
    dataset_factory: Callable[..., Any] | None = None
    candidate_cuts: Sequence[str] | Callable[[Any], Sequence[str]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = str(self.name).strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("TaskSpec.name must not be empty.")
        family = self.family if isinstance(self.family, TaskFamily) else TaskFamily(self.family)
        metrics = {str(key).strip().lower(): value for key, value in self.metrics.items()}
        primary = str(self.primary_metric).strip().lower()
        if not callable(self.adapter_factory):
            raise TypeError("TaskSpec.adapter_factory must be callable.")
        if not metrics or primary not in metrics:
            raise ValueError("TaskSpec.primary_metric must name one configured metric.")
        if any(key != metric.name for key, metric in metrics.items()):
            raise ValueError("TaskSpec metric mapping keys must equal MetricSpec.name.")
        if self.model_factory is not None and not callable(self.model_factory):
            raise TypeError("TaskSpec.model_factory must be callable when provided.")
        if self.dataset_factory is not None and not callable(self.dataset_factory):
            raise TypeError("TaskSpec.dataset_factory must be callable when provided.")
        if not callable(self.candidate_cuts) and any(
            not isinstance(value, str) or not value.strip() for value in self.candidate_cuts
        ):
            raise ValueError("TaskSpec candidate cuts must be non-empty strings.")
        object.__setattr__(self, "name", normalized)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "primary_metric", primary)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not callable(self.candidate_cuts):
            object.__setattr__(self, "candidate_cuts", tuple(self.candidate_cuts))

    def make_adapter(self, **kwargs: Any) -> TaskAdapter:
        return self.adapter_factory(**kwargs)

    def build_model(self, **kwargs: Any) -> Any:
        if self.model_factory is None:
            raise RuntimeError(f"Task {self.name!r} has no model_factory; supply one in a derived TaskSpec.")
        return self.model_factory(**kwargs)

    def build_datasets(self, **kwargs: Any) -> Any:
        if self.dataset_factory is None:
            raise RuntimeError(f"Task {self.name!r} has no dataset_factory; supply one in a derived TaskSpec.")
        return self.dataset_factory(**kwargs)

    def resolve_candidate_cuts(self, model: Any) -> tuple[str, ...]:
        values = self.candidate_cuts(model) if callable(self.candidate_cuts) else self.candidate_cuts
        resolved = tuple(str(value) for value in values)
        if len(resolved) != len(set(resolved)):
            raise ValueError(f"Task {self.name!r} produced duplicate candidate cuts.")
        return resolved

    def evaluate(self, targets: Any, predictions: Any) -> dict[str, float]:
        return {
            name: float(metric.evaluate(targets, predictions))
            for name, metric in self.metrics.items()
        }


class TaskRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, TaskSpec] = {}
        self._aliases: dict[str, str] = {}

    @staticmethod
    def _key(name: str) -> str:
        key = str(name).strip().lower().replace("-", "_")
        if not key:
            raise ValueError("Task name must not be empty.")
        return key

    def register(
        self,
        spec: TaskSpec,
        *,
        aliases: Sequence[str] = (),
        replace: bool = False,
    ) -> None:
        key = self._key(spec.name)
        alias_keys = tuple(self._key(alias) for alias in aliases)
        collisions = [name for name in (key, *alias_keys) if name in self._specs or name in self._aliases]
        if collisions and not replace:
            raise KeyError(f"Task names are already registered: {sorted(collisions)}")
        if replace:
            foreign = [
                name for name in (key, *alias_keys)
                if (name in self._specs and name != key)
                or (name in self._aliases and self._aliases[name] != key)
            ]
            if foreign:
                raise KeyError(f"Task names belong to another registered task: {sorted(set(foreign))}")
            self._specs.pop(key, None)
            for alias, target in list(self._aliases.items()):
                if alias in alias_keys or target == key:
                    self._aliases.pop(alias, None)
        self._specs[key] = spec
        for alias in alias_keys:
            if alias == key:
                continue
            self._aliases[alias] = key

    def get(self, name: str) -> TaskSpec:
        key = self._key(name)
        key = self._aliases.get(key, key)
        try:
            return self._specs[key]
        except KeyError as exc:
            raise KeyError(f"Unknown task {name!r}; available: {list(self.names())}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def aliases(self) -> Mapping[str, str]:
        return MappingProxyType(dict(sorted(self._aliases.items())))


def _segmentation_metric(function: Callable[..., float]) -> Callable[[Any, Any], float]:
    def evaluate(targets: Any, predictions: Any) -> float:
        expected = _array(targets)
        observed = _array(predictions)
        valid_mask = expected != 255
        if not np.any(valid_mask):
            raise ValueError("Cannot infer segmentation classes from an empty/ignored target.")
        num_classes = int(max(expected[valid_mask].max(), observed[valid_mask].max())) + 1
        return function(expected, observed, num_classes=num_classes)

    return evaluate


TASK_SPECS = TaskRegistry()
TASK_SPECS.register(
    TaskSpec(
        name="image_classification",
        family=TaskFamily.IMAGE_CLASSIFICATION,
        adapter_factory=ImageClassificationTask,
        metrics={
            "accuracy": MetricSpec("accuracy", classification_accuracy),
            "macro_f1": MetricSpec("macro_f1", macro_f1),
        },
        primary_metric="accuracy",
    ),
    aliases=("image_cls",),
)
TASK_SPECS.register(
    TaskSpec(
        name="text_classification",
        family=TaskFamily.TEXT_CLASSIFICATION,
        adapter_factory=TextClassificationTask,
        metrics={
            "accuracy": MetricSpec("accuracy", classification_accuracy),
            "macro_f1": MetricSpec("macro_f1", macro_f1),
        },
        primary_metric="macro_f1",
    ),
    aliases=("text_cls",),
)
TASK_SPECS.register(
    TaskSpec(
        name="object_detection",
        family=TaskFamily.OBJECT_DETECTION,
        adapter_factory=DetectionTask,
        metrics={"map50": MetricSpec("map50", detection_map50)},
        primary_metric="map50",
    ),
    aliases=("detection",),
)
TASK_SPECS.register(
    TaskSpec(
        name="semantic_segmentation",
        family=TaskFamily.SEMANTIC_SEGMENTATION,
        adapter_factory=SemanticSegmentationTask,
        metrics={
            "miou": MetricSpec("miou", _segmentation_metric(mean_iou)),
            "dice": MetricSpec("dice", _segmentation_metric(mean_dice)),
        },
        primary_metric="miou",
    ),
    aliases=("segmentation",),
)
TASK_SPECS.register(
    TaskSpec(
        name="instance_segmentation",
        family=TaskFamily.INSTANCE_SEGMENTATION,
        adapter_factory=InstanceSegmentationTask,
        # This evaluates boxes only. Mask AP needs a separate prediction/target
        # evaluator and must not be inferred from a passing split-gradient test.
        metrics={"box_map50": MetricSpec("box_map50", detection_map50)},
        primary_metric="box_map50",
        metadata={"mask_ap_available": False},
    ),
    aliases=("instance_seg",),
)


__all__ = ["MetricSpec", "TaskFamily", "TaskSpec", "TaskRegistry", "TASK_SPECS"]
