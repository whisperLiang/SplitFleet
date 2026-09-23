from __future__ import annotations

import numpy as np
import pytest

from splitfleet.tasks import (
    TASK_SPECS,
    MetricSpec,
    TaskFamily,
    TaskRegistry,
    TaskSpec,
    classification_accuracy,
    detection_map,
    detection_map50,
    macro_f1,
    mean_dice,
    mean_iou,
)


def test_builtin_task_registry_exposes_the_unified_task_families() -> None:
    assert TASK_SPECS.names() == (
        "image_classification",
        "instance_segmentation",
        "object_detection",
        "semantic_segmentation",
        "text_classification",
    )
    assert TASK_SPECS.get("detection").family is TaskFamily.OBJECT_DETECTION
    assert TASK_SPECS.get("segmentation").primary_metric == "miou"
    assert TASK_SPECS.get("instance_seg").primary_metric == "box_map50"


def test_task_spec_can_bind_model_data_and_candidate_factories() -> None:
    spec = TaskSpec(
        name="custom",
        family=TaskFamily.IMAGE_CLASSIFICATION,
        adapter_factory=TASK_SPECS.get("image_classification").adapter_factory,
        metrics={"accuracy": MetricSpec("accuracy", classification_accuracy)},
        primary_metric="accuracy",
        model_factory=lambda width=4: {"width": width},
        dataset_factory=lambda split="train": [split],
        candidate_cuts=lambda model: [f"after:{model['width']}", "50%"],
    )
    registry = TaskRegistry()
    registry.register(spec, aliases=("alias",))

    assert registry.get("alias") is spec
    assert spec.build_model(width=8) == {"width": 8}
    assert spec.build_datasets(split="test") == ["test"]
    assert spec.resolve_candidate_cuts({"width": 8}) == ("after:8", "50%")
    assert spec.evaluate([0, 1], [0, 1]) == {"accuracy": 1.0}


def test_registry_rejects_duplicate_names_and_unknown_tasks() -> None:
    registry = TaskRegistry()
    spec = TASK_SPECS.get("image_classification")
    registry.register(spec, aliases=("vision",))
    with pytest.raises(KeyError, match="already registered"):
        registry.register(spec)
    with pytest.raises(KeyError, match="Unknown task"):
        registry.get("missing")


def test_registry_replace_cannot_shadow_another_canonical_name() -> None:
    registry = TaskRegistry()
    registry.register(TASK_SPECS.get("image_classification"))
    registry.register(TASK_SPECS.get("text_classification"))
    custom = TaskSpec(
        name="custom",
        family=TaskFamily.IMAGE_CLASSIFICATION,
        adapter_factory=TASK_SPECS.get("image_classification").adapter_factory,
        metrics={"accuracy": MetricSpec("accuracy", classification_accuracy)},
        primary_metric="accuracy",
    )
    with pytest.raises(KeyError, match="another registered task"):
        registry.register(custom, aliases=("image_classification",), replace=True)
    assert registry.get("image_classification").name == "image_classification"
    with pytest.raises(KeyError, match="Unknown task"):
        registry.get("custom")


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA unavailable")
def test_builtin_segmentation_metric_accepts_cuda_tensors() -> None:
    import torch

    targets = torch.tensor([[0, 1]], device="cuda")
    assert TASK_SPECS.get("segmentation").evaluate(targets, targets)["miou"] == 1.0


def test_classification_metrics_have_expected_values() -> None:
    targets = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    assert classification_accuracy(targets, predictions) == 0.75
    assert macro_f1(targets, predictions) == pytest.approx((2 / 3 + 0.8) / 2)


def test_segmentation_metrics_ignore_void_pixels() -> None:
    targets = np.array([[0, 0, 1], [1, 255, 1]])
    predictions = np.array([[0, 1, 1], [1, 0, 0]])
    # class 0: IoU=1/3, Dice=1/2; class 1: IoU=2/4, Dice=2/3.
    assert mean_iou(targets, predictions, num_classes=2) == pytest.approx(5 / 12)
    assert mean_dice(targets, predictions, num_classes=2) == pytest.approx(7 / 12)


def test_detection_map50_matches_ranked_predictions_and_penalizes_false_positive() -> None:
    targets = [
        {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "labels": np.array([1])},
        {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "labels": np.array([1])},
    ]
    perfect = [
        {
            "boxes": np.array([[0.0, 0.0, 1.0, 1.0]]),
            "labels": np.array([1]),
            "scores": np.array([0.9]),
        },
        {
            "boxes": np.array([[0.0, 0.0, 1.0, 1.0]]),
            "labels": np.array([1]),
            "scores": np.array([0.8]),
        },
    ]
    false_first = [
        {
            "boxes": np.array([[2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 1.0, 1.0]]),
            "labels": np.array([1, 1]),
            "scores": np.array([0.99, 0.9]),
        },
        perfect[1],
    ]

    assert detection_map50(targets, perfect) == 1.0
    assert detection_map50(targets, false_first) == pytest.approx(2 / 3)


def test_voc07_11point_ap_and_difficult_ground_truth() -> None:
    targets = [
        {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "labels": np.array([1]), "difficult": np.array([False])},
        {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "labels": np.array([1]), "difficult": np.array([False])},
        {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "labels": np.array([1]), "difficult": np.array([True])},
    ]
    predictions = [
        {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "labels": np.array([1]), "scores": np.array([0.9])},
        {"boxes": np.empty((0, 4)), "labels": np.array([], dtype=int), "scores": np.array([])},
        {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "labels": np.array([1]), "scores": np.array([0.8])},
    ]
    assert detection_map(targets, predictions) == pytest.approx(0.5)
    assert detection_map(targets, predictions, ap_mode="voc07_11point") == pytest.approx(6 / 11)
