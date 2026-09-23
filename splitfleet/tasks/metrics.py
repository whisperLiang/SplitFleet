"""Framework-neutral reference metrics for SplitFleet task specifications.

The implementations intentionally depend only on NumPy.  They are suitable for
small and medium validation sets and provide deterministic reference values for
the experiment harness; large production evaluations may replace them with a
streaming or official dataset evaluator through :class:`TaskSpec`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def classification_accuracy(targets: Any, predictions: Any) -> float:
    """Return example accuracy for aligned integer class labels."""

    expected = _array(targets).reshape(-1)
    observed = _array(predictions).reshape(-1)
    if expected.size == 0 or expected.shape != observed.shape:
        raise ValueError("Classification targets and predictions must be non-empty and aligned.")
    return float(np.mean(expected == observed))


def macro_f1(targets: Any, predictions: Any, *, labels: Sequence[int] | None = None) -> float:
    """Return unweighted per-class F1, including a zero for an unpredicted class."""

    expected = _array(targets).reshape(-1)
    observed = _array(predictions).reshape(-1)
    if expected.size == 0 or expected.shape != observed.shape:
        raise ValueError("Classification targets and predictions must be non-empty and aligned.")
    classes = np.asarray(
        list(labels) if labels is not None else sorted(set(expected.tolist()) | set(observed.tolist()))
    )
    if classes.size == 0:
        raise ValueError("At least one class label is required.")
    scores: list[float] = []
    for label in classes:
        true_positive = int(np.sum((expected == label) & (observed == label)))
        false_positive = int(np.sum((expected != label) & (observed == label)))
        false_negative = int(np.sum((expected == label) & (observed != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def segmentation_confusion_matrix(
    targets: Any,
    predictions: Any,
    *,
    num_classes: int,
    ignore_index: int | None = 255,
) -> np.ndarray:
    """Build a ``[target, prediction]`` confusion matrix for semantic masks."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    expected = _array(targets).astype(np.int64, copy=False).reshape(-1)
    observed = _array(predictions).astype(np.int64, copy=False).reshape(-1)
    if expected.size == 0 or expected.shape != observed.shape:
        raise ValueError("Segmentation targets and predictions must be non-empty and aligned.")
    valid = (expected >= 0) & (expected < num_classes)
    if ignore_index is not None:
        valid &= expected != int(ignore_index)
    if np.any((observed[valid] < 0) | (observed[valid] >= num_classes)):
        raise ValueError("Segmentation predictions contain labels outside num_classes.")
    encoded = num_classes * expected[valid] + observed[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def mean_iou(
    targets: Any,
    predictions: Any,
    *,
    num_classes: int,
    ignore_index: int | None = 255,
) -> float:
    """Return mean intersection-over-union across classes present in the union."""

    matrix = segmentation_confusion_matrix(
        targets, predictions, num_classes=num_classes, ignore_index=ignore_index
    )
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    valid = union > 0
    if not np.any(valid):
        raise ValueError("No valid segmentation pixels remain after filtering.")
    return float(np.mean(intersection[valid] / union[valid]))


def mean_dice(
    targets: Any,
    predictions: Any,
    *,
    num_classes: int,
    ignore_index: int | None = 255,
) -> float:
    """Return macro Dice across classes present in targets or predictions."""

    matrix = segmentation_confusion_matrix(
        targets, predictions, num_classes=num_classes, ignore_index=ignore_index
    )
    intersection = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    valid = denominator > 0
    if not np.any(valid):
        raise ValueError("No valid segmentation pixels remain after filtering.")
    return float(np.mean(2.0 * intersection[valid] / denominator[valid]))


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + areas - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _validate_detection_record(record: Mapping[str, Any], *, prediction: bool) -> None:
    required = {"boxes", "labels"}
    if prediction:
        required.add("scores")
    missing = required - set(record)
    if missing:
        raise ValueError(f"Detection record is missing fields: {sorted(missing)}")
    boxes = _array(record["boxes"])
    labels = _array(record["labels"])
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or labels.ndim != 1:
        raise ValueError("Detection boxes must be [N, 4] and labels must be [N].")
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError("Detection boxes and labels must contain the same number of objects.")
    if prediction and _array(record["scores"]).reshape(-1).shape[0] != boxes.shape[0]:
        raise ValueError("Detection scores must contain one value per predicted box.")
    if not prediction and "difficult" in record and _array(record["difficult"]).reshape(-1).shape[0] != boxes.shape[0]:
        raise ValueError("Detection difficult flags must contain one value per ground-truth box.")


def detection_map(
    targets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
    ap_mode: Literal["all_point", "voc07_11point"] = "all_point",
) -> float:
    """Compute detection mAP at one IoU threshold.

    Classes without any ground-truth object are excluded, matching common mAP
    practice. Difficult ground-truth boxes are ignored when flagged, and each
    non-difficult ground-truth box can be matched at most once.
    """

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1].")
    if ap_mode not in ("all_point", "voc07_11point"):
        raise ValueError("ap_mode must be 'all_point' or 'voc07_11point'.")
    if not targets or len(targets) != len(predictions):
        raise ValueError("Detection targets and predictions must be non-empty and aligned by image.")
    for record in targets:
        _validate_detection_record(record, prediction=False)
    for record in predictions:
        _validate_detection_record(record, prediction=True)

    classes = sorted(
        {
            int(label)
            for record in targets
            for label in _array(record["labels"]).reshape(-1).tolist()
        }
    )
    if not classes:
        raise ValueError("Detection mAP requires at least one ground-truth object.")
    average_precisions: list[float] = []
    for label in classes:
        ground_truth: dict[int, np.ndarray] = {}
        difficult: dict[int, np.ndarray] = {}
        matched: dict[int, np.ndarray] = {}
        total_ground_truth = 0
        for image_index, record in enumerate(targets):
            labels = _array(record["labels"]).astype(np.int64, copy=False)
            boxes = _array(record["boxes"]).astype(np.float64, copy=False)[labels == label]
            flags = _array(record.get("difficult", np.zeros(len(labels), dtype=bool))).astype(bool, copy=False)
            flags = flags.reshape(-1)[labels == label]
            ground_truth[image_index] = boxes
            difficult[image_index] = flags
            matched[image_index] = np.zeros(len(boxes), dtype=bool)
            total_ground_truth += int(np.count_nonzero(~flags))

        if total_ground_truth == 0:
            continue

        ranked: list[tuple[float, int, np.ndarray]] = []
        for image_index, record in enumerate(predictions):
            labels = _array(record["labels"]).astype(np.int64, copy=False)
            boxes = _array(record["boxes"]).astype(np.float64, copy=False)
            scores = _array(record["scores"]).astype(np.float64, copy=False).reshape(-1)
            for box, score in zip(boxes[labels == label], scores[labels == label], strict=True):
                ranked.append((float(score), image_index, box))
        ranked.sort(key=lambda item: item[0], reverse=True)

        true_positive = np.zeros(len(ranked), dtype=np.float64)
        false_positive = np.zeros(len(ranked), dtype=np.float64)
        for index, (_score, image_index, box) in enumerate(ranked):
            candidates = ground_truth[image_index]
            if candidates.size == 0:
                false_positive[index] = 1.0
                continue
            overlaps = _box_iou(box, candidates)
            best = int(np.argmax(overlaps))
            if overlaps[best] >= iou_threshold and difficult[image_index][best]:
                continue
            if overlaps[best] >= iou_threshold and not matched[image_index][best]:
                true_positive[index] = 1.0
                matched[image_index][best] = True
            else:
                false_positive[index] = 1.0

        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(false_positive)
        recall = cumulative_tp / total_ground_truth
        precision = np.divide(
            cumulative_tp,
            cumulative_tp + cumulative_fp,
            out=np.zeros_like(cumulative_tp),
            where=(cumulative_tp + cumulative_fp) > 0,
        )
        if ap_mode == "voc07_11point":
            average_precisions.append(float(np.mean([
                float(np.max(precision[recall >= level])) if np.any(recall >= level) else 0.0
                for level in np.linspace(0.0, 1.0, 11)
            ])))
        else:
            recall = np.concatenate(([0.0], recall, [1.0]))
            precision = np.concatenate(([0.0], precision, [0.0]))
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            changes = np.flatnonzero(recall[1:] != recall[:-1]) + 1
            average_precisions.append(
                float(np.sum((recall[changes] - recall[changes - 1]) * precision[changes]))
            )
    if not average_precisions:
        raise ValueError("Detection mAP requires at least one non-difficult ground-truth object.")
    return float(np.mean(average_precisions))


def detection_map50(targets: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> float:
    return detection_map(targets, predictions, iou_threshold=0.5)


__all__ = [
    "classification_accuracy",
    "macro_f1",
    "segmentation_confusion_matrix",
    "mean_iou",
    "mean_dice",
    "detection_map",
    "detection_map50",
]
