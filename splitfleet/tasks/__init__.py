"""Common task contracts for all SplitFleet model backends."""

from splitfleet.tasks.base import ModelInputs, TaskAdapter, TaskBatch, infer_num_examples, prepare_task_batch
from splitfleet.tasks.adapters import (
    DetectionTask, ImageClassificationTask, InstanceSegmentationTask,
    SemanticSegmentationTask, TextClassificationTask,
)
from splitfleet.tasks.metrics import (
    classification_accuracy, detection_map, detection_map50, macro_f1,
    mean_dice, mean_iou, segmentation_confusion_matrix,
)
from splitfleet.tasks.registry import MetricSpec, TASK_SPECS, TaskFamily, TaskRegistry, TaskSpec

__all__ = [
    "ModelInputs", "TaskAdapter", "TaskBatch", "infer_num_examples", "prepare_task_batch",
    "ImageClassificationTask", "TextClassificationTask", "DetectionTask",
    "SemanticSegmentationTask", "InstanceSegmentationTask",
    "MetricSpec", "TaskFamily", "TaskSpec", "TaskRegistry", "TASK_SPECS",
    "classification_accuracy", "macro_f1", "segmentation_confusion_matrix",
    "mean_iou", "mean_dice", "detection_map", "detection_map50",
]
