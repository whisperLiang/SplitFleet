"""Common task contracts for all SplitFleet model backends."""

from splitfleet.tasks.base import ModelInputs, TaskAdapter, TaskBatch, infer_num_examples, prepare_task_batch
from splitfleet.tasks.adapters import (
    DetectionTask, ImageClassificationTask, InstanceSegmentationTask,
    SemanticSegmentationTask, TextClassificationTask,
)

__all__ = [
    "ModelInputs", "TaskAdapter", "TaskBatch", "infer_num_examples", "prepare_task_batch",
    "ImageClassificationTask", "TextClassificationTask", "DetectionTask",
    "SemanticSegmentationTask", "InstanceSegmentationTask",
]
