"""Measured and statistical metrics for RA-SplitFed."""

from __future__ import annotations

import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

import torch

T = TypeVar("T")


def timed_call(fn: Callable[[], T], device: torch.device | str = "cpu") -> tuple[T, float]:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        value = fn()
        end.record()
        torch.cuda.synchronize(device)
        return value, float(start.elapsed_time(end))
    started = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - started) / 1_000_000.0


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> dict[str, object]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Targets and predictions must be non-empty and aligned.")
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions):
        matrix[int(target)][int(prediction)] += 1
    recalls: list[float | None] = []
    f1_values: list[float] = []
    for label in range(num_classes):
        tp = matrix[label][label]
        fn = sum(matrix[label]) - tp
        fp = sum(row[label] for row in matrix) - tp
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        recalls.append(recall)
        if recall is not None and precision is not None and recall + precision:
            f1_values.append(2.0 * recall * precision / (recall + precision))
        else:
            f1_values.append(0.0)
    correct = sum(matrix[index][index] for index in range(num_classes))
    return {
        "test_accuracy": correct / len(targets),
        "macro_f1": statistics.fmean(f1_values),
        "per_class_recall": recalls,
    }

def jain_fairness_index(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    if not data or sum(value * value for value in data) == 0:
        return None
    return sum(data) ** 2 / (len(data) * sum(value * value for value in data))


def descriptive_stats(values: Iterable[float]) -> dict[str, float | int | list[float] | None]:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
            "confidence_interval_95": None,
            "number_of_valid_runs": 0,
        }
    mean = statistics.fmean(data)
    std = statistics.stdev(data) if len(data) >= 2 else 0.0
    p95_index = max(0, math.ceil(0.95 * len(data)) - 1)
    ci = None
    if len(data) >= 3:
        half = 1.96 * std / math.sqrt(len(data))
        ci = [mean - half, mean + half]
    return {
        "mean": mean,
        "std": std,
        "median": statistics.median(data),
        "p95": data[p95_index],
        "confidence_interval_95": ci,
        "number_of_valid_runs": len(data),
    }
