"""Reporting controls distinguish convergence, time to target and runtime speed."""

import numpy as np
import pytest

from experiments.resnet50_quality.convergence import convergence_comparison, convergence_metrics
from experiments.resnet50_quality.protocol import SEEDS

SETTINGS = {"target_val_top1": .7, "sustained_rounds": 5}


def curve(accuracies, losses=None):
    return [{"round": index + 1, "val_top1": accuracy, "elapsed_wall_sec": (index + 1) * 2.,
             "val_ce": losses[index] if losses is not None else 2. - index / 20,
             "train_online_ce": 2. - index / 20} for index, accuracy in enumerate(accuracies)]


def test_time_to_target_requires_five_consecutive_observed_validations():
    rows = curve([.3, .71, .72, .6, .71, .72, .73, .74, .75, .76])
    result = convergence_metrics(rows, SETTINGS, 22.)
    assert result["converged"]
    assert result["target_confirmation_round"] == 9
    assert result["time_to_target_seconds"] == 18.


@pytest.mark.parametrize("accuracies,losses", [
    ([.2] * 10, None),
    ([.8] * 5 + [.5] * 5, None),
    ([.8] * 10, [2.] * 10),
])
def test_low_accuracy_regression_or_flat_loss_cannot_count_as_convergence(accuracies, losses):
    result = convergence_metrics(curve(accuracies, losses), SETTINGS, 22.)
    assert not result["converged"]


def test_unreached_target_is_missing_and_cannot_generate_time_speedup():
    result = convergence_metrics(curve([.6] * 10), SETTINGS, 22.)
    assert result["time_to_target_seconds"] is None and result["target_confirmation_round"] is None
    outcomes = {(seed, method): {"convergence": result} for seed in SEEDS for method in ("fedavg", "splitfleet")}
    comparison = convergence_comparison(outcomes, SEEDS)
    assert not comparison["both_converged"]
    assert comparison["time_to_target"]["geometric_mean_speedup"] is None


@pytest.mark.parametrize("field,value", [("val_top1", np.nan), ("val_top1", 1.1), ("val_ce", -1.),
                                           ("elapsed_wall_sec", 0.), ("elapsed_wall_sec", 23.)])
def test_nonfinite_invalid_or_unmeasured_curves_are_rejected(field, value):
    rows = curve([.8] * 10)
    rows[5][field] = value
    with pytest.raises(ValueError, match="finite"):
        convergence_metrics(rows, SETTINGS, 22.)
