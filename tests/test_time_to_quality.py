import math

import pytest

from experiments.analysis.time_to_quality import time_to_quality, summarize_paired_ttq


def evaluations(*values):
    return [dict(round_id=i, elapsed_sec=elapsed, metrics={"map50": quality})
            for i, (elapsed, quality) in enumerate(values, 1)]


def test_ttq_uses_first_recorded_crossing_including_evaluation_time():
    result = time_to_quality(evaluations((50, .8), (85, .9), (120, .82), (160, .95)),
                             metric="map50", threshold=.9)
    assert result["status"] == "reached"
    assert result["time_sec"] == 85
    assert result["round_id"] == 2


def test_unreached_threshold_is_censored_at_last_evaluation():
    result = time_to_quality(evaluations((40, .7), (90, .8)), metric="map50", threshold=.9)
    assert result["status"] == "not_reached"
    assert result["time_sec"] is None
    assert result["censor_time_sec"] == 90


def test_legacy_evaluations_do_not_invent_monotonic_timestamps():
    result = time_to_quality([dict(round_id=1, metrics={"map50": .95})], metric="map50", threshold=.9)
    assert result["status"] == "unavailable"
    assert result["time_sec"] is None


@pytest.mark.parametrize("rows", [evaluations((10, .8), (5, .9)), evaluations((10, .8), (20, math.nan))])
def test_invalid_checkpoints_are_rejected_even_after_an_earlier_crossing(rows):
    with pytest.raises(ValueError):
        time_to_quality(rows, metric="map50", threshold=.7)


def test_paired_ttq_summary_preserves_censored_and_excluded_population():
    identity = dict(stage="nano", scheme="splitfleet", metric="map50", threshold=.9,
                    valid=True, complete_paired_seed=True)
    rows = [{**identity, "seed": 1, "status": "reached", "time_sec": 10},
            {**identity, "seed": 2, "status": "not_reached", "time_sec": None},
            {**identity, "seed": 3, "status": "reached", "time_sec": 1, "complete_paired_seed": False}]
    result = summarize_paired_ttq(rows)[0]
    assert result["mean_ttq_sec"] is None
    assert result["complete_paired_n"] == 2 and result["reached_n"] == 1 and result["censored_n"] == 1
    rows[1].update(status="reached", time_sec=20)
    assert summarize_paired_ttq(rows)[0]["mean_ttq_sec"] == 15


def test_tasks_with_the_same_quality_metric_keep_separate_paired_populations():
    identity = dict(stage="joint", scheme="splitfleet", metric="accuracy", threshold=.8,
                    valid=True, complete_paired_seed=True, seed=1, status="reached")
    rows = [{**identity, "task": "image_classification", "time_sec": 10},
            {**identity, "task": "text_classification", "time_sec": 20}]
    reports = {row["task"]: row for row in summarize_paired_ttq(rows)}
    assert set(reports) == {"image_classification", "text_classification"}
    assert reports["image_classification"]["mean_ttq_sec"] == 10
    assert reports["text_classification"]["mean_ttq_sec"] == 20
    assert all(row["complete_paired_n"] == 1 for row in reports.values())
