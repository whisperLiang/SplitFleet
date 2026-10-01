from __future__ import annotations

import pytest

from experiments.common.statistics import holm_adjust, paired_effect


def test_paired_effect_is_deterministic_and_preserves_pairing() -> None:
    effect = paired_effect([8.0, 9.0, 10.0, 11.0, 12.0], [10.0] * 5, seed=7)
    repeated = paired_effect([8.0, 9.0, 10.0, 11.0, 12.0], [10.0] * 5, seed=7)

    assert effect == repeated
    assert effect.paired_n == 5
    assert effect.mean_difference == 0.0
    assert effect.median_difference == 0.0
    assert effect.relative_difference == 0.0
    assert effect.confidence_interval_95[0] < 0 < effect.confidence_interval_95[1]
    assert effect.sign_flip_p_value == 1.0


def test_exact_sign_flip_detects_consistent_direction_but_not_with_five_pairs_at_point05() -> None:
    effect = paired_effect([1, 1, 1, 1, 1], [2, 2, 2, 2, 2])

    assert effect.mean_difference == -1.0
    assert effect.confidence_interval_95 == (-1.0, -1.0)
    assert effect.sign_flip_p_value == pytest.approx(2 / 32)
    assert effect.cohens_dz is None


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
