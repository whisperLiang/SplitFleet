"""The canonical fixture actually checks all six numerical contract properties."""

from experiments.analysis.canonical_backend_correctness import CHECKS, validate_backend


def test_canonical_torch_contract_exercises_every_enumerated_training_boundary():
    result = validate_backend("torch")
    assert result["status"] == "passed", result
    assert result["counts"]["passed"] > 1
    assert sum(result["counts"].values()) == len(result["nodes"])
    for row in result["nodes"]:
        if row["status"] == "passed":
            assert set(row["checks"]) == set(CHECKS)
            assert row["split_id"] == row["boundary"]
        else:
            assert row["status"] == "unsupported" and row["reason"]


def test_boolean_frontiers_keep_numerical_failures_and_valid_error_statistics():
    import numpy as np
    import pytest
    from experiments.analysis.canonical_backend_correctness import _compare

    assert _compare(np.array([True, False]), np.array([True, False]), rtol=0, atol=0) == 0
    with pytest.raises(AssertionError):
        _compare(np.array([True, False]), np.array([False, False]), rtol=0, atol=0)
