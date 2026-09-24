from __future__ import annotations

import inspect

import numpy as np
import pytest

from splitfleet.server.placement.cosplit_ucb import DiscountedLinUCB


def _model(*, gamma: float = 0.98) -> DiscountedLinUCB:
    return DiscountedLinUCB(
        2,
        ridge_lambda=2.0,
        discount_gamma=gamma,
        alpha=1.0,
        target_scale=1.0,
        feature_schema_version="test-v1",
    )


def test_ridge_initialization_and_update_math() -> None:
    model = _model(gamma=0.5)
    assert np.array_equal(model.A, 2.0 * np.eye(2))
    assert np.array_equal(model.b, np.zeros(2))

    x = np.array([1.0, 2.0])
    model.update(x, 3.0, round_id=4)

    expected_A = 0.5 * (2.0 * np.eye(2)) + np.outer(x, x) + 0.5 * 2.0 * np.eye(2)
    assert np.allclose(model.A, expected_A)
    assert np.allclose(model.b, x * 3.0)
    assert model.num_updates == 1
    assert model.last_update_round == 4


def test_discount_forgets_old_observation_and_repetition_reduces_uncertainty() -> None:
    model = _model(gamma=0.5)
    x = np.array([1.0, 0.0])
    initial = model.predict(x).uncertainty
    model.update(x, 20.0, round_id=1)
    once = model.predict(x).uncertainty
    old_b = model.b.copy()
    model.update(np.array([0.0, 1.0]), 0.0, round_id=2)

    assert once < initial
    assert model.b[0] == pytest.approx(old_b[0] * 0.5)

    repeated = _model()
    before = repeated.predict(x).uncertainty
    for round_id in range(1, 8):
        repeated.update(x, 5.0, round_id=round_id)
    assert repeated.predict(x).uncertainty < before


def test_multiple_clients_in_one_round_are_not_discounted_again() -> None:
    model = _model(gamma=0.5)
    x = np.array([1.0, 0.0])
    model.update(x, 10.0, round_id=1)
    after_first = model.b.copy()
    model.update(x, 20.0, round_id=1)
    assert model.b[0] == pytest.approx(after_first[0] + 20.0)
    assert model.A[0, 0] == pytest.approx(4.0)

    model.update(x, 0.0, round_id=3)
    assert model.b[0] == pytest.approx(30.0 * 0.5**2)
    with pytest.raises(ValueError, match="round order"):
        model.update(x, 0.0, round_id=2)


def test_unobserved_rounds_age_confidence_before_prediction_without_double_discount() -> None:
    model = _model(gamma=0.5)
    x = np.array([1.0, 0.0])
    model.update(x, 10.0, round_id=1)
    before = model.predict(x).uncertainty
    model.advance_round(3)
    assert model.predict(x).uncertainty > before
    assert model.b[0] == pytest.approx(10.0 * 0.5**2)
    model.update(x, 4.0, round_id=3)
    assert model.b[0] == pytest.approx(10.0 * 0.5**2 + 4.0)
    assert model.last_update_round == 3
    assert model.last_discount_round == 3

    restored = _model(gamma=0.5)
    restored.load_state_dict(model.state_dict())
    assert restored.predict(x) == model.predict(x)


def test_implementation_never_calls_explicit_matrix_inverse() -> None:
    source = inspect.getsource(DiscountedLinUCB)
    assert "linalg.inv" not in source
    assert "linalg.solve" in source


def test_state_restore_preserves_prediction() -> None:
    model = _model()
    model.update(np.array([1.0, 0.25]), 17.0, round_id=3)
    restored = _model()
    restored.load_state_dict(model.state_dict())
    assert restored.predict(np.array([1.0, 0.25])) == model.predict(np.array([1.0, 0.25]))


def test_restore_rejects_incompatible_confidence_and_corrupt_covariance() -> None:
    model = _model()
    state = model.state_dict()
    incompatible = DiscountedLinUCB(
        2,
        ridge_lambda=2.0,
        discount_gamma=0.98,
        alpha=2.0,
        target_scale=1.0,
        feature_schema_version="test-v1",
    )
    with pytest.raises(ValueError, match="alpha mismatch"):
        incompatible.load_state_dict(state)

    state["A"] = [[-1.0, 0.0], [0.0, 2.0]]
    with pytest.raises(ValueError, match="positive definite"):
        model.load_state_dict(state)
