"""Discounted linear upper-confidence models used by CoSplit-UCB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LinearPrediction:
    """Mean and confidence radius in the caller's target unit."""

    mean: float
    uncertainty: float

    @property
    def lcb(self) -> float:
        return max(self.mean - self.uncertainty, 0.0)

    @property
    def ucb(self) -> float:
        return self.mean + self.uncertainty


class DiscountedLinUCB:
    """Non-stationary LinUCB regressor for non-negative cost minimization."""

    def __init__(
        self,
        dimension: int,
        *,
        ridge_lambda: float = 1.0,
        discount_gamma: float = 0.98,
        alpha: float = 1.0,
        target_scale: float = 1000.0,
        feature_schema_version: str,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if ridge_lambda <= 0 or not 0 < discount_gamma <= 1:
            raise ValueError("invalid ridge/discount parameters")
        if alpha < 0 or target_scale <= 0:
            raise ValueError("invalid alpha/target scale")
        if not feature_schema_version:
            raise ValueError("feature_schema_version must not be empty")
        self.dimension = int(dimension)
        self.ridge_lambda = float(ridge_lambda)
        self.discount_gamma = float(discount_gamma)
        self.alpha = float(alpha)
        self.target_scale = float(target_scale)
        self.feature_schema_version = str(feature_schema_version)
        self.A = self.ridge_lambda * np.eye(self.dimension, dtype=np.float64)
        self.b = np.zeros(self.dimension, dtype=np.float64)
        self.num_updates = 0
        self.last_update_round = -1
        self.last_discount_round = -1

    def _vector(self, context: np.ndarray) -> np.ndarray:
        value = np.asarray(context, dtype=np.float64).reshape(-1)
        if value.shape != (self.dimension,):
            raise ValueError(f"expected context shape {(self.dimension,)}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("context contains non-finite values")
        return value

    def _solve(self, rhs: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(self.A, rhs)
        except np.linalg.LinAlgError:
            # A discounted covariance should remain SPD. Jitter handles small
            # numerical drift without replacing the estimator with an inverse.
            jitter = max(self.ridge_lambda, 1.0) * 1e-10
            return np.linalg.solve(self.A + jitter * np.eye(self.dimension), rhs)

    def predict(self, context: np.ndarray) -> LinearPrediction:
        """Predict non-negative mean cost and confidence radius."""

        x = self._vector(context)
        theta = self._solve(self.b)
        mean_scaled = float(x @ theta)
        variance = max(float(x @ self._solve(x)), 0.0)
        return LinearPrediction(
            mean=max(mean_scaled * self.target_scale, 0.0),
            uncertainty=max(self.alpha * np.sqrt(variance) * self.target_scale, 0.0),
        )

    def advance_round(self, round_id: int) -> None:
        """Age previous observations before making this round's predictions."""

        current = int(round_id)
        if current < self.last_discount_round:
            raise ValueError("bandit observations must arrive in round order")
        if self.last_discount_round < 0:
            self.last_discount_round = current
            return
        elapsed_rounds = current - self.last_discount_round
        if not elapsed_rounds:
            return
        gamma = self.discount_gamma ** elapsed_rounds
        identity = np.eye(self.dimension, dtype=np.float64)
        self.A = gamma * self.A + (1.0 - gamma) * self.ridge_lambda * identity
        self.b = gamma * self.b
        self.last_discount_round = current

    def update(self, context: np.ndarray, target: float, *, round_id: int) -> None:
        """Apply one observation after discounting any elapsed rounds."""

        x = self._vector(context)
        y = float(target)
        if not np.isfinite(y) or y < 0:
            raise ValueError("cost target must be finite and non-negative")
        self.advance_round(round_id)
        self.A = self.A + np.outer(x, x)
        self.b = self.b + x * (y / self.target_scale)
        self.num_updates += 1
        self.last_update_round = int(round_id)

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-compatible sufficient statistics."""

        return {
            "dimension": self.dimension,
            "ridge_lambda": self.ridge_lambda,
            "discount_gamma": self.discount_gamma,
            "alpha": self.alpha,
            "target_scale": self.target_scale,
            "feature_schema_version": self.feature_schema_version,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "num_updates": self.num_updates,
            "last_update_round": self.last_update_round,
            "last_discount_round": self.last_discount_round,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state only when dimensions and feature schema match."""

        if int(state["dimension"]) != self.dimension:
            raise ValueError("bandit state dimension mismatch")
        if str(state["feature_schema_version"]) != self.feature_schema_version:
            raise ValueError("bandit feature schema mismatch")
        for name in ("ridge_lambda", "discount_gamma", "alpha", "target_scale"):
            if float(state[name]) != float(getattr(self, name)):
                raise ValueError(f"bandit {name} mismatch")
        A = np.asarray(state["A"], dtype=np.float64)
        b = np.asarray(state["b"], dtype=np.float64)
        if A.shape != (self.dimension, self.dimension) or b.shape != (self.dimension,):
            raise ValueError("invalid bandit sufficient-statistic shapes")
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError("bandit state contains non-finite values")
        if not np.allclose(A, A.T, rtol=1e-10, atol=1e-10):
            raise ValueError("bandit covariance must be symmetric")
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError as exc:
            raise ValueError("bandit covariance must be positive definite") from exc
        self.A = A.copy()
        self.b = b.copy()
        self.num_updates = int(state.get("num_updates", 0))
        self.last_update_round = int(state.get("last_update_round", -1))
        self.last_discount_round = int(state.get("last_discount_round", self.last_update_round))
        if self.last_discount_round < self.last_update_round:
            raise ValueError("bandit discount round cannot precede last observation")


__all__ = ["DiscountedLinUCB", "LinearPrediction"]
