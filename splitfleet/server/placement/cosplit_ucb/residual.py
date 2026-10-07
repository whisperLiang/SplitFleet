"""Measured local residuals around the shared linear cost model."""

from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np

from .bandit import DiscountedLinUCB, LinearPrediction


class ResidualLinUCB(DiscountedLinUCB):
    """Correct local underfitting using discounted, deployment-only observations.

    Linear models still share information across contexts. Nearby measured costs
    correct their residuals, while confidence radii are never reduced. No model
    names, boundaries or historical benchmark timings enter this estimator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._anchors = OrderedDict()
        self._anchor_round = -1

    def advance_round(self, round_id):
        super().advance_round(round_id)
        current = int(round_id)
        if self._anchor_round >= 0 and current > self._anchor_round:
            discount = self.discount_gamma ** (current - self._anchor_round)
            for key, row in list(self._anchors.items()):
                row[1] *= discount
                row[2] *= discount
                if row[2] < 1e-12:
                    del self._anchors[key]
        self._anchor_round = current

    def update(self, context, target, *, round_id, sample_weight=1.0):
        super().update(context, target, round_id=round_id, sample_weight=sample_weight)
        x = self._vector(context).copy()
        key = x.tobytes()
        if key not in self._anchors:
            self._anchors[key] = [x, 0.0, 0.0]
        row = self._anchors[key]
        row[1] += float(target) * float(sample_weight)
        row[2] += float(sample_weight)
        self._anchors.move_to_end(key)
        if len(self._anchors) > 256:
            self._anchors.popitem(last=False)

    def predict(self, context):
        x = self._vector(context)
        base = super().predict(x)
        if not self._anchors:
            return base
        neighbors = sorted(
            ((float(np.dot(x - row[0], x - row[0])), row) for row in self._anchors.values()),
            key=lambda pair: pair[0],
        )[:3]
        if neighbors[0][0] <= 1e-20:
            row = neighbors[0][1]
            return LinearPrediction(row[1] / row[2], base.uncertainty)
        weights = np.asarray([1.0 / distance for distance, _ in neighbors])
        weights /= weights.sum()
        residuals = np.asarray([
            row[1] / row[2] - super(ResidualLinUCB, self).predict(row[0]).mean
            for _, row in neighbors
        ])
        correction = float(weights @ residuals)
        spread = float(np.sqrt(weights @ ((residuals - correction) ** 2)))
        return LinearPrediction(max(base.mean + correction, 0.0), base.uncertainty + spread)

    def state_dict(self):
        state = super().state_dict()
        state.update(residual_anchor_round=self._anchor_round,
                     residual_anchors=[{"context": row[0].tolist(), "sum_ms": row[1], "weight": row[2]}
                                       for row in self._anchors.values()])
        return state

    def load_state_dict(self, state):
        anchors = OrderedDict()
        for row in state.get("residual_anchors", []):
            x = self._vector(row["context"])
            total, weight = float(row["sum_ms"]), float(row["weight"])
            if not math.isfinite(total) or total < 0 or not math.isfinite(weight) or weight <= 0:
                raise ValueError("invalid residual calibration statistics")
            anchors[x.tobytes()] = [x.copy(), total, weight]
        if len(anchors) > 256:
            raise ValueError("too many residual calibration anchors")
        super().load_state_dict(state)
        self._anchors = anchors
        self._anchor_round = int(state.get("residual_anchor_round", self.last_discount_round))


__all__ = ["ResidualLinUCB"]
