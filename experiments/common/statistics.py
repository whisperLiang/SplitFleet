"""Small-sample paired inference used by the confirmatory experiment report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairedEffect:
    paired_n: int
    mean_difference: float
    median_difference: float
    relative_difference: float | None
    confidence_interval_95: tuple[float, float]
    cohens_dz: float | None
    sign_flip_p_value: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def paired_effect(
    left: Iterable[float],
    right: Iterable[float],
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 2026,
) -> PairedEffect:
    """Estimate a paired mean difference without a large-sample t assumption."""

    left_values = np.asarray(list(left), dtype=np.float64)
    right_values = np.asarray(list(right), dtype=np.float64)
    if left_values.ndim != 1 or left_values.shape != right_values.shape or left_values.size < 2:
        raise ValueError("Paired effects require aligned one-dimensional samples with n >= 2.")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("Paired effects do not accept NaN or infinite observations.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    differences = left_values - right_values
    mean = float(np.mean(differences))
    denominator = float(np.mean(np.abs(right_values)))
    relative = mean / denominator if denominator > 0 else None
    std = float(np.std(differences, ddof=1))
    dz = mean / std if std > 0 else None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(bootstrap_samples, differences.size))
    bootstrap = differences[indices].mean(axis=1)
    interval = tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))
    return PairedEffect(
        paired_n=int(differences.size),
        mean_difference=mean,
        median_difference=float(np.median(differences)),
        relative_difference=relative,
        confidence_interval_95=(interval[0], interval[1]),
        cohens_dz=dz,
        sign_flip_p_value=_sign_flip_p_value(differences, seed=seed),
    )


def _sign_flip_p_value(differences: np.ndarray, *, seed: int) -> float:
    """Two-sided paired randomization test under exchangeability of signs."""

    observed = abs(float(np.mean(differences)))
    n = int(differences.size)
    if n <= 20:
        statistics = np.fromiter(
            (
                abs(float(np.mean(differences * np.asarray(signs, dtype=np.float64))))
                for signs in product((-1.0, 1.0), repeat=n)
            ),
            dtype=np.float64,
            count=2**n,
        )
        return float(np.count_nonzero(statistics >= observed - 1e-15) / len(statistics))
    else:
        rng = np.random.default_rng(seed + 1)
        signs = rng.choice((-1.0, 1.0), size=(100_000, n))
        statistics = np.abs(np.mean(signs * differences, axis=1))
    # The +1 correction gives only the Monte Carlo path a valid non-zero p-value.
    return float((np.count_nonzero(statistics >= observed - 1e-15) + 1) / (len(statistics) + 1))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in original order."""

    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1].")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(values[index]) * (count - rank)))
        adjusted[index] = running
    return adjusted.tolist()


__all__ = ["PairedEffect", "paired_effect", "holm_adjust"]
