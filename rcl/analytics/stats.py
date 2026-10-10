"""Confidence intervals across seeds (DEFINITIONS.md, "Aggregation across seeds").

Default: two-sided 95% Student t-interval, mean +/- t * s / sqrt(n) with ddof = 1.
Percentile bootstrap (10,000 resamples, seeded RNG) for skewed, censored metrics.
"""

from __future__ import annotations

import math

import numpy as np

# t_{0.975, df} for df = 1..30 (3 decimals; error in the interval below 0.05% of s / sqrt(n)).
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip
_Z975 = 1.959964


def t_critical(df: int) -> float:
    """Two-sided 95% critical value of Student's t with ``df`` degrees of freedom.

    Exact table up to 30; beyond that the Cornish-Fisher series (error < 1e-4 for df > 30).
    """
    if df < 1:
        raise ValueError("df must be >= 1")
    if df <= 30:
        return _T975[df - 1]
    z = _Z975
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def t_interval(values: list[float]) -> tuple[float, float]:
    """95% t-interval of the mean; (NaN, NaN) for fewer than two values."""
    n = len(values)
    if n < 2:
        return math.nan, math.nan
    a = np.asarray(values, dtype=np.float64)
    half = t_critical(n - 1) * float(a.std(ddof=1)) / math.sqrt(n)
    mean = float(a.mean())
    return mean - half, mean + half


def bootstrap_interval(
    values: list[float], resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile-bootstrap interval of the mean; (NaN, NaN) for fewer than two values.
    The RNG is seeded, so the same values always give the same interval."""
    n = len(values)
    if n < 2:
        return math.nan, math.nan
    a = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)
