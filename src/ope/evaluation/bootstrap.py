"""Percentile bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    y, p, metric: Callable, n: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Point estimate and (1 - alpha) percentile CI, resampling rows with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, p)), float(lo), float(hi)
