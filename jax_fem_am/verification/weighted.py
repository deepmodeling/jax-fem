"""Explicit volume-weighted statistics for mesh fields."""

from __future__ import annotations

import numpy as np


def _validated(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be finite one-dimensional arrays")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("values and weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("total weight must be positive")
    return values, weights


def weighted_mean(values, weights):
    values, weights = _validated(values, weights)
    return float(np.dot(values, weights) / weights.sum())


def weighted_quantile(values, weights, quantile):
    """Return the left-continuous inverse of the weighted empirical CDF."""
    values, weights = _validated(values, weights)
    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    target = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])

