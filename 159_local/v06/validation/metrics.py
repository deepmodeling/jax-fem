"""Metrics that keep numerical error and measurement uncertainty explicit."""

from __future__ import annotations

import numpy as np


def _paired(observed, predicted):
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.ndim != 1 or predicted.shape != observed.shape or not len(observed):
        raise ValueError("observed and predicted must be nonempty equal-length vectors")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("observed and predicted values must be finite")
    return observed, predicted


def field_error_metrics(observed, predicted, uncertainty=None, *, fitted_parameters=0):
    """Return bias/MAE/RMSE plus uncertainty-aware comparison statistics."""
    if (
        isinstance(fitted_parameters, (bool, np.bool_))
        or not isinstance(fitted_parameters, (int, np.integer))
        or int(fitted_parameters) < 0
    ):
        raise ValueError("fitted_parameters must be a nonnegative integer")
    observed, predicted = _paired(observed, predicted)
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    dynamic_range = float(observed.max() - observed.min())
    metrics = {
        "count": int(len(observed)),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": rmse,
        "nrmse_range": rmse / dynamic_range if dynamic_range > 0.0 else None,
    }
    if uncertainty is None:
        metrics["reduced_chi_square"] = None
        metrics["coverage_2sigma"] = None
        return metrics

    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if uncertainty.shape != observed.shape or not np.all(np.isfinite(uncertainty)):
        raise ValueError("uncertainty must be a finite vector matching observations")
    if np.any(uncertainty <= 0.0):
        raise ValueError("uncertainty values must be positive")
    dof = len(observed) - int(fitted_parameters)
    if dof <= 0:
        raise ValueError("degrees of freedom must be positive")
    normalized = residual / uncertainty
    metrics["reduced_chi_square"] = float(np.sum(normalized**2) / dof)
    metrics["coverage_2sigma"] = float(np.mean(np.abs(normalized) <= 2.0))
    return metrics


def evaluate_anchors(anchors, predictions):
    """Compare scalar predictions with explicitly reported paper anchors."""
    report = {}
    for anchor in anchors:
        anchor_id = anchor["id"]
        if anchor_id not in predictions:
            report[anchor_id] = {"status": "missing_prediction"}
            continue
        predicted = float(predictions[anchor_id])
        if not np.isfinite(predicted):
            raise ValueError(f"prediction for {anchor_id} must be finite")
        if anchor["kind"] == "target":
            observed = float(anchor["value"])
            tolerance = float(anchor["screening_tolerance_microstrain"])
            if not np.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(
                    f"screening tolerance for {anchor_id} must be positive"
                )
            absolute_error = abs(predicted - observed)
            report[anchor_id] = {
                "status": "evaluated",
                "predicted": predicted,
                "observed": observed,
                "absolute_error": absolute_error,
                "screening_tolerance": tolerance,
                "within_screening_band": absolute_error <= tolerance,
            }
        elif anchor["kind"] == "range":
            lower, upper = (float(value) for value in anchor["range"])
            report[anchor_id] = {
                "status": "evaluated",
                "predicted": predicted,
                "lower": lower,
                "upper": upper,
                "within_range": lower <= predicted <= upper,
                "distance_to_range": max(lower - predicted, predicted - upper, 0.0),
            }
        else:
            raise ValueError(f"unsupported anchor kind: {anchor['kind']!r}")
    return report
