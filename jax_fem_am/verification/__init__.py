"""Numerical verification utilities for paper-facing quantities.

Includes forward measurement operators matching simulated fields to
experimental resolution (XRD gauge prediction).
"""

from .mesh_quality import audit_tet_mesh
from .weighted import weighted_mean, weighted_quantile
from .xrd import (
    GaugeWeights,
    TetBoxIntersection,
    compute_gauge_weights,
    gauge_volume_average,
    predict_gauge_microstrain,
    project_normal_microstrain,
    project_normal_strain,
    tetra_box_intersection,
)

__all__ = [
    "GaugeWeights",
    "TetBoxIntersection",
    "audit_tet_mesh",
    "compute_gauge_weights",
    "gauge_volume_average",
    "predict_gauge_microstrain",
    "project_normal_microstrain",
    "project_normal_strain",
    "tetra_box_intersection",
    "weighted_mean",
    "weighted_quantile",
]
