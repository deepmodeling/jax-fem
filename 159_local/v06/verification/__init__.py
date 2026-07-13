"""Numerical verification utilities for paper-facing quantities."""

from .mesh_quality import audit_tet_mesh
from .weighted import weighted_mean, weighted_quantile

__all__ = ["audit_tet_mesh", "weighted_mean", "weighted_quantile"]

