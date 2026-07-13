"""TET4 quality metrics used to gate stress reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EDGES = np.asarray(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    dtype=np.int64,
)


@dataclass(frozen=True)
class TetMeshQualityReport:
    signed_volume: np.ndarray
    volume: np.ndarray
    mean_ratio: np.ndarray
    edge_ratio: np.ndarray
    quality_threshold: float
    inverted_count: int
    degenerate_count: int
    below_threshold_count: int


def _validate_tet_mesh(points, cells):
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (num_points, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("point coordinates must be finite")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("cells must have shape (num_cells, 4)")
    if cells.size and (cells.min() < 0 or cells.max() >= len(points)):
        raise ValueError("cells contain out-of-range point indices")
    return points, cells


def audit_tet_mesh(points, cells, *, quality_threshold=0.05):
    """Compute normalized TET4 quality, volume and orientation diagnostics.

    The mean-ratio metric is 6*sqrt(2)*V/l_rms^3 and equals one for a
    regular tetrahedron. Volumes retain SI units implied by the input mesh.
    """
    points, cells = _validate_tet_mesh(points, cells)
    tet = points[cells]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        np.cross(tet[:, 1] - tet[:, 0], tet[:, 2] - tet[:, 0]),
        tet[:, 3] - tet[:, 0],
    )
    signed_volume = signed_six_volume / 6.0
    volume = np.abs(signed_volume)

    edge_vectors = tet[:, _EDGES[:, 1]] - tet[:, _EDGES[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=-1)
    rms_edge = np.sqrt(np.mean(edge_lengths * edge_lengths, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_ratio = np.where(
            rms_edge > 0.0,
            6.0 * np.sqrt(2.0) * volume / rms_edge**3,
            0.0,
        )
        edge_ratio = np.where(
            edge_lengths.min(axis=1) > 0.0,
            edge_lengths.max(axis=1) / edge_lengths.min(axis=1),
            np.inf,
        )

    span = float(np.max(np.ptp(points, axis=0))) if len(points) else 0.0
    volume_tolerance = 100.0 * np.finfo(np.float64).eps * max(
        span**3, np.finfo(np.float64).tiny
    )
    degenerate = volume <= volume_tolerance
    inverted = signed_volume < -volume_tolerance
    below_threshold = mean_ratio < float(quality_threshold)
    return TetMeshQualityReport(
        signed_volume=signed_volume,
        volume=volume,
        mean_ratio=mean_ratio,
        edge_ratio=edge_ratio,
        quality_threshold=float(quality_threshold),
        inverted_count=int(np.count_nonzero(inverted)),
        degenerate_count=int(np.count_nonzero(degenerate)),
        below_threshold_count=int(np.count_nonzero(below_threshold)),
    )
