"""TET4/HEX8 quality metrics used to gate stress reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TET_EDGES = np.asarray(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    dtype=np.int64,
)
_HEX_EDGES = np.asarray(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int64,
)
_HEX_REFERENCE_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
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

    edge_vectors = tet[:, _TET_EDGES[:, 1]] - tet[:, _TET_EDGES[:, 0]]
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


def _validate_hex_mesh(points, cells):
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (num_points, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("point coordinates must be finite")
    if cells.ndim != 2 or cells.shape[1] != 8:
        raise ValueError("cells must have shape (num_cells, 8)")
    if cells.size and (cells.min() < 0 or cells.max() >= len(points)):
        raise ValueError("cells contain out-of-range point indices")
    return points, cells


def _hex_shape_gradients():
    coordinate = 1.0 / np.sqrt(3.0)
    quadrature_points = coordinate * _HEX_REFERENCE_SIGNS
    gradients = np.empty((8, 8, 3), dtype=np.float64)
    for quad_id, (xi, eta, zeta) in enumerate(quadrature_points):
        sx = _HEX_REFERENCE_SIGNS[:, 0]
        sy = _HEX_REFERENCE_SIGNS[:, 1]
        sz = _HEX_REFERENCE_SIGNS[:, 2]
        gradients[quad_id, :, 0] = (
            0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
        )
        gradients[quad_id, :, 1] = (
            0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
        )
        gradients[quad_id, :, 2] = (
            0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
        )
    return gradients


_HEX_SHAPE_GRADIENTS = _hex_shape_gradients()


def audit_hex_mesh(points, cells, *, quality_threshold=0.05):
    """Compute HEX8 volume, scaled-Jacobian and edge diagnostics."""

    points, cells = _validate_hex_mesh(points, cells)
    hexahedra = points[cells]
    jacobian = np.einsum(
        "cna,qnb->cqab",
        hexahedra,
        _HEX_SHAPE_GRADIENTS,
    )
    determinant = np.linalg.det(jacobian)
    signed_volume = np.sum(determinant, axis=1)
    volume = np.abs(signed_volume)
    column_norms = np.linalg.norm(jacobian, axis=-2)
    denominator = np.prod(column_norms, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_jacobian = np.where(
            denominator > 0.0,
            determinant / denominator,
            0.0,
        )
        frobenius_ratio = np.where(
            determinant > 0.0,
            3.0
            * determinant ** (2.0 / 3.0)
            / np.sum(column_norms * column_norms, axis=-1),
            0.0,
        )
    mean_ratio = np.maximum(
        np.min(np.minimum(scaled_jacobian, frobenius_ratio), axis=1),
        0.0,
    )

    edge_vectors = (
        hexahedra[:, _HEX_EDGES[:, 1]]
        - hexahedra[:, _HEX_EDGES[:, 0]]
    )
    edge_lengths = np.linalg.norm(edge_vectors, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_ratio = np.where(
            edge_lengths.min(axis=1) > 0.0,
            edge_lengths.max(axis=1) / edge_lengths.min(axis=1),
            np.inf,
        )

    span = float(np.max(np.ptp(points, axis=0))) if len(points) else 0.0
    jacobian_tolerance = 100.0 * np.finfo(np.float64).eps * max(
        span**3,
        np.finfo(np.float64).tiny,
    )
    inverted = np.any(determinant < -jacobian_tolerance, axis=1)
    degenerate = np.any(
        np.abs(determinant) <= jacobian_tolerance,
        axis=1,
    ) | (volume <= jacobian_tolerance)
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


def audit_solid_mesh(points, cells, *, quality_threshold=0.05):
    """Dispatch quality auditing by supported first-order solid topology."""

    cells = np.asarray(cells)
    if cells.ndim != 2:
        raise ValueError("cells must be a two-dimensional array")
    if cells.shape[1] == 4:
        return audit_tet_mesh(
            points,
            cells,
            quality_threshold=quality_threshold,
        )
    if cells.shape[1] == 8:
        return audit_hex_mesh(
            points,
            cells,
            quality_threshold=quality_threshold,
        )
    raise ValueError("only TET4 and HEX8 solid cells are supported")
