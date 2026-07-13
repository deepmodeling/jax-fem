"""Minimal XRD gauge-volume operators for elastic-strain comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TetBoxIntersection:
    volume: float
    centroid: Optional[np.ndarray]
    vertex_count: int


@dataclass(frozen=True)
class GaugeWeights:
    cell_weights: np.ndarray
    nominal_gauge_volume: float
    material_intersection_volume: float
    material_fill_fraction: float
    contributing_cell_count: int
    effective_cell_count: float
    max_cell_weight_fraction: float


def gauge_volume_average(elastic_strain, weights):
    """Average 3x3 elastic-strain tensors over a physical gauge volume."""
    elastic_strain = np.asarray(elastic_strain, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if elastic_strain.ndim != 3 or elastic_strain.shape[1:] != (3, 3):
        raise ValueError("elastic_strain must have shape (num_samples, 3, 3)")
    if weights.shape != (len(elastic_strain),):
        raise ValueError("weights must have one value per strain sample")
    if not np.all(np.isfinite(elastic_strain)) or not np.all(np.isfinite(weights)):
        raise ValueError("strain and weights must be finite")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("weights must be nonnegative with a positive total")
    return np.tensordot(weights, elastic_strain, axes=(0, 0)) / weights.sum()


def project_normal_strain(elastic_strain, direction):
    """Project elastic strain onto the normalized diffraction direction."""
    elastic_strain = np.asarray(elastic_strain, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if elastic_strain.shape != (3, 3) or direction.shape != (3,):
        raise ValueError("expected a (3, 3) strain tensor and a 3-vector")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("measurement direction must have positive finite length")
    unit = direction / norm
    return float(unit @ elastic_strain @ unit)


def project_normal_microstrain(elastic_strain, direction):
    """Project dimensionless elastic strain and return explicit microstrain."""
    return 1.0e6 * project_normal_strain(elastic_strain, direction)


def _rotation_matrix(rotation_gauge_to_specimen):
    if rotation_gauge_to_specimen is None:
        return np.eye(3)
    rotation = np.asarray(rotation_gauge_to_specimen, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("gauge rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10):
        raise ValueError("gauge rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10):
        raise ValueError("gauge rotation must be right handed")
    return rotation


def _polyhedron_volume_centroid(vertices):
    from scipy.spatial import ConvexHull, QhullError

    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) < 4:
        return 0.0, None
    try:
        hull = ConvexHull(vertices)
    except QhullError:
        return 0.0, None
    if not np.isfinite(hull.volume) or hull.volume <= 1.0e-14:
        return 0.0, None
    interior = vertices.mean(axis=0)
    volume_sum = 0.0
    moment = np.zeros(3)
    for triangle in hull.simplices:
        a, b, c = vertices[triangle]
        volume = abs(
            np.linalg.det(np.column_stack((a - interior, b - interior, c - interior)))
        ) / 6.0
        volume_sum += volume
        moment += volume * (interior + a + b + c) / 4.0
    if volume_sum <= 1.0e-14:
        return 0.0, None
    return volume_sum, moment / volume_sum


def _axis_aligned_intersection(local_tetra, half_size):
    from scipy.spatial import ConvexHull, QhullError

    tetra = np.asarray(local_tetra, dtype=np.float64)
    half_size = np.asarray(half_size, dtype=np.float64)
    if np.any(tetra.max(axis=0) < -half_size) or np.any(
        tetra.min(axis=0) > half_size
    ):
        return TetBoxIntersection(0.0, None, 0)

    scale = max(float(np.max(np.abs(tetra))), float(np.max(half_size)), 1.0e-30)
    tetra_n = tetra / scale
    half_n = half_size / scale
    try:
        tetra_hull = ConvexHull(tetra_n)
    except QhullError as error:
        raise ValueError("tetrahedron must have positive three-dimensional volume") from error

    tetra_a = tetra_hull.equations[:, :3]
    tetra_b = -tetra_hull.equations[:, 3]
    box_a = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    box_b = np.repeat(half_n, 2)
    constraints_a = np.vstack([tetra_a, box_a])
    constraints_b = np.concatenate([tetra_b, box_b])

    vertices = []
    feasibility_tolerance = 2.0e-10
    dedup_tolerance = 5.0e-9
    for indices in combinations(range(len(constraints_a)), 3):
        matrix = constraints_a[list(indices)]
        if abs(float(np.linalg.det(matrix))) <= 1.0e-12:
            continue
        candidate = np.linalg.solve(matrix, constraints_b[list(indices)])
        if np.all(
            constraints_a @ candidate
            <= constraints_b + feasibility_tolerance
        ) and not any(
            np.linalg.norm(candidate - existing) <= dedup_tolerance
            for existing in vertices
        ):
            vertices.append(candidate)

    volume_n, centroid_n = _polyhedron_volume_centroid(vertices)
    if centroid_n is None:
        return TetBoxIntersection(0.0, None, len(vertices))
    return TetBoxIntersection(
        volume=float(volume_n * scale**3),
        centroid=np.asarray(centroid_n) * scale,
        vertex_count=len(vertices),
    )


def tetra_box_intersection(
    tetra,
    *,
    center,
    size,
    rotation_gauge_to_specimen=None,
):
    """Return exact convex intersection of one TET4 and an oriented gauge box."""
    tetra = np.asarray(tetra, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    if tetra.shape != (4, 3) or not np.all(np.isfinite(tetra)):
        raise ValueError("tetra must contain four finite 3D vertices")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("gauge center must be a finite 3-vector")
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("gauge size must be a positive finite 3-vector")
    rotation = _rotation_matrix(rotation_gauge_to_specimen)
    local_tetra = (tetra - center) @ rotation
    local = _axis_aligned_intersection(local_tetra, 0.5 * size)
    if local.centroid is None:
        return local
    centroid_specimen = center + local.centroid @ rotation.T
    return TetBoxIntersection(local.volume, centroid_specimen, local.vertex_count)


def compute_gauge_weights(
    points,
    cells,
    *,
    center,
    size,
    rotation_gauge_to_specimen=None,
):
    """Compute exact TET4 material-volume weights for one XRD gauge box."""
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    rotation = _rotation_matrix(rotation_gauge_to_specimen)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must have shape (num_points, 3) and be finite")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("cells must have shape (num_cells, 4)")
    if cells.size and (cells.min() < 0 or cells.max() >= len(points)):
        raise ValueError("cells contain out-of-range point indices")
    if (
        center.shape != (3,)
        or size.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(size))
        or np.any(size <= 0.0)
    ):
        raise ValueError(
            "gauge center/size must be finite 3-vectors with positive size"
        )

    local_points = (points - center) @ rotation
    tetra_local = local_points[cells]
    half_size = 0.5 * size
    candidates = np.all(tetra_local.max(axis=1) >= -half_size, axis=1) & np.all(
        tetra_local.min(axis=1) <= half_size, axis=1
    )
    weights = np.zeros(len(cells), dtype=np.float64)
    for cell_id in np.flatnonzero(candidates):
        weights[cell_id] = _axis_aligned_intersection(
            tetra_local[cell_id], half_size
        ).volume

    nominal = float(np.prod(size))
    material = float(weights.sum())
    contributing = int(np.count_nonzero(weights > 0.0))
    if material > 0.0:
        effective_count = float(material**2 / np.sum(weights * weights))
        max_fraction = float(weights.max() / material)
    else:
        effective_count = 0.0
        max_fraction = 0.0
    return GaugeWeights(
        cell_weights=weights,
        nominal_gauge_volume=nominal,
        material_intersection_volume=material,
        material_fill_fraction=material / nominal,
        contributing_cell_count=contributing,
        effective_cell_count=effective_count,
        max_cell_weight_fraction=max_fraction,
    )


def predict_gauge_microstrain(
    cell_elastic_strain,
    weights: GaugeWeights,
    *,
    direction,
    valid_mask=None,
    minimum_material_fill=0.95,
):
    """Apply a P0 elastic-strain gauge operator with explicit coverage gates."""
    strain = np.asarray(cell_elastic_strain, dtype=np.float64)
    if strain.ndim != 3 or strain.shape[1:] != (3, 3):
        raise ValueError("cell_elastic_strain must have shape (num_cells, 3, 3)")
    if len(strain) != len(weights.cell_weights):
        raise ValueError("elastic strain and gauge weights must use the same cells")
    finite = np.all(np.isfinite(strain), axis=(1, 2))
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (len(strain),):
            raise ValueError("valid_mask must contain one flag per cell")
        finite &= valid_mask
    material = weights.material_intersection_volume
    minimum_material_fill = float(minimum_material_fill)
    if (
        not np.isfinite(minimum_material_fill)
        or minimum_material_fill <= 0.0
        or minimum_material_fill > 1.0
    ):
        raise ValueError("minimum_material_fill must lie in (0, 1]")
    valid_volume = float(weights.cell_weights[finite].sum())
    valid_fraction = valid_volume / material if material > 0.0 else 0.0

    if material <= 0.0:
        status = "no_material_coverage"
        predicted = None
    elif weights.material_fill_fraction > 1.0 + 1.0e-8:
        status = "overlapping_material_coverage"
        predicted = None
    elif weights.material_fill_fraction < minimum_material_fill:
        status = "low_material_coverage"
        predicted = None
    elif valid_fraction < 1.0 - 1.0e-12:
        status = "invalid_field_coverage"
        predicted = None
    else:
        averaged = gauge_volume_average(strain[finite], weights.cell_weights[finite])
        predicted = project_normal_microstrain(averaged, direction)
        status = "ok"

    return {
        "status": status,
        "operator_order": "P0_exact_geometry",
        "input_unit": "1",
        "output_unit": "microstrain",
        "nominal_gauge_volume_m3": weights.nominal_gauge_volume,
        "material_intersection_volume_m3": weights.material_intersection_volume,
        "material_fill_fraction": weights.material_fill_fraction,
        "valid_field_fraction": valid_fraction,
        "contributing_cell_count": weights.contributing_cell_count,
        "effective_cell_count": weights.effective_cell_count,
        "max_cell_weight_fraction": weights.max_cell_weight_fraction,
        "predicted_microstrain": predicted,
    }
