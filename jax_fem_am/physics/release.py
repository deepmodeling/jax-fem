"""Mechanics boundary conditions for build and release stages.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import jax
import jax.numpy as np
import numpy as onp


def make_full_bottom_mechanics_bc(bottom):
    def zero(_point):
        return 0.0

    return [[bottom, bottom, bottom], [0, 1, 2], [zero, zero, zero]]


def make_paper_minimal_bottom_mechanics_bc(
    points,
    bottom,
    *,
    build_axis_id,
    plane_axis_ids,
    anchor_corner="min_min",
    return_metadata=False,
):
    """Restrain the bottom normal and only the three remaining rigid modes.

    Kaess et al. (2023), Section 2.3, restrains every bottom node in the
    build direction while partially permitting in-plane motion:
    https://doi.org/10.3390/ma16062321. The paper does not publish the exact
    in-plane nodes, so this helper implements four deterministic corner
    variants for the required sensitivity study.
    """

    points = onp.asarray(points, dtype=onp.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not onp.all(onp.isfinite(points))
    ):
        raise ValueError("paper minimal bottom BC requires finite 3D points")

    build_axis_id = int(build_axis_id)
    plane_axis_ids = tuple(int(axis) for axis in plane_axis_ids)
    if (
        build_axis_id not in (0, 1, 2)
        or len(plane_axis_ids) != 2
        or len(set(plane_axis_ids)) != 2
        or set(plane_axis_ids) | {build_axis_id} != {0, 1, 2}
    ):
        raise ValueError(
            "build_axis_id and plane_axis_ids must partition the three axes"
        )

    corner_sides = {
        "min_min": ("min", "min"),
        "max_min": ("max", "min"),
        "max_max": ("max", "max"),
        "min_max": ("min", "max"),
    }
    if anchor_corner not in corner_sides:
        raise ValueError(
            "paper minimal anchor corner must be one of "
            f"{sorted(corner_sides)}, got {anchor_corner!r}"
        )

    try:
        bottom_mask = onp.asarray(
            jax.vmap(bottom)(np.asarray(points)),
            dtype=bool,
        )
    # JAX trace/concretization failures inherit TypeError. Location
    # predicates are also allowed to be ordinary host-side callables, so
    # retry them point-by-point without hiding non-TypeError geometry bugs.
    except TypeError:
        bottom_mask = onp.asarray(
            [bool(onp.asarray(bottom(point))) for point in points],
            dtype=bool,
        )
    bottom_node_ids = onp.flatnonzero(bottom_mask)
    if bottom_node_ids.size < 3:
        raise ValueError(
            "paper minimal bottom BC needs at least three bottom nodes"
        )
    bottom_mask_jax = np.asarray(bottom_mask)

    def resolved_bottom(_point, node_id):
        return bottom_mask_jax[node_id]

    plane_points = points[bottom_node_ids][:, plane_axis_ids]
    plane_span = onp.ptp(plane_points, axis=0)
    geometry_scale = max(float(onp.max(plane_span)), 1.0e-30)
    coordinate_scale = max(float(onp.max(onp.abs(points))), geometry_scale)
    alignment_tolerance = max(
        1.0e-10 * geometry_scale,
        32.0 * onp.finfo(onp.float64).eps * coordinate_scale,
    )
    if onp.linalg.matrix_rank(
        plane_points - plane_points[0],
        tol=alignment_tolerance,
    ) < 2:
        raise ValueError(
            "paper minimal bottom BC requires a non-collinear bottom surface"
        )

    side0, side1 = corner_sides[anchor_corner]
    target = onp.asarray(
        [
            (
                onp.min(plane_points[:, axis])
                if side == "min"
                else onp.max(plane_points[:, axis])
            )
            for axis, side in enumerate((side0, side1))
        ],
        dtype=onp.float64,
    )
    normalized_distance = onp.sum(
        ((plane_points - target) / plane_span) ** 2,
        axis=1,
    )
    anchor0_local_id = int(onp.argmin(normalized_distance))
    anchor0_id = int(bottom_node_ids[anchor0_local_id])
    anchor0_plane = plane_points[anchor0_local_id]

    same_second_coordinate = (
        onp.abs(plane_points[:, 1] - anchor0_plane[1])
        <= alignment_tolerance
    )
    separated_first_coordinate = (
        onp.abs(plane_points[:, 0] - anchor0_plane[0])
        > alignment_tolerance
    )
    anchor1_candidates = onp.flatnonzero(
        same_second_coordinate & separated_first_coordinate
    )
    if anchor1_candidates.size:
        separation = onp.abs(
            plane_points[anchor1_candidates, 0] - anchor0_plane[0]
        )
        rotation_component = plane_axis_ids[1]
    else:
        same_first_coordinate = (
            onp.abs(plane_points[:, 0] - anchor0_plane[0])
            <= alignment_tolerance
        )
        separated_second_coordinate = (
            onp.abs(plane_points[:, 1] - anchor0_plane[1])
            > alignment_tolerance
        )
        anchor1_candidates = onp.flatnonzero(
            same_first_coordinate & separated_second_coordinate
        )
        if not anchor1_candidates.size:
            raise ValueError(
                "paper minimal bottom BC could not find an axis-aligned "
                "second anchor on the bottom surface"
            )
        separation = onp.abs(
            plane_points[anchor1_candidates, 1] - anchor0_plane[1]
        )
        rotation_component = plane_axis_ids[0]

    anchor1_local_id = int(
        anchor1_candidates[int(onp.argmax(separation))]
    )
    anchor1_id = int(bottom_node_ids[anchor1_local_id])

    def at_node(target_node_id):
        def location(_point, node_id):
            return node_id == target_node_id

        return location

    def zero(_point):
        return 0.0

    bc = [
        [
            resolved_bottom,
            at_node(anchor0_id),
            at_node(anchor0_id),
            at_node(anchor1_id),
        ],
        [
            build_axis_id,
            plane_axis_ids[0],
            plane_axis_ids[1],
            rotation_component,
        ],
        [zero, zero, zero, zero],
    ]
    metadata = {
        "mode": "paper_minimal",
        "anchor_corner": anchor_corner,
        "build_axis_id": build_axis_id,
        "plane_axis_ids": list(plane_axis_ids),
        "bottom_node_count": int(bottom_node_ids.size),
        "anchor_node_ids": [anchor0_id, anchor1_id],
        "anchor_coordinates": [
            points[anchor0_id].tolist(),
            points[anchor1_id].tolist(),
        ],
        "rotation_component": int(rotation_component),
    }
    print(
        "paper minimal bottom BC: "
        f"{metadata['bottom_node_count']} normal restraints; "
        f"anchors={metadata['anchor_node_ids']}; "
        f"rotation_component={metadata['rotation_component']}; "
        f"corner={anchor_corner}"
    )
    if return_metadata:
        return bc, metadata
    return bc


def make_anchor_mechanics_bc(points, candidate_node_ids=None):
    # Anchors must sit on load-bearing (printed) material: the mesh extremes
    # can land on void cells whose near-zero stiffness makes the release
    # system singular (observed NaN). Pass the printed-node subset.
    if candidate_node_ids is not None and len(candidate_node_ids) >= 3:
        candidates = points[onp.asarray(candidate_node_ids)]
    else:
        candidates = points
    span = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1.0)
    atol = 1e-8 * span
    anchor0_id = int(onp.argmin(candidates[:, 0]))
    anchor0 = candidates[anchor0_id]
    dist0 = onp.linalg.norm(candidates - anchor0, axis=1)
    anchor1_id = int(onp.argmax(dist0))
    anchor1 = candidates[anchor1_id]
    axis = anchor1 - anchor0
    axis_norm = max(float(onp.linalg.norm(axis)), 1e-12)
    cross_dist = onp.linalg.norm(onp.cross(candidates - anchor0, axis / axis_norm), axis=1)
    anchor2_id = int(onp.argmax(cross_dist))
    anchor2 = candidates[anchor2_id]

    def at_node(target):
        target = np.asarray(target)

        def location(point):
            return np.linalg.norm(point - target) <= atol

        return location

    def zero(_point):
        return 0.0

    return [
        [at_node(anchor0), at_node(anchor0), at_node(anchor0), at_node(anchor1), at_node(anchor1), at_node(anchor2)],
        [0, 1, 2, 1, 2, 2],
        [zero, zero, zero, zero, zero, zero],
    ]


def make_box_anchor_mechanics_bc(points, box):
    # Partial-cut release: nodes inside the axis-aligned box stay clamped
    # (u=0, all components), modeling the un-cut root attachment that an
    # EDM/saw cut leaves behind (e.g. the Kaess 2023 cantilever separation).
    # The box is given in mesh coordinates (after --mesh-length-scale).
    box = [float(v) for v in box]
    if len(box) != 6:
        raise ValueError("release anchor box needs 6 floats: xmin xmax ymin ymax zmin zmax")
    lo = onp.asarray(box[0::2])
    hi = onp.asarray(box[1::2])
    if onp.any(hi <= lo):
        raise ValueError(f"release anchor box is empty: min={lo}, max={hi}")
    span = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1e-30)
    atol = 1e-8 * span
    inside = onp.all((points >= lo - atol) & (points <= hi + atol), axis=1)
    count = int(onp.sum(inside))
    if count < 3:
        raise ValueError(
            f"release anchor box contains only {count} mesh nodes; "
            "a partial-cut anchor needs at least 3 to suppress rigid modes"
        )
    print(f"release anchor box: clamping {count} nodes in {box}")

    def location(point):
        return np.all((point >= lo - atol) & (point <= hi + atol))

    def zero(_point):
        return 0.0

    return [
        [location, location, location],
        [0, 1, 2],
        [zero, zero, zero],
    ]
