"""Release-step mechanics boundary conditions (clamp, rigid-body anchors, box anchor).

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import jax.numpy as np
import numpy as onp


def make_full_bottom_mechanics_bc(bottom):
    def zero(_point):
        return 0.0

    return [[bottom, bottom, bottom], [0, 1, 2], [zero, zero, zero]]


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
