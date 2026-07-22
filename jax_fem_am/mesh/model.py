"""Build-domain geometry helpers (box locations, part bounds, layer/cell classification).

Origin: 159_local/v03/am_thermal_stress_macro_intersection_mech100.py
(AXIS_TO_ID, make_box_locations, make_part_build_box, coord_from_frac,
resolve_axis_range, compute_cell_temperature, classify_cells,
compute_layer_id, compute_physical_layer_id_cell, compute_cell_build_interval,
cells_intersect_distance_band, compute_nominal_layer_id_from_interval).
Moved verbatim in the 2026-07-22 restructure.
"""
import jax.numpy as np
import numpy as onp


AXIS_TO_ID = {"x": 0, "y": 1, "z": 2}


def make_box_locations(points, build_axis="x", base_side="min", tol_ratio=1e-8, abs_tol=None):
    pmin = onp.min(points, axis=0)
    pmax = onp.max(points, axis=0)
    span = max(float(onp.max(pmax - pmin)), 1.0)
    # Real CAD meshes rarely have their base-face nodes exactly coplanar; the
    # legacy 1e-8 relative tolerance can select only a handful of nodes (8 of
    # ~1225 on the 0119 part). Pass abs_tol (--boundary-tol) to widen it.
    atol = float(abs_tol) if abs_tol is not None and abs_tol > 0.0 else tol_ratio * span
    build_axis_id = AXIS_TO_ID[build_axis]
    plane_axis_ids = tuple(i for i in range(3) if i != build_axis_id)
    if base_side == "min":
        base_coord = pmin[build_axis_id]
        exposed_coord = pmax[build_axis_id]
    else:
        base_coord = pmax[build_axis_id]
        exposed_coord = pmin[build_axis_id]

    def bottom(point):
        return np.isclose(point[build_axis_id], base_coord, rtol=0.0, atol=atol)

    def exposed(point):
        return np.isclose(point[build_axis_id], exposed_coord, rtol=0.0, atol=atol)

    def walls(point):
        a0, a1 = plane_axis_ids
        return (
            np.isclose(point[a0], pmin[a0], rtol=0.0, atol=atol)
            | np.isclose(point[a0], pmax[a0], rtol=0.0, atol=atol)
            | np.isclose(point[a1], pmin[a1], rtol=0.0, atol=atol)
            | np.isclose(point[a1], pmax[a1], rtol=0.0, atol=atol)
        )

    return pmin, pmax, bottom, exposed, walls, build_axis_id, plane_axis_ids, float(base_coord), float(exposed_coord)


def make_part_build_box(pmin, pmax, build_axis_id, base_side, substrate_thickness, support_thickness):
    """Return the build bounds used for part layers and raster paths.

    If substrate/support are included in the mesh, layer fronts should advance
    through the printed part only, not through the substrate/support thickness.
    Thickness values are assumed to be in the same internal units as the scaled
    mesh coordinates.
    """
    part_pmin = onp.array(pmin, dtype=onp.float64).copy()
    part_pmax = onp.array(pmax, dtype=onp.float64).copy()
    total_offset = max(float(substrate_thickness), 0.0) + max(float(support_thickness), 0.0)
    build_span = float(pmax[build_axis_id] - pmin[build_axis_id])
    if total_offset >= build_span and build_span > 0.0:
        raise ValueError("substrate_thickness + support_thickness must be smaller than the build-axis span")
    if base_side == "min":
        part_pmin[build_axis_id] = pmin[build_axis_id] + total_offset
    else:
        part_pmax[build_axis_id] = pmax[build_axis_id] - total_offset
    return part_pmin, part_pmax


def coord_from_frac(pmin, pmax, axis_id, frac):
    return float(pmin[axis_id] + frac * (pmax[axis_id] - pmin[axis_id]))


def resolve_axis_range(pmin, pmax, axis_id, start_value, end_value, start_frac, end_frac):
    start = float(start_value) if start_value is not None else coord_from_frac(pmin, pmax, axis_id, start_frac)
    end = float(end_value) if end_value is not None else coord_from_frac(pmin, pmax, axis_id, end_frac)
    return start, end


def compute_cell_temperature(T_nodes, cells):
    return onp.mean(onp.asarray(T_nodes)[cells, 0], axis=1)


def classify_cells(points, cells, build_axis_id, build_sign, base_coord, args):
    centroids = onp.mean(points[cells], axis=1)
    cell_build_coord = centroids[:, build_axis_id]
    dist_from_base = build_sign * (cell_build_coord - base_coord)
    substrate = dist_from_base <= args.substrate_thickness if args.substrate_thickness > 0 else onp.zeros(len(cells), dtype=bool)
    support = (
        (dist_from_base > args.substrate_thickness)
        & (dist_from_base <= args.substrate_thickness + args.support_thickness)
        if args.support_thickness > 0
        else onp.zeros(len(cells), dtype=bool)
    )
    return centroids, cell_build_coord, substrate, support


def compute_layer_id(cell_build_coord, build_axis_id, pmin, pmax, args):
    build_min = float(pmin[build_axis_id])
    build_max = float(pmax[build_axis_id])
    if args.base_side == "min":
        frac = (cell_build_coord - build_min) / max(build_max - build_min, 1e-15)
    else:
        frac = (build_max - cell_build_coord) / max(build_max - build_min, 1e-15)
    return onp.clip(onp.ceil(frac * args.layers), 1, args.layers).astype(onp.int32)


def compute_physical_layer_id_cell(cell_build_coord, build_axis_id, part_pmin, part_pmax, build_sign, args):
    """Return physical layer id for each part cell.

    Unlike compute_layer_id(), this function is based on layer_thickness when
    available and does not clip to --max-print-layers. Cells beyond the printed
    test window therefore keep layer ids larger than the simulated layer count.
    Fixture cells are assigned outside this function.
    """
    if args.layer_thickness is None or args.layer_thickness <= 0.0:
        return compute_layer_id(cell_build_coord, build_axis_id, part_pmin, part_pmax, args)

    if args.base_side == "min":
        part_base_coord = float(part_pmin[build_axis_id])
    else:
        part_base_coord = float(part_pmax[build_axis_id])

    dist_from_part_base = build_sign * (cell_build_coord - part_base_coord)
    layer_id = onp.ceil(dist_from_part_base / float(args.layer_thickness)).astype(onp.int32)
    return onp.maximum(layer_id, 1).astype(onp.int32)


def compute_cell_build_interval(points, cells, build_axis_id, build_sign, part_base_coord):
    """Return each tetra cell's build-direction distance interval from part base.

    The previous layer assignment used the cell centroid only. That can miss
    thin layers when no centroid lies inside the layer band. This interval
    representation marks a cell as intersecting a layer if any part of its
    vertex span crosses that layer band.
    """
    cell_axis = onp.asarray(points[cells, build_axis_id], dtype=onp.float64)
    cell_dist = float(build_sign) * (cell_axis - float(part_base_coord))
    cell_d_min = onp.min(cell_dist, axis=1)
    cell_d_max = onp.max(cell_dist, axis=1)
    return cell_d_min, cell_d_max


def cells_intersect_distance_band(cell_d_min, cell_d_max, lower, upper, tol=1e-12):
    """Boolean mask: cell build-axis interval intersects [lower, upper]."""
    return (cell_d_max >= float(lower) - tol) & (cell_d_min <= float(upper) + tol)


def compute_nominal_layer_id_from_interval(cell_d_min, cell_d_max, args):
    """Assign a representative layer id for visualization under interval activation.

    This is not used for activation. It is the first physical layer intersected
    by the cell, useful for ParaView coloring when tetrahedra span multiple
    thin layers.
    """
    if args.layer_thickness is None or args.layer_thickness <= 0.0:
        return onp.ones_like(cell_d_min, dtype=onp.int32)
    lt = float(args.layer_thickness)
    layer_id = onp.floor(onp.maximum(cell_d_min, 0.0) / lt).astype(onp.int32) + 1
    return onp.maximum(layer_id, 1).astype(onp.int32)
