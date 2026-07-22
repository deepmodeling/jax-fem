"""Solid-mesh inp readers (Abaqus via meshio).

Origin: 159_local/v01/inp_initial_guess_smoke.py (read_tet4_inp, compact_mesh,
connected_cell_ids, orient_tet4) and 159_local/v03/
am_thermal_stress_macro_intersection_mech100.py (read_solid_inp,
read_inp_cell_set). Moved verbatim in the 2026-07-22 restructure.
"""
from collections import deque

import numpy as onp


def connected_cell_ids(cells, max_cells):
    if max_cells is None or max_cells <= 0 or max_cells >= len(cells):
        return onp.arange(len(cells), dtype=onp.int64)

    node_to_cells = {}
    for cell_id, cell in enumerate(cells):
        for node_id in cell:
            node_to_cells.setdefault(int(node_id), []).append(cell_id)

    selected = {0}
    queue = deque([0])
    while queue and len(selected) < max_cells:
        cell_id = queue.popleft()
        for node_id in cells[cell_id]:
            for next_cell_id in node_to_cells[int(node_id)]:
                if next_cell_id not in selected:
                    selected.add(next_cell_id)
                    queue.append(next_cell_id)
                    if len(selected) >= max_cells:
                        break
            if len(selected) >= max_cells:
                break

    return onp.array(sorted(selected), dtype=onp.int64)


def compact_mesh(points, cells, max_cells):
    cell_ids = connected_cell_ids(cells, max_cells)
    sub_cells_old = cells[cell_ids]
    used_nodes = onp.unique(sub_cells_old.reshape(-1))
    old_to_new = -onp.ones(len(points), dtype=onp.int64)
    old_to_new[used_nodes] = onp.arange(len(used_nodes), dtype=onp.int64)
    sub_points = points[used_nodes]
    sub_cells = old_to_new[sub_cells_old]
    return sub_points, orient_tet4(sub_points, sub_cells), len(cell_ids)


def orient_tet4(points, cells):
    pts = points[cells]
    signed = onp.einsum(
        "ij,ij->i",
        onp.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]),
        pts[:, 3] - pts[:, 0],
    )
    neg = signed < 0.0
    if onp.any(neg):
        cells = cells.copy()
        tmp = cells[neg, 1].copy()
        cells[neg, 1] = cells[neg, 2]
        cells[neg, 2] = tmp
    return cells


def read_tet4_inp(path, max_cells):
    import meshio

    meshio_mesh = meshio.read(path)
    if "tetra" not in meshio_mesh.cells_dict:
        available = ", ".join(sorted(meshio_mesh.cells_dict))
        raise ValueError(f"Expected tetra cells in {path}; available cells: {available}")

    points = onp.asarray(meshio_mesh.points, dtype=onp.float64)
    cells = onp.asarray(meshio_mesh.cells_dict["tetra"], dtype=onp.int64)
    return compact_mesh(points, cells, max_cells)


SOLID_BLOCKS = (("tetra", "TET4"), ("hexahedron", "HEX8"))


def read_solid_inp(path, max_cells):
    """Read a C3D4 (tetra) or C3D8 (hexahedron) inp.

    Returns (points, cells, selected_cells, ele_type). The hexahedron path
    keeps meshio cell order (VTK == Abaqus C3D8 node order, no reordering)
    so ELSET indices from read_inp_cell_set() stay aligned.
    """
    import meshio

    meshio_mesh = meshio.read(path)
    if "hexahedron" in meshio_mesh.cells_dict:
        if "tetra" in meshio_mesh.cells_dict:
            raise ValueError(f"{path} mixes tetra and hexahedron blocks")
        if max_cells:
            raise ValueError("--max-cells truncation is TET4-only")
        points = onp.asarray(meshio_mesh.points, dtype=onp.float64)
        cells = onp.asarray(meshio_mesh.cells_dict["hexahedron"], dtype=onp.int64)
        used = onp.unique(cells.reshape(-1))
        if len(used) != len(points):
            old_to_new = -onp.ones(len(points), dtype=onp.int64)
            old_to_new[used] = onp.arange(len(used), dtype=onp.int64)
            points = points[used]
            cells = old_to_new[cells]
        return points, cells, len(cells), "HEX8"
    points, cells, selected_cells = read_tet4_inp(path, max_cells)
    return points, cells, selected_cells, "TET4"


def read_inp_cell_set(path, set_name, num_cells):
    """Boolean mask over solid cells for a named inp ELSET.

    Cell order must match read_solid_inp(): both read the same meshio solid
    block and neither reorders cells (orientation fixes swap nodes in place),
    so the 0-based set indices align with the solver cell array.
    """
    import meshio

    meshio_mesh = meshio.read(path)
    block = next((b for b, _ in SOLID_BLOCKS if b in meshio_mesh.cells_dict), None)
    sets = getattr(meshio_mesh, "cell_sets_dict", None) or {}
    if block is None or set_name not in sets or block not in sets[set_name]:
        available = ", ".join(sorted(sets)) or "(none)"
        raise ValueError(
            f"ELSET {set_name!r} with {block or 'solid'} cells not found in "
            f"{path}; available sets: {available}")
    indices = onp.asarray(sets[set_name][block], dtype=onp.int64)
    if len(meshio_mesh.cells_dict[block]) != num_cells:
        raise ValueError(
            f"cell count mismatch between solver mesh ({num_cells}) and "
            f"{path} {block} block ({len(meshio_mesh.cells_dict[block])}); "
            "--max-cells truncation is incompatible with --powder-elset")
    mask = onp.zeros(num_cells, dtype=bool)
    mask[indices] = True
    return mask
