"""Mechanics boundary conditions for build and release stages.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp


@dataclass(frozen=True)
class ReleaseCellSet:
    """Content-addressed exact release partition for one frozen mesh."""

    document: dict
    removed_cell_ids: onp.ndarray
    retained_root_cell_ids: onp.ndarray
    cell_mask: onp.ndarray
    artifact_sha256: str


def zero_exact_release_cells(mechanics_factor, removed_quad_mask):
    """Apply physical element deletion with no residual ersatz stiffness."""

    mechanics_factor = np.asarray(mechanics_factor)
    removed_quad_mask = np.asarray(removed_quad_mask)
    if mechanics_factor.shape != removed_quad_mask.shape:
        raise ValueError(
            "exact release mask must match the mechanics factor shape"
        )
    return np.where(
        removed_quad_mask > 0.5,
        np.zeros_like(mechanics_factor),
        mechanics_factor,
    )


def _release_id_array(name, values, num_cells):
    if (
        not isinstance(values, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
    ):
        raise ValueError(f"{name} must be a list of integer cell ids")
    ids = onp.asarray(values, dtype=onp.int64)
    if not len(ids):
        raise ValueError(f"{name} must be non-empty")
    if len(onp.unique(ids)) != len(ids):
        raise ValueError(f"{name} cell ids must be unique")
    if onp.any(ids < 0) or onp.any(ids >= num_cells):
        raise ValueError(
            f"{name} cell id is outside the valid range [0, {num_cells})"
        )
    return ids


def _release_mask_from_base64(name, encoded, num_cells):
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"{name} must be a non-empty base64 string")
    try:
        packed = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} is not valid base64") from exc
    expected_bytes = (num_cells + 7) // 8
    if len(packed) != expected_bytes:
        raise ValueError(
            f"{name} has {len(packed)} bytes; expected {expected_bytes} "
            f"for {num_cells} cells"
        )
    bits = onp.unpackbits(
        onp.frombuffer(packed, dtype=onp.uint8),
        bitorder="little",
    )
    if onp.any(bits[num_cells:] != 0):
        raise ValueError(f"{name} has nonzero padding bits")
    return bits[:num_cells].astype(bool)


def _release_ids_and_mask(document, *, id_key, mask_key, num_cells):
    has_ids = id_key in document
    has_mask = mask_key in document
    if has_ids == has_mask:
        raise ValueError(
            f"release artifact must define exactly one of {id_key!r} "
            f"or {mask_key!r}"
        )
    if has_ids:
        ids = _release_id_array(id_key, document[id_key], num_cells)
        mask = onp.zeros(num_cells, dtype=bool)
        mask[ids] = True
        return ids, mask
    mask = _release_mask_from_base64(
        mask_key,
        document[mask_key],
        num_cells,
    )
    ids = onp.flatnonzero(mask).astype(onp.int64)
    if not len(ids):
        raise ValueError(f"{mask_key} must select a non-empty cell set")
    return ids, mask


def _validate_release_mask_metadata(
    document,
    *,
    count_key,
    mask_sha_key,
    ids,
    mask,
):
    expected_count = document.get(count_key)
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(ids)
    ):
        raise ValueError(
            f"release cell-set {count_key} does not match the explicit set"
        )
    expected_mask_sha = document.get(mask_sha_key)
    if expected_mask_sha is not None:
        packed = onp.packbits(
            mask.astype(onp.uint8),
            bitorder="little",
        ).tobytes()
        if hashlib.sha256(packed).hexdigest() != expected_mask_sha:
            raise ValueError(
                f"release cell-set {mask_sha_key} does not match"
            )


def load_release_cell_set(
    path,
    *,
    expected_mesh_sha256,
    num_cells,
):
    """Load and verify an exact, content-addressed release cell set."""

    path = Path(path)
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"release cell-set artifact is not valid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("release cell-set artifact must be a JSON object")
    if document.get("schema_version") != "kaess.release-cellset/1":
        raise ValueError(
            "release cell-set schema_version must be "
            "'kaess.release-cellset/1'"
        )
    if document.get("protocol_id") != "kaess-2023-public-v1":
        raise ValueError(
            "release cell-set protocol_id must be kaess-2023-public-v1"
        )
    if document.get("source_class") not in {
        "author_artifact",
        "figure_digitized",
        "inferred",
        "assumption",
    } or not str(document.get("source_locator", "")).strip():
        raise ValueError(
            "release cell-set source class and locator must be explicit"
        )
    if document.get("cell_id_basis") != "solver_zero_based":
        raise ValueError(
            "release cell-set cell_id_basis must be solver_zero_based"
        )
    num_cells = int(num_cells)
    if num_cells <= 0:
        raise ValueError("release cell-set solver cell count must be positive")
    if document.get("mesh_num_cells") != num_cells:
        raise ValueError(
            "release cell-set mesh cell count does not match the solver mesh"
        )
    expected_mesh_sha256 = str(expected_mesh_sha256).lower()
    if (
        len(expected_mesh_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_mesh_sha256)
    ):
        raise ValueError("expected mesh SHA-256 must contain 64 lowercase hex digits")
    if document.get("mesh_sha256") != expected_mesh_sha256:
        raise ValueError("release cell-set mesh SHA identity does not match")

    removed_ids, removed_mask = _release_ids_and_mask(
        document,
        id_key="removed_cell_ids",
        mask_key="removed_cell_mask_base64",
        num_cells=num_cells,
    )
    root_ids, root_mask = _release_ids_and_mask(
        document,
        id_key="retained_root_cell_ids",
        mask_key="retained_root_cell_mask_base64",
        num_cells=num_cells,
    )
    _validate_release_mask_metadata(
        document,
        count_key="expected_removed_count",
        mask_sha_key="removed_cell_mask_sha256",
        ids=removed_ids,
        mask=removed_mask,
    )
    if "expected_retained_root_count" in document:
        _validate_release_mask_metadata(
            document,
            count_key="expected_retained_root_count",
            mask_sha_key="retained_root_cell_mask_sha256",
            ids=root_ids,
            mask=root_mask,
        )
    if onp.intersect1d(removed_ids, root_ids).size:
        raise ValueError("release removed set overlaps the retained root set")

    canonical_ids = json.dumps(
        removed_ids.tolist(),
        separators=(",", ":"),
    ).encode("utf-8")
    expected_ids_sha = document.get("removed_cell_ids_sha256")
    actual_ids_sha = hashlib.sha256(canonical_ids).hexdigest()
    if expected_ids_sha is not None and expected_ids_sha != actual_ids_sha:
        raise ValueError("release removed cell-id SHA-256 does not match")
    return ReleaseCellSet(
        document=document,
        removed_cell_ids=removed_ids,
        retained_root_cell_ids=root_ids,
        cell_mask=removed_mask,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
    )


_SOLID_FACE_LOCAL_NODES = {
    4: (
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ),
    8: (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}


def _release_face_component_count(cells, cell_mask):
    """Return the number of face-connected retained solid components."""

    local_faces = _SOLID_FACE_LOCAL_NODES.get(cells.shape[1])
    if local_faces is None:
        raise ValueError(
            "release connectivity supports only TET4 and HEX8 solid cells"
        )
    parent = onp.arange(len(cells), dtype=onp.int64)
    tree_rank = onp.zeros(len(cells), dtype=onp.int8)

    def find(cell_id):
        cell_id = int(cell_id)
        while parent[cell_id] != cell_id:
            parent[cell_id] = parent[parent[cell_id]]
            cell_id = int(parent[cell_id])
        return cell_id

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if tree_rank[left_root] < tree_rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if tree_rank[left_root] == tree_rank[right_root]:
            tree_rank[left_root] += 1

    face_owners = {}
    face_counts = {}
    retained_ids = onp.flatnonzero(cell_mask)
    for cell_id in retained_ids:
        cell = cells[cell_id]
        for local_face in local_faces:
            face = tuple(sorted(int(cell[index]) for index in local_face))
            count = face_counts.get(face, 0) + 1
            if count > 2:
                raise ValueError(
                    "release load-bearing mesh contains a non-manifold face"
                )
            face_counts[face] = count
            if face in face_owners:
                union(cell_id, face_owners[face])
            else:
                face_owners[face] = int(cell_id)
    return len({find(cell_id) for cell_id in retained_ids})


def _rigid_body_constraint_rank(points, anchor_dof_pairs):
    points = onp.asarray(points, dtype=onp.float64)
    pairs = onp.asarray(anchor_dof_pairs, dtype=onp.int64)
    center = onp.mean(points[onp.unique(pairs[:, 0])], axis=0)
    rows = []
    for node_id, component in pairs:
        point = points[node_id] - center
        x, y, z = point
        if component == 0:
            rows.append([1.0, 0.0, 0.0, 0.0, z, -y])
        elif component == 1:
            rows.append([0.0, 1.0, 0.0, -z, 0.0, x])
        else:
            rows.append([0.0, 0.0, 1.0, y, -x, 0.0])
    return int(onp.linalg.matrix_rank(onp.asarray(rows, dtype=onp.float64)))


def validate_release_cell_set(
    release_set,
    *,
    cells,
    points,
    removable_cell_mask,
    protected_cell_mask,
    anchor_node_ids,
    anchor_dof_pairs=None,
):
    """Reject wrong-body cuts, floating components and deficient anchors."""

    cells = onp.asarray(cells)
    points = onp.asarray(points, dtype=onp.float64)
    removable = onp.asarray(removable_cell_mask, dtype=bool)
    protected = onp.asarray(protected_cell_mask, dtype=bool)
    num_cells = len(cells)
    if cells.ndim != 2 or removable.shape != (num_cells,) or protected.shape != (
        num_cells,
    ):
        raise ValueError("release partition arrays do not match the cell mesh")
    if points.ndim != 2 or points.shape[1] != 3 or not onp.all(
        onp.isfinite(points)
    ):
        raise ValueError("release validation requires finite 3D mesh points")
    if (
        not onp.issubdtype(cells.dtype, onp.integer)
        or onp.any(cells < 0)
        or onp.any(cells >= len(points))
        or onp.any(onp.diff(onp.sort(cells, axis=1), axis=1) == 0)
    ):
        raise ValueError("release validation requires valid solid-cell nodes")
    removed = onp.asarray(release_set.cell_mask, dtype=bool)
    if removed.shape != (num_cells,):
        raise ValueError("release cell mask does not match the cell mesh")
    if onp.any(removed & protected):
        raise ValueError("release cell set cuts protected part cells")
    if onp.any(removed & (~removable)):
        raise ValueError("release cell set contains cells outside the removable support")
    root_ids = onp.asarray(
        release_set.retained_root_cell_ids,
        dtype=onp.int64,
    )
    if onp.any(removed[root_ids]):
        raise ValueError("release cell set removes a required retained root cell")
    if onp.any(~removable[root_ids]) or onp.any(protected[root_ids]):
        raise ValueError("retained root ids must identify removable support cells")

    load_bearing = (removable | protected) & (~removed)
    component_count = _release_face_component_count(cells, load_bearing)
    if component_count != 1:
        raise ValueError(
            "release load-bearing cells must form one face-connected "
            f"component; found {component_count} floating components"
        )
    retained_nodes = onp.zeros(len(points), dtype=bool)
    if onp.any(load_bearing):
        retained_nodes[onp.unique(cells[load_bearing])] = True
    anchors = onp.asarray(anchor_node_ids, dtype=onp.int64).reshape(-1)
    if (
        len(onp.unique(anchors)) != len(anchors)
        or onp.any(anchors < 0)
        or onp.any(anchors >= len(points))
        or onp.any(~retained_nodes[anchors])
    ):
        raise ValueError(
            "release anchor nodes must be unique nodes in the retained domain"
        )
    root_nodes = onp.unique(cells[root_ids])
    if onp.any(~onp.isin(anchors, root_nodes)):
        raise ValueError(
            "release anchor nodes must belong to the retained root cell set"
        )
    if anchor_dof_pairs is None:
        pairs = onp.asarray(
            [(int(node_id), component) for node_id in anchors for component in range(3)],
            dtype=onp.int64,
        )
    else:
        raw_pairs = onp.asarray(anchor_dof_pairs)
        if (
            raw_pairs.ndim != 2
            or raw_pairs.shape[1] != 2
            or not onp.issubdtype(raw_pairs.dtype, onp.integer)
        ):
            raise ValueError(
                "release anchor DOF pairs must be integer [node, component] rows"
            )
        pairs = raw_pairs.astype(onp.int64)
    if (
        not len(pairs)
        or len(onp.unique(pairs, axis=0)) != len(pairs)
        or onp.any(pairs[:, 0] < 0)
        or onp.any(pairs[:, 0] >= len(points))
        or onp.any(pairs[:, 1] < 0)
        or onp.any(pairs[:, 1] >= 3)
        or not onp.array_equal(onp.unique(pairs[:, 0]), anchors)
    ):
        raise ValueError(
            "release anchor DOF pairs must exactly cover the anchor node set"
        )
    if len(anchors) < 3 or _rigid_body_constraint_rank(points, pairs) < 6:
        raise ValueError("release anchor constraints do not suppress all rigid modes")
    return removed.copy()


def validate_release_direction(
    displacement,
    *,
    measurement_node_ids,
    build_axis,
    expected_sign,
    minimum_magnitude,
):
    """Validate the signed peak displacement of an analytic/solver cantilever."""

    displacement = onp.asarray(displacement, dtype=onp.float64)
    node_ids = onp.asarray(measurement_node_ids, dtype=onp.int64).reshape(-1)
    build_axis = int(build_axis)
    expected_sign = int(expected_sign)
    minimum_magnitude = float(minimum_magnitude)
    if (
        displacement.ndim != 2
        or build_axis < 0
        or build_axis >= displacement.shape[1]
        or not onp.all(onp.isfinite(displacement))
        or not len(node_ids)
        or onp.any(node_ids < 0)
        or onp.any(node_ids >= len(displacement))
        or expected_sign not in (-1, 1)
        or not onp.isfinite(minimum_magnitude)
        or minimum_magnitude <= 0.0
    ):
        raise ValueError("invalid release direction validation inputs")
    values = displacement[node_ids, build_axis]
    peak = float(values[int(onp.argmax(onp.abs(values)))])
    if expected_sign * peak < minimum_magnitude:
        raise ValueError(
            "release displacement direction/sign does not match the "
            "registered cantilever direction"
        )
    return abs(peak)


def make_full_bottom_mechanics_bc(bottom):
    def zero(_point):
        return 0.0

    return [[bottom, bottom, bottom], [0, 1, 2], [zero, zero, zero]]


def make_paper_minimal_bottom_mechanics_bc(
    points,
    bottom=None,
    *,
    build_axis_id,
    plane_axis_ids,
    anchor_corner="min_min",
    bottom_node_ids=None,
    anchor_candidate_node_ids=None,
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

    if bottom_node_ids is not None:
        raw_bottom_node_ids = onp.asarray(bottom_node_ids)
        if (
            raw_bottom_node_ids.ndim != 1
            or not onp.issubdtype(raw_bottom_node_ids.dtype, onp.integer)
        ):
            raise ValueError("explicit bottom node ids must be a 1D integer array")
        resolved_bottom_node_ids = raw_bottom_node_ids.astype(onp.int64)
        if (
            len(onp.unique(resolved_bottom_node_ids))
            != len(resolved_bottom_node_ids)
            or onp.any(resolved_bottom_node_ids < 0)
            or onp.any(resolved_bottom_node_ids >= len(points))
        ):
            raise ValueError(
                "explicit bottom node ids must be unique in-range mesh nodes"
            )
        bottom_mask = onp.zeros(len(points), dtype=bool)
        bottom_mask[resolved_bottom_node_ids] = True
    else:
        if bottom is None:
            raise ValueError(
                "paper minimal bottom BC requires a location or explicit node ids"
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
        resolved_bottom_node_ids = onp.flatnonzero(bottom_mask)
    if resolved_bottom_node_ids.size < 3:
        raise ValueError(
            "paper minimal bottom BC needs at least three bottom nodes"
        )
    if anchor_candidate_node_ids is None:
        resolved_anchor_candidate_ids = resolved_bottom_node_ids
    else:
        raw_anchor_candidate_ids = onp.asarray(anchor_candidate_node_ids)
        if (
            raw_anchor_candidate_ids.ndim != 1
            or not onp.issubdtype(
                raw_anchor_candidate_ids.dtype,
                onp.integer,
            )
        ):
            raise ValueError(
                "anchor candidate node ids must be a 1D integer array"
            )
        resolved_anchor_candidate_ids = (
            raw_anchor_candidate_ids.astype(onp.int64)
        )
        if (
            len(onp.unique(resolved_anchor_candidate_ids))
            != len(resolved_anchor_candidate_ids)
            or onp.any(resolved_anchor_candidate_ids < 0)
            or onp.any(resolved_anchor_candidate_ids >= len(points))
            or onp.any(
                ~onp.isin(
                    resolved_anchor_candidate_ids,
                    resolved_bottom_node_ids,
                )
            )
        ):
            raise ValueError(
                "anchor candidates must be unique bottom-surface mesh nodes"
            )
    if resolved_anchor_candidate_ids.size < 3:
        raise ValueError(
            "paper minimal bottom BC needs at least three anchor candidates"
        )
    bottom_mask_jax = np.asarray(bottom_mask)

    def resolved_bottom(_point, node_id):
        return bottom_mask_jax[node_id]

    plane_points = points[resolved_anchor_candidate_ids][:, plane_axis_ids]
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
    anchor0_id = int(resolved_anchor_candidate_ids[anchor0_local_id])
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
    anchor1_id = int(resolved_anchor_candidate_ids[anchor1_local_id])

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
        "bottom_node_count": int(resolved_bottom_node_ids.size),
        "anchor_candidate_node_count": int(
            resolved_anchor_candidate_ids.size
        ),
        "constrained_dof_count": int(resolved_bottom_node_ids.size + 3),
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


def make_root_minimal_release_mechanics_bc(
    points,
    cells,
    retained_root_cell_ids,
    *,
    build_axis_id,
    plane_axis_ids,
    base_coord,
    base_tolerance,
    anchor_corner="min_min",
    return_metadata=False,
):
    """Apply paper-minimal restraints only to the retained release root.

    The retained W3 support root remains attached to the plate after the
    Figure-7 cut. Every root node on the plate is restrained only along the
    build axis; three deterministic in-plane DOFs remove the remaining rigid
    modes without clamping the root surface.
    """

    points = onp.asarray(points, dtype=onp.float64)
    cells = onp.asarray(cells)
    root_ids = onp.asarray(retained_root_cell_ids)
    build_axis_id = int(build_axis_id)
    base_coord = float(base_coord)
    base_tolerance = float(base_tolerance)
    if (
        cells.ndim != 2
        or not onp.issubdtype(cells.dtype, onp.integer)
        or root_ids.ndim != 1
        or not onp.issubdtype(root_ids.dtype, onp.integer)
        or not len(root_ids)
        or len(onp.unique(root_ids)) != len(root_ids)
        or onp.any(root_ids < 0)
        or onp.any(root_ids >= len(cells))
    ):
        raise ValueError(
            "retained root cell ids must be unique in-range solid cells"
        )
    if (
        build_axis_id not in (0, 1, 2)
        or not onp.isfinite(base_coord)
        or not onp.isfinite(base_tolerance)
        or base_tolerance < 0.0
    ):
        raise ValueError("invalid retained-root base-plane definition")

    root_ids = root_ids.astype(onp.int64)
    root_node_ids = onp.unique(cells[root_ids].reshape(-1))
    if onp.any(root_node_ids < 0) or onp.any(root_node_ids >= len(points)):
        raise ValueError("retained root cells reference invalid mesh nodes")
    root_bottom_node_ids = root_node_ids[
        onp.abs(points[root_node_ids, build_axis_id] - base_coord)
        <= base_tolerance
    ]
    bc, metadata = make_paper_minimal_bottom_mechanics_bc(
        points,
        None,
        build_axis_id=build_axis_id,
        plane_axis_ids=plane_axis_ids,
        anchor_corner=anchor_corner,
        bottom_node_ids=root_bottom_node_ids,
        return_metadata=True,
    )
    constrained_dof_pairs = [
        [int(node_id), build_axis_id] for node_id in root_bottom_node_ids
    ]
    constrained_dof_pairs.extend(
        [
            [metadata["anchor_node_ids"][0], int(plane_axis_ids[0])],
            [metadata["anchor_node_ids"][0], int(plane_axis_ids[1])],
            [
                metadata["anchor_node_ids"][1],
                metadata["rotation_component"],
            ],
        ]
    )
    rigid_body_rank = _rigid_body_constraint_rank(
        points,
        onp.asarray(constrained_dof_pairs, dtype=onp.int64),
    )
    metadata.update(
        {
            "mode": "paper_minimal_root",
            "retained_root_cell_count": int(len(root_ids)),
            "retained_root_node_count": int(len(root_node_ids)),
            "base_coord": base_coord,
            "base_tolerance": base_tolerance,
            "root_bottom_node_ids": [
                int(node_id) for node_id in root_bottom_node_ids
            ],
            "constrained_dof_pairs": constrained_dof_pairs,
            "rigid_body_rank": rigid_body_rank,
        }
    )
    print(
        "paper minimal release root: "
        f"{metadata['bottom_node_count']} normal restraints + 3 in-plane "
        f"DOFs; root_cells={metadata['retained_root_cell_count']}"
    )
    if return_metadata:
        return bc, metadata
    return bc


def validate_release_anchor_protocol(
    release_set,
    resolved_metadata,
    *,
    anchor_corner,
):
    """Bind resolved root restraints to the content-addressed input."""

    protocol = release_set.document.get("anchor_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(
            "formal release artifact requires an anchor_protocol object"
        )
    required_variants = {
        "min_min",
        "max_min",
        "max_max",
        "min_max",
    }
    variants = protocol.get("variants")
    if (
        protocol.get("mode") != "paper_minimal_root"
        or protocol.get("build_axis_id")
        != resolved_metadata.get("build_axis_id")
        or protocol.get("plane_axis_ids")
        != resolved_metadata.get("plane_axis_ids")
        or protocol.get("primary_corner") not in required_variants
        or not isinstance(variants, dict)
        or set(variants) != required_variants
        or anchor_corner not in variants
    ):
        raise ValueError(
            "release anchor protocol mode, axes, or variants do not match"
        )

    root_bottom_ids = protocol.get("root_bottom_node_ids")
    if (
        not isinstance(root_bottom_ids, list)
        or any(
            isinstance(node_id, bool) or not isinstance(node_id, int)
            for node_id in root_bottom_ids
        )
        or root_bottom_ids
        != resolved_metadata.get("root_bottom_node_ids")
    ):
        raise ValueError(
            "release anchor protocol root-bottom nodes do not match"
        )
    canonical_ids = json.dumps(
        root_bottom_ids,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        hashlib.sha256(canonical_ids).hexdigest()
        != protocol.get("root_bottom_node_ids_sha256")
        or protocol.get("expected_root_bottom_node_count")
        != resolved_metadata.get("bottom_node_count")
        or protocol.get("expected_physical_release_dof_count")
        != resolved_metadata.get("constrained_dof_count")
    ):
        raise ValueError(
            "release anchor protocol count or node hash does not match"
        )
    try:
        protocol_base_coord = float(protocol["base_coord_m"])
        protocol_base_tolerance = float(protocol["base_tolerance_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "release anchor protocol base plane does not match"
        ) from exc
    if (
        not onp.isfinite(protocol_base_coord)
        or not onp.isfinite(protocol_base_tolerance)
        or protocol_base_tolerance < 0.0
        or protocol_base_coord != resolved_metadata.get("base_coord")
        or protocol_base_tolerance
        != resolved_metadata.get("base_tolerance")
    ):
        raise ValueError(
            "release anchor protocol base plane does not match"
        )

    variant = variants[anchor_corner]
    if not isinstance(variant, dict):
        raise ValueError("release anchor protocol variant must be an object")
    expected_in_plane_pairs = [
        pair
        for pair in resolved_metadata.get("constrained_dof_pairs", [])
        if pair[1] != resolved_metadata.get("build_axis_id")
    ]
    if (
        variant.get("anchor_node_ids")
        != resolved_metadata.get("anchor_node_ids")
        or variant.get("in_plane_dof_pairs") != expected_in_plane_pairs
        or variant.get("rigid_body_rank") != 6
        or resolved_metadata.get("rigid_body_rank") != 6
    ):
        raise ValueError(
            "resolved release anchors differ from the registered variant"
        )
    expected_coordinates = onp.asarray(
        variant.get("anchor_coordinates_m"),
        dtype=onp.float64,
    )
    resolved_coordinates = onp.asarray(
        resolved_metadata.get("anchor_coordinates"),
        dtype=onp.float64,
    )
    if (
        expected_coordinates.shape != (2, 3)
        or resolved_coordinates.shape != (2, 3)
        or not onp.array_equal(expected_coordinates, resolved_coordinates)
    ):
        raise ValueError(
            "resolved release anchor coordinates differ from the artifact"
        )
    return protocol


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
