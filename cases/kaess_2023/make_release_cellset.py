"""Generate the frozen Kaess Figure-7 release cell-set artifact."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

import meshio
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jax_fem_am.physics.release import (
    make_root_minimal_release_mechanics_bc,
)


DEFAULT_MESH = Path(__file__).with_name(
    "kaess_cantilever_c3d8_powder_margin.inp"
)
DEFAULT_OUTPUT = Path(__file__).with_name("inputs") / "release-cellset.json"
ANCHOR_CORNERS = ("min_min", "max_min", "max_max", "min_max")


def _box(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("box requires xmin xmax ymin ymax zmin zmax")
    lo = values[0::2]
    hi = values[1::2]
    if np.any(hi <= lo):
        raise ValueError("box maxima must exceed minima")
    return lo, hi


def _packed_mask(mask):
    packed = np.packbits(
        np.asarray(mask, dtype=np.uint8),
        bitorder="little",
    ).tobytes()
    return (
        base64.b64encode(packed).decode("ascii"),
        hashlib.sha256(packed).hexdigest(),
    )


def _canonical_json_sha256(value):
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _ids_sha256(mask):
    return _canonical_json_sha256(np.flatnonzero(mask).tolist())


def _anchor_protocol(points, cells, retained_root):
    """Freeze every deterministic paper-minimal root-anchor variant."""

    build_axis_id = 2
    plane_axis_ids = (0, 1)
    base_coord = 0.0
    base_tolerance = 1.0e-12
    retained_root_ids = np.flatnonzero(retained_root).astype(np.int64)
    variants = {}
    root_bottom_node_ids = None
    expected_dof_count = None

    for anchor_corner in ANCHOR_CORNERS:
        _, metadata = make_root_minimal_release_mechanics_bc(
            points,
            cells,
            retained_root_ids,
            build_axis_id=build_axis_id,
            plane_axis_ids=plane_axis_ids,
            base_coord=base_coord,
            base_tolerance=base_tolerance,
            anchor_corner=anchor_corner,
            return_metadata=True,
        )
        resolved_root_bottom_ids = metadata["root_bottom_node_ids"]
        resolved_dof_count = int(metadata["constrained_dof_count"])
        if root_bottom_node_ids is None:
            root_bottom_node_ids = resolved_root_bottom_ids
            expected_dof_count = resolved_dof_count
        elif (
            root_bottom_node_ids != resolved_root_bottom_ids
            or expected_dof_count != resolved_dof_count
        ):
            raise ValueError(
                "paper-minimal anchor variants disagree on the retained-root "
                "bottom nodes or physical release DOF count"
            )
        if int(metadata["rigid_body_rank"]) != 6:
            raise ValueError(
                f"paper-minimal {anchor_corner} anchors have deficient "
                f"rigid-body rank {metadata['rigid_body_rank']}"
            )

        variants[anchor_corner] = {
            "anchor_node_ids": metadata["anchor_node_ids"],
            "anchor_coordinates_m": metadata["anchor_coordinates"],
            "in_plane_dof_pairs": [
                pair
                for pair in metadata["constrained_dof_pairs"]
                if pair[1] != build_axis_id
            ],
            "rigid_body_rank": int(metadata["rigid_body_rank"]),
        }

    return {
        "mode": "paper_minimal_root",
        "build_axis_id": build_axis_id,
        "plane_axis_ids": list(plane_axis_ids),
        "base_coord_m": base_coord,
        "base_tolerance_m": base_tolerance,
        "root_bottom_node_ids": root_bottom_node_ids,
        "root_bottom_node_ids_sha256": _canonical_json_sha256(
            root_bottom_node_ids
        ),
        "expected_root_bottom_node_count": len(root_bottom_node_ids),
        "expected_physical_release_dof_count": expected_dof_count,
        "primary_corner": "min_min",
        "variants": variants,
    }


def build_document(
    mesh_path,
    *,
    cut_box,
    retained_root_box,
    powder_elset,
):
    mesh_path = Path(mesh_path)
    mesh = meshio.read(mesh_path)
    if "hexahedron" not in mesh.cells_dict:
        raise ValueError("Kaess release artifact requires the frozen HEX8 mesh")
    cells = np.asarray(mesh.cells_dict["hexahedron"], dtype=np.int64)
    points = np.asarray(mesh.points, dtype=np.float64)
    centroids = np.mean(points[cells], axis=1)
    sets = getattr(mesh, "cell_sets_dict", None) or {}
    try:
        powder_ids = np.asarray(
            sets[powder_elset]["hexahedron"],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"powder ELSET {powder_elset!r} is absent from the frozen mesh"
        ) from exc
    powder = np.zeros(len(cells), dtype=bool)
    powder[powder_ids] = True

    cut_lo, cut_hi = _box(cut_box)
    root_lo, root_hi = _box(retained_root_box)
    removed = np.all(
        (centroids >= cut_lo[None, :])
        & (centroids <= cut_hi[None, :]),
        axis=1,
    ) & (~powder)
    retained_root = np.all(
        (centroids >= root_lo[None, :])
        & (centroids <= root_hi[None, :]),
        axis=1,
    ) & (~powder)
    if not np.any(removed) or not np.any(retained_root):
        raise ValueError("release selection produced an empty removed/root set")
    if np.any(removed & retained_root):
        raise ValueError("release removed and retained-root selections overlap")
    support_band = (
        (centroids[:, 2] >= min(cut_lo[2], root_lo[2]))
        & (centroids[:, 2] <= max(cut_hi[2], root_hi[2]))
        & (~powder)
    )
    if not np.array_equal(removed | retained_root, support_band):
        raise ValueError(
            "release removed/root sets must partition every non-powder "
            "support-band cell"
        )

    removed_b64, removed_mask_sha = _packed_mask(removed)
    root_b64, root_mask_sha = _packed_mask(retained_root)
    anchor_protocol = _anchor_protocol(points, cells, retained_root)
    return {
        "schema_version": "kaess.release-cellset/1",
        "protocol_id": "kaess-2023-public-v1",
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "mesh_num_cells": int(len(cells)),
        "cell_id_basis": "solver_zero_based",
        "source_class": "inferred",
        "source_locator": (
            "Kaess et al. 2023 Figure 7 registered to the frozen reconstructed "
            "mesh; W1/W2 are removed and W3 remains attached"
        ),
        "claim_boundary": (
            "approved registered assumption; author element labels remain "
            "unavailable and release-set sensitivity is mandatory"
        ),
        "selection_provenance": {
            "generator": "cases/kaess_2023/make_release_cellset.py",
            "cut_box_m": [float(value) for value in cut_box],
            "retained_root_box_m": [
                float(value) for value in retained_root_box
            ],
            "excluded_powder_elset": powder_elset,
        },
        "removed_cell_mask_encoding": "numpy.packbits(bitorder=little)",
        "removed_cell_mask_base64": removed_b64,
        "removed_cell_mask_sha256": removed_mask_sha,
        "removed_cell_ids_sha256": _ids_sha256(removed),
        "expected_removed_count": int(np.sum(removed)),
        "retained_root_cell_mask_encoding": (
            "numpy.packbits(bitorder=little)"
        ),
        "retained_root_cell_mask_base64": root_b64,
        "retained_root_cell_mask_sha256": root_mask_sha,
        "expected_retained_root_count": int(np.sum(retained_root)),
        "anchor_protocol": anchor_protocol,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cut-box",
        type=float,
        nargs=6,
        default=(0.0, 7.0e-4, 0.0, 5.0e-4, 0.0, 2.999e-4),
    )
    parser.add_argument(
        "--retained-root-box",
        type=float,
        nargs=6,
        default=(7.75e-4, 9.75e-4, 0.0, 5.0e-4, 0.0, 2.999e-4),
    )
    parser.add_argument("--powder-elset", default="POWDER")
    args = parser.parse_args(argv)

    document = build_document(
        args.mesh,
        cut_box=args.cut_box,
        retained_root_box=args.retained_root_box,
        powder_elset=args.powder_elset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: removed={document['expected_removed_count']} "
        f"retained_root={document['expected_retained_root_count']}"
    )


if __name__ == "__main__":
    main()
