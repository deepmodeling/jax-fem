"""Reproducible JSON audit for tetrahedral simulation meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .mesh_quality import audit_tet_mesh


def _percentiles(values):
    levels = (0, 1, 5, 50, 95, 99, 100)
    return {str(level): float(np.percentile(values, level)) for level in levels}


def summarize_tet_mesh(points, cells, *, thresholds=(0.05, 0.1, 0.2), worst_count=10):
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    if len(cells) == 0:
        raise ValueError("mesh must contain at least one tetrahedral cell")
    report = audit_tet_mesh(points, cells)
    thresholds = tuple(float(value) for value in thresholds)
    order = np.argsort(report.mean_ratio)
    worst = []
    for cell_id in order[: max(int(worst_count), 0)]:
        node_ids = cells[cell_id]
        coordinates = points[node_ids]
        worst.append(
            {
                "cell_id": int(cell_id),
                "node_ids": [int(value) for value in node_ids],
                "centroid": [float(value) for value in coordinates.mean(axis=0)],
                "node_coordinates": coordinates.tolist(),
                "mean_ratio": float(report.mean_ratio[cell_id]),
                "edge_ratio": float(report.edge_ratio[cell_id]),
                "volume": float(report.volume[cell_id]),
                "signed_volume": float(report.signed_volume[cell_id]),
            }
        )
    finite_edge_ratio = report.edge_ratio[np.isfinite(report.edge_ratio)]
    return {
        "schema_version": "v06.mesh-audit.1",
        "quality_metric": "6*sqrt(2)*abs(V)/l_rms^3; regular tetrahedron = 1",
        "mesh": {
            "num_points": int(len(points)),
            "num_cells": int(len(cells)),
            "bbox_min": points.min(axis=0).tolist(),
            "bbox_max": points.max(axis=0).tolist(),
            "bbox_span": np.ptp(points, axis=0).tolist(),
        },
        "validity": {
            "inverted_count": report.inverted_count,
            "degenerate_count": report.degenerate_count,
            "valid_for_fem": report.inverted_count == 0
            and report.degenerate_count == 0,
        },
        "volume": {
            "total": float(report.volume.sum()),
            "minimum": float(report.volume.min()),
            "maximum": float(report.volume.max()),
            "percentiles": _percentiles(report.volume),
        },
        "quality": {
            "mean_ratio_percentiles": _percentiles(report.mean_ratio),
            "edge_ratio_percentiles": _percentiles(finite_edge_ratio)
            if len(finite_edge_ratio)
            else {},
            "below_threshold": {
                format(threshold, "g"): int(
                    np.count_nonzero(report.mean_ratio < threshold)
                )
                for threshold in thresholds
            },
            "worst_cells": worst,
        },
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tetra(path):
    import meshio

    mesh = meshio.read(path)
    if "tetra" not in mesh.cells_dict:
        available = ", ".join(sorted(mesh.cells_dict))
        raise ValueError(f"expected tetra cells; available: {available}")
    return np.asarray(mesh.points), np.asarray(mesh.cells_dict["tetra"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2")
    parser.add_argument("--worst-count", type=int, default=10)
    parser.add_argument("--fail-below-quality", type=float)
    args = parser.parse_args(argv)

    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    points, cells = _load_tetra(args.mesh)
    summary = summarize_tet_mesh(
        points, cells, thresholds=thresholds, worst_count=args.worst_count
    )
    summary["source"] = {
        "path": str(args.mesh.resolve()),
        "sha256": _sha256(args.mesh),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "num_cells": summary["mesh"]["num_cells"],
        "minimum_quality": summary["quality"]["mean_ratio_percentiles"]["0"],
        "inverted": summary["validity"]["inverted_count"],
        "degenerate": summary["validity"]["degenerate_count"],
    }, sort_keys=True))
    if not summary["validity"]["valid_for_fem"]:
        return 2
    if args.fail_below_quality is not None:
        count = int(
            np.count_nonzero(
                audit_tet_mesh(points, cells).mean_ratio
                < args.fail_below_quality
            )
        )
        if count:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
