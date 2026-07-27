#!/usr/bin/env python3
"""QoI extraction for the Kaess 2023 benchmark (code-to-code comparison).

Reference QoIs (paper Figures 7-9):
  1. bending line: u_z along the beam top surface (z=0.6 mm) vs x, on the
     RELEASED state (after the partial saw cut) - paper red lines;
  2. residual stress profile: sigma_xx vs depth z at the beam mid-span,
     on the CONSTRAINED state before separation - paper black arrow
     (exact x/y position pending Figure 7 digitization; mid-span assumed);
  3. scalar: maximum deflection at the cantilever front (x=1.0 mm).

Usage:
  python analyze_kaess.py RUN_DIR [RUN_DIR ...] [--json OUT.json]

Multiple run dirs (e.g. a preheat ladder) are reported side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as onp

try:
    import meshio
except ImportError:  # pragma: no cover
    meshio = None

BEAM_Z0 = 0.3e-3
BEAM_Z1 = 0.6e-3
BEAM_X1 = 1.0e-3
BEAM_YMID = 0.25e-3
# mesh v2: root wall at x in [0.775, 0.975] mm; FREE end at x=0.
# Paper coordinates run the other way (x_paper = 0 at the root).

FIG9A_REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "references"
    / "digitized"
    / "fig9_bending.csv"
)
VOLUME_CELL_TYPES = frozenset(("tetra", "hexahedron"))


def load_figure9a_reference(path: Path = FIG9A_REFERENCE_PATH) -> dict[int, float]:
    """Load the one authoritative Figure 9a series from the frozen CSV."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["series_id"] == "plate_temperature_fixed_p250_v850"
        ]
    return {
        int(float(row["build_plate_temp_c"])): float(
            row["max_front_bending_um"]
        )
        for row in rows
    }


FIG9A_REFERENCE_UM = load_figure9a_reference()


def _volume_cell_blocks(mesh):
    blocks = []
    for block_index, block in enumerate(mesh.cells):
        cell_type = block.type
        if cell_type in VOLUME_CELL_TYPES:
            blocks.append(
                (
                    block_index,
                    onp.asarray(block.data, dtype=onp.int64),
                )
            )
    if not blocks:
        supported = ", ".join(sorted(VOLUME_CELL_TYPES))
        raise ValueError(
            f"VTU contains no supported volume cells ({supported})"
        )
    return blocks


def cell_centroids(mesh):
    """Return supported volume-cell centroids in mesh cell-block order."""
    return onp.concatenate(
        [
            onp.asarray(mesh.points, dtype=float)[cells].mean(axis=1)
            for _block_index, cells in _volume_cell_blocks(mesh)
        ],
        axis=0,
    )


def _volume_cell_field(mesh, data_blocks):
    volume_values = []
    for block_index, cells in _volume_cell_blocks(mesh):
        if block_index >= len(data_blocks):
            raise ValueError(
                "cell-data blocks do not align with mesh cell blocks"
            )
        values = onp.asarray(data_blocks[block_index], dtype=float)
        if values.shape[0] != len(cells):
            raise ValueError(
                "cell-data length does not match its volume cell block"
            )
        if values.ndim == 1:
            scalar_values = values
        elif values.ndim == 2 and values.shape[1] == 1:
            scalar_values = values[:, 0]
        else:
            raise ValueError(
                "quadrature cell field must be scalar per cell"
            )
        volume_values.append(scalar_values)
    return onp.concatenate(volume_values, axis=0)


def pooled_quad_field(mesh, prefix, component=None):
    """Average one scalar field across its per-quadrature cell arrays."""
    component_suffix = (
        "" if component is None else f"_{re.escape(component)}"
    )
    field_pattern = re.compile(
        rf"^{re.escape(prefix)}_quad(?P<quad>\d*){component_suffix}$"
    )
    matches = []
    for name in (mesh.cell_data or {}):
        match = field_pattern.fullmatch(name)
        if match:
            matches.append((name, match.group("quad")))
    if not matches:
        return None

    unindexed = [name for name, quad in matches if quad == ""]
    numbered = [
        (int(quad), name)
        for name, quad in matches
        if quad != ""
    ]
    if unindexed and numbered:
        raise ValueError(
            "quadrature fields mix unindexed and numbered names"
        )
    if unindexed:
        ordered_names = unindexed
    else:
        numbered.sort()
        quad_indices = [quad for quad, _name in numbered]
        if quad_indices != list(range(len(numbered))):
            raise ValueError(
                "numbered quadrature fields must start at zero and be "
                "contiguous"
            )
        ordered_names = [name for _quad, name in numbered]

    chunks = [
        _volume_cell_field(mesh, mesh.cell_data[name])
        for name in ordered_names
    ]
    return onp.mean(onp.stack(chunks, axis=0), axis=0)


def bending_line(release_path):
    mesh = meshio.read(release_path)
    key = next((k for k in ("u", "sol", "displacement")
                if k in mesh.point_data), None)
    if key is None:
        return None
    u = onp.asarray(mesh.point_data[key], dtype=float)
    pts = mesh.points
    on_top = (
        (onp.abs(pts[:, 2] - BEAM_Z1) < 1.0e-6)
        & (onp.abs(pts[:, 1] - BEAM_YMID) < 13.0e-6)
    )
    order = onp.argsort(pts[on_top, 0])
    u_line = u[on_top, 2][order]
    x_line = pts[on_top, 0][order]
    # mesh v2: free end at x=0, root at large x -> deflection relative to
    # the root end (paper plots bending starting at 0 over the root)
    u_rel = u_line - u_line[-1]
    return {
        "x_m": x_line.tolist(),
        "u_z_m": u_line.tolist(),
        "front_deflection_m": float(u_rel[0]),
        "max_deflection_m": float(onp.max(u_rel)),
    }


def stress_depth_profile(constrained_path, x_pos=0.475e-3):
    # default x_pos = center of support wall W2 (paper's black-arrow path
    # position is between mid-span and the root; Fig 7 perspective gives
    # ~0.3-0.5 mm from the root -> W2 center chosen, documented estimate)
    mesh = meshio.read(constrained_path)
    sxx = pooled_quad_field(mesh, "stress", component="xx")
    if sxx is None:
        return None
    cent = cell_centroids(mesh)
    if sxx.shape != (len(cent),):
        raise ValueError(
            "sigma_xx cell count does not match volume-cell centroids"
        )
    near = (
        (onp.abs(cent[:, 0] - x_pos) < 26.0e-6)
        & (onp.abs(cent[:, 1] - BEAM_YMID) < 26.0e-6)
    )
    z = cent[near, 2]
    order = onp.argsort(z)[::-1]  # top -> down (paper: negative z direction)
    return {
        "z_m": z[order].tolist(),
        "sigma_xx_pa": sxx[near][order].tolist(),
        "x_pos_m": x_pos,
    }


def summarize(run_dir: Path) -> dict:
    out = {"dir": str(run_dir), "status": "missing"}
    release = run_dir / "release.vtu"
    coolings = sorted(run_dir.glob("step_*_cooling.vtu"))
    if meshio is None:
        out["status"] = "meshio_unavailable"
        return out
    if release.is_file():
        out["bending"] = bending_line(release)
        out["status"] = "ok"
    if coolings:
        out["stress_profile_constrained"] = stress_depth_profile(coolings[-1])
        out["constrained_vtu"] = coolings[-1].name
    cfg = run_dir / "used_config.json"
    if cfg.is_file():
        c = json.loads(cfg.read_text())
        out["plate_temp_k"] = c.get("ambient")
        out["laser_power_w"] = c.get("laser_power")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rows = [summarize(d) for d in args.run_dirs]
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))

    print(f"{'run':>40} {'T_plate[K]':>10} {'front u_z[um]':>14} "
          f"{'Fig9a[um]':>10} {'ratio':>6} "
          f"{'sxx_top[MPa]':>12} {'sxx_min[MPa]':>12}")
    for r in rows:
        b = r.get("bending") or {}
        s = r.get("stress_profile_constrained") or {}
        sx = onp.asarray(s.get("sigma_xx_pa", [onp.nan]))
        front_um = 1e6 * (b.get("front_deflection_m") or float("nan"))
        tk = r.get("plate_temp_k") or float("nan")
        ref = FIG9A_REFERENCE_UM.get(round(tk - 273.15), float("nan"))
        ratio = front_um / ref if ref == ref and ref else float("nan")
        print(f"{Path(r['dir']).name[:40]:>40} "
              f"{tk:>10.1f} "
              f"{front_um:>14.3f} "
              f"{ref:>10.1f} {ratio:>6.2f} "
              f"{1e-6 * (sx[0] if sx.size else float('nan')):>12.2f} "
              f"{1e-6 * float(onp.nanmin(sx)) if sx.size else float('nan'):>12.2f}")

    # trend gates (paper anchors, see kaess_2023.json reported_anchors)
    ok_rows = [r for r in rows if (r.get("bending") or {}).get("front_deflection_m") is not None]
    if len(ok_rows) >= 2 and all(r.get("plate_temp_k") for r in ok_rows):
        srt = sorted(ok_rows, key=lambda r: r["plate_temp_k"])
        defl = [r["bending"]["front_deflection_m"] for r in srt]
        upward = all(d > 0 for d in defl)
        monotone_down = all(defl[i] >= defl[i + 1] for i in range(len(defl) - 1))
        print(f"\nanchor 'cantilever_deflects_upward_after_cut': "
              f"{'PASS' if upward else 'FAIL'}")
        print(f"anchor 'preheat_reduces_residual_stress' (deflection proxy, "
              f"monotone decreasing): {'PASS' if monotone_down else 'FAIL'}")


if __name__ == "__main__":
    main()
