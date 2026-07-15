#!/usr/bin/env python3
"""Summarize a k_p sensitivity ladder produced by run_kp_ladder.sh.

For each kp_* run directory under the ladder root, report:

  - release |u| max / p99 (part-scale distortion QoI)
  - release von Mises p50/p90 (field-level QoI; the global max is NOT
    reported on the raw mesh -- sliver cells make it unquotable, see the
    v06 mesh audit)
  - plastic footprint: quadrature points with eqp above threshold
  - build-phase sentinel: max u_max over driver log summary lines
    (linear-spring validity indicator)

Usage: python analyze_kp_ladder.py <ladder_root> [--eqp-threshold 1e-8]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as onp

try:
    import meshio
except ImportError:  # pragma: no cover
    meshio = None

U_MAX_RE = re.compile(r"u_max=([0-9.eE+-]+)")


def _finite_percentiles(values, percentiles):
    values = onp.asarray(values, dtype=float).ravel()
    values = values[onp.isfinite(values)]
    if values.size == 0:
        return {p: float("nan") for p in percentiles}
    return {p: float(onp.percentile(values, p)) for p in percentiles}


def _first_key(data, *candidates):
    for key in candidates:
        if key in data:
            return key
    return None


def summarize_run(run_dir: Path, eqp_threshold: float) -> dict:
    out = {"dir": run_dir.name, "status": "missing"}
    release = run_dir / "release.vtu"
    if meshio is None:
        out["status"] = "meshio_unavailable"
        return out
    if not release.is_file():
        return out
    mesh = meshio.read(release)
    out["status"] = "ok"

    u_key = _first_key(mesh.point_data, "u", "sol", "displacement")
    if u_key is not None:
        u = onp.asarray(mesh.point_data[u_key], dtype=float)
        u_norm = onp.linalg.norm(u, axis=1)
        pct = _finite_percentiles(u_norm, (99,))
        out["release_u_max"] = float(onp.nanmax(u_norm))
        out["release_u_p99"] = pct[99]

    # von Mises lives as per-quadrature cell fields (vm_quad0..3); pool all
    # quad values for the percentiles.
    vm_chunks = []
    vm_names = []
    for name, data in (mesh.cell_data or {}).items():
        if name.lower().startswith("vm"):
            vm_names.append(name)
            vm_chunks.extend(
                onp.asarray(block, dtype=float).ravel() for block in data
            )
    vm_data = onp.concatenate(vm_chunks) if vm_chunks else None
    if vm_data is None and (key := _first_key(mesh.point_data, "vm", "von_mises")):
        vm_data = onp.asarray(mesh.point_data[key], dtype=float).ravel()
        vm_names = [key]
    if vm_data is not None:
        pct = _finite_percentiles(vm_data, (50, 90))
        out["release_vm_p50"] = pct[50]
        out["release_vm_p90"] = pct[90]
        out["vm_field"] = ",".join(sorted(vm_names))

    # Equivalent plastic strain: 'eq_plastic_strain' cell field in the v05
    # output convention (match 'plastic'/'eqp' to stay robust to renames).
    eqp_values = None
    for source in (mesh.point_data, mesh.cell_data or {}):
        for name, data in source.items():
            lowered = name.lower()
            if "eqp" in lowered or "plastic" in lowered:
                if isinstance(data, list):
                    data = onp.concatenate(
                        [onp.asarray(b, dtype=float).ravel() for b in data]
                    )
                eqp_values = onp.asarray(data, dtype=float).ravel()
                out["eqp_field"] = name
                break
        if eqp_values is not None:
            break
    if eqp_values is not None:
        out["plastic_points"] = int(onp.sum(eqp_values > eqp_threshold))
        out["eqp_max"] = float(onp.nanmax(eqp_values))

    return out


def build_sentinel(console_log: Path) -> float:
    if not console_log.is_file():
        return float("nan")
    peak = float("nan")
    for line in console_log.read_text(errors="replace").splitlines():
        match = U_MAX_RE.search(line)
        if match:
            value = float(match.group(1))
            peak = value if not (peak == peak) else max(peak, value)
    return peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder_root", type=Path)
    ap.add_argument("--eqp-threshold", type=float, default=1e-8)
    args = ap.parse_args()

    root = args.ladder_root
    runs = sorted(
        (d for d in root.glob("kp_*") if d.is_dir()),
        key=lambda d: float(d.name.removeprefix("kp_")),
    )
    if not runs:
        print(f"no kp_* run dirs under {root}", file=sys.stderr)
        return 1

    rows = []
    for run_dir in runs:
        kp = float(run_dir.name.removeprefix("kp_"))
        summary = summarize_run(run_dir, args.eqp_threshold)
        summary["kp"] = kp
        summary["build_u_max_sentinel"] = build_sentinel(
            root / f"{run_dir.name}_console.log"
        )
        rows.append(summary)

    (root / "ladder_summary.json").write_text(json.dumps(rows, indent=1))

    def fmt(row, key, spec="{:.4e}"):
        value = row.get(key)
        if value is None or value != value:
            return "-"
        if isinstance(value, int):
            return str(value)
        return spec.format(value)

    header = (
        f"{'kp':>8} {'status':>8} {'rel_u_max':>12} {'rel_u_p99':>12} "
        f"{'vm_p50':>12} {'vm_p90':>12} {'plastic_pts':>11} "
        f"{'eqp_max':>10} {'build_u_sent':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['kp']:>8.0e} {row['status']:>8} "
            f"{fmt(row, 'release_u_max'):>12} {fmt(row, 'release_u_p99'):>12} "
            f"{fmt(row, 'release_vm_p50'):>12} {fmt(row, 'release_vm_p90'):>12} "
            f"{fmt(row, 'plastic_points'):>11} {fmt(row, 'eqp_max'):>10} "
            f"{fmt(row, 'build_u_max_sentinel'):>12}"
        )

    baseline = next((r for r in rows if r["kp"] == 0.0 and r["status"] == "ok"), None)
    if baseline and baseline.get("release_u_max"):
        print("\nrelative drift vs kp=0 baseline (release_u_max):")
        for row in rows:
            if row["kp"] == 0.0 or row.get("release_u_max") is None:
                continue
            drift = (
                row["release_u_max"] - baseline["release_u_max"]
            ) / baseline["release_u_max"]
            print(f"  kp={row['kp']:.0e}: {drift:+.3%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
