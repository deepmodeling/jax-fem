#!/usr/bin/env python3
"""Post-run analysis for a Kaess phase2 run: locking fingerprint at the
accumulated states (last scan / last cooling / release), front-end release
deflection, run_audit gate summary, and profile.json breakdown.

Usage: python post_smoke_analysis.py <run_dir> [<report_path>]
Writes a markdown report (default: <run_dir>/POST_RUN_REPORT.md).
"""
import glob
import json
import os
import re
import sys

import meshio
import numpy as np

BEAM_BASE_Z = 3.0e-4          # support top / beam bottom
FRONT_X_MAX = 1.0e-4          # "front end" = x < 0.1 mm (free end)


def quad_mean(m, pattern):
    pat = re.compile(pattern)
    arrs = [np.asarray(m.cell_data[n][0]) for n in m.cell_data if pat.match(n)]
    return np.mean(arrs, axis=0) if arrs else None


def fingerprint(path):
    m = meshio.read(path)
    block = next(iter(m.cells_dict))
    cells = m.cells_dict[block]
    centroids = m.points[cells].mean(axis=1)
    sxx = quad_mean(m, r"^stress_quad\d+_xx$")
    syy = quad_mean(m, r"^stress_quad\d+_yy$")
    szz = quad_mean(m, r"^stress_quad\d+_zz$")
    vm = quad_mean(m, r"^vm_quad\d+$")
    if sxx is None:
        return None
    p = (sxx + syy + szz) / 3.0
    active = quad_mean(m, r"^active$")
    mask = np.ones(len(p), dtype=bool)
    if active is not None:
        mask &= active > 0.5
    mask &= vm > 5e6
    layer = np.round(centroids[:, 2] / 2.5e-5).astype(int)

    def dstd(v):
        out = v[mask].astype(float).copy()
        g = layer[mask]
        for gv in np.unique(g):
            mm = g == gv
            out[mm] -= out[mm].mean()
        return float(np.std(out))

    ps, vs = dstd(p), dstd(vm)
    return {
        "cells_used": int(mask.sum()),
        "p_std_MPa": ps / 1e6,
        "vm_std_MPa": vs / 1e6,
        "ratio": ps / max(vs, 1.0),
        "p_min_MPa": float(p[mask].min()) / 1e6,
        "p_max_MPa": float(p[mask].max()) / 1e6,
        "vm_max_MPa": float(vm[mask].max()) / 1e6,
    }


def release_deflection(path):
    m = meshio.read(path)
    if "u" not in m.point_data:
        return None
    u = np.asarray(m.point_data["u"])
    pts = m.points
    beam = pts[:, 2] > BEAM_BASE_Z + 1e-9
    front = beam & (pts[:, 0] < FRONT_X_MAX)
    root = beam & (pts[:, 0] > 9.0e-4)
    return {
        "front_uz_mean_um": float(u[front, 2].mean()) * 1e6,
        "front_uz_max_um": float(u[front, 2].max()) * 1e6,
        "front_uz_min_um": float(u[front, 2].min()) * 1e6,
        "root_uz_mean_um": float(u[root, 2].mean()) * 1e6,
        "front_nodes": int(front.sum()),
    }


def main():
    run_dir = sys.argv[1]
    report_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        run_dir, "POST_RUN_REPORT.md")
    lines = [f"# Post-run report: {os.path.basename(run_dir)}", ""]

    # --- locking fingerprint at accumulated states
    lines.append("## Locking fingerprint (in-layer demeaned p-std / vm-std)")
    lines.append("")
    lines.append("| frame | cells | p std (MPa) | vm std (MPa) | RATIO | p range (MPa) | vm max |")
    lines.append("|---|---|---|---|---|---|---|")
    scans = sorted(glob.glob(os.path.join(run_dir, "step_*_scan.vtu")))
    cools = sorted(glob.glob(os.path.join(run_dir, "step_*_cooling.vtu")))
    frames = []
    if scans:
        frames.append(("last_scan " + os.path.basename(scans[-1]), scans[-1]))
    if cools:
        frames.append(("last_cooling " + os.path.basename(cools[-1]), cools[-1]))
    rel = os.path.join(run_dir, "release.vtu")
    if os.path.exists(rel):
        frames.append(("release", rel))
    for label, f in frames:
        r = fingerprint(f)
        if r is None:
            lines.append(f"| {label} | (no stress arrays) | | | | | |")
            continue
        lines.append(
            f"| {label} | {r['cells_used']} | {r['p_std_MPa']:.1f} | "
            f"{r['vm_std_MPa']:.1f} | **{r['ratio']:.2f}** | "
            f"[{r['p_min_MPa']:.0f}, {r['p_max_MPa']:.0f}] | {r['vm_max_MPa']:.0f} |")
    lines.append("")
    lines.append("Reference (locked TET4, 10-layer T150C): last_cooling 8.40, release 9.31; healthy = O(1).")
    lines.append("")

    # --- release deflection
    if os.path.exists(rel):
        d = release_deflection(rel)
        if d:
            lines.append("## Release front-end deflection (sign evidence only - "
                         "2-layer beam is NOT comparable to the paper's 10-layer +14 um)")
            lines.append("")
            lines.append(f"- front (x<0.1mm, beam) uz: mean {d['front_uz_mean_um']:+.2f} um, "
                         f"range [{d['front_uz_min_um']:+.2f}, {d['front_uz_max_um']:+.2f}] um "
                         f"({d['front_nodes']} nodes)")
            lines.append(f"- root (x>0.9mm, beam) uz: mean {d['root_uz_mean_um']:+.2f} um")
            lines.append("")

    # --- run audit
    audit = os.path.join(run_dir, "v06_run_audit.json")
    if os.path.exists(audit):
        with open(audit) as f:
            a = json.load(f)
        lines.append("## run_audit gates")
        lines.append("")
        lines.append("```json")
        keep = {k: v for k, v in a.items() if isinstance(v, (bool, int, float, str))}
        lines.append(json.dumps(keep, indent=1, default=str)[:2000])
        lines.append("```")
        lines.append("")

    # --- profile breakdown
    prof = os.path.join(run_dir, "profile.json")
    if os.path.exists(prof):
        with open(prof) as f:
            p = json.load(f)
        lines.append("## profile.json (top-level timings)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(p, indent=1, default=str)[:4000])
        lines.append("```")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {report_path}")
    print("\n".join(lines[:24]))


if __name__ == "__main__":
    main()
