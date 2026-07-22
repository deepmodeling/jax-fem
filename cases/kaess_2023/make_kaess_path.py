#!/usr/bin/env python3
"""Scan path for the Kaess 2023 benchmark phase 2 (moving heat source).

Reference process (paper section 2.6-2.8): linear hatch pattern rotated 67
degrees per layer, hatch spacing 100 um, beam radius 50 um, scan speed default
850 mm/s, manufactured area 1.0 x 0.5 mm, 10 layers of 30 um (or 5 x 60 um).
The absolute start angle of layer 1 is NOT stated in the text (Figure 6);
0 degrees (along x) is assumed and documented.

Output CSV columns match the v03 --path-file schema:
  time,x,y,z,power,laser_on,layer,hatch,mode[,front_coord,scan_id]
Units: meters/seconds/watts (path-length-scale 1). Laser-off jump rows connect
consecutive tracks so the time axis stays strictly increasing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as onp

AREA_X = 1.0e-3
AREA_Y = 0.5e-3
Z_BASE = 0.3e-3  # support block top = bottom of first layer


def hatch_lines(angle_rad, hatch, cx=AREA_X / 2, cy=AREA_Y / 2):
    """Parallel scan vectors at `angle_rad` covering the rectangle, clipped."""
    d = onp.array([onp.cos(angle_rad), onp.sin(angle_rad)])  # scan direction
    n = onp.array([-d[1], d[0]])                             # hatch normal
    corners = onp.array([[0, 0], [AREA_X, 0], [0, AREA_Y], [AREA_X, AREA_Y]])
    offs = (corners - [cx, cy]) @ n
    lines = []
    for o in onp.arange(offs.min() + hatch / 2, offs.max(), hatch):
        p0 = onp.array([cx, cy]) + o * n
        ts = []
        for lo, hi, comp in ((0.0, AREA_X, 0), (0.0, AREA_Y, 1)):
            if abs(d[comp]) > 1e-12:
                ts.extend([(lo - p0[comp]) / d[comp], (hi - p0[comp]) / d[comp]])
        ts = sorted(t for t in ts
                    if onp.all((p0 + t * d >= -1e-12))
                    and onp.all((p0 + t * d <= [AREA_X + 1e-12, AREA_Y + 1e-12])))
        if len(ts) >= 2 and ts[-1] - ts[0] > 1e-9:
            lines.append((p0 + ts[0] * d, p0 + ts[-1] * d))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--layer-thickness", type=float, default=30.0e-6)
    ap.add_argument("--power", type=float, default=250.0)
    ap.add_argument("--speed", type=float, default=0.850)  # m/s
    ap.add_argument("--hatch", type=float, default=100.0e-6)
    ap.add_argument("--sample-step", type=float, default=25.0e-6,
                    help="path discretization along track (beam r/2 default)")
    ap.add_argument("--jump-speed", type=float, default=5.0)
    ap.add_argument("--rotation-deg", type=float, default=67.0)
    ap.add_argument("--start-angle-deg", type=float, default=46.0,
                    help="layer-1 hatch angle 46 deg, back-solved from Figure 6 "
                         "(layer 3/8 is exactly horizontal: 46+2*67=180=0)")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "kaess_path_10x30um.csv")
    args = ap.parse_args()

    rows = []
    t = 0.0
    prev_end = None
    for layer in range(1, args.layers + 1):
        z = Z_BASE + layer * args.layer_thickness
        angle = onp.deg2rad(args.start_angle_deg
                            + (layer - 1) * args.rotation_deg)
        for hatch_idx, (p0, p1) in enumerate(hatch_lines(angle, args.hatch),
                                             start=1):
            if hatch_idx % 2 == 0:
                p0, p1 = p1, p0  # serpentine: alternate direction (Fig 6)
            if prev_end is not None:
                jump = float(onp.linalg.norm(
                    onp.append(p0, z) - prev_end
                ))
                t += max(jump / args.jump_speed, 1.0e-5)
                rows.append((t, p0[0], p0[1], z, 0.0, 0, layer, hatch_idx,
                             "scan"))
            length = float(onp.linalg.norm(p1 - p0))
            n_seg = max(int(onp.ceil(length / args.sample_step)), 1)
            for s in range(1, n_seg + 1):
                frac = s / n_seg
                p = p0 + frac * (p1 - p0)
                t += (length / n_seg) / args.speed
                rows.append((t, p[0], p[1], z, args.power, 1, layer, hatch_idx,
                             "scan"))
            prev_end = onp.array([p1[0], p1[1], z])

    with open(args.output, "w") as f:
        f.write("time,x,y,z,power,laser_on,layer,hatch,mode,front_coord\n")
        for (tt, x, y, z, p, on, layer, hatch_idx, mode) in rows:
            f.write(f"{tt:.9f},{x:.9e},{y:.9e},{z:.9e},{p},{on},"
                    f"{layer},{hatch_idx},{mode},{z:.9e}\n")

    n_on = sum(1 for r in rows if r[5] == 1)
    print(f"wrote {args.output}: {len(rows)} rows ({n_on} laser-on), "
          f"{args.layers} layers, t_end={t*1e3:.2f} ms")


if __name__ == "__main__":
    main()
