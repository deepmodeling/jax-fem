#!/usr/bin/env python3
"""Single-track scan path for V1 (CBM case B: 195 W, 800 mm/s).

CSV schema matches the v03 --path-file reader (same as kaess):
  time,x,y,z,power,laser_on,layer,hatch,mode,front_coord
Units m/s/W.

Track: along +x at domain mid-width (y = 0.24 mm), z = layer top (0.30 mm),
from x = 0.05 mm to x = 0.95 mm (0.05 mm margins at both ends; Balbaa does
not state the track start/end -> registered D-V1-17). Case parameters from
inputs/nist-meltpool.json triangle_anchor (D-V1-03 resolution).
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--power", type=float, default=195.0)
ap.add_argument("--speed", type=float, default=0.800)      # m/s
ap.add_argument("--x0", type=float, default=0.05e-3)
ap.add_argument("--x1", type=float, default=0.95e-3)
ap.add_argument("--y", type=float, default=0.24e-3)
ap.add_argument("--z", type=float, default=0.30e-3)
ap.add_argument("--sample-step", type=float, default=12.5e-6)  # r/4
ap.add_argument("--output", type=Path,
                default=Path(__file__).parent / "v1_path_cbmB.csv")
args = ap.parse_args()

length = args.x1 - args.x0
n_seg = max(int(round(length / args.sample_step)), 1)
rows = [(0.0, args.x0, args.y, args.z, args.power, 1, 1, 1, "scan")]
t = 0.0
for s in range(1, n_seg + 1):
    frac = s / n_seg
    t = frac * length / args.speed
    rows.append((t, args.x0 + frac * length, args.y, args.z,
                 args.power, 1, 1, 1, "scan"))

with open(args.output, "w") as f:
    f.write("time,x,y,z,power,laser_on,layer,hatch,mode,front_coord\n")
    for (tt, x, y, z, p, on, layer, hatch, mode) in rows:
        f.write(f"{tt:.9f},{x:.9e},{y:.9e},{z:.9e},{p},{on},"
                f"{layer},{hatch},{mode},{z:.9e}\n")

print(f"wrote {args.output}: {len(rows)} rows, track {length*1e3:.2f} mm, "
      f"t_end = {t*1e3:.4f} ms at {args.speed*1e3:.0f} mm/s, P = {args.power} W")
