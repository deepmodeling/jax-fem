#!/usr/bin/env python3
"""M2: multi-layer cube mesh generator, Balbaa2022 Sec 2.6.3 / Fig 4 (SPEC.md).

Part 10 x 10 x 10 mm @ uniform 200 um (his DC3D8 200 um lumped-layer mesh)
centered on a substrate whose extent is figure-derived (D-V2-07, default
30 x 30 x 6 mm). Balbaa's substrate uses ~2 mm elements + tie constraint; our
runner needs ONE conforming mesh, so the default meshes the substrate at
200 um too (~800k cells) -- the final variant is an L3-time decision
(D-V2-07). Use --count-only to size alternatives without writing.
Units: meters, build axis z. C3D8.
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=float, default=200.0e-6, help="uniform cell size")
ap.add_argument("--part-xy", type=float, default=10.0e-3)
ap.add_argument("--part-z", type=float, default=10.0e-3)
ap.add_argument("--sub-xy", type=float, default=30.0e-3, help="substrate footprint (D-V2-07)")
ap.add_argument("--sub-z", type=float, default=6.0e-3, help="substrate thickness (D-V2-07)")
ap.add_argument("--sub-grading", type=float, default=None, metavar="R",
                help="geometric z-grading ratio for the substrate (e.g. 1.4): "
                     "cells start at --res under the part and grow toward the "
                     "bottom until --sub-z is filled; bottom cells reach the "
                     "~2 mm scale of Balbaa's substrate mesh while staying "
                     "conforming in-plane. None = uniform --res.")
ap.add_argument("--count-only", action="store_true")
ap.add_argument("--output", type=Path, default=Path(__file__).parent / "v2_cube_c3d8.inp")
args = ap.parse_args()

r = args.res
nsx = round(args.sub_xy / r)
npx = round(args.part_xy / r)
npz = round(args.part_z / r)
off = (nsx - npx) // 2  # part centered on the substrate footprint

# substrate z ladder, bottom->top (top cell = --res, conforming to the part)
if args.sub_grading:
    ladder = [r]
    while sum(ladder) < args.sub_z - 1e-12:
        ladder.append(min(ladder[-1] * args.sub_grading, args.sub_z - sum(ladder)))
    sub_dz = sorted(ladder, reverse=True)               # coarse at bottom
    scale = args.sub_z / sum(sub_dz)
    sub_dz = [d * scale for d in sub_dz]
else:
    sub_dz = [r] * round(args.sub_z / r)
nsz = len(sub_dz)

n_sub = nsx * nsx * nsz
n_part = npx * npx * npz
print(f"substrate: {nsx} x {nsx} x {nsz} = {n_sub:,} cells"
      + (f" (z ladder um: {[round(d*1e6) for d in sub_dz]})" if args.sub_grading else ""))
print(f"part:      {npx} x {npx} x {npz} = {n_part:,} cells (columns offset {off})")
print(f"total:     {n_sub + n_part:,} cells @ {r*1e6:.0f} um")
print(f"runner hint: --support-thickness {args.sub_z:.1e} --layer-thickness 2.0e-4 "
      f"--layers {npz} (5:1 lumped; D-V2-11 for event-series granularity)")
if args.count_only:
    raise SystemExit(0)

# Node grid: substrate occupies full footprint for k in [0, nsz]; above the
# substrate only the part's column block continues. To keep node ids simple,
# use the full (nsx+1)^2 grid at every level and simply omit elements outside
# the part above the substrate (orphan nodes are dropped by a compaction pass).
NXY = nsx + 1
NLEV = nsz + npz + 1
Z_LEVELS = [0.0]
for dz in sub_dz:
    Z_LEVELS.append(Z_LEVELS[-1] + dz)
for _ in range(npz):
    Z_LEVELS.append(Z_LEVELS[-1] + r)

def nid(i, j, k):
    return 1 + i + j * NXY + k * NXY * NXY

used = {}
elements = []
def emit(i, j, k):
    plane = NXY * NXY
    n = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k)]
    n += [x + plane for x in n]
    for x in n:
        used.setdefault(x, None)
    elements.append(n)

for k in range(nsz):
    for j in range(nsx):
        for i in range(nsx):
            emit(i, j, k)
for k in range(nsz, nsz + npz):
    for j in range(off, off + npx):
        for i in range(off, off + npx):
            emit(i, j, k)

for new, old in enumerate(sorted(used), start=1):
    used[old] = new

with open(args.output, "w") as f:
    f.write("*HEADING\n")
    f.write(f"V2 multi-layer cube Balbaa2022-parity mesh: {args.part_xy*1e3:.0f}mm cube @"
            f"{r*1e6:.0f}um on {args.sub_xy*1e3:.0f}x{args.sub_xy*1e3:.0f}x{args.sub_z*1e3:.0f}mm"
            " substrate (D-V2-07). Units m, build axis z.\n")
    f.write("*NODE\n")
    for old in sorted(used):
        rem = old - 1
        k, rem = divmod(rem, NXY * NXY)
        j, i = divmod(rem, NXY)
        f.write(f"{used[old]}, {i*r:.10e}, {j*r:.10e}, {Z_LEVELS[k]:.10e}\n")
    f.write("*ELEMENT, TYPE=C3D8\n")
    for eid, n in enumerate(elements, start=1):
        f.write(f"{eid}, " + ", ".join(str(used[x]) for x in n) + "\n")

print(f"wrote {args.output}: {len(used):,} nodes, {len(elements):,} elements")
