#!/usr/bin/env python3
"""Structured TET4 mesh for the Kaess 2023 cantilever benchmark.

Geometry v2 (Figure 3/5/7 of the paper, digitized 2026-07-16):

  - beam (manufactured area): 1.0 x 0.5 x 0.3 mm, 10 layers of 30 um
  - support: THREE walls (full y-depth, 0.3 mm tall) under the beam,
    powder between them (unmeshed - v06 void semantics). Wall x-ranges in
    beam coordinates (x=0 at the free/front end, root at x=1.0), estimated
    by element counting in Fig 3/5 (+-1 element = 25 um):
      W1 [0.025, 0.125]  (4 columns)
      W2 [0.425, 0.525]  (4 columns)
      W3 [0.775, 0.975]  (8 columns - the wide ROOT wall, stays attached
                          after the saw cut per Fig 7)
  - in-plane resolution 25 um; z = 12 x 25 um (support) + 10 x 30 um (build)
  - lateral powder margins (0.1 mm) are NOT meshed.

Units are meters (v03 mesh convention). Build axis: z.
Output: kaess_cantilever_c3d4.inp (ABAQUS format, C3D4 only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Kuhn (Freudenthal) 6-tet decomposition: every tet contains the main
# diagonal v0->v6 ((0,0,0)->(1,1,1) in VTK corner ordering). Applied with the
# SAME orientation to every hex of a structured grid, all shared faces get
# matching diagonals, so the tet mesh is globally conforming. (A naive split
# is mirror-inconsistent in one axis and silently disconnects the mesh -
# observed as an exactly singular stiffness matrix.)
HEX_TO_TETS = (
    (0, 1, 2, 6),
    (0, 1, 5, 6),
    (0, 3, 2, 6),
    (0, 3, 7, 6),
    (0, 4, 5, 6),
    (0, 4, 7, 6),
)


# support wall x-ranges in meters (beam coords, x=0 = free end)
SUPPORT_WALLS = ((0.025e-3, 0.125e-3),
                 (0.425e-3, 0.525e-3),
                 (0.775e-3, 0.975e-3))


def build_mesh(nx=40, ny=20, support_layers=12, build_layers=10,
               dx=25.0e-6, dy=25.0e-6, dz_support=25.0e-6, dz_build=30.0e-6,
               walls=SUPPORT_WALLS, powder_fill=False):
    """powder_fill=True meshes the inter-wall gaps of the support region and
    reports their (1-based) element ids so they can be written as a POWDER
    elset. The reference model (Kaess 2023) meshes powder as a weak solid;
    lateral margins remain unmeshed (documented deviation)."""
    xs = np.arange(nx + 1) * dx
    ys = np.arange(ny + 1) * dy
    z_support = np.arange(support_layers + 1) * dz_support
    z_build = z_support[-1] + np.arange(1, build_layers + 1) * dz_build
    zs = np.concatenate([z_support, z_build])
    nz = len(zs) - 1

    nid = lambda i, j, k: (k * (ny + 1) + j) * (nx + 1) + i + 1  # 1-based
    points = np.array(
        [[xs[i], ys[j], zs[k]]
         for k in range(nz + 1)
         for j in range(ny + 1)
         for i in range(nx + 1)],
        dtype=np.float64,
    )

    def in_wall(i):
        xc = (xs[i] + xs[i + 1]) / 2.0
        return any(lo - 1e-9 < xc < hi + 1e-9 for lo, hi in walls)

    tets = []
    powder_eids = []
    for k in range(nz):
        support_region = k < support_layers
        for j in range(ny):
            for i in range(nx):
                gap = support_region and not in_wall(i)
                if gap and not powder_fill:
                    continue  # powder gap between walls: unmeshed (void)
                corners = (
                    nid(i, j, k), nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                )
                for tet in HEX_TO_TETS:
                    tets.append(tuple(corners[c] for c in tet))
                    if gap:
                        powder_eids.append(len(tets))  # 1-based element id

    # compact node numbering: keep only referenced nodes
    used = np.unique(np.asarray(tets).reshape(-1))
    remap = np.zeros(int(used.max()) + 1, dtype=np.int64)
    remap[used] = np.arange(1, len(used) + 1)
    points = points[used - 1]
    tets = [tuple(int(remap[n]) for n in t) for t in tets]
    return points, tets, powder_eids


def signed_volumes(points, tets):
    p = points[np.asarray(tets) - 1]
    a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "kaess_cantilever_c3d4.inp")
    ap.add_argument("--powder-fill", action="store_true",
                    help="mesh the inter-wall support gaps and emit them as "
                         "an *ELSET, ELSET=POWDER block (weak-solid powder)")
    args = ap.parse_args()

    points, tets, powder_eids = build_mesh(powder_fill=args.powder_fill)
    vols = signed_volumes(points, tets)
    flipped = int(np.sum(vols <= 0.0))
    if flipped:
        # fix orientation by swapping two nodes of inverted tets
        tets = [
            (t[0], t[2], t[1], t[3]) if v <= 0 else t
            for t, v in zip(tets, vols)
        ]
        vols = signed_volumes(points, tets)
        assert int(np.sum(vols <= 0.0)) == 0, "orientation fix failed"

    wall_width = sum(hi - lo for lo, hi in SUPPORT_WALLS)
    expected = (1.0e-3 * 0.5e-3 * 0.3e-3          # beam
                + wall_width * 0.5e-3 * 0.3e-3)   # support walls
    if args.powder_fill:
        expected += (1.0e-3 - wall_width) * 0.5e-3 * 0.3e-3  # powder gaps
    total = float(np.sum(vols))
    assert abs(total - expected) / expected < 1e-9, (total, expected)

    with open(args.output, "w") as f:
        f.write("*HEADING\nKaess 2023 cantilever benchmark structured TET4 mesh"
                " (units: m, build axis z)\n")
        f.write("*NODE\n")
        for n, (x, y, z) in enumerate(points, start=1):
            f.write(f"{n}, {x:.10e}, {y:.10e}, {z:.10e}\n")
        f.write("*ELEMENT, TYPE=C3D4, ELSET=PART\n")
        for e, t in enumerate(tets, start=1):
            f.write(f"{e}, {t[0]}, {t[1]}, {t[2]}, {t[3]}\n")
        if powder_eids:
            f.write("*ELSET, ELSET=POWDER\n")
            for start in range(0, len(powder_eids), 12):
                f.write(", ".join(str(e) for e in powder_eids[start:start + 12]) + "\n")

    print(f"nodes={len(points)} tets={len(tets)} flipped_fixed={flipped} "
          f"powder_tets={len(powder_eids)}")
    print(f"total_volume={total:.6e} m^3 (expected {expected:.6e})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
