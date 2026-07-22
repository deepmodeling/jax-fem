#!/usr/bin/env python3
"""Structured mesh for the Kaess 2023 cantilever benchmark (TET4 or HEX8).

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
  - lateral powder margins (0.1 mm): meshed as POWDER only with
    --margin-fill; margin coordinates go NEGATIVE (x in [-0.1, 1.1] mm)
    so beam coordinates - and every downstream anchor/cut/gauge box -
    are unchanged.

Element type:
  - c3d4 (default, legacy): Kuhn 6-tet split of every hex. Volumetric
    locking under J2 flow (diagnosed 2026-07-21) - kept for comparison arms.
  - c3d8: the hexes themselves, Abaqus C3D8 node order (= VTK). Reference
    parity: Kaess 2023 used C3D8, whose built-in selective reduced
    volumetric integration is the B-bar the v03 solver applies to HEX8.
    Full parity target: --element-type c3d8 --powder-fill --margin-fill
    -> 48 x 28 x 22 = 29,568 elements, cell-for-cell the reference mesh.

Units are meters (v03 mesh convention). Build axis: z.
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

BEAM_X = 1.0e-3
BEAM_Y = 0.5e-3
MARGIN = 0.1e-3


def build_mesh(nx=40, ny=20, support_layers=12, build_layers=10,
               dx=25.0e-6, dy=25.0e-6, dz_support=25.0e-6, dz_build=30.0e-6,
               walls=SUPPORT_WALLS, powder_fill=False, margin_fill=False):
    """Returns (points, elems, powder_eids) where elems are hex corner
    8-tuples (VTK/C3D8 order). powder_fill meshes the inter-wall support
    gaps; margin_fill meshes the 0.1 mm lateral margins (all layers). Both
    are reported in powder_eids (1-based HEX ids) for the POWDER elset.
    The reference model (Kaess 2023) meshes powder as a weak solid."""
    margin_cells_x = int(round(MARGIN / dx)) if margin_fill else 0
    margin_cells_y = int(round(MARGIN / dy)) if margin_fill else 0
    nx_tot = nx + 2 * margin_cells_x
    ny_tot = ny + 2 * margin_cells_y
    # beam stays at x in [0, BEAM_X], y in [0, BEAM_Y]; margins go negative
    xs = (np.arange(nx_tot + 1) - margin_cells_x) * dx
    ys = (np.arange(ny_tot + 1) - margin_cells_y) * dy
    z_support = np.arange(support_layers + 1) * dz_support
    z_build = z_support[-1] + np.arange(1, build_layers + 1) * dz_build
    zs = np.concatenate([z_support, z_build])
    nz = len(zs) - 1

    nid = lambda i, j, k: (k * (ny_tot + 1) + j) * (nx_tot + 1) + i + 1  # 1-based
    points = np.array(
        [[xs[i], ys[j], zs[k]]
         for k in range(nz + 1)
         for j in range(ny_tot + 1)
         for i in range(nx_tot + 1)],
        dtype=np.float64,
    )

    def in_wall(i):
        xc = (xs[i] + xs[i + 1]) / 2.0
        return any(lo - 1e-9 < xc < hi + 1e-9 for lo, hi in walls)

    def in_margin(i, j):
        xc = (xs[i] + xs[i + 1]) / 2.0
        yc = (ys[j] + ys[j + 1]) / 2.0
        return not (0.0 < xc < BEAM_X and 0.0 < yc < BEAM_Y)

    elems = []
    powder_eids = []
    for k in range(nz):
        support_region = k < support_layers
        for j in range(ny_tot):
            for i in range(nx_tot):
                if in_margin(i, j):
                    powder = True          # lateral margin: powder, all layers
                    if not margin_fill:
                        continue
                elif support_region and not in_wall(i):
                    powder = True          # inter-wall support gap
                    if not powder_fill:
                        continue           # legacy void semantics
                else:
                    powder = False
                elems.append((
                    nid(i, j, k), nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ))
                if powder:
                    powder_eids.append(len(elems))  # 1-based element id

    # compact node numbering: keep only referenced nodes
    used = np.unique(np.asarray(elems).reshape(-1))
    remap = np.zeros(int(used.max()) + 1, dtype=np.int64)
    remap[used] = np.arange(1, len(used) + 1)
    points = points[used - 1]
    elems = [tuple(int(remap[n]) for n in e) for e in elems]
    return points, elems, powder_eids


def hexes_to_tets(elems, powder_eids):
    """Kuhn split; powder hex ids map to the 6 child tet ids each."""
    powder = set(powder_eids)
    tets, tet_powder_eids = [], []
    for e, hex_nodes in enumerate(elems, start=1):
        for tet in HEX_TO_TETS:
            tets.append(tuple(hex_nodes[c] for c in tet))
            if e in powder:
                tet_powder_eids.append(len(tets))
    return tets, tet_powder_eids


def tet_signed_volumes(points, tets):
    p = points[np.asarray(tets) - 1]
    a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def expected_volume(powder_fill, margin_fill):
    wall_width = sum(hi - lo for lo, hi in SUPPORT_WALLS)
    support_h = 0.3e-3
    total_h = 0.6e-3
    vol = BEAM_X * BEAM_Y * 0.3e-3          # beam
    vol += wall_width * BEAM_Y * support_h  # support walls
    if powder_fill:
        vol += (BEAM_X - wall_width) * BEAM_Y * support_h  # inter-wall gaps
    if margin_fill:
        ring_area = (BEAM_X + 2 * MARGIN) * (BEAM_Y + 2 * MARGIN) - BEAM_X * BEAM_Y
        vol += ring_area * total_h          # lateral margins, all layers
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None,
                    help="default: kaess_cantilever_<type>[_powder][_margin].inp "
                         "next to this script")
    ap.add_argument("--element-type", choices=("c3d4", "c3d8"), default="c3d4",
                    help="c3d4 = legacy Kuhn 6-tet split (volumetric locking "
                         "under J2, kept for comparison arms); c3d8 = reference-"
                         "parity hexes (v03 applies B-bar on HEX8)")
    ap.add_argument("--powder-fill", action="store_true",
                    help="mesh the inter-wall support gaps and emit them as "
                         "an *ELSET, ELSET=POWDER block (weak-solid powder)")
    ap.add_argument("--margin-fill", action="store_true",
                    help="mesh the 0.1 mm lateral powder margins (all layers) "
                         "into the POWDER elset (reference mesh parity)")
    args = ap.parse_args()

    points, hexes, powder_eids = build_mesh(
        powder_fill=args.powder_fill, margin_fill=args.margin_fill)

    if args.element_type == "c3d4":
        elems, powder_eids = hexes_to_tets(hexes, powder_eids)
        vols = tet_signed_volumes(points, elems)
        flipped = int(np.sum(vols <= 0.0))
        if flipped:
            # fix orientation by swapping two nodes of inverted tets
            elems = [
                (t[0], t[2], t[1], t[3]) if v <= 0 else t
                for t, v in zip(elems, vols)
            ]
            vols = tet_signed_volumes(points, elems)
            assert int(np.sum(vols <= 0.0)) == 0, "orientation fix failed"
        total = float(np.sum(vols))
        abq_type = "C3D4"
    else:
        elems = hexes
        # hex volume via its own Kuhn split. Half the HEX_TO_TETS corner
        # listings are negatively oriented by construction (the c3d4 path
        # node-swaps them), so |vol| is the right per-child measure; a
        # degenerate hex would show up as a zero child volume.
        child_tets, _ = hexes_to_tets(hexes, [])
        vols = np.abs(tet_signed_volumes(points, child_tets))
        assert float(vols.min()) > 0.0, "degenerate hex (zero child tet volume)"
        total = float(np.sum(vols))
        abq_type = "C3D8"

    expected = expected_volume(args.powder_fill, args.margin_fill)
    assert abs(total - expected) / expected < 1e-9, (total, expected)

    output = args.output
    if output is None:
        name = f"kaess_cantilever_{args.element_type}"
        if args.powder_fill:
            name += "_powder"
        if args.margin_fill:
            name += "_margin"
        output = Path(__file__).parent / f"{name}.inp"

    with open(output, "w") as f:
        f.write(f"*HEADING\nKaess 2023 cantilever benchmark structured "
                f"{abq_type} mesh (units: m, build axis z)\n")
        f.write("*NODE\n")
        for n, (x, y, z) in enumerate(points, start=1):
            f.write(f"{n}, {x:.10e}, {y:.10e}, {z:.10e}\n")
        f.write(f"*ELEMENT, TYPE={abq_type}, ELSET=PART\n")
        for e, t in enumerate(elems, start=1):
            f.write(f"{e}, " + ", ".join(str(n) for n in t) + "\n")
        if powder_eids:
            f.write("*ELSET, ELSET=POWDER\n")
            for start in range(0, len(powder_eids), 12):
                f.write(", ".join(str(e) for e in powder_eids[start:start + 12]) + "\n")

    print(f"nodes={len(points)} {abq_type.lower()}s={len(elems)} "
          f"powder_elems={len(powder_eids)}")
    print(f"total_volume={total:.6e} m^3 (expected {expected:.6e})")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
