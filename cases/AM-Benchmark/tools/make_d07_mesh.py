"""D-07 domain mesh generator for AMB2018-01 (case-local tooling).

Structured, tensor-product graded C3D8 mesh of the D-07 domain:
  x : [-20.34, 79.66] mm  (substrate real extent; part occupies [0, 75])
  y : [-10, 10] mm        (20 mm periodic cell, part |y| <= 2.5)
  z : [-12.7, +12.48] mm  (substrate below 0; build = 624 x 20 um layers)

Element classification at centroids (D-03 compaction convention):
  SUBSTRATE (z < 0) | PART (inside the part plan geometry at that height)
  | POWDER (everything else in the cell up to full build height; runtime
  activation makes not-yet-recoated material void, D-04).

Plan geometry (legs / overhang interpolation / bridge / ridges / 45-degree
tip) is imported from layer_schedule_generator.py — single source of truth.

Presets (in-plane part resolution x layer lumping N):
  L0: dx=0.50, dy=0.50, N=50   (pipeline test, ~10^5 elements)
  L1: dx=0.25, dy=0.25, N=25   (energy closure)
  L2: dx=0.25, dy=0.25, N=10   (trusted thermal, ~10^6 elements)
Thin legs (0.5 mm) get 2 elements across at 0.25 mm; the 3-across variant
(D.4 note) is available via --dx-part 0.1666...

Output: derived/meshes/amb_d07_<level>.inp (NOT committed; regenerable) +
derived/meshes/amb_d07_<level>_summary.json (committed).
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from layer_schedule_generator import (  # noqa: E402
    band_of, features_at, TOTAL_LAYERS, Y_DEPTH)

DZ_PHYS = 0.02          # mm, physical layer height
X_MIN, X_MAX = -20.34, 79.66
Y_MIN, Y_MAX = -10.0, 10.0
Z_SUB = 12.7
Z_BUILD = TOTAL_LAYERS * DZ_PHYS      # 12.48 (B1: 624-layer basis)

PRESETS = {
    'L0': {'dx_part': 0.50, 'dy_part': 0.50, 'N': 50},
    'L1': {'dx_part': 0.25, 'dy_part': 0.25, 'N': 25},
    'L2': {'dx_part': 0.25, 'dy_part': 0.25, 'N': 10},
}


def graded(length, a, b):
    """Cell sizes over `length` growing geometrically from a to b."""
    if length <= 0:
        return []
    n = max(1, round(2 * length / (a + b)))
    r = (b / a) ** (1 / max(n - 1, 1))
    sizes = [a * r ** i for i in range(n)]
    scale = length / sum(sizes)
    return [s * scale for s in sizes]


def axis(segments):
    """Cumulative node coordinates from [(x_start, [sizes...]), ...]."""
    xs = [segments[0][0]]
    for start, sizes in segments:
        assert abs(xs[-1] - start) < 1e-9
        for s in sizes:
            xs.append(xs[-1] + s)
    return xs


def in_part(x, y, layer):
    if abs(y) > Y_DEPTH / 2:
        return False
    band = band_of(layer)
    yl = y + Y_DEPTH / 2          # geometry module uses y in [0, 5]
    for x0, x1 in features_at(layer, band):
        xr = x1(yl) if callable(x1) else x1
        if x0 <= x <= xr:
            return True
    return False


def build_mesh(level, dx_part=None, out_dir=None):
    p = PRESETS[level]
    dxp = dx_part or p['dx_part']
    dyp = p['dy_part']
    N = p['N']

    xs = axis([(X_MIN, list(reversed(graded(20.34, dxp * 1.6, 1.2)))),
               (0.0, [dxp] * round(75.0 / dxp)),
               (75.0, graded(4.66, dxp * 1.6, 1.2))])
    ys_wing = graded(7.5, dyp * 1.4, 1.2)
    ys = axis([(Y_MIN, list(reversed(ys_wing))),
               (-Y_DEPTH / 2, [dyp] * round(Y_DEPTH / dyp)),
               (Y_DEPTH / 2, ys_wing)])
    dz_comp = N * DZ_PHYS
    n_full = TOTAL_LAYERS // N
    z_build_sizes = [dz_comp] * n_full
    rem = TOTAL_LAYERS - n_full * N
    if rem:
        z_build_sizes.append(rem * DZ_PHYS)
    zs = axis([(-Z_SUB, list(reversed(graded(Z_SUB, 0.4, 2.0)))),
               (0.0, z_build_sizes)])

    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    n_elems = nx * ny * nz

    def nid(i, j, k):
        return 1 + i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    out_dir = out_dir or os.path.join(CASE, 'derived', 'meshes')
    os.makedirs(out_dir, exist_ok=True)
    inp = os.path.join(out_dir, f'amb_d07_{level}.inp')

    sets = {'SUBSTRATE': [], 'PART': [], 'POWDER': []}
    with open(inp, 'w') as f:
        f.write('*HEADING\n')
        f.write(f'AMB2018-01 D-07 domain, level {level}, N={N}, '
                f'units mm, build axis z, Dirichlet face NSET=BOTTOM\n')
        f.write('*NODE\n')
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    f.write(f'{nid(i, j, k)}, {xs[i]:.6f}, '
                            f'{ys[j]:.6f}, {zs[k]:.6f}\n')
        f.write('*ELEMENT, TYPE=C3D8\n')
        eid = 0
        off = (nx + 1) * (ny + 1)
        for k in range(nz):
            zc = 0.5 * (zs[k] + zs[k + 1])
            layer = None
            if zc > 0:
                layer = min(TOTAL_LAYERS, int(zc / DZ_PHYS) + 1)
            for j in range(ny):
                yc = 0.5 * (ys[j] + ys[j + 1])
                for i in range(nx):
                    xc = 0.5 * (xs[i] + xs[i + 1])
                    eid += 1
                    n0, n1 = nid(i, j, k), nid(i + 1, j, k)
                    n2, n3 = nid(i + 1, j + 1, k), nid(i, j + 1, k)
                    f.write(f'{eid}, {n0}, {n1}, {n2}, {n3}, '
                            f'{n0 + off}, {n1 + off}, {n2 + off}, {n3 + off}\n')
                    if zc < 0:
                        sets['SUBSTRATE'].append(eid)
                    elif in_part(xc, yc, layer):
                        sets['PART'].append(eid)
                    else:
                        sets['POWDER'].append(eid)
        for name, ids in sets.items():
            f.write(f'*ELSET, ELSET={name}\n')
            for a in range(0, len(ids), 10):
                f.write(', '.join(map(str, ids[a:a + 10])) + '\n')
        f.write('*NSET, NSET=BOTTOM\n')
        bottom = [nid(i, j, 0) for j in range(ny + 1) for i in range(nx + 1)]
        for a in range(0, len(bottom), 10):
            f.write(', '.join(map(str, bottom[a:a + 10])) + '\n')
        for name, jj in (('YMIN', 0), ('YMAX', ny)):
            f.write(f'*NSET, NSET={name}\n')
            side = [nid(i, jj, k) for k in range(nz + 1)
                    for i in range(nx + 1)]
            for a in range(0, len(side), 10):
                f.write(', '.join(map(str, side[a:a + 10])) + '\n')

    summary = {
        'level': level, 'N': N,
        'dx_part_mm': dxp, 'dy_part_mm': dyp,
        'grid': [nx, ny, nz], 'nodes': n_nodes, 'elements': n_elems,
        'sets': {k: len(v) for k, v in sets.items()},
        'computational_layers': len(z_build_sizes),
        'domain_mm': {'x': [X_MIN, X_MAX], 'y': [Y_MIN, Y_MAX],
                      'z': [-Z_SUB, Z_BUILD]},
        'notes': [
            'B1: 624-layer basis (z_build 12.48 mm)',
            '45-degree tip stair-stepped at dx_part',
            'thin 0.5 mm legs: '
            f'{round(0.5 / dxp)} elements across at dx_part={dxp}',
            'PART/POWDER classified per computational-layer centroid via '
            'layer_schedule_generator plan geometry',
        ],
    }
    sfile = os.path.join(out_dir, f'amb_d07_{level}_summary.json')
    with open(sfile, 'w') as f:
        json.dump(summary, f, indent=1)
        f.write('\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'notes'},
                     indent=1))
    print(f'wrote {inp} ({os.path.getsize(inp)/1e6:.1f} MB) and summary')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('level', choices=list(PRESETS), nargs='?', default='L0')
    ap.add_argument('--dx-part', type=float, default=None)
    args = ap.parse_args()
    build_mesh(args.level, dx_part=args.dx_part)
