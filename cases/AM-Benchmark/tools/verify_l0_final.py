"""Post-run verification for L0 pipeline runs (rev-4 wiring).

Checks the final VTU of an L0 output directory against the six registered
success criteria (PREREQUISITES.md section L):
  C1 build region: PART cells SOLID, permanent-powder cells stay POWDER
  C2 stress_free_temperature == 1273.15 K for consolidated part cells
  C3 eq_plastic_strain significantly nonzero in the part
  C4 part von Mises at yield order surviving the final uniform cooldown
  C5 final temperature field back at plate temperature 347.05 K
(C6, CPU/GPU step equivalence, is checked from run logs, not here.)

Usage: python verify_l0_final.py <output_dir>
Writes <output_dir>/verify_final.json.
"""
import glob
import json
import os
import re
import sys

import meshio
import numpy as np

PLATE_K = 347.05
T_REF_EXPECT = 1273.15
EXPECT_STATES = {1: 43810, 2: 15340}   # D-07 L0 mesh: POWDER / PART


def cell_field(mesh, *names):
    for n in names:
        if n in mesh.cell_data:
            return np.concatenate([np.asarray(a) for a in mesh.cell_data[n]])
    return None


def stats(a):
    a = np.asarray(a, dtype=float)
    return dict(min=float(a.min()), max=float(a.max()),
                mean=float(a.mean()), p95=float(np.percentile(a, 95)))


def main(d):
    vtus = sorted(glob.glob(os.path.join(d, 'step_*.vtu')),
                  key=lambda f: int(re.search(r'step_(\d+)', f).group(1)))
    final = vtus[-1]
    print(f'final VTU: {os.path.basename(final)} (of {len(vtus)})')
    mesh = meshio.read(final)

    cells = np.concatenate([b.data for b in mesh.cells if b.type == 'hexahedron'])
    zc = mesh.points[cells][:, :, 2].mean(axis=1)
    build = zc > 1.0e-6                     # substrate occupies z < 0
    state = cell_field(mesh, 'material_state')
    solid = build & (state == 2)
    powder = build & (state == 1)

    result = {}
    vals, cnt = np.unique(state[build], return_counts=True)
    result['build_state_counts'] = {int(v): int(c) for v, c in zip(vals, cnt)}
    print('C1 build states:', result['build_state_counts'],
          f'[expect {EXPECT_STATES}]')

    tref = cell_field(mesh, 'stress_free_temperature')
    result['T_ref_part'] = stats(tref[solid])
    ok = max(abs(result['T_ref_part']['max'] - T_REF_EXPECT),
             abs(result['T_ref_part']['min'] - T_REF_EXPECT)) < 1.0
    print(f"C2 T_ref (part): {result['T_ref_part']}  "
          f"[{'OK' if ok else 'CHECK'} expect {T_REF_EXPECT}]")

    eqp = cell_field(mesh, 'eq_plastic_strain')
    result['eqp_part'] = stats(eqp[solid])
    print(f"C3 eqp (part): {result['eqp_part']}  "
          f"[{'OK' if result['eqp_part']['max'] > 1e-4 else 'FAIL: ~zero'}]")

    vm = np.max(np.stack([cell_field(mesh, f'vm_quad{q}') for q in range(8)]),
                axis=0)
    result['vm_part_MPa'] = {k: v / 1e6 for k, v in stats(vm[solid]).items()}
    print(f"C4 von Mises part [MPa]: {result['vm_part_MPa']}  "
          f"[{'OK' if result['vm_part_MPa']['p95'] > 100 else 'CHECK: low'}]")
    result['vm_powder_MPa_max'] = float(vm[powder].max() / 1e6)
    print(f"   powder vm max [MPa]: {result['vm_powder_MPa_max']:.2f} "
          f"(weak-solid regularization scale)")

    T = np.asarray(mesh.point_data['T'])
    result['T_final'] = stats(T)
    print(f"C5 final T: {result['T_final']}  "
          f"[{'OK' if abs(result['T_final']['max'] - PLATE_K) < 5 else 'CHECK'}"
          f" expect ~{PLATE_K}]")

    out = os.path.join(d, 'verify_final.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=1)
        f.write('\n')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
