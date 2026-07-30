#!/usr/bin/env python3
"""M1: multi-track mesh, Balbaa2022 Sec 2.6.2 / Fig 3 parity (see SPEC.md).

Powder layer 4 x 4 x 0.04 mm @ uniform 40 um (100 x 100 x 1) on a flush
4 x 4 x 0.4 mm substrate, in-plane 40 um (conforming, no tie; D-V2-05),
vertical grading 40 -> 140 um as the registered geometric ladder D-V2-06:
bottom -> top [136.0, 99.4, 72.7, 53.1, 38.8] um (sums to 400.0 um exactly).
Units: meters, build axis z. C3D8.
"""
from pathlib import Path

DXY = 40.0e-6
NX = NY = 100
# substrate sublayer thicknesses bottom->top (D-V2-06), then the powder layer
DZ_LADDER = [136.0e-6, 99.4e-6, 72.7e-6, 53.1e-6, 38.8e-6, 40.0e-6]
assert abs(sum(DZ_LADDER[:-1]) - 400.0e-6) < 1e-12

Z_LEVELS = [0.0]
for dz in DZ_LADDER:
    Z_LEVELS.append(Z_LEVELS[-1] + dz)
NZ = len(DZ_LADDER)

out = Path(__file__).parent / "v2_multitrack_c3d8.inp"

def nid(i, j, k):
    return 1 + i + j * (NX + 1) + k * (NX + 1) * (NY + 1)

with open(out, "w") as f:
    f.write("*HEADING\n")
    f.write("V2 multi-track Balbaa2022-parity mesh: 4x4x0.04mm layer @40um on "
            "4x4x0.4mm substrate, graded 40->140um (D-V2-05/06). Units m, build axis z.\n")
    f.write("*NODE\n")
    for k, z in enumerate(Z_LEVELS):
        for j in range(NY + 1):
            for i in range(NX + 1):
                f.write(f"{nid(i,j,k)}, {i*DXY:.10e}, {j*DXY:.10e}, {z:.10e}\n")
    f.write("*ELEMENT, TYPE=C3D8\n")
    eid = 0
    plane = (NX + 1) * (NY + 1)
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                eid += 1
                n0, n1 = nid(i, j, k), nid(i + 1, j, k)
                n2, n3 = nid(i + 1, j + 1, k), nid(i, j + 1, k)
                f.write(f"{eid}, {n0}, {n1}, {n2}, {n3}, "
                        f"{n0+plane}, {n1+plane}, {n2+plane}, {n3+plane}\n")

print(f"wrote {out}: {(NX+1)*(NY+1)*(NZ+1)} nodes, {eid} elements")
print(f"z levels (um): {[round(z*1e6,1) for z in Z_LEVELS]}")
print(f"substrate top = {Z_LEVELS[-2]*1e6:.1f} um; layer top = {Z_LEVELS[-1]*1e6:.1f} um")
print("runner hint: --support-thickness 4.0e-4 --layer-thickness 4.0e-5 --layers 1")
