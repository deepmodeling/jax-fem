#!/usr/bin/env python3
"""M1: multi-track mesh, Balbaa2022 Sec 2.6.2 / Fig 3 parity (see SPEC.md).

Powder layer 4 x 4 x 0.04 mm @ uniform 40 um (100 x 100 x 1) on a flush
4 x 4 x 0.4 mm substrate, in-plane 40 um (conforming, no tie; D-V2-05),
vertical grading 40 -> 140 um as the registered geometric ladder D-V2-06:
bottom -> top [136.0, 99.4, 72.7, 53.1, 38.8] um (sums to 400.0 um exactly).
Units: meters, build axis z. C3D8.
"""
import argparse
from pathlib import Path

DXY = 40.0e-6
NX = NY = 100
SUB_TOTAL = 400.0e-6

# 基板 z 向剖分的三个变体(D-V2-07 敏感性研究)。件内应力若对三者不敏感,
# 则"共形分级替代 Balbaa 的两尺度+tie"得证,可关闭 D-V2-07。
SUB_LADDERS = {
    # D-V2-06 登记的几何级数梯子:界面 38.8 -> 底部 136 um,闭合 400 um
    "graded": [136.0e-6, 99.4e-6, 72.7e-6, 53.1e-6, 38.8e-6],
    # 上界:全 40 um(比 Balbaa 的 40->140 um 分级更细,视为收敛参照)
    "fine": [40.0e-6] * 10,
    # 下界:2 x 200 um(比 Balbaa 底部 140 um 更粗)
    "coarse": [200.0e-6, 200.0e-6],
}

ap = argparse.ArgumentParser()
ap.add_argument("--substrate-mode", choices=sorted(SUB_LADDERS), default="graded")
args = ap.parse_args()

sub = SUB_LADDERS[args.substrate_mode]
assert abs(sum(sub) - SUB_TOTAL) < 1e-12, f"{args.substrate_mode} 基板厚度不闭合"
DZ_LADDER = sub + [40.0e-6]  # 顶部 40 um 粉层

Z_LEVELS = [0.0]
for dz in DZ_LADDER:
    Z_LEVELS.append(Z_LEVELS[-1] + dz)
NZ = len(DZ_LADDER)

suffix = "" if args.substrate_mode == "graded" else f"_{args.substrate_mode}"
out = Path(__file__).parent / f"v2_multitrack_c3d8{suffix}.inp"

def nid(i, j, k):
    return 1 + i + j * (NX + 1) + k * (NX + 1) * (NY + 1)

with open(out, "w") as f:
    f.write("*HEADING\n")
    f.write(f"V2 multi-track Balbaa2022-parity mesh: 4x4x0.04mm layer @40um on "
            f"4x4x0.4mm substrate, substrate-mode={args.substrate_mode} "
            f"(D-V2-05/06/07). Units m, build axis z.\n")
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
