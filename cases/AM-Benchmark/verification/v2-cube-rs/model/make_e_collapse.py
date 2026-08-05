#!/usr/bin/env python3
"""D-V2-17 方案 (c) 的文献对齐实现:E(T) 随固相线坍塌 + 有限地板。

依据(Fable5 2026-08-05 文献调研,IET-6):
  - Proell/Wall/Meier (arXiv:2107.11067) 三相 Voigt 均匀化:熔体相基本无力学
    强度,E 随相态一起变 —— 方案 (c) 的方向;
  - 316L 热应力建模(S175558172030002X)的"残余刚度法":熔融单元 E 降到很小
    但**绝不为零**,并用一个温度区间平滑过渡 —— 否则把"屈服为零、刚度还在"
    换成"刚度为零、矩阵奇异",一个病换另一个病。

现状的不自洽(病灶本身):E(T) 线性外推到固相线仍有 61.6 GPa,而 J-C 屈服在
同一温度已坍到 1 MPa 地板。近乎流体的东西顶着固体刚度,H/E 必然趋零。

本表的构造(全部声明式,不向任何实测回调):
  T <= 1144.15 K   数据表 [63] 原值,逐字节不动;
  1144.15 -> T_c   沿数据表自身的末段斜率线性延伸(与原表同口径);
  T_c -> 固相线    线性坍塌到地板,T_c = COLLAPSE_FRAC * T_solidus;
  >= 固相线        地板值(表末端钳制,再由 mushy/liquid 力学因子继续缩放)。
"""
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).parent
TABLES = HERE / "tables"

T_SOLIDUS = 1563.15
COLLAPSE_FRAC = 0.8          # 坍塌起点 = 0.8 * T_sol
FLOOR_FRAC = 0.01            # 地板 = 1 % 的室温 E


def read_E(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["T"]), float(r["value"]), r["source"]))
    rows.sort()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(TABLES / "E.csv"))
    ap.add_argument("--out", dest="dst", default=str(TABLES / "E_collapse.csv"))
    ap.add_argument("--collapse-frac", type=float, default=COLLAPSE_FRAC)
    ap.add_argument("--floor-frac", type=float, default=FLOOR_FRAC)
    args = ap.parse_args()

    src = read_E(args.src)
    # 数据表实测段:[63] 的 9 行止于 1144.15 K,其后是 D-V2-17 的线性外推行
    sheet = [r for r in src if r[0] <= 1144.15]
    E_rt = sheet[0][1]
    floor = args.floor_frac * E_rt
    t_c = args.collapse_frac * T_SOLIDUS

    # 末段斜率:用数据表最后两行,保持与原表同一外推口径
    (Ta, va, _), (Tb, vb, _) = sheet[-2], sheet[-1]
    slope = (vb - va) / (Tb - Ta)
    e_at_tc = vb + slope * (t_c - Tb)

    note = (f"D-V2-17 option (c): stiffness collapse toward the solidus with a "
            f"finite floor. Sheet rows (<=1144.15 K) unchanged from [63]; "
            f"linear sheet-slope extension to T_c={t_c:.2f} K "
            f"(={args.collapse_frac:g}*T_sol); linear collapse T_c->solidus to a "
            f"floor of {args.floor_frac:g}*E(RT)={floor:.4g} Pa (NOT zero -- a zero "
            f"floor trades a degenerate tangent for a singular matrix). "
            f"Declarative numerical convention, nothing calibrated to measurement.")

    rows = [(T, v, s) for T, v, s in sheet]
    rows.append((t_c, e_at_tc, note))
    rows.append((T_SOLIDUS, floor, note))

    with open(args.dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T", "value", "source"])
        for T, v, s in rows:
            w.writerow([f"{T:.2f}", f"{v:.6g}", s])

    print(f"wrote {Path(args.dst).name}")
    print(f"  E(RT)          = {E_rt:.4g} Pa")
    print(f"  T_c            = {t_c:.2f} K  ({args.collapse_frac:g} * T_sol)")
    print(f"  E(T_c)         = {e_at_tc:.4g} Pa   (sheet slope {slope:.4g} Pa/K)")
    print(f"  floor at T_sol = {floor:.4g} Pa   ({args.floor_frac:g} * E_RT)")
    print()
    print(f"{'T(K)':>9} {'E_old GPa':>11} {'E_new GPa':>11} {'ratio':>8}")
    old = dict((T, v) for T, v, _ in src)

    def interp(tbl, T):
        ks = sorted(tbl)
        if T <= ks[0]:
            return tbl[ks[0]]
        if T >= ks[-1]:
            return tbl[ks[-1]]
        for a, b in zip(ks, ks[1:]):
            if a <= T <= b:
                return tbl[a] + (T - a) / (b - a) * (tbl[b] - tbl[a])

    new = dict((T, v) for T, v, _ in rows)
    for T in [294.15, 1033.15, 1144.15, 1250.52, 1273.15, 1423.15, 1563.15]:
        o, n = interp(old, T), interp(new, T)
        print(f"{T:>9.2f} {o/1e9:>11.2f} {n/1e9:>11.3f} {n/o:>8.3f}")


if __name__ == "__main__":
    main()
