#!/usr/bin/env python3
"""V2 力学材料表生成 + 两项偏差定量(D-V2-02 速率项、D-V2-18 割线膨胀系数)。

数据源:../inputs/specialmetals-in625-tables.json(Balbaa 引用的 [63])与
../inputs/balbaa-v2-model.json(constitutive_actually_used,即 p.33 修正后的 J-C)。

高温外推(D-V2-17)必须由 --hight-mode 显式给出:数据表止于 871-927 C,
而模型跑到固相线 1290 C。不提供默认值——静默 hold-last-value 正是要避免的。

用法:
  python make_v2_mech_tables.py --quantify              # 只打印定量结果,不写表
  python make_v2_mech_tables.py --hight-mode hold       # 写表(需登记该选择)
"""
import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
INP = HERE.parent / "inputs"
SM = json.loads((INP / "specialmetals-in625-tables.json").read_text(encoding="utf-8"))
BAL = json.loads((INP / "balbaa-v2-model.json").read_text(encoding="utf-8"))

JC = BAL["constitutive_actually_used"]["johnson_cook_used"]
A, B, N = JC["A_MPa"], JC["B_MPa"], JC["n"]
C, M = JC["C"], JC["m"]
TM, TR = JC["Tm_C"], JC["Tr_C"]
EDOT0 = JC["edot0_per_s"]
SOLIDUS_C = 1290.0
K = 273.15

E_ROWS = SM["table4_modulus_elevated_temperature"]["rows"]
ALPHA_ROWS = SM["table3_mean_linear_expansion"]["rows"]
ALPHA_TREF = SM["table3_mean_linear_expansion"]["reference_temperature_C"]
E_OFFSET = SM["balbaa_modulus_reconstruction"]["annealed_branch"]["offset_GPa"]

ap = argparse.ArgumentParser()
ap.add_argument("--hight-mode", choices=["hold", "linear", "collapse"],
                help="D-V2-17 高温外推方式(871/927 C 以上)。必须显式给出。")
ap.add_argument("--quantify", action="store_true", help="只做定量报告,不写表")
args = ap.parse_args()


def jc_flow(eps_p, T_C, edot=None):
    """修正后的 J-C 流动应力 (MPa)。edot=None 表示省略速率项(=参考速率)。"""
    hard = A + B * (eps_p ** N if eps_p > 0 else 0.0)
    rate = 1.0 if edot is None else (1.0 + C * math.log(edot / EDOT0))
    tstar = (T_C - TR) / (TM - TR)
    soft = 1.0 - tstar ** M if tstar > 0 else 1.0
    return hard * rate * max(soft, 0.0)


# ---------- D-V2-02:省略应变率项的影响 ----------
print("=== D-V2-02 量化:省略 J-C 速率项 (1 + C*ln(edot/edot0)) ===")
print(f"C = {C}, edot0 = {EDOT0} /s")
for edot, label in ((1e-3, "准静态 1e-3"), (1.0, "1 /s"), (1e3, "1e3 /s"),
                    (1e5, "1e5 /s(LPBF 冷却量级上限)")):
    factor = 1.0 + C * math.log(edot / EDOT0)
    print(f"  edot = {label:<24s} -> 系数 {factor:.6f}  (偏差 {100*(factor-1):+.3f} %)")
print("  结论:整个 8 个数量级的速率跨度内偏差 < 0.3 %,远低于任何 RS 判据精度;")
print("       省略速率项等价于把所有点取在参考速率 1670 /s。")

# ---------- D-V2-18:割线 vs 瞬时热膨胀系数 ----------
print("\n=== D-V2-18 量化:[63] 的 alpha 是 21 C 参考的割线值 ===")
print(f"{'T(C)':>6} {'alpha_mean':>11} {'alpha_inst':>11} {'相对差':>9}")
prev_T, prev_a = ALPHA_TREF, ALPHA_ROWS[0][1]
worst = 0.0
for T_C, a_mean in ALPHA_ROWS:
    # 瞬时 = d/dT[ alpha_mean(T)*(T-Tref) ],用相邻行差分
    if T_C == ALPHA_ROWS[0][0]:
        a_inst = a_mean
    else:
        f_now = a_mean * (T_C - ALPHA_TREF)
        f_prev = prev_a * (prev_T - ALPHA_TREF)
        a_inst = (f_now - f_prev) / (T_C - prev_T)
    rel = (a_inst - a_mean) / a_mean * 100
    worst = max(worst, abs(rel))
    print(f"{T_C:>6} {a_mean:>11.2f} {a_inst:>11.2f} {rel:>8.1f} %")
    prev_T, prev_a = T_C, a_mean
print(f"  最大相对差 {worst:.1f} %(高温段);我方求解器用 alpha*(T - T_ref_quad),")
print("  T_ref 在凝固时重锚定而非 21 C,故应使用瞬时值。上表右列即转换结果。")

if args.quantify:
    raise SystemExit(0)

if not args.hight_mode:
    raise SystemExit(
        "错误:必须显式给出 --hight-mode(D-V2-17 未决,不接受静默默认值)。\n"
        "       先与主线协调高温外推方式,再登记并生成。"
    )

OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)
src = f"SpecialMetals [63] annealed, Balbaa offset {E_OFFSET} GPa; hight-mode={args.hight_mode} (D-V2-17)"


def extend(rows, mode, collapse_factor=1e-3):
    """把表延伸到固相线。rows: [(T_C, value)]"""
    out = list(rows)
    T_last, v_last = rows[-1]
    T_prev, v_prev = rows[-2]
    if mode == "hold":
        out.append((SOLIDUS_C, v_last))
    elif mode == "linear":
        slope = (v_last - v_prev) / (T_last - T_prev)
        out.append((SOLIDUS_C, max(v_last + slope * (SOLIDUS_C - T_last), 0.0)))
    elif mode == "collapse":
        out.append((SOLIDUS_C, v_last * collapse_factor))
    return out


def write(name, rows, unit):
    p = OUT / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T", "value", "source"])
        for T_C, v in rows:
            w.writerow([f"{T_C + K:.2f}", f"{v:.6g}", src])
    print(f"{p.name}: {len(rows)} rows, {rows[0][1]:.4g}..{rows[-1][1]:.4g} {unit}")


write("E.csv", extend([(r[0], (r[1] + E_OFFSET) * 1e9) for r in E_ROWS],
                      args.hight_mode), "Pa")
write("poisson.csv", extend([(r[0], r[5]) for r in E_ROWS], "hold"), "-")
# alpha:瞬时值(D-V2-18)
inst = []
prev_T, prev_a = ALPHA_TREF, ALPHA_ROWS[0][1]
for T_C, a_mean in ALPHA_ROWS:
    if T_C == ALPHA_ROWS[0][0]:
        a_inst = a_mean
    else:
        a_inst = (a_mean * (T_C - ALPHA_TREF) - prev_a * (prev_T - ALPHA_TREF)) / (T_C - prev_T)
    inst.append((T_C, a_inst * 1e-6))
    prev_T, prev_a = T_C, a_mean
write("alpha.csv", extend(inst, args.hight_mode), "1/K")

# J-C 流动曲线(省略速率项,D-V2-02)
eps_grid = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
T_grid = [r[0] for r in E_ROWS] + [1000, 1150, SOLIDUS_C]
p = OUT / "flow_curve.csv"
with open(p, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T", "eps_p", "sigma_Pa", "source"])
    for T_C in T_grid:
        for e in eps_grid:
            w.writerow([f"{T_C + K:.2f}", f"{e:g}", f"{jc_flow(e, T_C)*1e6:.6g}",
                        f"modified J-C (Balbaa p.33) A={A} B={B} n={N} m={M}; rate term omitted (D-V2-02)"])
print(f"{p.name}: {len(T_grid)*len(eps_grid)} rows, "
      f"sigma(0,21C)={jc_flow(0,21):.0f} MPa .. sigma(0,{SOLIDUS_C:.0f}C)={jc_flow(0,SOLIDUS_C):.1f} MPa")
