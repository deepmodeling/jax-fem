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

# alpha:割线 -> 瞬时,经平滑拟合(D-V2-18 子问题)。
# 源表只印到 0.1e-6 精度,逐点差分会放大舍入噪声(非单调,且线性外推到固相线
# 会给出 42.6e-6/K 这种非物理值)。因此先对 alpha_mean(T) 做一次线性最小二乘
# 拟合(最少假设、外推最稳),再解析求导:
#   alpha_inst(T) = d/dT[ alpha_mean(T)*(T - Tref) ] = alpha_mean(T) + (T - Tref)*slope
Ts = [r[0] for r in ALPHA_ROWS]
As = [r[1] for r in ALPHA_ROWS]
n = len(Ts)
mT, mA = sum(Ts) / n, sum(As) / n
slope = sum((t - mT) * (a - mA) for t, a in zip(Ts, As)) / sum((t - mT) ** 2 for t in Ts)
icept = mA - slope * mT
resid = max(abs(a - (icept + slope * t)) for t, a in zip(Ts, As))
print(f"\nalpha_mean 线性拟合: {icept:.4f} + {slope:.6f}*T  (最大残差 {resid:.3f}e-6,"
      f" 源表分辨率 0.1e-6)")


def alpha_inst(T_C):
    a_mean = icept + slope * T_C
    return (a_mean + (T_C - ALPHA_TREF) * slope) * 1e-6


# alpha 的高温延伸用同一解析式(拟合本身已是线性外推),不再叠加 extend()
alpha_rows = [(T_C, alpha_inst(T_C)) for T_C in Ts + [SOLIDUS_C]]
write("alpha.csv", alpha_rows, "1/K")
print(f"  alpha_inst: {alpha_inst(93)*1e6:.2f}e-6 (93 C) -> "
      f"{alpha_inst(927)*1e6:.2f}e-6 (927 C, 数据末端) -> "
      f"{alpha_inst(SOLIDUS_C)*1e6:.2f}e-6 (固相线,外推)")

# J-C 流动曲线(省略速率项,D-V2-02)
eps_grid = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
T_grid = [r[0] for r in E_ROWS] + [1000, 1150, SOLIDUS_C]
p = OUT / "flow_curve.csv"
jc_src = (f"modified J-C (Balbaa2022 p.33) A={A} B={B} n={N} C={C} m={M} "
          f"Tm={TM}C Tr={TR}C; rate term omitted (D-V2-02, <0.3%)")
with open(p, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["temperature_K", "equivalent_plastic_strain", "flow_stress_Pa", "source"])
    for T_C in T_grid:
        for e in eps_grid:
            # D-V2-19:固相线处 J-C 软化为 0,原 1e-3 MPa 平地板复现了仓库
            # c475f7e 记录的病理(sigma_y 小 + H=0 -> 切线 PSD -> Newton 停滞/
            # 发散,首跑 u_max 炸至 1e11)。按 kaess 同一先例正则化:
            # 地板 1 MPa(= kaess powder-solid-yield)+ 全曲线附加
            # H_reg = 1e7 Pa 硬化斜率(= kaess powder-solid-hardening,
            # 消除一切零斜率段)。对低温段影响 <0.2%(1e7*0.2/650e6)。
            sigma_pa = max(jc_flow(e, T_C) * 1e6, 1.0e6) + 1.0e7 * e
            w.writerow([f"{T_C + K:.2f}", f"{e:g}", f"{sigma_pa:.6g}", jc_src])
print(f"{p.name}: {len(T_grid)*len(eps_grid)} rows, "
      f"sigma(0,21C)={jc_flow(0,21):.0f} MPa .. sigma(0,{SOLIDUS_C:.0f}C)={jc_flow(0,SOLIDUS_C):.1f} MPa")
