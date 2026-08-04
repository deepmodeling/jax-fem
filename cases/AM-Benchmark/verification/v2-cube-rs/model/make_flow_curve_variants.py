#!/usr/bin/env python3
"""D-V2-22: J-C 幂律在 eps_p -> 0 的切线奇异,及其三个正则化臂。

诊断(2026-08-03,V 会话):三个基板变体(v2_sub5_*)全部在力学 Newton 上失败,
而把 flow_curve_table 换成常数 yield/hardening 占位表的 T1 变体跑通 108+ 步。
病因量化在 flow_curve.csv 上直接可见:

    室温 eps_p: 0 -> 0.002 段  H = 178.7 GPa  >  E = 171.0 GPa   (H/E = 1.04)
    下一段 (0.002 -> 0.005)    H =  29.7 GPa                     (骤降 6 倍)

这不是网格、不是基板变体、不是功率,而是 J-C 的解析性质:
    dsigma/deps_p = B * n * eps_p^(n-1),  n = 0.243  ->  eps_p^(-0.757) -> +inf
首格取 0.002 只是把这个奇异采样成了 178 GPa。**加密网格会让它更糟而非更好**
(H_seg1 随首格宽度 ^(n-1) 发散),所以"表太粗"不是可行解释,必须正则化幂律本身。

三个臂(全部保留 D-V2-19 的高温处理:1 MPa 地板 + H_reg = 1e7 Pa):

  asis    原样(对照,预期发散)。
  offset  幂律原点移到 0.2% 偏移点:
              sigma = A + B*[(eps_p + e_off)^n - e_off^n],  e_off = 0.002
          依据:A 是拉伸曲线的 0.2% 偏移屈服值(Balbaa p.33 取自自测拉伸),
          故幂律的塑性应变原点是该偏移点而非零。sigma(0) = A 恰好守住屈服点;
          H(0) = B*n*e_off^(n-1) = 43.4 GPa = 0.25 E,可容许;大应变分支几乎
          不动(eps_p = 0.2 处相差 0.24 %),UTS 区不受影响。
  cap     斜率限制:从 sigma(0) = A 起按 dsigma/deps_p = min(JC 斜率, f*E(T))
          积分,f = 0.10。作为 offset 的括号臂(近零段更软)。

--------------------------------------------------------------------------
2026-08-04 增补 —— D-V2-22 的因果已被证伪,本脚本改为同时服务 D-V2-19 试验梯
--------------------------------------------------------------------------
A 相四臂(asis / offset / cap / asis + 求解器参数)**全部**在同一处失败
(ledger=12, global_step=10),近零切线跨度 H1/E = 2.98 -> 0.23 -> 0.10 对收敛
行为毫无影响。原因在本脚本的 build() 里一眼可见:**近固相线行上三臂逐点相同**
—— J-C 软化因子 soft(T) 在 Tm 处归零,把三臂唯一不同的硬化项 B*(...)^n 一起
乘没了,只剩下三臂共用的 D-V2-19 对(1 MPa 地板 + H_reg 1e7 Pa)。所以变的
从来不是失败的那一段。

因此新增 --floor-pa / --h-reg-pa / --name,用于在**固定 offset 近零处理**的前提下
单变量地扫 D-V2-19 这一对:

  python make_flow_curve_variants.py --arm offset --name hy10  --floor-pa 1e7
  python make_flow_curve_variants.py --arm offset --name hy100 --floor-pa 1e8
  python make_flow_curve_variants.py --arm offset --name htan  --h-reg-pa 1e9

不带新参数时输出与 2026-08-03 已提交的三张表**逐字节相同**(已核验:先备份三张表,
再 `--arm all` 重新生成,`cmp -s` 三张全部 IDENTICAL)。

采纳的生产臂(D-V2-19-R1,--min-tangent-frac 见 build() 的说明):

  python make_flow_curve_variants.py --arm offset --name offset_mt --min-tangent-frac 0.01
  python make_flow_curve_variants.py --arm cap    --name cap_mt    --min-tangent-frac 0.01

用法:
  python make_flow_curve_variants.py --arm offset --quantify   # 只打印
  python make_flow_curve_variants.py --arm offset              # 写表+配置
  python make_flow_curve_variants.py --arm all                 # 全部
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
INP = HERE.parent / "inputs"
TABLES = HERE / "tables"

BAL = json.loads((INP / "balbaa-v2-model.json").read_text(encoding="utf-8"))
JC = BAL["constitutive_actually_used"]["johnson_cook_used"]
A, B, N = JC["A_MPa"], JC["B_MPa"], JC["n"]
M = JC["m"]
TM, TR = JC["Tm_C"], JC["Tr_C"]
SOLIDUS_C = 1290.0
K = 273.15

# D-V2-19 的两项高温正则化,三臂共用(现在可由 CLI 覆盖,见模块 docstring)
FLOOR_PA = 1.0e6
H_REG_PA = 1.0e7

E_OFF = 0.002          # offset 臂:0.2 % 偏移屈服约定
CAP_FRAC = 0.10        # cap 臂:H <= 0.10 * E(T)

# 输出网格:沿用原 eps 网格,并在近零加两点以解析初始曲率
EPS_GRID = [0.0, 5.0e-4, 1.0e-3, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# E(T) 表(cap 臂需要)
E_TAB = []
with open(TABLES / "E.csv") as f:
    for r in csv.DictReader(f):
        E_TAB.append((float(r["T"]), float(r["value"])))
E_TAB.sort()


def E_at(T_K):
    if T_K <= E_TAB[0][0]:
        return E_TAB[0][1]
    if T_K >= E_TAB[-1][0]:
        return E_TAB[-1][1]
    for (Ta, va), (Tb, vb) in zip(E_TAB, E_TAB[1:]):
        if Ta <= T_K <= Tb:
            return va + (T_K - Ta) / (Tb - Ta) * (vb - va)
    raise AssertionError


def soft(T_C):
    """J-C 温度软化因子。"""
    tstar = (T_C - TR) / (TM - TR)
    return max(1.0 - tstar ** M if tstar > 0 else 1.0, 0.0)


def hard_asis(e):
    return A + B * (e ** N if e > 0 else 0.0)


def hard_offset(e):
    return A + B * ((e + E_OFF) ** N - E_OFF ** N)


def hard_cap(e, T_K, n_sub=4000, e_max=0.2):
    """斜率限制积分:dsigma/deps = min(JC 斜率, CAP_FRAC*E) / soft 之前的硬化空间。

    注意在硬化空间(未乘软化因子)内限幅,再统一乘 soft,使限幅判据与温度相关的
    E(T) 直接可比(sigma_final = hard * soft, 而 H_final = H_hard * soft <= H_hard)。
    """
    cap = CAP_FRAC * E_at(T_K) / 1e6      # MPa / 单位应变
    de = e_max / n_sub
    s = A
    x = 0.0
    while x < e - 1e-15:
        step = min(de, e - x)
        xm = x + 0.5 * step
        jc_slope = B * N * (xm ** (N - 1.0)) if xm > 0 else float("inf")
        s += min(jc_slope, cap) * step
        x += step
    return s


ARMS = ("asis", "offset", "cap")

# 温度网格:必须与 make_v2_mech_tables.py 逐点一致(E.csv 的 9 个数据行 + 固相线,
# 外加 1000/1150 C 两个高温插值行),否则三臂与 asis 的实际运行不可比。
T_GRID_K = sorted(set([t for t, _ in E_TAB] + [1000.0 + K, 1150.0 + K]))


def build(arm, floor_pa=FLOOR_PA, h_reg_pa=H_REG_PA, mt_frac=0.0):
    """mt_frac > 0 时,逐段强制**最小塑性切线** H >= mt_frac * E(T)。

    这是 D-V2-19 的生产形态修订(2026-08-04):把正则化硬化从"固定绝对值
    H_reg = 1e7 Pa"改为"与 E(T) 成比例的切线下限"。理由是病态量本来就是**比值**
    H/E —— 固相线行上 1e7/61.6e9 = 1.6e-4,近乎理想塑性,几千个积分点一起把整体
    切线压成近奇异;而同一个 1e7 Pa 在室温(J-C 自带 35-40 GPa 切线)完全无关痛痒。

    用"下限"而不是 htan 那样的"加法项"很关键:加法项会把室温 sigma(0.2) 从
    1391 抬到 1589 MPa(+14 %),那是在动产生残余应力的那一段材料本构;下限则只在
    J-C 已经软化坍塌的高温行上起作用。

    实测扰动(完整运行 eqp 峰值 0.0233、p99 6.5e-4 处逐行核验):294 K 到 1273 K
    的所有温度行**恰好为 0**(CSV 字面量逐字节相同),1423 K 行最大 0.41 %,
    其余改动全部落在固相线那一行——而那一行从 D-V2-19 起就已声明是正则化而非材料数据。
    表格末端 eps_p = 0.2 处扰动更大(室温 +0.16 %,1423 K +46 %),但完整运行里
    越过 eps_p = 0.2 的单元数为 0 / 30000,那一段根本没有被消费。
    """
    rows = []
    for T_K in T_GRID_K:
        T_C = T_K - K
        sf = soft(T_C)
        Ev = E_at(T_K)
        prev_e = None
        prev_s = None
        for e in EPS_GRID:
            if arm == "asis":
                h = hard_asis(e)
            elif arm == "offset":
                h = hard_offset(e)
            else:
                h = hard_cap(e, T_K)
            sigma = max(h * sf * 1e6, floor_pa) + h_reg_pa * e
            if mt_frac > 0.0 and prev_s is not None:
                sigma = max(sigma, prev_s + mt_frac * Ev * (e - prev_e))
            rows.append((T_K, e, sigma))
            prev_e, prev_s = e, sigma
    return rows


def diagnose(arm, rows):
    by_T = {}
    for T_K, e, s in rows:
        by_T.setdefault(T_K, []).append((e, s))
    print(f"\n=== arm={arm} ===")
    print(f"{'T(K)':>9} {'sig(0)MPa':>10} {'E GPa':>7} {'H1 GPa':>8} {'H1/E':>6} "
          f"{'H2 GPa':>8} {'sig(0.2)MPa':>12}")
    worst = 0.0
    for T_K in sorted(by_T):
        pts = sorted(by_T[T_K])
        H1 = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
        H2 = (pts[2][1] - pts[1][1]) / (pts[2][0] - pts[1][0])
        Ev = E_at(T_K)
        worst = max(worst, H1 / Ev)
        print(f"{T_K:>9.2f} {pts[0][1]/1e6:>10.1f} {Ev/1e9:>7.1f} {H1/1e9:>8.1f} "
              f"{H1/Ev:>6.2f} {H2/1e9:>8.1f} {pts[-1][1]/1e6:>12.1f}")
    verdict = "可容许 (H1 < E)" if worst < 1.0 else "不可容许 (H1 >= E)"
    print(f"  worst H1/E = {worst:.2f}  -> {verdict}")
    return worst


def write(arm, rows, name=None, floor_pa=FLOOR_PA, h_reg_pa=H_REG_PA, mt_frac=0.0):
    label = name or arm
    src = (f"modified J-C (Balbaa2022 p.33) A={A} B={B} n={N} m={M} Tm={TM}C Tr={TR}C; "
           f"rate term omitted (D-V2-02); floor {floor_pa:.0e} Pa + H_reg {h_reg_pa:.0e} Pa "
           f"(D-V2-19); near-zero tangent regularization arm={arm} (D-V2-22"
           + (f", e_off={E_OFF}" if arm == "offset" else "")
           + (f", cap={CAP_FRAC}*E" if arm == "cap" else "") + ")")
    if mt_frac > 0.0:
        src += (f"; D-V2-19-R1 minimum plastic tangent H >= {mt_frac:g}*E(T)")
    if label != arm:
        src += (f"; D-V2-19 single-variable ladder arm '{label}' "
                f"(floor={floor_pa:.0e} Pa, H_reg={h_reg_pa:.0e} Pa)")
    p = TABLES / f"flow_curve_{label}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature_K", "equivalent_plastic_strain", "flow_stress_Pa", "source"])
        for T_K, e, s in rows:
            w.writerow([f"{T_K:.2f}", f"{e:g}", f"{s:.6g}", src])
    print(f"  wrote {p.name}: {len(rows)} rows")

    base = json.loads((HERE / "v2_material_config.json").read_text(encoding="utf-8"))
    base["_comment"] = (f"D-V2-22 arm '{label}': flow_curve 的近零切线正则化臂。"
                        f"其余字段与 v2_material_config.json 逐字相同。"
                        + base["_comment"])
    base["flow_curve_table"] = str((TABLES / f"flow_curve_{label}.csv").resolve())
    q = HERE / f"v2_material_config_fc_{label}.json"
    q.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {q.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS + ("all",), required=True)
    ap.add_argument("--quantify", action="store_true", help="只打印切线诊断,不写文件")
    ap.add_argument("--name", default=None,
                    help="输出臂名(默认 = --arm)。用于 D-V2-19 单变量试验梯。")
    ap.add_argument("--floor-pa", type=float, default=FLOOR_PA,
                    help=f"D-V2-19 高温流动应力地板,默认 {FLOOR_PA:.0e} Pa")
    ap.add_argument("--h-reg-pa", type=float, default=H_REG_PA,
                    help=f"D-V2-19 附加硬化模量,默认 {H_REG_PA:.0e} Pa")
    ap.add_argument("--min-tangent-frac", type=float, default=0.0,
                    help="D-V2-19-R1:逐段最小塑性切线 H >= frac*E(T)。0 = 关闭。")
    args = ap.parse_args()

    if args.name and args.arm == "all":
        ap.error("--name 只能与单个 --arm 一起用")

    targets = ARMS if args.arm == "all" else (args.arm,)
    for arm in targets:
        rows = build(arm, args.floor_pa, args.h_reg_pa, args.min_tangent_frac)
        diagnose(args.name or arm, rows)
        if not args.quantify:
            write(arm, rows, args.name, args.floor_pa, args.h_reg_pa,
                  args.min_tangent_frac)


if __name__ == "__main__":
    main()
