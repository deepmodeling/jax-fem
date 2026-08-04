#!/usr/bin/env python3
"""D-V2-22 后续:把"flow curve 是病因"这个结论再切一刀。

2026-08-04 的实况:A 相四臂(asis / offset / cap / asis_iter)**全部失败**,而且
全部停在同一处(ledger=12, global_step=10)。三臂在近零端的跨度是 H1/E = 2.98
(asis) → 0.23 (offset) → 0.10 (cap),即近零切线被从"比弹性还硬"一路括到"十分之一
弹性",**收敛行为纹丝不动**。所以 D-V2-22 的近零奇异虽然真实(物理侧 1746 MPa 的
交叉核对仍然成立),但它**不是这个不收敛的原因**——这条因果被证伪了。

上一轮的 T1 对照之所以看起来指向 flow curve,是因为它一次换掉了三样东西:

    T1 占位表:  sigma_y = 490 MPa 常数(全温区),H = 1 GPa 常数,
                 且走的是 yield_table + hardening_table 这条**另一条代码路径**。

    三臂:       sigma_y(T) 从 650 MPa 软化到 1 MPa(D-V2-19 的固相线地板),
                 H_reg = 1e7 Pa,走 flow_curve_table 路径。

一次动三样,分不出是哪样。本脚本生成两个把它们逐一分开的臂:

  flat      sigma = 490 MPa + 1e9 * eps_p,全温区常数——**与 T1 物理上逐点等价**,
            但经由 flow_curve_table 交付。
            → 若 flat 也失败,病因就是 flow_curve_table 这条**消费路径本身**
              (共享求解器代码),而不是任何表值。按用户既定约束:停下来报告,
              不自行修改共享代码。
            → 若 flat 通过,消费路径无罪,病因在表值,进入下一刀。

  hotfloor  sigma = max(offset 臂值, 490 MPa + 1e9 * eps_p)。
            近零端与 offset 逐点相同(室温 650 MPa 远高于 490 MPa 地板,不受影响),
            只把**高温软化段**抬到 T1 的水平(1273 K 处 167 → 490 MPa,
            固相线处 1 → 490 MPa)。连续,无跳变。
            → 若 hotfloor 通过,病因是 D-V2-19 的高温正则化对(1 MPa 地板 +
              H_reg 1e7 Pa)在这个载荷下不足:H/E = 1e7/6.16e10 = 1.6e-4,
              近乎理想塑性,成千上万个这样的单元把整体切线压成近奇异。
            → 若 hotfloor 也失败,病因在亚固相线的温度依赖本身。

两个臂都是**诊断用的二分臂,不是物理配置**,任何一个都不得直接采纳为 V2 的
生产表——采纳需要另行登记(D-V2-19 的修订)。
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
TABLES = HERE / "tables"

# T1 占位对(V1 yield_placeholder.csv / hardening_placeholder.csv 的值)
FLAT_SIGMA_PA = 4.9e8
FLAT_H_PA = 1.0e9


def read_arm(name):
    """读回已生成的臂表,返回 {(T_K, eps): sigma_Pa} 与有序键。"""
    rows = []
    with open(TABLES / f"flow_curve_{name}.csv") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["temperature_K"]),
                         float(r["equivalent_plastic_strain"]),
                         float(r["flow_stress_Pa"])))
    return rows


def write(arm, rows, src):
    p = TABLES / f"flow_curve_{arm}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature_K", "equivalent_plastic_strain",
                    "flow_stress_Pa", "source"])
        for T_K, e, s in rows:
            w.writerow([f"{T_K:.2f}", f"{e:g}", f"{s:.6g}", src])
    print(f"  wrote {p.name}: {len(rows)} rows")

    base = json.loads((HERE / "v2_material_config.json").read_text(encoding="utf-8"))
    base["_comment"] = (
        f"D-V2-22 二分臂 '{arm}'(诊断用,非物理配置,不得直接采纳)。" + base["_comment"])
    base["flow_curve_table"] = str((TABLES / f"flow_curve_{arm}.csv").resolve())
    q = HERE / f"v2_material_config_fc_{arm}.json"
    q.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {q.name}")


offset = read_arm("offset")

# --- flat:与 T1 逐点等价,经 flow_curve_table 交付 -------------------------
flat = [(T, e, FLAT_SIGMA_PA + FLAT_H_PA * e) for T, e, _ in offset]
print("=== arm=flat ===")
print(f"  sigma(eps_p) = {FLAT_SIGMA_PA:.3g} + {FLAT_H_PA:.3g}*eps_p,全温区常数")
print(f"  与 T1 的 yield_placeholder/hardening_placeholder 逐点等价")
write("flat", flat,
      "BISECTION ARM (not physics): T1 placeholder pair 4.9e8 Pa yield + 1e9 Pa "
      "hardening, delivered via flow_curve_table to isolate the consumption path "
      "from the table values (D-V2-22 follow-up 2026-08-04)")

# --- hotfloor:offset 的近零端 + T1 的高温地板 ------------------------------
hotfloor = [(T, e, max(s, FLAT_SIGMA_PA + FLAT_H_PA * e)) for T, e, s in offset]
print("\n=== arm=hotfloor ===")
print(f"{'T(K)':>9} {'offset sig0':>12} {'hotfloor sig0':>14} {'lifted':>7}")
seen = set()
for (T, e, s0), (_, _, s1) in zip(offset, hotfloor):
    if e != 0.0 or T in seen:
        continue
    seen.add(T)
    print(f"{T:>9.2f} {s0/1e6:>12.1f} {s1/1e6:>14.1f} {'yes' if s1 > s0 else '-':>7}")
write("hotfloor", hotfloor,
      "BISECTION ARM (not physics): offset arm with its high-temperature branch "
      "floored at the T1 placeholder level (4.9e8 Pa + 1e9 Pa*eps_p) to separate "
      "the D-V2-19 hot-end regularization from the near-zero tangent "
      "(D-V2-22 follow-up 2026-08-04)")
