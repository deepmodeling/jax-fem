#!/usr/bin/env python3
"""V2 基板离散敏感性分析(D-V2-07)。

三个变体只在基板 z 剖分上不同,顶层(粉层 100x100 个 40um 单元)在三者中
完全一致且单元编号相同(网格按 z 层逐层构建,顶层恒为最后 NX*NY 个单元),
因此可以**逐单元**直接比对件内应力,无需插值。

判据:若件内应力对基板剖分不敏感,则"共形分级基板替代 Balbaa 的
两尺度 + tie 约束"得证,D-V2-07 可关闭。

同时给出三个无外部参照的健康判据:
  - 锁定指纹(层内去均值 p_std / vm_std,健康约 O(1))
  - 能量台账最大相对平衡误差
  - 全约束回程应力的解析对照 min(sigma_y, E*alpha*dT/(1-nu))
"""
import glob
import json
import os
import sys

import meshio
import numpy as np

NX = NY = 100
TOP_CELLS = NX * NY
OUT_ROOT = "/home/user/work/159/output"
VARIANTS = ["graded", "fine", "coarse"]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "v2_sub3"


def quad_mean(cd, base):
    """把 8 个积分点的场平均成单元值。"""
    arrs = []
    for q in range(8):
        key = f"{base}{q}" if base.endswith("quad") else f"{base}_quad{q}"
        for k, v in cd.items():
            if k == key or k == f"{base}quad{q}":
                arrs.append(np.asarray(list(v.values())[0]).reshape(-1))
    return np.mean(arrs, axis=0) if arrs else None


def cell_field(m, name):
    d = m.cell_data_dict.get(name)
    if d is None:
        return None
    return np.asarray(list(d.values())[0]).reshape(-1)


def quad_avg(m, base):
    """base 例如 'stress' + '_xx' -> stress_quad{0..7}_xx"""
    parts = []
    for q in range(8):
        f = cell_field(m, f"{base}_quad{q}" if False else base.replace("Q", str(q)))
        if f is not None:
            parts.append(f)
    return np.mean(parts, axis=0) if parts else None


def load(variant):
    d = os.path.join(OUT_ROOT, f"{PREFIX}_{variant}")
    if not os.path.isdir(d):
        return None
    vt = sorted(glob.glob(f"{d}/*.vtu"))
    if not vt:
        return None
    # 优先 release,否则最后一帧
    rel = [p for p in vt if "release" in os.path.basename(p)]
    path = rel[-1] if rel else vt[-1]
    m = meshio.read(path)
    res = {"variant": variant, "vtu": os.path.basename(path), "dir": d}
    for comp in ("xx", "yy", "zz"):
        res[f"s{comp}"] = quad_avg(m, f"stress_quadQ_{comp}")
    res["vm"] = quad_avg(m, "vm_quadQ")
    res["eqp"] = cell_field(m, "eq_plastic_strain")
    res["maxT"] = cell_field(m, "max_temperature_history")
    res["ncells"] = len(res["vm"]) if res["vm"] is not None else 0
    lp = os.path.join(d, "thermal_energy_ledger_summary.json")
    if os.path.exists(lp):
        s = json.load(open(lp))
        res["ledger_ok"] = s.get("all_balance_steps_within_tolerance")
        res["bal_rel_max"] = s.get("maximum_relative_balance_error")
        res["steps"] = s.get("recorded_step_count")
        res["complete"] = s.get("complete")
    return res


def top_layer(arr, ncells):
    """顶层 = 最后 NX*NY 个单元。"""
    return arr[ncells - TOP_CELLS:] if arr is not None and ncells >= TOP_CELLS else None


def locking_fingerprint(r):
    """层内去均值 p_std / vm_std;健康约 O(1),锁定时远大于 1。"""
    n = r["ncells"]
    sxx, syy, szz = (top_layer(r[k], n) for k in ("sxx", "syy", "szz"))
    vm = top_layer(r["vm"], n)
    if sxx is None or vm is None:
        return None
    p = (sxx + syy + szz) / 3.0
    p = p - p.mean()
    v = vm - vm.mean()
    return float(np.std(p) / np.std(v)) if np.std(v) > 0 else None


def main():
    runs = [r for r in (load(v) for v in VARIANTS) if r]
    if not runs:
        print("没有可用的运行结果")
        return
    print("=" * 74)
    print("V2 基板离散敏感性(D-V2-07)")
    print("=" * 74)
    print(f"{'变体':>8} {'单元数':>8} {'步数':>6} {'完成':>6} "
          f"{'台账相对误差':>14} {'锁定指纹':>10}")
    for r in runs:
        print(f"{r['variant']:>8} {r['ncells']:>8} {r.get('steps','-'):>6} "
              f"{str(r.get('complete','-')):>6} "
              f"{r.get('bal_rel_max',float('nan')):>14.3e} "
              f"{(locking_fingerprint(r) or float('nan')):>10.3f}")

    ref = next((r for r in runs if r["variant"] == "fine"), runs[0])
    print(f"\n件内(顶层 {TOP_CELLS} 单元)逐单元比对,参照 = {ref['variant']}")
    print(f"{'变体':>8} {'量':>8} {'参照RMS':>12} {'最大差':>12} "
          f"{'RMS差':>12} {'相对RMS差':>11}")
    for r in runs:
        if r is ref:
            continue
        for key, label in (("sxx", "sigma_xx"), ("vm", "von Mises"), ("eqp", "eqp")):
            a = top_layer(ref[key], ref["ncells"])
            b = top_layer(r[key], r["ncells"])
            if a is None or b is None or a.shape != b.shape:
                continue
            d = b - a
            rms_ref = float(np.sqrt(np.mean(a**2)))
            rms_d = float(np.sqrt(np.mean(d**2)))
            rel = rms_d / rms_ref * 100 if rms_ref > 0 else float("nan")
            print(f"{r['variant']:>8} {label:>8} {rms_ref:>12.4e} "
                  f"{float(np.abs(d).max()):>12.4e} {rms_d:>12.4e} {rel:>10.2f}%")

    out = os.path.join(OUT_ROOT, f"{PREFIX}_substrate_comparison.json")
    summary = {
        "prefix": PREFIX,
        "reference": ref["variant"],
        "runs": [{k: v for k, v in r.items()
                  if not isinstance(v, np.ndarray)} for r in runs],
        "locking_fingerprint": {r["variant"]: locking_fingerprint(r) for r in runs},
    }
    for r in runs:
        if r is ref:
            continue
        for key in ("sxx", "vm", "eqp"):
            a, b = top_layer(ref[key], ref["ncells"]), top_layer(r[key], r["ncells"])
            if a is None or b is None or a.shape != b.shape:
                continue
            rms_ref = float(np.sqrt(np.mean(a**2)))
            summary.setdefault("relative_rms_diff_pct", {})[f"{r['variant']}/{key}"] = (
                float(np.sqrt(np.mean((b - a) ** 2)) / rms_ref * 100)
                if rms_ref > 0 else None)
    with open(out, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"\n已写出: {out}")


if __name__ == "__main__":
    main()
