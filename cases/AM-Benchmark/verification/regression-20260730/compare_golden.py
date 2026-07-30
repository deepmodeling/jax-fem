#!/usr/bin/env python3
"""Golden-equivalence comparison: new run vs stored golden dir.

Gate design per GOLDEN_EQUIVALENCE.txt verdict (2026-07-22):
  - thermal chain must be bitwise identical (T, max_temperature_history);
  - u carries a documented same-code reproducibility band (multithreaded MKL
    x locked TET4 x j2 stall); report max/rel diffs, do not pass/fail on it
    beyond the recorded band;
  - ledger numeric fields compared per step (metadata like timestamps and
    wall-clock excluded); summary gates must all be true in the new run.
"""
import json
import sys

import meshio
import numpy as np

GOLD, NEW = sys.argv[1], sys.argv[2]
report = {}

def cmp_vtu(name, fields_point=("T", "u"), fields_cell=("max_temperature_history",)):
    g = meshio.read(f"{GOLD}/{name}")
    n = meshio.read(f"{NEW}/{name}")
    out = {}
    for f in fields_point:
        if f in g.point_data and f in n.point_data:
            a, b = np.asarray(g.point_data[f]), np.asarray(n.point_data[f])
            d = np.abs(a - b)
            scale = max(np.abs(a).max(), 1e-30)
            out[f] = {"max_abs": float(d.max()), "max_rel": float(d.max() / scale),
                      "bitwise": bool((a == b).all())}
    for f in fields_cell:
        gc = g.cell_data_dict.get(f, {})
        nc = n.cell_data_dict.get(f, {})
        for key in gc:
            if key in nc:
                a, b = np.asarray(gc[key]), np.asarray(nc[key])
                d = np.abs(a - b)
                out[f] = {"max_abs": float(d.max()), "bitwise": bool((a == b).all())}
    return out

for vtu in ("step_000000_scan.vtu", "step_000200_scan.vtu", "release.vtu"):
    try:
        report[vtu] = cmp_vtu(vtu)
    except FileNotFoundError as e:
        report[vtu] = {"error": str(e)}

# ledger: numeric fields per step, excluding metadata
NUMERIC_SKIP = {"step_state", "schema_version", "claim_level"}
def ledger_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({k: v for k, v in d.items()
                         if isinstance(v, (int, float)) and k not in NUMERIC_SKIP})
    return rows

lg = ledger_rows(f"{GOLD}/thermal_energy_ledger.jsonl")
ln = ledger_rows(f"{NEW}/thermal_energy_ledger.jsonl")
led = {"steps_gold": len(lg), "steps_new": len(ln), "worst": {}}
for i, (a, b) in enumerate(zip(lg, ln)):
    for k in a:
        if k in b:
            diff = abs(a[k] - b[k])
            scale = max(abs(a[k]), 1e-30)
            rel = diff / scale
            w = led["worst"].get(k)
            if w is None or rel > w["rel"]:
                led["worst"][k] = {"rel": rel, "abs": diff, "step": i}
led["worst"] = {k: v for k, v in sorted(led["worst"].items(),
                key=lambda kv: -kv[1]["rel"])[:8]}
report["ledger"] = led

for tag, path in (("summary_new", f"{NEW}/thermal_energy_ledger_summary.json"),
                  ("summary_gold", f"{GOLD}/thermal_energy_ledger_summary.json")):
    s = json.load(open(path))
    report[tag] = {k: v for k, v in s.items() if isinstance(v, bool) or "error" in k}

print(json.dumps(report, indent=1, default=float))
