#!/usr/bin/env python3
"""Extract melt-pool metrics from a V1 run directory.

Measurement definitions (inputs/nist-meltpool.json):
  - melt-pool boundary: solidus 1563 K (= 1290 C), same convention as both
    Balbaa2022 and Lane2020.
  - width/depth: from the ever-melted region (cell max_temperature_history
    >= solidus) == etched cross-section fusion zone. Width is reported at
    mid-track (x = 0.5 mm) and as the max over the steady window
    (x in [0.4, 0.7] mm). Depth is reported both from the powder-layer top
    (z = 300 um) and from the substrate top (z = 280 um).
  - length: instantaneous x-extent of T >= solidus while the laser is on
    (in-situ thermography analogue), evaluated per output frame; steady
    value = median over frames with the laser in x in [0.5, 0.9] mm.
  - cooling rate: at the track-center surface node (0.5, 0.24, 0.30) mm,
    time from crossing 1290 C down to 1190 C (NIST reference band) and to
    1000 C (Lane Table 3 second band), linear interpolation between frames.
Zero-calibration: no knob here is fitted; definitions fixed a priori.
"""
import argparse
import glob
import json
import os
import re

import meshio
import numpy as np

SOLIDUS = 1563.0
C2K = 273.15
LAYER_TOP = 300.0e-6
SUBSTRATE_TOP = 280.0e-6
PROBE = np.array([0.5e-3, 0.24e-3, LAYER_TOP])

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--out", default=None)
args = ap.parse_args()

vtus = sorted(glob.glob(os.path.join(args.run_dir, "*.vtu")))
if not vtus:
    raise SystemExit(f"no VTU files in {args.run_dir}")

# exact per-step times from the solver's own record (the scan dt is set by
# the path-row spacing, NOT by --dt; discovered on the first CBM-B run)
step_time = {}
with open(os.path.join(args.run_dir, "path_used.csv")) as f:
    import csv as _csv
    for row in _csv.DictReader(f):
        step_time[int(row["step"])] = float(row["time"])

def step_of(path):
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[-1]) if m else -1

frames = sorted(((step_of(p), p) for p in vtus), key=lambda t: t[0])

first = meshio.read(frames[0][1])
pts = first.points
cells = first.cells_dict.get("hexahedron")
cell_centers = pts[cells].mean(axis=1)
probe_node = int(np.argmin(np.linalg.norm(pts - PROBE, axis=1)))
probe_err = float(np.linalg.norm(pts[probe_node] - PROBE))

# structured-grid shape (make_v1_mesh.py: i fastest, then j, then k)
NX, NY, NZ = 100, 48, 30
XS = np.arange(NX + 1) * 10.0e-6
YS = np.arange(NY + 1) * 10.0e-6
ZS = np.arange(NZ + 1) * 10.0e-6

records = []
Tmax_nodal = None
for step, path in frames:
    m = meshio.read(path)
    T = np.asarray(m.point_data["T"]).reshape(-1)
    Tmax_nodal = T.copy() if Tmax_nodal is None else np.maximum(Tmax_nodal, T)
    maxTh = None
    for key in ("max_temperature_history",):
        if key in m.cell_data_dict and "hexahedron" in m.cell_data_dict[key]:
            maxTh = np.asarray(m.cell_data_dict[key]["hexahedron"]).reshape(-1)
    molten = T >= SOLIDUS
    rec = {
        "step": step,
        "time": step_time.get(step, float("nan")),
        "Tmax": float(T.max()),
        "T_probe": float(T[probe_node]),
        "pool_len": 0.0,
        "pool_x_max": None,
    }
    if molten.any():
        xm = pts[molten, 0]
        rec["pool_len"] = float(xm.max() - xm.min())
        rec["pool_x_max"] = float(xm.max())
    if maxTh is not None:
        rec["_maxTh"] = maxTh
    records.append(rec)

# --- fusion-zone geometry from the last frame's max_temperature_history ---
last = records[-1]
geom = {}
if "_maxTh" in last:
    fused = last["_maxTh"] >= SOLIDUS
    fc = cell_centers[fused]
    if len(fc):
        mid = (fc[:, 0] > 0.45e-3) & (fc[:, 0] < 0.55e-3)
        steady = (fc[:, 0] > 0.4e-3) & (fc[:, 0] < 0.7e-3)
        def width(sel):
            if not sel.any():
                return 0.0
            # cell centers are half a cell from the faces; add one cell (10 um)
            return float(fc[sel, 1].max() - fc[sel, 1].min() + 10.0e-6)
        geom = {
            "n_fused_cells": int(fused.sum()),
            "width_mid_um": width(mid) * 1e6,
            "width_steady_max_um": width(steady) * 1e6,
            "depth_from_layer_top_um": (LAYER_TOP - float(fc[steady, 2].min() - 5.0e-6)) * 1e6 if steady.any() else 0.0,
            "depth_from_substrate_top_um": (SUBSTRATE_TOP - float(fc[steady, 2].min() - 5.0e-6)) * 1e6 if steady.any() else 0.0,
            "fusion_x_extent_mm": [float(fc[:, 0].min()) * 1e3, float(fc[:, 0].max()) * 1e3],
        }

# --- sub-cell fusion geometry from nodal max-T over output frames ---
# (frames sample every 2nd step, so nodal peaks can be slightly missed;
#  the cell max_temperature_history numbers above are the exact-but-
#  quantized companion. Report both.)
grid = Tmax_nodal.reshape(NZ + 1, NY + 1, NX + 1)

def extent_interp(vals, coords):
    """max extent of vals >= SOLIDUS along coords with linear interpolation"""
    above = vals >= SOLIDUS
    if not above.any():
        return None, None
    i = np.where(above)[0]
    lo, hi = i[0], i[-1]
    x_lo = coords[lo]
    if lo > 0:
        f = (SOLIDUS - vals[lo - 1]) / (vals[lo] - vals[lo - 1])
        x_lo = coords[lo - 1] + f * (coords[lo] - coords[lo - 1])
    x_hi = coords[hi]
    if hi < len(vals) - 1:
        f = (vals[hi] - SOLIDUS) / (vals[hi] - vals[hi + 1])
        x_hi = coords[hi] + f * (coords[hi + 1] - coords[hi])
    return x_lo, x_hi

interp = {}
mid_ix = [i for i, x in enumerate(XS) if 0.45e-3 <= x <= 0.55e-3]
widths, depths = [], []
for ix in mid_ix:
    for iz in range(NZ + 1):
        lo, hi = extent_interp(grid[iz, :, ix], YS)
        if lo is not None:
            widths.append(hi - lo)
    for iy in range(NY + 1):
        zlo, _ = extent_interp(grid[:, iy, ix], ZS)
        if zlo is not None:
            depths.append(zlo)
if widths:
    interp["width_interp_um"] = float(max(widths)) * 1e6
if depths:
    zmin = float(min(depths))
    interp["depth_interp_from_layer_top_um"] = (LAYER_TOP - zmin) * 1e6
    interp["depth_interp_from_substrate_top_um"] = (SUBSTRATE_TOP - zmin) * 1e6

# --- pool length: median over frames with pool front in the steady window ---
lens = [r["pool_len"] for r in records
        if r["pool_x_max"] is not None and 0.5e-3 <= r["pool_x_max"] <= 0.9e-3]
length_um = float(np.median(lens) * 1e6) if lens else 0.0

# --- steadiness: Tmax over laser-on frames ---
tmax_series = [(r["time"], r["Tmax"]) for r in records]

# --- cooling rate at probe ---
times = np.array([r["time"] for r in records])
Tp = np.array([r["T_probe"] for r in records])
def crossing(level):
    """first downward crossing after the global max"""
    i0 = int(np.argmax(Tp))
    for i in range(i0, len(Tp) - 1):
        if Tp[i] >= level > Tp[i + 1]:
            f = (Tp[i] - level) / (Tp[i] - Tp[i + 1])
            return times[i] + f * (times[i + 1] - times[i])
    return None

cr = {}
t1290 = crossing(1290.0 + C2K)
for label, level_c in (("1190", 1190.0), ("1000", 1000.0)):
    t_lo = crossing(level_c + C2K)
    if t1290 is not None and t_lo is not None and t_lo > t1290:
        cr[f"cooling_rate_1290_{label}_C_per_s"] = (1290.0 - level_c) / (t_lo - t1290)
    else:
        cr[f"cooling_rate_1290_{label}_C_per_s"] = None

result = {
    "run_dir": args.run_dir,
    "frames": len(records),
    "probe_node_offset_m": probe_err,
    "peak_T_probe_K": float(Tp.max()),
    "Tmax_last5": [round(t, 1) for _, t in tmax_series[-5:]],
    "Tmax_overall_K": float(max(t for _, t in tmax_series)),
    "pool_length_median_um": length_um,
    "pool_length_frames_used": len(lens),
    **geom,
    **interp,
    **cr,
    "definitions": "solidus 1563 K boundary; see module docstring",
}
out = args.out or os.path.join(args.run_dir, "v1_meltpool_metrics.json")
with open(out, "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps(result, indent=1))
print(f"\nwritten: {out}")
