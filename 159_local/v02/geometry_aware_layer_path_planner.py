#!/usr/bin/env python3
"""
Geometry-aware layer scanner/path planner for AM thermal simulations.

This standalone tool pre-scans a TET4 INP mesh, records per-layer sample
point/cell distributions, and generates a path_file CSV that follows the actual
occupied geometry in each layer instead of scanning a full rectangular control
plane.

It is intended to be used before am_thermal_stress_upgraded.py:
    python geometry_aware_layer_path_planner.py ... --path-output path_geometry_aware.csv
    python am_thermal_stress_upgraded.py ... --path-file path_geometry_aware.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from inp_initial_guess_smoke import read_tet4_inp
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import read_tet4_inp. Run with PYTHONPATH pointing to "
        "159_local/v01, e.g.\n"
        "PYTHONPATH=/home/user/work/159/jax-fem/159_local/v01:/home/user/work/159/jax-fem python3 ...\n"
        f"Original error: {exc}"
    )

AXIS_TO_ID = {"x": 0, "y": 1, "z": 2}
ID_TO_AXIS = ("x", "y", "z")


@dataclass
class PathRow:
    step: int
    time: float
    mode: str
    layer: int
    hatch: int
    scan: int
    center: np.ndarray
    front_coord: float
    power: float
    laser_on: int
    dt: float


@dataclass
class LayerInfo:
    layer: int
    front_coord: float
    band: float
    selected_count: int
    scan_min: float
    scan_max: float
    hatch_min: float
    hatch_max: float
    hatch_count: int
    segment_count: int
    path_points: int
    covered_count: int
    missing_count: int
    missing_fraction: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate geometry-aware layer scan path from actual mesh occupancy.")
    p.add_argument("--inp", required=True)
    p.add_argument("--max-cells", type=int, default=0)
    p.add_argument("--mesh-length-scale", type=float, default=1.0)

    p.add_argument("--build-axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--base-side", choices=("min", "max"), default="min")
    p.add_argument("--scan-axis", choices=("auto", "x", "y", "z"), default="auto")
    p.add_argument("--scan-rotation-per-layer", type=float, default=0.0)

    p.add_argument("--layer-thickness", type=float, required=True)
    p.add_argument("--layers", type=int, default=None, help="Derived from geometry if omitted.")
    p.add_argument("--max-print-layers", type=int, default=10)

    p.add_argument("--planning-entity", choices=("cell", "node", "both"), default="cell",
                   help="Samples used to infer occupied layer geometry. Cell is closest to TET4 one-point quadrature.")
    p.add_argument("--layer-band", type=float, default=None,
                   help="Thickness used to collect samples below current front. Defaults to max(layer_thickness, source_depth).")
    p.add_argument("--auto-expand-layer-band", action="store_true", default=True)
    p.add_argument("--no-auto-expand-layer-band", dest="auto_expand_layer_band", action="store_false")
    p.add_argument("--max-layer-band", type=float, default=5.0e-3)
    p.add_argument("--min-samples-per-layer", type=int, default=20)

    p.add_argument("--scan-start-frac", type=float, default=0.0)
    p.add_argument("--scan-end-frac", type=float, default=1.0)
    p.add_argument("--hatch-start-frac", type=float, default=0.0)
    p.add_argument("--hatch-end-frac", type=float, default=1.0)
    p.add_argument("--scan-start", type=float, default=None)
    p.add_argument("--scan-end", type=float, default=None)
    p.add_argument("--hatch-start", type=float, default=None)
    p.add_argument("--hatch-end", type=float, default=None)

    p.add_argument("--hatch-lines-per-layer", type=int, default=30)
    p.add_argument("--hatch-spacing", type=float, default=None,
                   help="If set, overrides hatch-lines-per-layer per layer based on occupied hatch span.")
    p.add_argument("--lane-half-width", type=float, default=None,
                   help="Sample association half-width for each hatch lane. Defaults to coverage_radius = factor*beam_radius.")
    p.add_argument("--segment-gap", type=float, default=None,
                   help="Split a hatch into multiple scan segments if projected sample gaps exceed this. Defaults to 4*coverage_radius.")
    p.add_argument("--segment-margin", type=float, default=None,
                   help="Extend each segment at both ends. Defaults to coverage_radius.")
    p.add_argument("--min-segment-length", type=float, default=1.0e-9)
    p.add_argument("--serpentine", action="store_true", default=True)
    p.add_argument("--no-serpentine", dest="serpentine", action="store_false")

    p.add_argument("--scan-speed", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=1.0e-4)
    p.add_argument("--auto-scan-steps-from-speed", action="store_true", default=True)
    p.add_argument("--no-auto-scan-steps-from-speed", dest="auto_scan_steps_from_speed", action="store_false")
    p.add_argument("--scan-steps-per-segment", type=int, default=100)
    p.add_argument("--jump-speed", type=float, default=0.0, help="Laser-off move speed between segments. 0 writes one jump row.")

    p.add_argument("--laser-power", type=float, default=800.0)
    p.add_argument("--beam-radius", type=float, default=1.0e-3)
    p.add_argument("--source-depth", type=float, default=2.5e-4)
    p.add_argument("--coverage-radius-factor", type=float, default=2.0)
    p.add_argument("--coverage-depth-factor", type=float, default=3.0)

    p.add_argument("--path-output", required=True)
    p.add_argument("--output-dir", default="/home/user/work/159/output/geometry_path_planner")
    p.add_argument("--coverage-chunk-size", type=int, default=2048)
    return p.parse_args()


def resolve_scan_and_hatch_axes(scan_axis: str, build_axis_id: int) -> Tuple[int, int]:
    plane = [i for i in range(3) if i != build_axis_id]
    if scan_axis == "auto":
        scan_axis_id = plane[0]
    else:
        scan_axis_id = AXIS_TO_ID[scan_axis]
    if scan_axis_id == build_axis_id:
        raise ValueError("scan_axis cannot equal build_axis")
    hatch_axis_id = [i for i in plane if i != scan_axis_id][0]
    return scan_axis_id, hatch_axis_id


def axis_range(pmin: np.ndarray, pmax: np.ndarray, axis_id: int, start: Optional[float], end: Optional[float], start_frac: float, end_frac: float) -> Tuple[float, float]:
    a0 = float(start) if start is not None else float(pmin[axis_id] + start_frac * (pmax[axis_id] - pmin[axis_id]))
    a1 = float(end) if end is not None else float(pmin[axis_id] + end_frac * (pmax[axis_id] - pmin[axis_id]))
    return min(a0, a1), max(a0, a1)


def layer_front(layer: int, pmin: np.ndarray, pmax: np.ndarray, build_axis_id: int, base_side: str, layer_thickness: float) -> float:
    base = float(pmin[build_axis_id] if base_side == "min" else pmax[build_axis_id])
    sign = 1.0 if base_side == "min" else -1.0
    return base + sign * layer * layer_thickness


def layer_distance_from_base(x: np.ndarray, base_coord: float, build_axis_id: int, build_sign: float) -> np.ndarray:
    return build_sign * (x[:, build_axis_id] - base_coord)


def make_layer_basis(scan_axis_id: int, hatch_axis_id: int, layer_idx_zero: int, rotation_per_layer_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    e0 = np.zeros(3)
    e1 = np.zeros(3)
    e0[scan_axis_id] = 1.0
    e1[hatch_axis_id] = 1.0
    theta = math.radians(rotation_per_layer_deg * layer_idx_zero)
    c = math.cos(theta)
    s = math.sin(theta)
    e_scan = c * e0 + s * e1
    e_hatch = -s * e0 + c * e1
    return e_scan, e_hatch


def select_samples_for_layer(samples: np.ndarray, layer: int, args: argparse.Namespace, pmin: np.ndarray, pmax: np.ndarray, build_axis_id: int, build_sign: float, base_coord: float, scan_axis_id: int, hatch_axis_id: int) -> Tuple[np.ndarray, float]:
    base_band = args.layer_band if args.layer_band is not None else max(args.layer_thickness, args.source_depth)
    band = float(base_band)

    scan_lo, scan_hi = axis_range(pmin, pmax, scan_axis_id, args.scan_start, args.scan_end, args.scan_start_frac, args.scan_end_frac)
    hatch_lo, hatch_hi = axis_range(pmin, pmax, hatch_axis_id, args.hatch_start, args.hatch_end, args.hatch_start_frac, args.hatch_end_frac)
    in_window = (
        (samples[:, scan_axis_id] >= scan_lo) & (samples[:, scan_axis_id] <= scan_hi) &
        (samples[:, hatch_axis_id] >= hatch_lo) & (samples[:, hatch_axis_id] <= hatch_hi)
    )
    dist = layer_distance_from_base(samples, base_coord, build_axis_id, build_sign)
    target_top = layer * args.layer_thickness

    while True:
        lo = max(0.0, target_top - band)
        hi = target_top
        mask = in_window & (dist >= lo) & (dist <= hi)
        count = int(mask.sum())
        if count >= args.min_samples_per_layer or not args.auto_expand_layer_band or band >= args.max_layer_band:
            return samples[mask], band
        band = min(args.max_layer_band, band * 1.5)


def hatch_offsets(h_vals: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    h_min = float(np.min(h_vals))
    h_max = float(np.max(h_vals))
    if h_max <= h_min:
        return np.array([0.5 * (h_min + h_max)])
    if args.hatch_spacing is not None and args.hatch_spacing > 0:
        n = max(1, int(math.floor((h_max - h_min) / args.hatch_spacing)) + 1)
        vals = [h_min + i * args.hatch_spacing for i in range(n)]
        if vals[-1] < h_max - 0.25 * args.hatch_spacing:
            vals.append(h_max)
        return np.asarray(vals, dtype=float)
    n = max(1, args.hatch_lines_per_layer)
    if n == 1:
        return np.array([0.5 * (h_min + h_max)])
    return np.linspace(h_min, h_max, n)


def split_segments(s_lane: np.ndarray, coverage_radius: float, args: argparse.Namespace) -> List[Tuple[float, float]]:
    if len(s_lane) == 0:
        return []
    s = np.sort(s_lane)
    gap = args.segment_gap if args.segment_gap is not None else 4.0 * coverage_radius
    margin = args.segment_margin if args.segment_margin is not None else coverage_radius
    segments: List[Tuple[float, float]] = []
    start = s[0]
    prev = s[0]
    for val in s[1:]:
        if val - prev > gap:
            a = start - margin
            b = prev + margin
            if abs(b - a) >= args.min_segment_length:
                segments.append((a, b))
            start = val
        prev = val
    a = start - margin
    b = prev + margin
    if abs(b - a) >= args.min_segment_length:
        segments.append((a, b))
    return segments


def add_jump(rows: List[PathRow], step: int, time: float, last_center: Optional[np.ndarray], next_center: np.ndarray, layer: int, hatch: int, front_coord: float, args: argparse.Namespace) -> Tuple[int, float]:
    if last_center is None:
        return step, time
    dist = float(np.linalg.norm(next_center - last_center))
    if dist <= 1e-15:
        return step, time
    if args.jump_speed > 0:
        jump_dt = dist / args.jump_speed
    else:
        jump_dt = args.dt
    time += jump_dt
    rows.append(PathRow(step, time, "jump", layer, hatch, 0, next_center.copy(), front_coord, 0.0, 0, jump_dt))
    return step + 1, time


def add_scan_segment(rows: List[PathRow], step: int, time: float, layer: int, hatch: int, front_coord: float, e_scan: np.ndarray, e_hatch: np.ndarray, h: float, s0: float, s1: float, args: argparse.Namespace) -> Tuple[int, float, np.ndarray, int]:
    length = abs(s1 - s0)
    if args.auto_scan_steps_from_speed:
        n = max(2, int(math.ceil(length / max(args.scan_speed * args.dt, 1e-30))) + 1)
    else:
        n = max(2, args.scan_steps_per_segment)
    last_center = None
    for i in range(n):
        frac = i / float(n - 1)
        s = s0 + frac * (s1 - s0)
        center = e_scan * s + e_hatch * h
        # e_scan/e_hatch have no build-axis component, so set below outside from nonzero? Find zero component as not enough.
        # We set all rows' build coordinate by overriding later in caller using front_coord.
        time += args.dt
        rows.append(PathRow(step, time, "scan", layer, hatch, i + 1, center.copy(), front_coord, args.laser_power, 1, args.dt))
        last_center = center.copy()
        step += 1
    return step, time, last_center, n


def coverage_for_layer(samples: np.ndarray, centers: np.ndarray, front_coord: float, build_axis_id: int, build_sign: float, beam_radius: float, source_depth: float, coverage_radius_factor: float, coverage_depth_factor: float, chunk_size: int) -> np.ndarray:
    if len(samples) == 0:
        return np.zeros(0, dtype=bool)
    if len(centers) == 0:
        return np.zeros(len(samples), dtype=bool)
    # In-plane distance is full xyz distance minus build-axis component.
    covered = np.zeros(len(samples), dtype=bool)
    cov_r2 = (coverage_radius_factor * beam_radius) ** 2
    max_depth = coverage_depth_factor * source_depth
    # Active layer samples must be at/under the laser front.
    depth = build_sign * (front_coord - samples[:, build_axis_id])
    depth_ok = (depth >= -1e-15) & (depth <= max_depth + 1e-15)
    idx_candidates = np.where(depth_ok)[0]
    if len(idx_candidates) == 0:
        return covered
    samples_ok = samples[idx_candidates]
    for start in range(0, len(samples_ok), chunk_size):
        block = samples_ok[start:start + chunk_size]
        # diff shape: B x C x 3. For moderate sizes okay; this is planner/diagnostic only.
        # To save memory, chunk over centers too.
        block_cov = np.zeros(len(block), dtype=bool)
        for c0 in range(0, len(centers), max(1, chunk_size)):
            c = centers[c0:c0 + chunk_size]
            diff = block[:, None, :] - c[None, :, :]
            diff[:, :, build_axis_id] = 0.0
            d2 = np.sum(diff * diff, axis=2)
            block_cov |= np.any(d2 <= cov_r2, axis=1)
            if np.all(block_cov):
                break
        covered[idx_candidates[start:start + len(block)]] = block_cov
    return covered


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isabs(args.path_output):
        args.path_output = os.path.join(args.output_dir, args.path_output)
    os.makedirs(os.path.dirname(args.path_output), exist_ok=True)

    points_raw, cells, selected_cells = read_tet4_inp(args.inp, args.max_cells)
    points = points_raw * args.mesh_length_scale
    cells = np.asarray(cells, dtype=np.int64)
    centroids = points[cells].mean(axis=1)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    span = pmax - pmin

    build_axis_id = AXIS_TO_ID[args.build_axis]
    scan_axis_id, hatch_axis_id = resolve_scan_and_hatch_axes(args.scan_axis, build_axis_id)
    build_sign = 1.0 if args.base_side == "min" else -1.0
    base_coord = float(pmin[build_axis_id] if args.base_side == "min" else pmax[build_axis_id])
    build_span = abs(float(pmax[build_axis_id] - pmin[build_axis_id]))
    derived_layers = int(math.ceil(build_span / args.layer_thickness))
    layers = args.layers if args.layers is not None else derived_layers
    layers = min(layers, args.max_print_layers)

    if args.planning_entity == "cell":
        samples = centroids
        sample_type = np.array(["cell"] * len(centroids), dtype=object)
        sample_ids = np.arange(len(centroids))
    elif args.planning_entity == "node":
        samples = points
        sample_type = np.array(["node"] * len(points), dtype=object)
        sample_ids = np.arange(len(points))
    else:
        samples = np.vstack([centroids, points])
        sample_type = np.array(["cell"] * len(centroids) + ["node"] * len(points), dtype=object)
        sample_ids = np.concatenate([np.arange(len(centroids)), np.arange(len(points))])

    coverage_radius = args.coverage_radius_factor * args.beam_radius
    lane_half_width = args.lane_half_width if args.lane_half_width is not None else coverage_radius

    print(f"read points={len(points)} cells={len(cells)} selected_cells={selected_cells}")
    print(f"pmin={pmin} pmax={pmax} span={span}")
    print(f"build_axis={args.build_axis} scan_axis={ID_TO_AXIS[scan_axis_id]} hatch_axis={ID_TO_AXIS[hatch_axis_id]}")
    print(f"derived_layers={derived_layers} planned_layers={layers}")
    print(f"beam_radius={args.beam_radius} source_depth={args.source_depth} coverage_radius={coverage_radius}")

    rows: List[PathRow] = []
    layer_infos: List[LayerInfo] = []
    unswept_rows = []
    step = 0
    time = 0.0
    last_center = None

    for layer in range(1, layers + 1):
        front = layer_front(layer, pmin, pmax, build_axis_id, args.base_side, args.layer_thickness)
        selected_samples, band = select_samples_for_layer(
            samples, layer, args, pmin, pmax, build_axis_id, build_sign, base_coord, scan_axis_id, hatch_axis_id
        )
        selected_indices = np.where(np.isin(samples, selected_samples).all(axis=1))[0] if False else None
        if len(selected_samples) == 0:
            print(f"layer={layer}: selected=0 band={band:.6e}; skipped")
            layer_infos.append(LayerInfo(layer, front, band, 0, math.nan, math.nan, math.nan, math.nan, 0, 0, 0, 0, 0, 1.0))
            continue

        e_scan, e_hatch = make_layer_basis(scan_axis_id, hatch_axis_id, layer - 1, args.scan_rotation_per_layer)
        # Project absolute coordinates into layer-local basis.
        s_vals = selected_samples @ e_scan
        h_vals = selected_samples @ e_hatch
        hatches = hatch_offsets(h_vals, args)
        segments_all: List[Tuple[int, float, float, float]] = []  # hatch idx, h, s0, s1
        for hi, h in enumerate(hatches, start=1):
            lane_mask = np.abs(h_vals - h) <= lane_half_width
            if not np.any(lane_mask):
                continue
            segs = split_segments(s_vals[lane_mask], coverage_radius, args)
            # Order segments according to serpentine direction.
            if args.serpentine and (hi % 2 == 0):
                segs = [(b, a) for (a, b) in reversed(segs)]
            for s0, s1 in segs:
                segments_all.append((hi, h, s0, s1))

        layer_centers = []
        for hi, h, s0, s1 in segments_all:
            # Jump to the start of this segment.
            start_center = e_scan * s0 + e_hatch * h
            start_center[build_axis_id] = front
            step, time = add_jump(rows, step, time, last_center, start_center, layer, hi, front, args)
            step_before = step
            step, time, last_center, n = add_scan_segment(rows, step, time, layer, hi, front, e_scan, e_hatch, h, s0, s1, args)
            # Fix build coordinate for rows just added and collect centers.
            for row in rows[step_before:step]:
                row.center[build_axis_id] = front
                layer_centers.append(row.center.copy())

        centers_arr = np.asarray(layer_centers) if layer_centers else np.empty((0, 3))
        covered = coverage_for_layer(
            selected_samples, centers_arr, front, build_axis_id, build_sign,
            args.beam_radius, args.source_depth, args.coverage_radius_factor,
            args.coverage_depth_factor, args.coverage_chunk_size
        )
        missing_count = int((~covered).sum())
        covered_count = int(covered.sum())
        missing_fraction = missing_count / max(len(selected_samples), 1)
        print(
            f"layer={layer:04d} selected={len(selected_samples)} band={band:.6e} "
            f"hatches={len(hatches)} segments={len(segments_all)} path_points={len(layer_centers)} "
            f"covered={covered_count} missing={missing_count} missing_frac={missing_fraction:.6e}",
            flush=True,
        )

        # Recompute selected sample ids/types for unswept output in a memory-cheap way.
        if missing_count > 0:
            # Locate selected rows by coordinates approximately only for reporting; not used by solver.
            missing_samples = selected_samples[~covered]
            for m in missing_samples[:100000]:
                unswept_rows.append([layer, float(m[0]), float(m[1]), float(m[2])])

        layer_infos.append(
            LayerInfo(
                layer, front, band, len(selected_samples), float(np.min(s_vals)), float(np.max(s_vals)),
                float(np.min(h_vals)), float(np.max(h_vals)), len(hatches), len(segments_all),
                len(layer_centers), covered_count, missing_count, missing_fraction
            )
        )

    # Write path file expected by am_thermal_stress_upgraded.py.
    with open(args.path_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z", "power", "laser_on", "layer", "hatch", "mode", "front_coord", "scan_id"])
        for r in rows:
            writer.writerow([
                f"{r.time:.12g}",
                f"{r.center[0]:.12g}", f"{r.center[1]:.12g}", f"{r.center[2]:.12g}",
                f"{r.power:.12g}", r.laser_on, r.layer, r.hatch, r.mode,
                f"{r.front_coord:.12g}", r.scan,
            ])

    layer_csv = os.path.join(args.output_dir, "layer_distribution_and_coverage.csv")
    with open(layer_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "front_coord", "selection_band", "selected_count", "scan_min", "scan_max", "hatch_min", "hatch_max",
            "hatch_count", "segment_count", "path_points", "covered_count", "missing_count", "missing_fraction",
        ])
        for info in layer_infos:
            writer.writerow([
                info.layer, info.front_coord, info.band, info.selected_count, info.scan_min, info.scan_max,
                info.hatch_min, info.hatch_max, info.hatch_count, info.segment_count, info.path_points,
                info.covered_count, info.missing_count, info.missing_fraction,
            ])

    unswept_csv = os.path.join(args.output_dir, "unswept_samples.csv")
    with open(unswept_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "x", "y", "z"])
        writer.writerows(unswept_rows)

    report = {
        "inp": args.inp,
        "points": int(len(points)),
        "cells": int(len(cells)),
        "pmin": pmin.tolist(),
        "pmax": pmax.tolist(),
        "span": span.tolist(),
        "build_axis": args.build_axis,
        "scan_axis": ID_TO_AXIS[scan_axis_id],
        "hatch_axis": ID_TO_AXIS[hatch_axis_id],
        "derived_layers": derived_layers,
        "planned_layers": layers,
        "beam_radius": args.beam_radius,
        "source_depth": args.source_depth,
        "coverage_radius": coverage_radius,
        "path_output": args.path_output,
        "layer_csv": layer_csv,
        "unswept_csv": unswept_csv,
        "total_path_rows": len(rows),
        "total_missing": int(sum(i.missing_count for i in layer_infos)),
    }
    report_path = os.path.join(args.output_dir, "geometry_path_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"path_output: {args.path_output}")
    print(f"layer_distribution: {layer_csv}")
    print(f"unswept_samples: {unswept_csv}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
