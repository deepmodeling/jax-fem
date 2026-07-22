"""Scan-path generation: raster generator, path-file reader and helpers.

Extracted verbatim from 159_local/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import csv
import math

import numpy as onp

from jax_fem_am.config.loaders import parse_scalar
from jax_fem_am.domain.state import StepState
from jax_fem_am.mesh.model import AXIS_TO_ID, resolve_axis_range


def resolve_scan_and_hatch_axes(scan_axis, build_axis_id, plane_axis_ids):
    if scan_axis == "auto":
        scan_axis_id = plane_axis_ids[0]
    else:
        scan_axis_id = AXIS_TO_ID[scan_axis]
    if scan_axis_id == build_axis_id:
        raise ValueError("scan_axis cannot be the same as build_axis")
    hatch_axis_id = [axis for axis in plane_axis_ids if axis != scan_axis_id][0]
    return scan_axis_id, hatch_axis_id


def build_front_coord(layer_idx, layers, pmin, pmax, build_axis_id, base_side, layer_thickness=None):
    build_min = float(pmin[build_axis_id])
    build_max = float(pmax[build_axis_id])
    layer_frac = float(layer_idx + 1) / float(layers)
    if layer_thickness is not None and layer_thickness > 0.0:
        distance = min(float(layer_idx + 1) * float(layer_thickness), abs(build_max - build_min))
        if base_side == "min":
            coord = build_min + distance
        else:
            coord = build_max - distance
        return coord, layer_frac
    if base_side == "min":
        coord = build_min + layer_frac * (build_max - build_min)
    else:
        coord = build_max - layer_frac * (build_max - build_min)
    return coord, layer_frac


def update_layers_from_thickness(args, pmin, pmax, build_axis_id):
    if args.layer_thickness is None or args.layer_thickness <= 0.0:
        return
    build_span = abs(float(pmax[build_axis_id] - pmin[build_axis_id]))
    derived_layers = max(1, int(math.ceil(build_span / float(args.layer_thickness))))
    if derived_layers != args.layers:
        print(
            f"INFO: --layer-thickness overrides --layers: layers {args.layers} -> {derived_layers}",
            flush=True,
        )
    args.layers = derived_layers

    if args.max_print_layers is not None:
        if args.max_print_layers < 1:
            raise ValueError("--max-print-layers must be >= 1")
        limited_layers = min(args.layers, int(args.max_print_layers))
        if limited_layers != args.layers:
            print(
                f"INFO: --max-print-layers limits layers: {args.layers} -> {limited_layers}",
                flush=True,
            )
        args.layers = limited_layers


def path_bounds_by_axis(pmin, pmax, scan_axis_id, hatch_axis_id, args):
    bounds = {}
    scan_start, scan_end = resolve_axis_range(
        pmin, pmax, scan_axis_id, args.scan_start, args.scan_end, args.scan_start_frac, args.scan_end_frac
    )
    hatch_start, hatch_end = resolve_axis_range(
        pmin, pmax, hatch_axis_id, args.hatch_start, args.hatch_end, args.hatch_start_frac, args.hatch_end_frac
    )
    bounds[scan_axis_id] = (min(scan_start, scan_end), max(scan_start, scan_end))
    bounds[hatch_axis_id] = (min(hatch_start, hatch_end), max(hatch_start, hatch_end))
    return bounds, scan_start, scan_end, hatch_start, hatch_end


def make_layer_basis(scan_axis_id, hatch_axis_id, layer_idx, rotation_per_layer_deg):
    e_scan0 = onp.zeros(3, dtype=onp.float64)
    e_hatch0 = onp.zeros(3, dtype=onp.float64)
    e_scan0[scan_axis_id] = 1.0
    e_hatch0[hatch_axis_id] = 1.0
    theta = math.radians(float(rotation_per_layer_deg) * float(layer_idx))
    c = math.cos(theta)
    s = math.sin(theta)
    e_scan = c * e_scan0 + s * e_hatch0
    e_hatch = -s * e_scan0 + c * e_hatch0
    return e_scan, e_hatch


def make_path_center_from_bounds(pmin, pmax, bounds_by_axis):
    center = 0.5 * (onp.asarray(pmin, dtype=onp.float64) + onp.asarray(pmax, dtype=onp.float64))
    for axis, (lo, hi) in bounds_by_axis.items():
        center[axis] = 0.5 * (lo + hi)
    return center


def path_rectangle_corners(center, bounds_by_axis):
    axes = list(bounds_by_axis.keys())
    corners = []
    for v0 in bounds_by_axis[axes[0]]:
        for v1 in bounds_by_axis[axes[1]]:
            p = center.copy()
            p[axes[0]] = v0
            p[axes[1]] = v1
            corners.append(p)
    return corners


def hatch_offsets_for_layer(center, e_hatch, bounds_by_axis, args):
    corners = path_rectangle_corners(center, bounds_by_axis)
    h_vals = [float(onp.dot(c - center, e_hatch)) for c in corners]
    h_min, h_max = min(h_vals), max(h_vals)
    if args.hatch_fixed is not None:
        return [float(args.hatch_fixed)], [0.5]
    if args.hatch_spacing is not None and args.hatch_spacing > 0.0:
        count = max(1, int(math.floor((h_max - h_min) / float(args.hatch_spacing))) + 1)
        offsets = [h_min + i * float(args.hatch_spacing) for i in range(count)]
        if offsets[-1] < h_max - 0.25 * float(args.hatch_spacing):
            offsets.append(h_max)
    else:
        count = max(1, int(args.hatch_lines_per_layer))
        if count == 1:
            offsets = [0.5 * (h_min + h_max)]
        else:
            offsets = [h_min + i * (h_max - h_min) / float(count - 1) for i in range(count)]
    denom = max(h_max - h_min, 1e-15)
    fracs = [(h - h_min) / denom for h in offsets]
    return offsets, fracs


def clip_scan_line_to_bounds(center, e_scan, e_hatch, hatch_offset, bounds_by_axis):
    base = center + hatch_offset * e_hatch
    s_lo = -1e100
    s_hi = 1e100
    for axis, (lo, hi) in bounds_by_axis.items():
        a = float(e_scan[axis])
        b = float(base[axis])
        if abs(a) < 1e-14:
            if b < lo or b > hi:
                return None
            continue
        s1 = (lo - b) / a
        s2 = (hi - b) / a
        s_axis_lo, s_axis_hi = min(s1, s2), max(s1, s2)
        s_lo = max(s_lo, s_axis_lo)
        s_hi = min(s_hi, s_axis_hi)
    if s_lo > s_hi:
        return None
    return s_lo, s_hi


def append_jump_states(states, global_step, layer_idx, hatch_idx, start_center, end_center, front_coord, layer_frac, args):
    if args.jump_speed <= 0.0:
        return global_step
    dist = float(onp.linalg.norm(end_center - start_center))
    if dist <= 1e-15:
        return global_step
    jump_time = dist / float(args.jump_speed)
    n_steps = max(1, int(math.ceil(jump_time / max(float(args.dt), 1e-30))))
    dt_jump = jump_time / float(n_steps)
    for j in range(1, n_steps + 1):
        frac = j / float(n_steps)
        center = (1.0 - frac) * start_center + frac * end_center
        states.append(
            make_step_state(
                global_step,
                "jump",
                layer_idx,
                hatch_idx,
                j - 1,
                center,
                0.0,
                0.0,
                dt_jump,
                frac,
                0.0,
                front_coord,
                layer_frac,
            )
        )
        global_step += 1
    return global_step


def make_step_state(global_step, mode, layer_idx, hatch_idx, scan_idx, laser_center, power, switch, dt, scan_frac, hatch_frac, front_coord, layer_frac):
    return StepState(
        global_step=global_step,
        mode=mode,
        layer_idx=int(layer_idx),
        hatch_idx=int(hatch_idx),
        scan_idx=int(scan_idx),
        laser_center=onp.asarray(laser_center, dtype=onp.float64),
        laser_power=float(power),
        laser_switch=float(switch),
        dt=float(dt),
        scan_frac=float(scan_frac),
        hatch_frac=float(hatch_frac),
        front_coord=float(front_coord),
        layer_frac=float(layer_frac),
    )


def generate_raster_step_states(args, pmin, pmax, build_axis_id, scan_axis_id, hatch_axis_id):
    if args.scan_pattern != "raster":
        raise ValueError(f"Unsupported --scan-pattern: {args.scan_pattern}")

    bounds_by_axis, scan_start, scan_end, _, _ = path_bounds_by_axis(
        pmin, pmax, scan_axis_id, hatch_axis_id, args
    )
    nominal_scan_length = abs(scan_end - scan_start)
    if args.scan_speed > 0.0 and args.scan_speed * args.dt > 0.5 * max(args.beam_radius, 1e-12):
        print("WARNING: scan_speed * dt is larger than 0.5 * beam_radius; consider smaller dt.")

    states = []
    global_step = 0
    last_center = make_path_center_from_bounds(pmin, pmax, bounds_by_axis)
    scan_lengths = []
    scan_speeds = []

    for layer_idx in range(args.layers):
        front_coord, layer_frac = build_front_coord(
            layer_idx,
            args.layers,
            pmin,
            pmax,
            build_axis_id,
            args.base_side,
            args.layer_thickness,
        )
        e_scan, e_hatch = make_layer_basis(scan_axis_id, hatch_axis_id, layer_idx, args.scan_rotation_per_layer)
        rect_center = make_path_center_from_bounds(pmin, pmax, bounds_by_axis)
        rect_center[build_axis_id] = front_coord
        hatch_offsets, hatch_fracs = hatch_offsets_for_layer(rect_center, e_hatch, bounds_by_axis, args)
        args.hatch_lines_per_layer = max(args.hatch_lines_per_layer, len(hatch_offsets))

        for hatch_idx, (hatch_offset, hatch_frac) in enumerate(zip(hatch_offsets, hatch_fracs)):
            clipped = clip_scan_line_to_bounds(rect_center, e_scan, e_hatch, hatch_offset, bounds_by_axis)
            if clipped is None:
                continue
            s_start, s_end = clipped
            if args.serpentine and (hatch_idx % 2 == 1):
                s_start, s_end = s_end, s_start
            line_length = abs(s_end - s_start)
            if args.auto_scan_steps_from_speed:
                if args.scan_speed <= 0.0:
                    raise ValueError("--auto-scan-steps-from-speed requires --scan-speed > 0")
                scan_steps = int(math.ceil(line_length / (args.scan_speed * args.dt))) + 1
            else:
                scan_steps = args.scan_steps_per_layer
            scan_steps = max(1, scan_steps)
            actual_speed = line_length / max((scan_steps - 1) * args.dt, args.dt)
            scan_lengths.append(line_length)
            scan_speeds.append(actual_speed)

            first_center = rect_center + hatch_offset * e_hatch + s_start * e_scan
            first_center[build_axis_id] = front_coord
            if hatch_idx > 0:
                if args.jump_speed > 0.0:
                    global_step = append_jump_states(
                        states,
                        global_step,
                        layer_idx,
                        hatch_idx,
                        last_center,
                        first_center,
                        front_coord,
                        layer_frac,
                        args,
                    )
                else:
                    for _ in range(args.dwell_steps_between_hatches):
                        states.append(
                            make_step_state(
                                global_step,
                                "hatch_dwell",
                                layer_idx,
                                hatch_idx - 1,
                                scan_steps - 1,
                                last_center,
                                0.0,
                                0.0,
                                args.dt,
                                1.0,
                                hatch_frac,
                                front_coord,
                                layer_frac,
                            )
                        )
                        global_step += 1

            for scan_idx in range(scan_steps):
                scan_frac = 0.0 if scan_steps <= 1 else scan_idx / float(scan_steps - 1)
                s_pos = s_start + scan_frac * (s_end - s_start)
                center = rect_center + hatch_offset * e_hatch + s_pos * e_scan
                center[build_axis_id] = front_coord
                last_center = center.copy()
                states.append(
                    make_step_state(
                        global_step,
                        "scan",
                        layer_idx,
                        hatch_idx,
                        scan_idx,
                        center,
                        args.laser_power,
                        1.0,
                        args.dt,
                        scan_frac,
                        hatch_frac,
                        front_coord,
                        layer_frac,
                    )
                )
                global_step += 1

        if layer_idx < args.layers - 1:
            for _ in range(args.dwell_steps_between_layers):
                states.append(
                    make_step_state(
                        global_step,
                        "layer_dwell",
                        layer_idx,
                        max(len(hatch_offsets) - 1, 0),
                        args.scan_steps_per_layer - 1,
                        last_center,
                        0.0,
                        0.0,
                        args.dt,
                        1.0,
                        1.0,
                        front_coord,
                        layer_frac,
                    )
                )
                global_step += 1
            # Span the recoat interval with a few large implicit steps instead
            # of recoat_time/dt tiny ones (10 s at dt=1e-4 would be 100k steps).
            recoat_steps = max(int(getattr(args, "recoat_steps", 10)), 1)
            recoat_dt = max(args.recoat_time, 0.0) / recoat_steps
            for _ in range(recoat_steps if recoat_dt > 0.0 else 0):
                states.append(
                    make_step_state(
                        global_step,
                        "recoat",
                        layer_idx,
                        max(len(hatch_offsets) - 1, 0),
                        args.scan_steps_per_layer - 1,
                        last_center,
                        0.0,
                        0.0,
                        recoat_dt,
                        1.0,
                        1.0,
                        front_coord,
                        layer_frac,
                    )
                )
                global_step += 1

    final_front, final_frac = build_front_coord(
        args.layers - 1,
        args.layers,
        pmin,
        pmax,
        build_axis_id,
        args.base_side,
        args.layer_thickness,
    )
    cooling_dt = (
        float(args.cooling_dt)
        if getattr(args, "cooling_dt", None) is not None and args.cooling_dt > 0.0
        else args.dt
    )
    for _ in range(args.cooling_steps):
        states.append(make_step_state(global_step, "cooling", args.layers - 1, 0, 0, last_center, 0.0, 0.0, cooling_dt, 0.0, 0.0, final_front, final_frac))
        global_step += 1

    representative_scan_length = max(scan_lengths) if scan_lengths else nominal_scan_length
    representative_scan_speed = sum(scan_speeds) / len(scan_speeds) if scan_speeds else 0.0
    return states, representative_scan_length, representative_scan_speed


def generate_path_file_step_states(args, pmin, pmax, build_axis_id):
    states = []
    path_scale = args.mesh_length_scale if args.path_length_scale is None else args.path_length_scale
    with open(args.path_file, newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"time", "x", "y", "z", "power", "laser_on", "layer", "hatch", "mode"}
        if not required.issubset(fields):
            raise ValueError(f"--path-file must contain columns: {sorted(required)}")
        has_front_coord = "front_coord" in fields
        has_scan_id = "scan_id" in fields
        rows = list(reader)
    if not rows:
        raise ValueError("--path-file is empty")
    if not has_front_coord:
        print("WARNING: --path-file has no front_coord column; using laser center coordinate along build_axis as activation front.")
    times = [float(row["time"]) for row in rows]
    recoat_steps = max(int(getattr(args, "recoat_steps", 10)), 1)
    recoat_dt = max(args.recoat_time, 0.0) / recoat_steps if args.recoat_time > 0.0 else 0.0
    global_step = 0
    prev_layer_idx = None
    for i, row in enumerate(rows):
        if i > 0 and times[i] <= times[i - 1]:
            raise ValueError(
                "--path-file time must be strictly increasing: "
                f"row={i + 2}, time={times[i]}, previous={times[i - 1]}"
            )
        dt = args.dt if i == 0 else times[i] - times[i - 1]
        center = path_scale * onp.asarray([float(row["x"]), float(row["y"]), float(row["z"])], dtype=onp.float64)
        layer_idx = max(int(row["layer"]) - 1, 0)
        # Honor --layers / --max-print-layers in path-file mode: rows beyond
        # the layer limit are dropped (matching the raster generator), instead
        # of silently scanning and ACTIVATING all layers in the CSV.
        if layer_idx >= args.layers:
            break
        if has_front_coord and str(row.get("front_coord", "")).strip() != "":
            front_coord = path_scale * float(row["front_coord"])
        else:
            front_coord = float(center[build_axis_id])
        layer_frac = min(max((layer_idx + 1) / float(args.layers), 0.0), 1.0)
        scan_id = int(row["scan_id"]) if has_scan_id and str(row.get("scan_id", "")).strip() != "" else i

        # Machine path files often contain scan vectors only, with no recoater
        # dwell between layers. Insert laser-off recoat states at each layer
        # transition so interlayer cooling exists (dt = recoat_time/recoat_steps).
        if (
            recoat_dt > 0.0
            and prev_layer_idx is not None
            and layer_idx > prev_layer_idx
            and states
        ):
            last = states[-1]
            for _ in range(recoat_steps):
                states.append(
                    make_step_state(
                        global_step,
                        "recoat",
                        last.layer_idx,
                        last.hatch_idx,
                        last.scan_idx,
                        last.laser_center,
                        0.0,
                        0.0,
                        recoat_dt,
                        last.scan_frac,
                        last.hatch_frac,
                        last.front_coord,
                        last.layer_frac,
                    )
                )
                global_step += 1

        states.append(
            make_step_state(
                global_step,
                row["mode"] or "path",
                layer_idx,
                max(int(row["hatch"]) - 1, 0),
                scan_id,
                center,
                float(row["power"]),
                1.0 if parse_scalar(row["laser_on"]) else 0.0,
                dt,
                0.0,
                0.0,
                front_coord,
                layer_frac,
            )
        )
        global_step += 1
        prev_layer_idx = layer_idx

    last = states[-1]
    cooling_dt = (
        float(args.cooling_dt)
        if getattr(args, "cooling_dt", None) is not None and args.cooling_dt > 0.0
        else args.dt
    )
    for _ in range(args.cooling_steps):
        states.append(
            make_step_state(
                global_step,
                "cooling",
                last.layer_idx,
                last.hatch_idx,
                last.scan_idx,
                last.laser_center,
                0.0,
                0.0,
                cooling_dt,
                last.scan_frac,
                last.hatch_frac,
                last.front_coord,
                last.layer_frac,
            )
        )
        global_step += 1
    return states, 0.0, 0.0
