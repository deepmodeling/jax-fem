"""Cell activation masks: moving thermal window and layer-on-scan recoating.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import numpy as onp

from jax_fem_am.mesh.model import cells_intersect_distance_band


def compute_active_cell(state, cell_build_coord, substrate_cell, support_cell, build_sign, args):
    tol = 1e-12 * max(float(onp.max(onp.abs(cell_build_coord))), 1.0)
    part_active = build_sign * (state.front_coord - cell_build_coord) >= -tol
    return substrate_cell | support_cell | part_active


def compute_moving_window_cells_by_intersection(state, cell_d_min, cell_d_max, substrate_cell, support_cell, args):
    """Printed/window/cooling masks using cell-layer interval intersection."""
    fixture_cell = substrate_cell | support_cell
    current_layer = int(state.layer_idx) + 1
    if args.layer_thickness is None or args.layer_thickness <= 0.0:
        raise ValueError("intersection activation requires --layer-thickness > 0")
    lt = float(args.layer_thickness)

    printed_lower = 0.0
    printed_upper = current_layer * lt
    printed_part = cells_intersect_distance_band(cell_d_min, cell_d_max, printed_lower, printed_upper)

    if args.active_window_below_layers <= 0:
        window_lower = printed_lower
    else:
        lower_layer = max(1, current_layer - int(args.active_window_below_layers))
        window_lower = float(lower_layer - 1) * lt
    window_upper = printed_upper
    thermal_window_part = cells_intersect_distance_band(cell_d_min, cell_d_max, window_lower, window_upper)

    printed_cell = fixture_cell | printed_part
    active_cell = fixture_cell | thermal_window_part
    cooling_only_cell = printed_cell & (~active_cell)
    return printed_cell, active_cell, cooling_only_cell


def compute_layer_on_scan_cells_by_intersection(highest_printed_layer, cell_d_min, cell_d_max, substrate_cell, support_cell, args):
    """Whole-layer recoating activation using cell-layer interval intersection.

    When layer L starts scanning, all cells intersecting layers <= L become
    printed/powder. The active thermal window is the last N printed layers.
    """
    fixture_cell = substrate_cell | support_cell
    if args.layer_thickness is None or args.layer_thickness <= 0.0:
        raise ValueError("intersection activation requires --layer-thickness > 0")
    lt = float(args.layer_thickness)
    top_layer = int(highest_printed_layer)

    if top_layer <= 0:
        printed_part = onp.zeros_like(cell_d_min, dtype=bool)
        thermal_window_part = onp.zeros_like(cell_d_min, dtype=bool)
    else:
        printed_part = cells_intersect_distance_band(cell_d_min, cell_d_max, 0.0, top_layer * lt)
        if args.active_window_below_layers <= 0:
            window_lower = 0.0
        else:
            lower_layer = max(1, top_layer - int(args.active_window_below_layers))
            window_lower = float(lower_layer - 1) * lt
        thermal_window_part = cells_intersect_distance_band(cell_d_min, cell_d_max, window_lower, top_layer * lt)

    printed_cell = fixture_cell | printed_part
    active_cell = fixture_cell | thermal_window_part
    cooling_only_cell = printed_cell & (~active_cell)
    return printed_cell, active_cell, cooling_only_cell


def compute_moving_window_cells(state, physical_layer_id_cell, substrate_cell, support_cell, args):
    """Compute printed, thermal-window and cooling-only masks.

    active_window_below_layers = 10 means that when current layer is 12, layers
    2..12 are in the thermal window. Printed layers below that window are marked
    cooling_only: they keep thermal capacity, their conductivity is reduced by
    old_layer_thermal_factor, and they may receive old_layer_cooling_h sink.
    """
    fixture_cell = substrate_cell | support_cell
    current_layer = int(state.layer_idx) + 1

    printed_part = physical_layer_id_cell <= current_layer
    if args.active_window_below_layers <= 0:
        thermal_window_part = printed_part
    else:
        lower_layer = max(1, current_layer - int(args.active_window_below_layers))
        upper_layer = current_layer
        thermal_window_part = (physical_layer_id_cell >= lower_layer) & (physical_layer_id_cell <= upper_layer)

    printed_cell = fixture_cell | printed_part
    # Keep substrate/support in the thermal window so the part still has a
    # stable thermal sink and mechanical fixture in early layers.
    active_cell = fixture_cell | thermal_window_part
    cooling_only_cell = printed_cell & (~active_cell)
    return printed_cell, active_cell, cooling_only_cell


def should_activate_layer_for_state(state):
    """Return True when a state represents the start/continuation of laser scanning.

    This is used for the recoating-style activation model: a layer becomes
    printed/powder only when the laser actually starts scanning that layer,
    not merely because a front coordinate exists in the path.
    """
    return (float(state.laser_switch) > 0.5) and (state.mode in ("scan", "path"))


def compute_layer_on_scan_cells(highest_printed_layer, physical_layer_id_cell, substrate_cell, support_cell, args):
    """Compute printed/window/cooling masks for whole-layer recoating activation.

    If highest_printed_layer = 12 and active_window_below_layers = 10,
    layers 2..12 are in the active thermal window, and layer 1 is
    cooling_only. Future layers > 12 remain unprinted.
    """
    fixture_cell = substrate_cell | support_cell
    if int(highest_printed_layer) <= 0:
        printed_part = onp.zeros_like(physical_layer_id_cell, dtype=bool)
        thermal_window_part = onp.zeros_like(physical_layer_id_cell, dtype=bool)
    else:
        top_layer = int(highest_printed_layer)
        printed_part = physical_layer_id_cell <= top_layer
        if args.active_window_below_layers <= 0:
            thermal_window_part = printed_part
        else:
            lower_layer = max(1, top_layer - int(args.active_window_below_layers))
            thermal_window_part = (physical_layer_id_cell >= lower_layer) & (physical_layer_id_cell <= top_layer)

    printed_cell = fixture_cell | printed_part
    active_cell = fixture_cell | thermal_window_part
    cooling_only_cell = printed_cell & (~active_cell)
    return printed_cell, active_cell, cooling_only_cell
