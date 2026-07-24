"""Cell activation masks: moving thermal window and layer-on-scan recoating.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import jax.numpy as np
import numpy as onp

from jax_fem_am.mesh.model import cells_intersect_distance_band


def uses_strict_active_domain(args):
    """Return whether future, unprinted cells must contribute exactly zero.

    ``layer_on_scan`` plus ``future_layer_mode=void`` is the paper-reproduction
    contract: cells above the highest recoated layer do not yet exist
    physically. Other activation modes retain the historical ersatz behavior.
    """

    return (
        getattr(args, "layer_activation_mode", "front") == "layer_on_scan"
        and getattr(args, "future_layer_mode", "void") == "void"
    )


def resolve_surface_active_mask(args):
    """Resolve face masking without allowing flux on strict-domain voids."""

    configured = getattr(args, "surface_active_mask", None)
    if uses_strict_active_domain(args):
        if configured is False:
            raise ValueError(
                "--no-surface-active-mask is incompatible with "
                "layer_on_scan future-layer void semantics"
            )
        return True
    if configured is not None:
        return bool(configured)
    return getattr(args, "surface_selection", "box") == "exterior"


def physical_node_mask(cells, physical_cell, num_nodes=None):
    """Mark nodes incident to at least one physically present cell.

    Shared interface nodes remain physical even when one adjacent cell is
    inactive. Only nodes owned exclusively by inactive cells are excluded from
    the active solve.
    """

    cells = onp.asarray(cells)
    physical_cell = onp.asarray(physical_cell, dtype=bool)
    if cells.ndim != 2:
        raise ValueError("cells must have shape (num_cells, nodes_per_cell)")
    if physical_cell.shape != (len(cells),):
        raise ValueError(
            "physical_cell must have shape (num_cells,), "
            f"got {physical_cell.shape} for {len(cells)} cells"
        )
    if not onp.issubdtype(cells.dtype, onp.integer):
        raise ValueError("cells must contain integer node indices")

    if num_nodes is None:
        num_nodes = int(cells.max()) + 1 if cells.size else 0
    num_nodes = int(num_nodes)
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= num_nodes):
        raise ValueError("cell node index is outside [0, num_nodes)")

    mask = onp.zeros(num_nodes, dtype=bool)
    selected = cells[physical_cell]
    if selected.size:
        mask[onp.unique(selected)] = True
    return mask


def contributing_cell_mask(*quad_fields):
    """Return cells with at least one nonzero assembled material coefficient."""

    if not quad_fields:
        raise ValueError("at least one quadrature field is required")
    arrays = [onp.asarray(field) for field in quad_fields]
    num_cells = len(arrays[0])
    if any(array.ndim < 1 or len(array) != num_cells for array in arrays):
        raise ValueError(
            "quadrature fields must share a leading num_cells dimension"
        )

    mask = onp.zeros(num_cells, dtype=bool)
    for array in arrays:
        if array.ndim == 1:
            mask |= array != 0
        else:
            mask |= onp.any(
                array != 0,
                axis=tuple(range(1, array.ndim)),
            )
    return mask


def make_inactive_node_dirichlet_bc(inactive_node_mask, *, vec, value):
    """Create zero-contribution DOF constraints for inactive-only nodes."""

    inactive_node_mask = onp.asarray(inactive_node_mask, dtype=bool)
    if inactive_node_mask.ndim != 1:
        raise ValueError("inactive_node_mask must be one-dimensional")
    vec = int(vec)
    if vec < 1:
        raise ValueError("vec must be >= 1")
    mask = np.asarray(inactive_node_mask)

    def inactive_node(_point, node_id):
        return mask[node_id]

    def prescribed_value(_point):
        return value

    return [
        [inactive_node for _ in range(vec)],
        list(range(vec)),
        [prescribed_value for _ in range(vec)],
    ]


def merge_dirichlet_bcs(*conditions):
    """Merge BC triplets with deterministic first-condition precedence.

    JAX-FEM applies Dirichlet rows with ``unique_indices=True``.  Therefore,
    later conditions are masked by all earlier conditions on the same vector
    component instead of emitting duplicate degrees of freedom.  This also
    makes the precedence explicit: physical/base boundary conditions win over
    subsequently appended inactive-domain constraints.
    """

    merged = [[], [], []]
    earlier_locations = {}

    def call_location(location, point, node_id):
        num_args = location.__code__.co_argcount
        if num_args == 1:
            return location(point)
        if num_args == 2:
            return location(point, node_id)
        raise ValueError(
            "Dirichlet BC location functions must accept one or two arguments"
        )

    def without_earlier(location, earlier):
        def unique_location(point, node_id):
            selected = call_location(location, point, node_id)
            for previous in earlier:
                selected = np.logical_and(
                    selected,
                    np.logical_not(
                        call_location(previous, point, node_id)
                    ),
                )
            return selected

        return unique_location

    for condition in conditions:
        if condition is None:
            continue
        if len(condition) != 3:
            raise ValueError(
                "Dirichlet BC must contain [location_fns, vecs, value_fns]"
            )
        lengths = tuple(len(values) for values in condition)
        if len(set(lengths)) != 1:
            raise ValueError(
                "Dirichlet BC location, component and value lists must align"
            )
        locations, components, values = condition
        for location, component, value in zip(
            locations,
            components,
            values,
        ):
            component = int(component)
            earlier = tuple(earlier_locations.get(component, ()))
            merged[0].append(
                without_earlier(location, earlier)
                if earlier
                else location
            )
            merged[1].append(component)
            merged[2].append(value)
            earlier_locations.setdefault(component, []).append(location)
    return merged if merged[0] else None


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
