import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as np
import numpy as onp

from jax_fem_am.mesh.readers import (
    SOLID_BLOCKS,
    read_inp_cell_set,
    read_solid_inp,
    read_tet4_inp,
)
from jax_fem_am.materials.tables import (
    PropertyTable,
    eval_property,
    load_property_tables,
)
from jax_fem_am.materials.phases import (
    MODE_TO_ID,
    STATE_LIQUID,
    STATE_MUSHY,
    STATE_POWDER,
    STATE_SOLID,
    STATE_SUBSTRATE,
    STATE_SUPPORT,
    STATE_VOID,
    clamp_mechanics_temperature,
    initial_phase_cell,
    make_quad_scalar,
    material_cell_state,
    mechanics_material_quads,
    phase_cell_from_quad,
    thermal_material_quads,
)
from jax_fem_am.mesh.model import (
    AXIS_TO_ID,
    cells_intersect_distance_band,
    classify_cells,
    compute_cell_build_interval,
    compute_cell_temperature,
    compute_layer_id,
    compute_nominal_layer_id_from_interval,
    compute_physical_layer_id_cell,
    coord_from_frac,
    make_box_locations,
    make_part_build_box,
    resolve_axis_range,
)
from jax_fem_am.mesh.quadrature import apply_thermal_mass_lumping
from jax_fem_am.io.vtu import (
    STRESS_COMPONENTS,
    empty_quad_stress,
    make_quad_stress_cell_infos,
    print_startup,
    quad_field_name,
    save_step,
    von_mises_from_stress,
    write_calibration_template,
    write_path_output,
    write_used_config,
)
from jax_fem_am.config.loaders import cfg, parse_scalar, read_config
from jax_fem_am.config.schema import build_parser, parse_args
from jax_fem_am.domain.events import update_phase_reference_and_eqp
from jax_fem_am.domain.state import StepState, reset_new_cell_nodal_temperature
from jax_fem_am.physics.mechanics import ThermoMechanical
from jax_fem_am.physics.release import (
    load_release_cell_set,
    make_anchor_mechanics_bc,
    make_box_anchor_mechanics_bc,
    make_full_bottom_mechanics_bc,
    make_paper_minimal_bottom_mechanics_bc,
    make_root_minimal_release_mechanics_bc,
    validate_release_anchor_protocol,
    validate_release_cell_set,
    zero_exact_release_cells,
)
from jax_fem_am.physics.thermal import TransientThermal
from jax_fem_am.process.activation import (
    contributing_cell_mask,
    compute_active_cell,
    compute_layer_on_scan_cells,
    compute_layer_on_scan_cells_by_intersection,
    compute_moving_window_cells,
    compute_moving_window_cells_by_intersection,
    make_inactive_node_dirichlet_bc,
    merge_dirichlet_bcs,
    physical_node_mask,
    resolve_surface_active_mask,
    should_activate_layer_for_state,
    uses_strict_active_domain,
)
from jax_fem_am.process.scan_path import (
    append_jump_states,
    build_front_coord,
    clip_scan_line_to_bounds,
    generate_path_file_step_states,
    generate_raster_step_states,
    hatch_offsets_for_layer,
    make_layer_basis,
    make_path_center_from_bounds,
    make_step_state,
    path_bounds_by_axis,
    path_rectangle_corners,
    resolve_scan_and_hatch_axes,
    update_layers_from_thickness,
)
from jax_fem_am.process.schedule import (
    apply_stage_temperature_schedule,
    should_run_mechanics,
    should_save_step,
)
from jax_fem_am.solvers.nonlinear import mechanics_newton_overrides_from_args
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol


ID_TO_AXIS = ("x", "y", "z")


def run_mechanics(mechanics, u_guess, params, newton_overrides=None):
    mechanics.set_params(params)
    newton = {
        "initial_guess": u_guess,
        "linear": {"spsolve_solver": {}},
        "tol": 1e-9,
        "rel_tol": 1e-11,
    }
    if newton_overrides:
        newton.update(newton_overrides)
    return solver(mechanics, solver_options={"newton": newton})


def run_mechanics_with_cutback(mechanics, u_guess, params, newton_overrides, args,
                               T_prev_quad, active_prev_quad, T_ref_quad,
                               active_quad, phase_quad, tables):
    """Mechanics solve with Abaqus-style automatic load-increment cutback.

    Try the full increment first. On Newton failure, walk the thermal load
    from the last accepted mechanics state to the current one in 2,4,...,
    2**mechanics_max_cuts equal substeps (temperature - hence thermal strain
    and material tables - interpolated along the way). Substeps are pure
    Newton continuation: plastic state (params[-1]) is held fixed and the
    final substep IS the original problem (same params object), so an
    accepted cutback solution satisfies exactly the equations a direct
    solve would have. Cells activated since the last accepted solve ramp
    from their stress-free reference T_ref (dT 0 -> full).
    """
    try:
        return run_mechanics(mechanics, u_guess, params, newton_overrides)
    except RuntimeError as exc:
        if not getattr(args, "mechanics_max_cuts", 0):
            raise
        last_exc = exc
        print(f"mechanics cutback: full increment failed ({exc}); subdividing")

    T_cur_quad = params[0]
    if T_prev_quad is None:
        T_prev_eff = T_ref_quad
    else:
        T_prev_eff = np.where(active_prev_quad > 0, T_prev_quad, T_ref_quad)

    n = 2
    while n <= 2 ** int(args.mechanics_max_cuts):
        u = u_guess
        try:
            for k in range(1, n + 1):
                if k == n:
                    sub_params = params
                else:
                    lam = k / n
                    T_lam = T_prev_eff + lam * (T_cur_quad - T_prev_eff)
                    dT_lam = (T_lam - T_ref_quad) * active_quad
                    (active_factor_lam, E_lam, alpha_lam, poisson_lam,
                     yield_lam, hardening_lam) = mechanics_material_quads(
                        T_lam, active_quad, phase_quad, args, tables)
                    sub_params = [
                        T_lam,
                        dT_lam,
                        active_factor_lam,
                        E_lam,
                        alpha_lam,
                        poisson_lam,
                        yield_lam,
                        hardening_lam,
                        params[-1],
                    ]
                u = run_mechanics(mechanics, u, sub_params, newton_overrides)
            print(f"mechanics cutback: converged with {n} substeps")
            return u
        except RuntimeError as exc:
            last_exc = exc
            print(f"mechanics cutback: {n} substeps failed ({exc}); refining")
            n *= 2
    raise last_exc


def validate_release_configuration(args, strict_active_domain):
    """Fail closed when a formal exact cut is paired with a legacy anchor."""

    if args.release_cell_set is not None and args.release_cut_box is not None:
        raise ValueError(
            "--release-cell-set and --release-cut-box are mutually exclusive"
        )
    if args.release_cell_set is not None and not args.release_after_cooling:
        raise ValueError(
            "--release-cell-set requires --release-after-cooling"
        )
    if args.release_cell_set is not None and not strict_active_domain:
        raise ValueError(
            "--release-cell-set requires the strict layer_on_scan/void "
            "active domain so removed cells have exactly zero contribution"
        )
    if (
        args.release_cell_set is not None
        and args.release_anchor_mode != "paper_minimal_root"
    ):
        raise ValueError(
            "--release-cell-set requires --release-anchor-mode "
            "paper_minimal_root; rigid-body and fully clamped box anchors "
            "are diagnostic-only"
        )
    if (
        args.release_cell_set is not None
        and args.bottom_mechanics_bc != "paper_minimal"
    ):
        raise ValueError(
            "--release-cell-set requires --bottom-mechanics-bc "
            "paper_minimal so the surviving root anchors are continuous "
            "from build through release"
        )
    if (
        args.release_anchor_mode == "paper_minimal_root"
        and args.release_cell_set is None
    ):
        raise ValueError(
            "--release-anchor-mode paper_minimal_root requires "
            "--release-cell-set"
        )


def main():
    args = parse_args()
    strict_active_domain = uses_strict_active_domain(args)
    validate_release_configuration(args, strict_active_domain)
    if args.steps is not None:
        print("WARNING: --steps is treated as an alias for --layers in this version.")
        args.layers = args.steps
    if args.layers < 1 or args.scan_steps_per_layer < 1 or args.hatch_lines_per_layer < 1:
        raise ValueError("--layers, --scan-steps-per-layer and --hatch-lines-per-layer must be >= 1")
    if args.mechanics_every < 0:
        raise ValueError("--mechanics-every must be >= 0; use 0 for thermal-only output")
    if args.release_after_cooling and args.cooling_steps < 1:
        print("WARNING: --release-after-cooling requested without cooling steps; release will run after printing.")

    raw_points, cells, selected_cells, ele_type = read_solid_inp(args.inp, args.max_cells)
    if ele_type == "HEX8" and (args.quadrature_order is None or args.quadrature_order < 2):
        raise ValueError(
            "HEX8 meshes require --quadrature-order 2 (2x2x2 Gauss): the "
            "single-point rule is rank-deficient (hourglass modes)")
    bbar_enabled = {"auto": ele_type == "HEX8", "on": True, "off": False}[args.mechanics_bbar]
    print(f"mesh element type: {ele_type}; mechanics B-bar: "
          f"{'ON' if bbar_enabled else 'OFF'} (--mechanics-bbar {args.mechanics_bbar})")
    if ele_type == "HEX8" and not bbar_enabled:
        print("WARNING: HEX8 without B-bar volumetric-locks under J2 flow "
              "(checkerboard hydrostatic pressure); comparison arms only.")
    raw_pmin = onp.min(raw_points, axis=0)
    raw_pmax = onp.max(raw_points, axis=0)
    points = raw_points * args.mesh_length_scale

    (
        pmin,
        pmax,
        bottom,
        exposed,
        walls,
        build_axis_id,
        plane_axis_ids,
        base_coord,
        exposed_coord,
    ) = make_box_locations(points, build_axis=args.build_axis, base_side=args.base_side, abs_tol=args.boundary_tol)

    scan_axis_id, hatch_axis_id = resolve_scan_and_hatch_axes(args.scan_axis, build_axis_id, plane_axis_ids)
    build_sign = 1.0 if args.base_side == "min" else -1.0
    part_pmin, part_pmax = make_part_build_box(
        pmin,
        pmax,
        build_axis_id,
        args.base_side,
        args.substrate_thickness,
        args.support_thickness,
    )
    span = part_pmax - part_pmin
    plane_scale = max(float(span[plane_axis_ids[0]]), float(span[plane_axis_ids[1]]), 1e-12)
    build_span = max(float(span[build_axis_id]), 1e-12)
    update_layers_from_thickness(args, part_pmin, part_pmax, build_axis_id)
    args.beam_radius = args.beam_radius if args.beam_radius > 0 else 0.04 * plane_scale
    args.source_depth = args.source_depth if args.source_depth > 0 else max(0.5 * args.beam_radius, 0.05 * build_span)
    if args.front_surface_loss_h > 0.0 and args.front_surface_loss_thickness <= 0.0:
        args.front_surface_loss_thickness = args.source_depth

    if args.path_file:
        step_states, scan_length, actual_scan_speed = generate_path_file_step_states(args, part_pmin, part_pmax, build_axis_id)
    else:
        step_states, scan_length, actual_scan_speed = generate_raster_step_states(args, part_pmin, part_pmax, build_axis_id, scan_axis_id, hatch_axis_id)

    initial_temperature = args.preheat_temperature if args.preheat_temperature is not None else args.ambient
    bottom_temperature_effective = args.bottom_temperature if args.bottom_temperature is not None else initial_temperature
    final_cooldown_enabled = (
        args.final_cooldown_temperature is not None
        and args.bottom_thermal_bc == "fixed"
    )
    if (
        args.final_cooldown_temperature is not None
        and args.bottom_thermal_bc != "fixed"
    ):
        print(
            "WARNING: --final-cooldown-temperature requires "
            "--bottom-thermal-bc fixed; ignoring."
        )
    apply_stage_temperature_schedule(
        step_states,
        process_ambient=args.ambient,
        process_bottom_temperature=bottom_temperature_effective,
        final_cooldown_temperature=(
            args.final_cooldown_temperature
            if final_cooldown_enabled
            else None
        ),
    )
    surface_active_mask_enabled = resolve_surface_active_mask(args)
    mechanics_newton_overrides = mechanics_newton_overrides_from_args(args)

    if args.surface_selection == "exterior":
        # Kaess et al. (2023), Section 2.3, applies chamber heat loss to the
        # "top active element layer": https://doi.org/10.3390/ma16062321
        # Keep all potential upward faces in a fixed superset; the
        # owner/neighbor activity mask exposes the correct one each step.
        span_for_tol = max(float(onp.max(pmax - pmin)), 1.0)
        base_tol = (
            float(args.boundary_tol)
            if args.boundary_tol is not None and args.boundary_tol > 0.0
            else 1e-8 * span_for_tol
        )

        def exterior_above_base(point):
            return build_sign * (point[build_axis_id] - base_coord) > base_tol

        exterior_above_base.exterior_only = True
        exterior_above_base.active_domain_exterior = True
        exterior_above_base.active_domain_top_only = True
        exterior_above_base.active_domain_build_axis = build_axis_id
        exterior_above_base.active_domain_build_sign = build_sign
        location_fns = [exterior_above_base]

        def exterior_bottom(point):
            return build_sign * (point[build_axis_id] - base_coord) <= base_tol

        exterior_bottom.exterior_only = True
        bottom_for_flux = exterior_bottom
    else:
        location_fns = [exposed, walls]
        bottom_for_flux = bottom

    thermal_bc = None
    if args.bottom_thermal_bc == "fixed":
        def bottom_temperature_value(_point):
            return bottom_temperature_effective

        thermal_bc = [[bottom], [0], [bottom_temperature_value]]
    else:
        location_fns.append(bottom_for_flux)

    mesh = Mesh(points, cells, ele_type=ele_type)
    thermal = TransientThermal(
        mesh=mesh,
        vec=1,
        dim=3,
        ele_type=ele_type,
        quadrature_order=args.quadrature_order,
        dirichlet_bc_info=thermal_bc,
        location_fns=location_fns,
        additional_info=(
            args.convection_h,
            args.ambient,
            args.emissivity,
            args.stefan_boltzmann,
            build_axis_id,
            plane_axis_ids[0],
            plane_axis_ids[1],
            build_sign,
            len(location_fns),
            args.front_surface_loss_h,
            args.front_surface_loss_thickness,
            args.front_surface_loss_radiation,
            args.source_model,
        ),
    )
    if args.thermal_mass_lumping:
        apply_thermal_mass_lumping(thermal)

    release_cell_set = None
    exact_release_bc = None
    args.paper_minimal_release_resolved_bc = None
    if args.release_cell_set is not None:
        with open(args.inp, "rb") as mesh_stream:
            mesh_sha256 = hashlib.sha256(mesh_stream.read()).hexdigest()
        release_cell_set = load_release_cell_set(
            args.release_cell_set,
            expected_mesh_sha256=mesh_sha256,
            num_cells=len(cells),
        )
        args.release_cell_set_sha256 = release_cell_set.artifact_sha256
        args.release_cell_set_removed_count = int(
            len(release_cell_set.removed_cell_ids)
        )
        anchor_protocol = release_cell_set.document.get("anchor_protocol")
        if not isinstance(anchor_protocol, dict):
            raise ValueError(
                "formal release artifact requires an anchor_protocol object"
            )
        try:
            release_base_coord = float(anchor_protocol["base_coord_m"])
            release_base_tolerance = float(
                anchor_protocol["base_tolerance_m"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "release anchor protocol requires a finite base plane"
            ) from exc
        if (
            not onp.isfinite(release_base_coord)
            or not onp.isfinite(release_base_tolerance)
            or release_base_tolerance < 0.0
        ):
            raise ValueError(
                "release anchor protocol requires a finite base plane"
            )
        (
            exact_release_bc,
            args.paper_minimal_release_resolved_bc,
        ) = make_root_minimal_release_mechanics_bc(
            points,
            cells,
            release_cell_set.retained_root_cell_ids,
            build_axis_id=build_axis_id,
            plane_axis_ids=plane_axis_ids,
            base_coord=release_base_coord,
            base_tolerance=release_base_tolerance,
            anchor_corner=args.paper_minimal_anchor_corner,
            return_metadata=True,
        )
        validate_release_anchor_protocol(
            release_cell_set,
            args.paper_minimal_release_resolved_bc,
            anchor_corner=args.paper_minimal_anchor_corner,
        )
        print(
            "release cell set: "
            f"{args.release_cell_set_removed_count} exact cells; "
            f"sha256={args.release_cell_set_sha256}"
        )
    else:
        args.release_cell_set_sha256 = None
        args.release_cell_set_removed_count = 0

    # write_used_config() serializes vars(args), including the resolved node
    # identity needed to audit/reproduce a paper-minimal anchor selection.
    args.paper_minimal_resolved_bc = None
    if args.bottom_mechanics_bc == "elastic":
        mechanics_bc = None
        mechanics_location_fns = [bottom]
        mechanics_foundation = args.bottom_foundation_stiffness
    elif args.bottom_mechanics_bc == "paper_minimal":
        (
            mechanics_bc,
            args.paper_minimal_resolved_bc,
        ) = make_paper_minimal_bottom_mechanics_bc(
            points,
            bottom,
            build_axis_id=build_axis_id,
            plane_axis_ids=plane_axis_ids,
            anchor_corner=args.paper_minimal_anchor_corner,
            anchor_candidate_node_ids=(
                args.paper_minimal_release_resolved_bc[
                    "root_bottom_node_ids"
                ]
                if release_cell_set is not None
                else None
            ),
            return_metadata=True,
        )
        if (
            release_cell_set is not None
            and (
                args.paper_minimal_resolved_bc["anchor_node_ids"]
                != args.paper_minimal_release_resolved_bc[
                    "anchor_node_ids"
                ]
                or args.paper_minimal_resolved_bc["rotation_component"]
                != args.paper_minimal_release_resolved_bc[
                    "rotation_component"
                ]
            )
        ):
            raise ValueError(
                "build and release must preserve the same physical "
                "in-plane anchor DOFs"
            )
        if release_cell_set is not None:
            args.paper_minimal_release_resolved_bc[
                "constraint_continuity"
            ] = "release_physical_dofs_are_build_dof_subset"
        mechanics_location_fns = []
        mechanics_foundation = 0.0
    else:
        mechanics_bc = make_full_bottom_mechanics_bc(bottom)
        mechanics_location_fns = []
        mechanics_foundation = 0.0
    powder_foundation = 0.0
    if args.powder_mechanics_bc == "elastic":
        if args.surface_selection != "exterior":
            raise ValueError(
                "--powder-mechanics-bc elastic requires --surface-selection exterior "
                "(the side faces of a curved part are only correct in exterior mode)"
            )
        powder_foundation = args.powder_foundation_stiffness

        def powder_side(point):
            # Exterior faces strictly above the base plane: the printed
            # material embedded in the surrounding powder bed. Must stay the
            # LAST mechanics location fn (custom_init assumes it).
            return build_sign * (point[build_axis_id] - base_coord) > base_tol

        powder_side.exterior_only = True
        mechanics_location_fns = list(mechanics_location_fns) + [powder_side]
    if not mechanics_location_fns:
        mechanics_location_fns = None
    mechanics = ThermoMechanical(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        quadrature_order=args.quadrature_order,
        dirichlet_bc_info=mechanics_bc,
        location_fns=mechanics_location_fns,
        additional_info=(
            args.mechanics_model,
            args.yield_saturation_stress,
            mechanics_foundation,
            powder_foundation,
            tuple(plane_axis_ids),
            bbar_enabled,
        ),
    )

    tables = load_property_tables(args)
    cell_centroids, cell_build_coord, substrate_cell, support_cell = classify_cells(points, cells, build_axis_id, build_sign, base_coord, args)
    if args.powder_solid_E is not None and args.powder_elset is None:
        raise ValueError("--powder-solid-E requires --powder-elset")
    if args.powder_elset is not None:
        permanent_powder_cell = read_inp_cell_set(args.inp, args.powder_elset, len(cells))
        # Permanent powder is never substrate/support/printed; the geometric
        # classifiers cannot know this (gap cells share the support z-band).
        substrate_cell &= ~permanent_powder_cell
        support_cell &= ~permanent_powder_cell
        print(f"powder elset {args.powder_elset!r}: {int(permanent_powder_cell.sum())} "
              f"permanent powder cells (weak-solid mechanics: "
              f"{'ON, E=%g Pa' % args.powder_solid_E if args.powder_solid_E is not None else 'off'})")
    else:
        permanent_powder_cell = onp.zeros(len(cells), dtype=bool)
    layer_id_cell = compute_layer_id(cell_build_coord, build_axis_id, part_pmin, part_pmax, args)
    # Fixture cells are not printed part layers. Keep their layer id at 0 for
    # clearer ParaView interpretation.
    layer_id_cell = onp.asarray(layer_id_cell, dtype=onp.int32)
    layer_id_cell[
        substrate_cell | support_cell | permanent_powder_cell
    ] = 0
    physical_layer_id_cell = compute_physical_layer_id_cell(
        cell_build_coord,
        build_axis_id,
        part_pmin,
        part_pmax,
        build_sign,
        args,
    )
    physical_layer_id_cell = onp.asarray(physical_layer_id_cell, dtype=onp.int32)
    physical_layer_id_cell[
        substrate_cell | support_cell | permanent_powder_cell
    ] = 0
    if args.layer_thickness is not None and args.layer_thickness > 0.0:
        # For physical-layer runs, report the real layer id instead of mapping
        # the full build height into the truncated --max-print-layers range.
        layer_id_cell = physical_layer_id_cell.copy()

    if args.base_side == "min":
        part_base_coord_for_interval = float(part_pmin[build_axis_id])
    else:
        part_base_coord_for_interval = float(part_pmax[build_axis_id])
    cell_d_min, cell_d_max = compute_cell_build_interval(
        points,
        cells,
        build_axis_id,
        build_sign,
        part_base_coord_for_interval,
    )
    if args.layer_activation_geometry == "intersection" and args.layer_thickness is not None and args.layer_thickness > 0.0:
        layer_id_cell = compute_nominal_layer_id_from_interval(cell_d_min, cell_d_max, args)
        layer_id_cell[
            substrate_cell | support_cell | permanent_powder_cell
        ] = 0

    T_old = initial_temperature * np.ones((len(points), 1))
    u_guess = [np.zeros((len(points), 3))]
    eqp_quad = np.zeros((len(cells), thermal.fes[0].num_quads, 1))
    max_temperature_cell = initial_temperature * onp.ones(len(cells), dtype=onp.float64)
    initially_active = substrate_cell | support_cell
    activation_temperature_cell = initial_temperature * onp.ones(len(cells), dtype=onp.float64)
    activation_step_cell = -onp.ones(len(cells), dtype=onp.float64)
    activation_step_cell[initially_active] = 0
    solidification_temperature_cell = initial_temperature * onp.ones(len(cells), dtype=onp.float64)
    solidification_step_cell = -onp.ones(len(cells), dtype=onp.float64)
    solidification_step_cell[initially_active] = 0
    previous_active = initially_active.copy()
    phase_cell_init = initial_phase_cell(initially_active, substrate_cell, support_cell, args)
    phase_cell_init[permanent_powder_cell] = STATE_POWDER
    phase_quad = make_quad_scalar(phase_cell_init, thermal.fes[0].num_quads)
    T_ref_quad = initial_temperature * np.ones_like(eqp_quad)

    os.makedirs(args.output_dir, exist_ok=True)
    write_path_output(args, args.output_dir, step_states)
    write_calibration_template(args)
    derived = {
        "plane_axes": [ID_TO_AXIS[i] for i in plane_axis_ids],
        "scan_axis": ID_TO_AXIS[scan_axis_id],
        "hatch_axis": ID_TO_AXIS[hatch_axis_id],
        "total_steps": len(step_states),
        "scan_length": scan_length,
        "actual_scan_speed": actual_scan_speed,
        "base_coord": base_coord,
        "exposed_coord": exposed_coord,
        "part_pmin": part_pmin.tolist(),
        "part_pmax": part_pmax.tolist(),
        "path_length_scale": args.mesh_length_scale if args.path_length_scale is None else args.path_length_scale,
        "bottom_temperature_effective": bottom_temperature_effective,
        "front_surface_loss_enabled": args.front_surface_loss_h > 0.0,
        "layer_thickness": args.layer_thickness,
        "hatch_spacing": args.hatch_spacing,
        "scan_pattern": args.scan_pattern,
        "scan_rotation_per_layer": args.scan_rotation_per_layer,
        "jump_speed": args.jump_speed,
        "path_output": args.path_output,
        "active_window_below_layers": args.active_window_below_layers,
        "old_layer_thermal_factor": args.old_layer_thermal_factor,
        "old_layer_cooling_h": args.old_layer_cooling_h,
        "layer_activation_mode": args.layer_activation_mode,
        "future_layer_mode": args.future_layer_mode,
        "layer_activation_geometry": args.layer_activation_geometry,
        "strict_active_domain": strict_active_domain,
        "release_cell_set_sha256": args.release_cell_set_sha256,
        "release_cell_set_removed_count": args.release_cell_set_removed_count,
        "release_selection_mode": (
            "exact_cell_set"
            if release_cell_set is not None
            else (
                "geometric_box_diagnostic"
                if args.release_cut_box is not None
                else "none"
            )
        ),
        "paper_release_gate_eligible": release_cell_set is not None,
        "cooling_temperature_schedule": {
            "mode": (
                "linear_k_over_n_to_final"
                if final_cooldown_enabled
                else "constant_process_temperature"
            ),
            "process_ambient_k": float(args.ambient),
            "process_bottom_temperature_k": float(
                bottom_temperature_effective
            ),
            "final_temperature_k": (
                float(args.final_cooldown_temperature)
                if final_cooldown_enabled
                else None
            ),
            "cooling_step_count": sum(
                1 for state in step_states if state.mode == "cooling"
            ),
        },
    }
    write_used_config(args, args.output_dir, derived)
    print_startup(args, raw_pmin, raw_pmax, pmin, pmax, part_pmin, part_pmax, selected_cells, points, thermal, mechanics, derived)

    quad_stress = None
    last_mechanics_step = -1
    last_active_cell = previous_active.copy()
    last_printed_cell = previous_active.copy()
    last_cooling_only_cell = onp.zeros_like(previous_active, dtype=bool)
    last_material_state = material_cell_state(last_active_cell, substrate_cell, support_cell, args, phase_cell=phase_cell_from_quad(phase_quad))
    last_dT_quad = np.zeros((len(cells), thermal.fes[0].num_quads, 1))
    last_T_mech_quad = None
    last_mechanical_active_quad = None
    permanent_powder_quad = make_quad_scalar(
        permanent_powder_cell.astype(onp.float64), thermal.fes[0].num_quads)

    highest_printed_layer = 0

    for state in step_states:
        if args.layer_activation_mode == "layer_on_scan":
            current_layer = int(state.layer_idx) + 1
            if should_activate_layer_for_state(state):
                highest_printed_layer = max(highest_printed_layer, current_layer)
            if args.layer_activation_geometry == "intersection":
                printed_cell, active_cell, cooling_only_cell = compute_layer_on_scan_cells_by_intersection(
                    highest_printed_layer,
                    cell_d_min,
                    cell_d_max,
                    substrate_cell,
                    support_cell,
                    args,
                )
            else:
                printed_cell, active_cell, cooling_only_cell = compute_layer_on_scan_cells(
                    highest_printed_layer,
                    physical_layer_id_cell,
                    substrate_cell,
                    support_cell,
                    args,
                )
        elif args.active_window_below_layers > 0:
            if args.layer_activation_geometry == "intersection" and args.layer_thickness is not None and args.layer_thickness > 0.0:
                printed_cell, active_cell, cooling_only_cell = compute_moving_window_cells_by_intersection(
                    state,
                    cell_d_min,
                    cell_d_max,
                    substrate_cell,
                    support_cell,
                    args,
                )
            else:
                printed_cell, active_cell, cooling_only_cell = compute_moving_window_cells(
                    state,
                    physical_layer_id_cell,
                    substrate_cell,
                    support_cell,
                    args,
                )
        else:
            raw_active_cell = compute_active_cell(state, cell_build_coord, substrate_cell, support_cell, build_sign, args)
            printed_cell = previous_active | raw_active_cell
            active_cell = printed_cell
            cooling_only_cell = onp.zeros_like(active_cell, dtype=bool)

        # A named permanent-powder ELSET is a separate, step-0 physical domain:
        # it is never printed and never enters the part phase-history update.
        # Keep it out of layer masks even when its geometric layer interval
        # intersects the current recoated layer.
        printed_cell &= ~permanent_powder_cell
        active_cell &= ~permanent_powder_cell
        cooling_only_cell &= ~permanent_powder_cell
        thermal_active_cell = active_cell | permanent_powder_cell
        thermal_physical_cell = printed_cell | permanent_powder_cell

        active_quad = make_quad_scalar(thermal_active_cell.astype(onp.float64), thermal.fes[0].num_quads)
        printed_quad = make_quad_scalar(printed_cell.astype(onp.float64), thermal.fes[0].num_quads)
        thermal_physical_quad = make_quad_scalar(
            thermal_physical_cell.astype(onp.float64),
            thermal.fes[0].num_quads,
        )
        cooling_only_quad = make_quad_scalar(cooling_only_cell.astype(onp.float64), thermal.fes[0].num_quads)
        # Layer activation means the quadrature point now contains powder.
        # It becomes solid only after passing through liquid/mushy and cooling.
        phase_quad = np.where((printed_quad > 0.5) & (phase_quad == STATE_VOID), STATE_POWDER, phase_quad)
        if args.reset_activation_temperature:
            newly_printed_pre = printed_cell & (~previous_active)
            if newly_printed_pre.any():
                activation_reset_value = (
                    float(args.activation_reset_temperature)
                    if args.activation_reset_temperature is not None
                    else initial_temperature
                )
                T_old = reset_new_cell_nodal_temperature(
                    T_old, cells, newly_printed_pre, previous_active, activation_reset_value
                )
        T_old_quad = thermal.fes[0].convert_from_dof_to_quad(T_old)
        rho_quad, cp_quad, conductivity_quad, latent_cp_quad = thermal_material_quads(
            T_old_quad,
            active_quad,
            phase_quad,
            args,
            tables,
            printed_quad=thermal_physical_quad,
            cooling_only_quad=cooling_only_quad,
        )
        thermal_contributing_cell = contributing_cell_mask(
            rho_quad * (cp_quad + latent_cp_quad),
            conductivity_quad,
        )
        thermal_step_bc = thermal_bc
        if final_cooldown_enabled and state.mode == "cooling":
            def ramped_bottom_value(
                _point,
                _value=state.bottom_temperature,
            ):
                return _value

            thermal_step_bc = [[bottom], [0], [ramped_bottom_value]]

        if strict_active_domain:
            # Preserve the full, static mesh shape for JAX while making its
            # algebra identical to deleting future cells. Shared interface
            # nodes remain in the physical solve; only inactive-only nodes are
            # prescribed until their first recoating event.
            thermal_physical_nodes = physical_node_mask(
                cells,
                thermal_contributing_cell,
                num_nodes=len(points),
            )
            inactive_thermal_bc = make_inactive_node_dirichlet_bc(
                ~thermal_physical_nodes,
                vec=1,
                value=initial_temperature,
            )
            thermal_step_bc = merge_dirichlet_bcs(
                thermal_step_bc,
                inactive_thermal_bc,
            )

        if strict_active_domain or thermal_step_bc is not thermal_bc:
            thermal.fes[0].update_Dirichlet_boundary_conditions(
                thermal_step_bc
            )

        effective_laser_power = args.absorptivity * state.laser_power
        if surface_active_mask_enabled:
            surface_mask_quad = thermal_physical_quad
        else:
            surface_mask_quad = np.ones_like(printed_quad)
        thermal.set_params(
            [
                T_old,
                state.dt,
                np.asarray(state.laser_center),
                effective_laser_power,
                args.beam_radius,
                args.source_depth,
                state.laser_switch,
                active_quad,
                rho_quad,
                cp_quad,
                conductivity_quad,
                latent_cp_quad,
                cooling_only_quad,
                args.old_layer_cooling_h,
                surface_mask_quad,
                state.ambient_temperature,
            ]
        )
        T_new = solver(thermal, solver_options={"newton": {"linear": {"spsolve_solver": {}}}})[0]

        cell_T = compute_cell_temperature(T_new, cells)
        newly_printed = printed_cell & (~previous_active)
        activation_temperature_cell[newly_printed] = cell_T[newly_printed]
        activation_step_cell[newly_printed] = state.global_step
        max_temperature_cell = onp.maximum(max_temperature_cell, cell_T)

        T_quad = mechanics.fes[0].convert_from_dof_to_quad(T_new)

        # Phase/solidification history belongs to all printed material, not only
        # to the current thermal active window. This is important when a moving
        # thermal window is used: old printed layers may leave the thermal window
        # but they still need to cool, solidify, and retain residual-stress history.
        phase_update_quad = printed_quad
        phase_quad, T_ref_quad, eqp_quad, newly_solidified_quad, entered_melted_quad = update_phase_reference_and_eqp(
            T_quad, phase_update_quad, phase_quad, T_ref_quad, eqp_quad, args
        )
        newly_solidified_cell = onp.any(onp.asarray(newly_solidified_quad)[:, :, 0], axis=1)
        if newly_solidified_cell.any():
            T_ref_cell = onp.mean(onp.asarray(T_ref_quad)[:, :, 0], axis=1)
            solidification_temperature_cell[newly_solidified_cell] = T_ref_cell[newly_solidified_cell]
            solidification_step_cell[newly_solidified_cell] = state.global_step

        # Mechanical participation is different from thermal-window participation.
        # thermal active_quad controls heat conduction/source localization, while
        # mechanical_active_quad allows all printed material to contribute to
        # displacement/stress. Phase-dependent stiffness inside
        # mechanics_material_quads() still weakens powder/mushy/liquid regions.
        mechanical_active_quad = printed_quad
        if args.powder_solid_E is not None:
            # weak-solid powder carries load during the build (Kaess 2023)
            mechanical_active_quad = np.maximum(printed_quad, permanent_powder_quad)
        T_mech_quad = clamp_mechanics_temperature(T_quad, args.mechanics_temperature_floor)
        dT_quad = (T_mech_quad - T_ref_quad) * mechanical_active_quad
        active_factor_quad, E_quad, alpha_quad, poisson_quad, yield_quad, hardening_quad = mechanics_material_quads(T_mech_quad, mechanical_active_quad, phase_quad, args, tables)
        mechanical_contributing_cell = contributing_cell_mask(
            active_factor_quad
        )
        mechanics_params = [
            T_mech_quad,
            dT_quad,
            active_factor_quad,
            E_quad,
            alpha_quad,
            poisson_quad,
            yield_quad,
            hardening_quad,
            eqp_quad,
        ]

        is_last = state.global_step == step_states[-1].global_step
        did_mechanics = should_run_mechanics(state.global_step, args) or (is_last and args.mechanics_every > 0)
        if did_mechanics:
            if strict_active_domain:
                mechanical_physical_nodes = physical_node_mask(
                    cells,
                    mechanical_contributing_cell,
                    num_nodes=len(points),
                )
                inactive_mechanics_bc = make_inactive_node_dirichlet_bc(
                    ~mechanical_physical_nodes,
                    vec=3,
                    value=0.0,
                )
                mechanics.fes[0].update_Dirichlet_boundary_conditions(
                    merge_dirichlet_bcs(
                        mechanics_bc,
                        inactive_mechanics_bc,
                    )
                )
            # Lateral powder springs follow the printed state: only faces of
            # printed cells are embedded in powder and receive support.
            mechanics.set_powder_surface_mask(printed_cell)
            u_guess = run_mechanics_with_cutback(
                mechanics, u_guess, mechanics_params, mechanics_newton_overrides, args,
                last_T_mech_quad, last_mechanical_active_quad, T_ref_quad,
                mechanical_active_quad, phase_quad, tables)
            quad_stress = mechanics.compute_cell_stress(u_guess[0], mechanics_params)
            eqp_quad = mechanics.compute_eqp_update(u_guess[0], mechanics_params)
            mechanics_params[-1] = eqp_quad
            last_mechanics_step = state.global_step
            last_T_mech_quad = T_mech_quad
            last_mechanical_active_quad = mechanical_active_quad

        material_state_cell = material_cell_state(active_cell, substrate_cell, support_cell, args, cell_T, phase_cell=phase_cell_from_quad(phase_quad))
        mechanics_is_current = last_mechanics_step == state.global_step
        if should_save_step(state.global_step, did_mechanics, is_last, args):
            vtk_path = os.path.join(args.output_dir, f"step_{state.global_step:06d}_{state.mode}.vtu")
            save_step(
                thermal.fes[0],
                T_new,
                u_guess[0] if mechanics_is_current else np.zeros_like(u_guess[0]),
                vtk_path,
                dT_quad,
                quad_stress if mechanics_is_current else None,
                active_cell,
                printed_cell,
                cooling_only_cell,
                layer_id_cell,
                activation_step_cell,
                activation_temperature_cell,
                solidification_temperature_cell,
                solidification_step_cell,
                material_state_cell,
                max_temperature_cell,
                eqp_quad,
                float(mechanics_is_current),
                last_mechanics_step,
                MODE_TO_ID.get(state.mode, 0),
            )
        else:
            vtk_path = ""

        if state.global_step % args.summary_every == 0 or is_last:
            vm_max_value = float(np.max(quad_stress["vm_quad"])) if quad_stress is not None else 0.0
            print(
                f"global_step={state.global_step} mode={state.mode} "
                f"layer={state.layer_idx + 1}/{args.layers} "
                f"highest_printed_layer={highest_printed_layer} "
                f"hatch={state.hatch_idx + 1}/{args.hatch_lines_per_layer} "
                f"scan={state.scan_idx + 1}/{args.scan_steps_per_layer} "
                f"front_{ID_TO_AXIS[build_axis_id]}={state.front_coord:.12g} "
                f"active_window_cells={int(active_cell.sum())}/{len(active_cell)} "
                f"printed_cells={int(printed_cell.sum())}/{len(printed_cell)} "
                f"cooling_only_cells={int(cooling_only_cell.sum())}/{len(cooling_only_cell)} "
                f"mechanics_current={int(last_mechanics_step == state.global_step)} "
                f"mechanics_source_step={last_mechanics_step} "
                f"T_min={float(np.min(T_new)):.12g} T_max={float(np.max(T_new)):.12g} "
                f"u_max={float(np.max(np.abs(u_guess[0]))):.12g} "
                f"vm_max={vm_max_value:.12g} "
                f"laser_center={state.laser_center} laser_switch={state.laser_switch:.6g} "
                f"effective_power={effective_laser_power:.12g} vtk={vtk_path}"
            )

        previous_active = printed_cell
        last_active_cell = active_cell
        last_printed_cell = printed_cell
        last_cooling_only_cell = cooling_only_cell
        last_material_state = material_state_cell
        last_dT_quad = dT_quad
        T_old = T_new

    if args.release_after_cooling:
        release_point_fields = None
        printed_node_ids = onp.unique(
            onp.asarray(cells)[onp.asarray(last_printed_cell, dtype=bool)].reshape(-1)
        )
        if release_cell_set is not None:
            release_bc = exact_release_bc
        elif args.release_anchor_mode == "box":
            if args.release_anchor_box is None:
                raise ValueError("--release-anchor-mode box requires --release-anchor-box")
            release_bc = make_box_anchor_mechanics_bc(points, args.release_anchor_box)
        else:
            release_bc = make_anchor_mechanics_bc(points, candidate_node_ids=printed_node_ids)
        release_cut_cell = onp.zeros(len(cells), dtype=bool)
        if release_cell_set is not None:
            release_cut_cell = onp.asarray(
                release_cell_set.cell_mask,
                dtype=bool,
            )
            anchor_nodes, anchor_components, _ = (
                mechanics.fes[0].Dirichlet_boundary_conditions(release_bc)
            )
            anchor_pair_blocks = [
                onp.column_stack(
                    (
                        onp.asarray(node_ids, dtype=onp.int64),
                        onp.asarray(components, dtype=onp.int64),
                    )
                )
                for node_ids, components in zip(
                    anchor_nodes,
                    anchor_components,
                )
                if len(node_ids)
            ]
            anchor_dof_pairs = (
                onp.concatenate(anchor_pair_blocks, axis=0)
                if anchor_pair_blocks
                else onp.empty((0, 2), dtype=onp.int64)
            )
            expected_dof_pairs = {
                tuple(pair)
                for pair in args.paper_minimal_release_resolved_bc[
                    "constrained_dof_pairs"
                ]
            }
            actual_dof_pairs = {
                tuple(pair) for pair in anchor_dof_pairs.tolist()
            }
            if actual_dof_pairs != expected_dof_pairs:
                raise ValueError(
                    "resolved release anchor DOFs differ from the frozen "
                    "paper-minimal-root contract"
                )
            release_point_fields = {}
            root_bottom_mask = onp.zeros(len(points), dtype=onp.float64)
            root_bottom_mask[
                args.paper_minimal_release_resolved_bc[
                    "root_bottom_node_ids"
                ]
            ] = 1.0
            release_point_fields[
                f"release_bottom_u{ID_TO_AXIS[build_axis_id]}"
            ] = root_bottom_mask
            for component in plane_axis_ids:
                component_mask = onp.zeros(
                    len(points),
                    dtype=onp.float64,
                )
                component_nodes = anchor_dof_pairs[
                    anchor_dof_pairs[:, 1] == component,
                    0,
                ]
                component_mask[component_nodes] = 1.0
                release_point_fields[
                    f"release_anchor_u{ID_TO_AXIS[component]}"
                ] = component_mask
            anchor_node_ids = onp.unique(anchor_dof_pairs[:, 0])
            validate_release_cell_set(
                release_cell_set,
                cells=cells,
                points=points,
                removable_cell_mask=support_cell,
                protected_cell_mask=last_printed_cell & (~support_cell),
                anchor_node_ids=anchor_node_ids,
                anchor_dof_pairs=anchor_dof_pairs,
            )
            print(
                "release exact cell set: deactivating "
                f"{int(release_cut_cell.sum())} validated support cells"
            )
        elif args.release_cut_box is not None:
            cut = [float(v) for v in args.release_cut_box]
            lo = onp.asarray(cut[0::2])
            hi = onp.asarray(cut[1::2])
            release_cut_cell = onp.all(
                (cell_centroids >= lo[None, :]) & (cell_centroids <= hi[None, :]),
                axis=1,
            )
            n_cut = int(release_cut_cell.sum())
            print(f"release cut box: deactivating {n_cut} cells in {cut}")
        if release_cut_cell.any():
            cut_quad = make_quad_scalar(
                release_cut_cell.astype(onp.float64),
                mechanics.fes[0].num_quads,
            )
            # Kill both stiffness and locked-in stress of sawed-off cells.
            mechanics_params = list(mechanics_params)
            if release_cell_set is not None:
                mechanics_params[2] = zero_exact_release_cells(
                    mechanics_params[2],
                    cut_quad,
                )
            else:
                mechanics_params[2] = np.where(
                    cut_quad > 0.5,
                    (
                        0.0
                        if strict_active_domain
                        else args.inactive_mechanics_factor
                    ),
                    mechanics_params[2],
                )
        if args.powder_solid_E is not None and permanent_powder_cell.any():
            # depowdering precedes the saw cut: weak-solid powder is removed
            # for the release solve exactly like sawed-off cells
            print(f"release depowder: deactivating "
                  f"{int(permanent_powder_cell.sum())} powder cells")
            mechanics_params = list(mechanics_params)
            mechanics_params[2] = np.where(
                permanent_powder_quad > 0.5,
                (
                    0.0
                    if strict_active_domain
                    else args.inactive_mechanics_factor
                ),
                mechanics_params[2],
            )
        if strict_active_domain:
            release_contributing_cell = contributing_cell_mask(
                mechanics_params[2]
            )
            release_physical_nodes = physical_node_mask(
                cells,
                release_contributing_cell,
                num_nodes=len(points),
            )
            release_bc = merge_dirichlet_bcs(
                release_bc,
                make_inactive_node_dirichlet_bc(
                    ~release_physical_nodes,
                    vec=3,
                    value=0.0,
                ),
            )
        release_mechanics = ThermoMechanical(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type=ele_type,
            quadrature_order=args.quadrature_order,
            dirichlet_bc_info=release_bc,
            additional_info=(args.mechanics_model, args.yield_saturation_stress,
                             0.0, 0.0, (), bbar_enabled),
        )
        u_release = run_mechanics(release_mechanics, u_guess, mechanics_params, mechanics_newton_overrides)
        quad_stress = release_mechanics.compute_cell_stress(u_release[0], mechanics_params)
        vtk_path = os.path.join(args.output_dir, "release.vtu")
        save_step(
            release_mechanics.fes[0],
            T_old,
            u_release[0],
            vtk_path,
            last_dT_quad,
            quad_stress,
            last_active_cell,
            last_printed_cell,
            last_cooling_only_cell,
            layer_id_cell,
            activation_step_cell,
            activation_temperature_cell,
            solidification_temperature_cell,
            solidification_step_cell,
            last_material_state,
            max_temperature_cell,
            eqp_quad,
            1.0,
            step_states[-1].global_step + 1 if step_states else 0,
            MODE_TO_ID["release"],
            release_cut_cell,
            release_point_fields,
        )
        print(f"release_vtk={vtk_path} release_u_max={float(np.max(np.abs(u_release[0]))):.12g}")


if __name__ == "__main__":
    main()
