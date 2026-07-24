"""VTU/ParaView output helpers and run metadata writers.

Origin: legacy/v03/am_thermal_stress_macro_intersection_mech100.py
(von_mises_from_stress, STRESS_COMPONENTS, empty_quad_stress, quad_field_name,
make_quad_stress_cell_infos, save_step, write_used_config, write_path_output,
write_calibration_template, print_startup). Moved verbatim in the 2026-07-22
restructure.
"""
import csv
import json
import os

import jax.numpy as np

from jax_fem.utils import save_sol


def von_mises_from_stress(stress):
    stress = np.asarray(stress)
    sxx = stress[..., 0, 0]
    syy = stress[..., 1, 1]
    szz = stress[..., 2, 2]
    sxy = 0.5 * (stress[..., 0, 1] + stress[..., 1, 0])
    syz = 0.5 * (stress[..., 1, 2] + stress[..., 2, 1])
    sxz = 0.5 * (stress[..., 0, 2] + stress[..., 2, 0])
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )


STRESS_COMPONENTS = (
    ("xx", 0, 0),
    ("yy", 1, 1),
    ("zz", 2, 2),
    ("xy", 0, 1),
    ("yz", 1, 2),
    ("xz", 0, 2),
)


def empty_quad_stress(num_cells, num_quads):
    return {
        "stress_quad": np.zeros((num_cells, num_quads, 3, 3)),
        "vm_quad": np.zeros((num_cells, num_quads)),
    }


def quad_field_name(base, quad_idx, num_quads):
    if num_quads == 1:
        return base
    return f"{base.replace('quad', f'quad{quad_idx}', 1)}"


def make_quad_stress_cell_infos(quad_stress):
    stress_quad = np.asarray(quad_stress["stress_quad"])
    vm_quad = np.asarray(quad_stress["vm_quad"])
    num_quads = stress_quad.shape[1]
    cell_infos = []
    for quad_idx in range(num_quads):
        for suffix, row, col in STRESS_COMPONENTS:
            value = stress_quad[:, quad_idx, row, col]
            if row != col:
                value = 0.5 * (value + stress_quad[:, quad_idx, col, row])
            cell_infos.append((quad_field_name(f"stress_quad_{suffix}", quad_idx, num_quads), value))
        cell_infos.append((quad_field_name("vm_quad", quad_idx, num_quads), vm_quad[:, quad_idx]))
    return cell_infos


def save_step(
    fe,
    T_new,
    u,
    vtk_path,
    dT_quad,
    quad_stress,
    active_cell,
    printed_cell,
    cooling_only_cell,
    layer_id_cell,
    activation_step_cell,
    activation_temperature_cell,
    stress_free_temperature_cell,
    solidification_step_cell,
    material_state_cell,
    max_temperature_cell,
    eqp_quad,
    mechanics_valid,
    mechanics_source_step,
    mode_id,
):
    if quad_stress is None:
        quad_stress = empty_quad_stress(fe.num_cells, dT_quad.shape[1])
    eqp_cell = np.mean(eqp_quad[:, :, 0], axis=1)
    point_infos = [("T", T_new), ("u", u)]
    cell_infos = [
        ("active", np.asarray(active_cell, dtype=np.float64)),
        ("printed", np.asarray(printed_cell, dtype=np.float64)),
        ("cooling_only", np.asarray(cooling_only_cell, dtype=np.float64)),
        ("layer_id", np.asarray(layer_id_cell, dtype=np.float64)),
        ("activation_step", np.asarray(activation_step_cell, dtype=np.float64)),
        ("activation_temperature", np.asarray(activation_temperature_cell, dtype=np.float64)),
        ("stress_free_temperature", np.asarray(stress_free_temperature_cell, dtype=np.float64)),
        ("solidification_step", np.asarray(solidification_step_cell, dtype=np.float64)),
        ("material_state", np.asarray(material_state_cell, dtype=np.float64)),
        ("dT", np.mean(dT_quad[:, :, 0], axis=1)),
        ("eq_plastic_strain", eqp_cell),
        ("max_temperature_history", np.asarray(max_temperature_cell, dtype=np.float64)),
        ("mechanics_valid", np.full(fe.num_cells, float(mechanics_valid), dtype=np.float64)),
        ("mechanics_source_step", np.full(fe.num_cells, float(mechanics_source_step), dtype=np.float64)),
        ("mode_id", np.full(fe.num_cells, float(mode_id), dtype=np.float64)),
    ]
    cell_infos.extend(make_quad_stress_cell_infos(quad_stress))
    save_sol(
        fe,
        T_new,
        vtk_path,
        point_infos=point_infos,
        cell_infos=cell_infos,
    )


def write_used_config(args, output_dir, derived):
    os.makedirs(output_dir, exist_ok=True)
    data = dict(vars(args))
    data["derived"] = derived
    with open(os.path.join(output_dir, "used_config.json"), "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_path_output(args, output_dir, step_states):
    if not args.path_output:
        return
    path = args.path_output
    if not os.path.isabs(path):
        path = os.path.join(output_dir, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    current_time = 0.0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "time",
            "mode",
            "layer",
            "hatch",
            "scan",
            "x",
            "y",
            "z",
            "front_coord",
            "power",
            "laser_on",
            "dt",
            "scan_frac",
            "hatch_frac",
        ])
        for state in step_states:
            current_time += state.dt
            writer.writerow([
                state.global_step,
                current_time,
                state.mode,
                state.layer_idx + 1,
                state.hatch_idx + 1,
                state.scan_idx + 1,
                float(state.laser_center[0]),
                float(state.laser_center[1]),
                float(state.laser_center[2]),
                state.front_coord,
                state.laser_power,
                state.laser_switch,
                state.dt,
                state.scan_frac,
                state.hatch_frac,
            ])
    print(f"path_output: {path}")


def write_calibration_template(args):
    if args.calibration_dir is None:
        return
    os.makedirs(args.calibration_dir, exist_ok=True)
    path = os.path.join(args.calibration_dir, "calibration_notes.md")
    if os.path.exists(path):
        return
    with open(path, "w") as f:
        f.write("# Calibration Notes\n\n")
        f.write("- melt pool width:\n- melt pool depth:\n- peak temperature:\n")
        f.write("- cooling rate:\n- final distortion:\n- residual stress:\n")


def print_startup(args, raw_pmin, raw_pmax, pmin, pmax, part_pmin, part_pmax, selected_cells, points, thermal, mechanics, derived):
    print("inp:", args.inp)
    print("selected_cells:", selected_cells)
    print("points:", len(points))
    print("raw_pmin:", raw_pmin)
    print("raw_pmax:", raw_pmax)
    print("mesh_length_scale:", args.mesh_length_scale)
    print("path_length_scale:", args.mesh_length_scale if args.path_length_scale is None else args.path_length_scale)
    print("scaled_pmin:", pmin)
    print("scaled_pmax:", pmax)
    print("part_pmin:", part_pmin)
    print("part_pmax:", part_pmax)
    print("thermal_dofs:", thermal.num_total_dofs_all_vars)
    print("mechanical_dofs:", mechanics.num_total_dofs_all_vars)
    print("build_axis:", args.build_axis)
    print("base_side:", args.base_side)
    print("build_direction:", f"+{args.build_axis}" if args.base_side == "min" else f"-{args.build_axis}")
    print("plane_axes:", derived["plane_axes"])
    print("scan_axis:", derived["scan_axis"])
    print("hatch_axis:", derived["hatch_axis"])
    print("layers:", args.layers)
    print("layer_thickness:", args.layer_thickness)
    print("hatch_lines_per_layer:", args.hatch_lines_per_layer)
    print("hatch_spacing:", args.hatch_spacing)
    print("scan_pattern:", args.scan_pattern)
    print("scan_rotation_per_layer:", args.scan_rotation_per_layer)
    print("jump_speed:", args.jump_speed)
    print("scan_steps_per_layer:", args.scan_steps_per_layer)
    print("total_thermal_steps:", derived["total_steps"])
    print("scan_length:", derived["scan_length"])
    print("scan_speed:", args.scan_speed)
    print("actual_scan_speed:", derived["actual_scan_speed"])
    print("dt:", args.dt)
    print("laser_power:", args.laser_power)
    print("absorptivity:", args.absorptivity)
    print("effective_laser_power_nominal:", args.absorptivity * args.laser_power)
    print("source_model:", args.source_model)
    if args.source_model == "paper_hemispherical":
        print(
            "heat_source_normalization:",
            "6*sqrt(3)*P_abs/(pi*sqrt(pi)*r_b^3)",
        )
    else:
        print("heat_source_normalization:", "2*P_abs/(pi*r_b^2*source_depth)")
    print("beam_radius:", args.beam_radius)
    print("source_depth:", args.source_depth)
    print("powder_mode:", args.powder_mode)
    print("inactive_thermal_factor:", args.inactive_thermal_factor)
    print("inactive_mechanics_factor:", args.inactive_mechanics_factor)
    print("rho_solid:", args.rho_solid if args.rho_solid is not None else args.rho)
    print("cp_solid:", args.cp_solid if args.cp_solid is not None else args.cp)
    print("conductivity_solid:", args.conductivity_solid if args.conductivity_solid is not None else args.conductivity)
    print("rho_powder:", args.rho_powder)
    print("cp_powder:", args.cp_powder)
    print("conductivity_powder:", args.conductivity_powder)
    print("rho_liquid:", args.rho_liquid if args.rho_liquid is not None else (args.rho_solid if args.rho_solid is not None else args.rho))
    print("cp_liquid:", args.cp_liquid if args.cp_liquid is not None else (args.cp_solid if args.cp_solid is not None else args.cp))
    print("conductivity_liquid:", args.conductivity_liquid if args.conductivity_liquid is not None else (args.conductivity_solid if args.conductivity_solid is not None else args.conductivity))
    print("emissivity:", args.emissivity)
    print("bottom_thermal_bc:", args.bottom_thermal_bc)
    print("bottom_temperature_effective:", derived["bottom_temperature_effective"])
    print("front_surface_loss_h:", args.front_surface_loss_h)
    print("front_surface_loss_thickness:", args.front_surface_loss_thickness)
    print("front_surface_loss_radiation:", args.front_surface_loss_radiation)
    print("mechanics_model:", args.mechanics_model)
    print("mushy_mechanics_factor:", args.mushy_mechanics_factor)
    print("liquid_mechanics_factor:", args.liquid_mechanics_factor)
    print("reset_plastic_on_melt:", args.reset_plastic_on_melt)
    print("active_window_below_layers:", args.active_window_below_layers)
    print("old_layer_thermal_factor:", args.old_layer_thermal_factor)
    print("old_layer_cooling_h:", args.old_layer_cooling_h)
    print("layer_activation_mode:", args.layer_activation_mode)
    print("future_layer_mode:", args.future_layer_mode)
    print("layer_activation_geometry:", args.layer_activation_geometry)
    if args.mechanics_model == "j2_plastic":
        print("WARNING: j2_plastic is a simplified stress-clipping/radial-return approximation; do not use as industrial-grade quantitative residual stress.")
    print("thermal_boundary_face_counts:", [len(x) for x in thermal.boundary_inds_list])
    print("thermal_dirichlet_node_counts:", [len(x) for x in thermal.fes[0].node_inds_list])
    print("mechanical_dirichlet_node_counts:", [len(x) for x in mechanics.fes[0].node_inds_list])
    print("output_dir:", args.output_dir)
