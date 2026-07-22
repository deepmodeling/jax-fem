import argparse
import os

import jax
import jax.numpy as np
import numpy as onp

from inp_initial_guess_smoke import read_tet4_inp
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol


AXIS_TO_ID = {"x": 0, "y": 1, "z": 2}
ID_TO_AXIS = ("x", "y", "z")


class TransientThermal(Problem):
    def custom_init(self, rho, cp, conductivity, convection_h, ambient, build_axis_id, plane_axis0_id, plane_axis1_id):
        self.rho = rho
        self.cp = cp
        self.conductivity = conductivity
        self.convection_h = convection_h
        self.ambient = ambient
        self.build_axis_id = int(build_axis_id)
        self.plane_axis0_id = int(plane_axis0_id)
        self.plane_axis1_id = int(plane_axis1_id)

    def get_tensor_map(self):
        def heat_flux(T_grad, T_old, dt, laser_center, laser_power, beam_radius, source_depth, laser_switch):
            return self.conductivity * T_grad

        return heat_flux

    def get_mass_map(self):
        def heat_capacity(T, x, T_old, dt, laser_center, laser_power, beam_radius, source_depth, laser_switch):
            r0 = x[self.plane_axis0_id] - laser_center[self.plane_axis0_id]
            r1 = x[self.plane_axis1_id] - laser_center[self.plane_axis1_id]
            depth = x[self.build_axis_id] - laser_center[self.build_axis_id]
            r2 = r0**2 + r1**2
            depth2 = depth**2
            q_vol = (
                laser_power
                / (np.pi * beam_radius**2 * source_depth)
                * np.exp(-2.0 * r2 / beam_radius**2)
                * np.exp(-2.0 * depth2 / source_depth**2)
                * laser_switch
            )
            return self.rho * self.cp * (T - T_old) / dt - q_vol

        return heat_capacity

    def get_surface_maps(self):
        def top_flux(_T, point, old_T, laser_center, laser_power, beam_radius, laser_switch):
            r0 = point[self.plane_axis0_id] - laser_center[self.plane_axis0_id]
            r1 = point[self.plane_axis1_id] - laser_center[self.plane_axis1_id]
            r2 = r0**2 + r1**2
            q_laser = (
                2.0
                * laser_power
                / (np.pi * beam_radius**2)
                * np.exp(-2.0 * r2 / beam_radius**2)
                * laser_switch
            )
            q_conv = self.convection_h * (self.ambient - old_T[0])
            return -np.array([q_laser + q_conv])

        def wall_flux(_T, _point, old_T):
            q_conv = self.convection_h * (self.ambient - old_T[0])
            return -np.array([q_conv])

        return [top_flux, wall_flux]

    def set_params(self, params):
        T_old, dt, laser_center, laser_power, beam_radius, source_depth, laser_switch = params
        dt_quad = dt * np.ones((self.fes[0].num_cells, self.fes[0].num_quads, 1))
        laser_center_quad = laser_center[None, None, :] * np.ones((self.fes[0].num_cells, self.fes[0].num_quads))[:, :, None]
        laser_power_quad = laser_power * np.ones((self.fes[0].num_cells, self.fes[0].num_quads, 1))
        beam_radius_quad = beam_radius * np.ones((self.fes[0].num_cells, self.fes[0].num_quads, 1))
        source_depth_quad = source_depth * np.ones((self.fes[0].num_cells, self.fes[0].num_quads, 1))
        laser_switch_quad = laser_switch * np.ones((self.fes[0].num_cells, self.fes[0].num_quads, 1))
        self.internal_vars = [
            self.fes[0].convert_from_dof_to_quad(T_old),
            dt_quad,
            laser_center_quad,
            laser_power_quad,
            beam_radius_quad,
            source_depth_quad,
            laser_switch_quad,
        ]

        T_old_top = self.fes[0].convert_from_dof_to_face_quad(T_old, self.boundary_inds_list[0])
        T_old_walls = self.fes[0].convert_from_dof_to_face_quad(T_old, self.boundary_inds_list[1])
        laser_center_top = laser_center[None, None, :] * np.ones(
            (len(self.boundary_inds_list[0]), self.fes[0].num_face_quads)
        )[:, :, None]
        laser_power_top = laser_power * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))
        beam_radius_top = beam_radius * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))
        laser_switch_top = laser_switch * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))
        self.internal_vars_surfaces = [
            [T_old_top, laser_center_top, laser_power_top, beam_radius_top, laser_switch_top],
            [T_old_walls],
        ]


class LinearThermoElasticity(Problem):
    def custom_init(self, young, poisson, alpha):
        self.young = young
        self.poisson = poisson
        self.alpha = alpha
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lmbda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.internal_vars = [
            np.zeros((len(self.fes[0].cells), self.fes[0].num_quads, 1))
        ]

    def stress_fn(self, u_grad, dT):
        eps = 0.5 * (u_grad + u_grad.T)
        thermal_eps = self.alpha * dT[0] * np.eye(self.dim)
        elastic_eps = eps - thermal_eps
        return self.lmbda * np.trace(elastic_eps) * np.eye(self.dim) + 2.0 * self.mu * elastic_eps

    def get_tensor_map(self):
        return self.stress_fn

    def set_params(self, params):
        self.internal_vars = params

    def compute_cell_stress(self, sol, dT_quad):
        u_grads = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)
        sigmas = jax.vmap(jax.vmap(self.stress_fn))(u_grads, dT_quad)
        sigma_mean = np.mean(sigmas, axis=1)
        sxx = sigma_mean[:, 0, 0]
        syy = sigma_mean[:, 1, 1]
        szz = sigma_mean[:, 2, 2]
        sxy = sigma_mean[:, 0, 1]
        syz = sigma_mean[:, 1, 2]
        sxz = sigma_mean[:, 0, 2]
        vm = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + sxz**2)
        )
        return sigma_mean, vm


def make_box_locations(points, build_axis="x", base_side="min", tol_ratio=1e-8):
    pmin = onp.min(points, axis=0)
    pmax = onp.max(points, axis=0)
    span = max(float(onp.max(pmax - pmin)), 1.0)
    atol = tol_ratio * span
    build_axis_id = AXIS_TO_ID[build_axis]
    plane_axis_ids = tuple(i for i in range(3) if i != build_axis_id)

    if base_side == "min":
        base_coord = pmin[build_axis_id]
        exposed_coord = pmax[build_axis_id]
    else:
        base_coord = pmax[build_axis_id]
        exposed_coord = pmin[build_axis_id]

    def bottom(point):
        return np.isclose(point[build_axis_id], base_coord, atol=atol)

    def top(point):
        return np.isclose(point[build_axis_id], exposed_coord, atol=atol)

    def walls(point):
        a0, a1 = plane_axis_ids
        return (
            np.isclose(point[a0], pmin[a0], atol=atol)
            | np.isclose(point[a0], pmax[a0], atol=atol)
            | np.isclose(point[a1], pmin[a1], atol=atol)
            | np.isclose(point[a1], pmax[a1], atol=atol)
        )

    return pmin, pmax, bottom, top, walls, build_axis_id, plane_axis_ids


def resolve_scan_axis(scan_axis, build_axis_id, plane_axis_ids):
    if scan_axis == "auto":
        return plane_axis_ids[0]

    axis_id = AXIS_TO_ID[scan_axis]
    if axis_id == build_axis_id:
        print(
            "WARNING: scan_axis is the same as build_axis. "
            "For layer-plane scanning, use --scan-axis auto or choose one of "
            f"{ID_TO_AXIS[plane_axis_ids[0]]}/{ID_TO_AXIS[plane_axis_ids[1]]}."
        )
    return axis_id


def make_laser_path(pmin, pmax, args, build_axis_id, plane_axis_ids):
    axis_id = resolve_scan_axis(args.scan_axis, build_axis_id, plane_axis_ids)
    span = pmax - pmin
    midpoint = 0.5 * (pmin + pmax)
    base = onp.array(midpoint, dtype=onp.float64)
    base[build_axis_id] = pmax[build_axis_id] if args.base_side == "min" else pmin[build_axis_id]
    fixed_values = [args.scan_fixed_x, args.scan_fixed_y, args.scan_fixed_z]

    for i, value in enumerate(fixed_values):
        if value is not None:
            base[i] = value

    start_axis = args.scan_start
    if start_axis is None:
        start_axis = pmin[axis_id] + args.scan_start_frac * span[axis_id]

    end_axis = args.scan_end
    if end_axis is None:
        end_axis = pmin[axis_id] + args.scan_end_frac * span[axis_id]

    start = base.copy()
    end = base.copy()
    start[axis_id] = start_axis
    end[axis_id] = end_axis
    return start, end, axis_id


def laser_state(step, args, scan_start, scan_end):
    path = scan_end - scan_start
    path_length = float(onp.linalg.norm(path))

    if args.scan_speed > 0.0 and path_length > 0.0:
        traveled = args.scan_speed * args.dt * (step + 1)
        raw_frac = traveled / path_length
        frac = min(raw_frac, 1.0)
        laser_switch = 1.0 if raw_frac <= 1.0 else 0.0
    else:
        frac = 0.0 if args.steps <= 1 else step / (args.steps - 1)
        laser_switch = 1.0

    laser_center = scan_start + frac * path
    return np.asarray(laser_center), laser_switch, frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default="/home/user/work/159/schema/0119_c3d4_only.inp")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--ambient", type=float, default=300.0)
    parser.add_argument("--rho", type=float, default=7800.0)
    parser.add_argument("--cp", type=float, default=500.0)
    parser.add_argument("--conductivity", type=float, default=20.0)
    parser.add_argument("--convection-h", type=float, default=10.0)
    parser.add_argument("--young", type=float, default=2.0e11)
    parser.add_argument("--poisson", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1.2e-5)
    parser.add_argument("--laser-power", type=float, default=1.0)
    parser.add_argument("--beam-radius", type=float, default=0.0)
    parser.add_argument("--source-depth", type=float, default=0.0)
    parser.add_argument("--build-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--base-side", choices=("min", "max"), default="min")
    parser.add_argument("--scan-axis", choices=("auto", "x", "y", "z"), default="auto")
    parser.add_argument("--scan-start", type=float, default=None)
    parser.add_argument("--scan-end", type=float, default=None)
    parser.add_argument("--scan-start-frac", type=float, default=0.25)
    parser.add_argument("--scan-end-frac", type=float, default=0.75)
    parser.add_argument("--scan-speed", type=float, default=0.0)
    parser.add_argument("--scan-fixed-x", type=float, default=None)
    parser.add_argument("--scan-fixed-y", type=float, default=None)
    parser.add_argument("--scan-fixed-z", type=float, default=None)
    parser.add_argument("--mechanics-every", type=int, default=1)
    parser.add_argument("--output-dir", default="/home/user/work/159/output/inp_thermal_stress_oneway")
    args = parser.parse_args()

    points, cells, selected_cells = read_tet4_inp(args.inp, args.max_cells)
    mesh = Mesh(points, cells, ele_type="TET4")
    pmin, pmax, bottom, top, walls, build_axis_id, plane_axis_ids = make_box_locations(
        points,
        build_axis=args.build_axis,
        base_side=args.base_side,
    )
    span = pmax - pmin
    plane_scale = max(float(span[plane_axis_ids[0]]), float(span[plane_axis_ids[1]]), 1e-12)
    build_span = max(float(span[build_axis_id]), 1e-12)
    beam_radius = args.beam_radius if args.beam_radius > 0 else 0.08 * plane_scale
    source_depth = args.source_depth if args.source_depth > 0 else max(0.5 * beam_radius, 0.1 * build_span)
    scan_start, scan_end, scan_axis_id = make_laser_path(pmin, pmax, args, build_axis_id, plane_axis_ids)

    def ambient_value(_point):
        return args.ambient

    def zero(_point):
        return 0.0

    thermal_bc = [[bottom], [0], [ambient_value]]
    mechanics_bc = [[bottom, bottom, bottom], [0, 1, 2], [zero, zero, zero]]

    thermal = TransientThermal(
        mesh=mesh,
        vec=1,
        dim=3,
        ele_type="TET4",
        dirichlet_bc_info=thermal_bc,
        location_fns=[top, walls],
        additional_info=(
            args.rho,
            args.cp,
            args.conductivity,
            args.convection_h,
            args.ambient,
            build_axis_id,
            plane_axis_ids[0],
            plane_axis_ids[1],
        ),
    )
    mechanics = LinearThermoElasticity(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type="TET4",
        dirichlet_bc_info=mechanics_bc,
        additional_info=(args.young, args.poisson, args.alpha),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    T_old = args.ambient * np.ones((len(points), 1))
    u_guess = [np.zeros((len(points), 3))]

    print("inp:", args.inp)
    print("selected_cells:", selected_cells)
    print("points:", len(points))
    print("thermal_dofs:", thermal.num_total_dofs_all_vars)
    print("mechanical_dofs:", mechanics.num_total_dofs_all_vars)
    print("thermal_boundary_face_counts:", [len(x) for x in thermal.boundary_inds_list])
    print("thermal_dirichlet_node_counts:", [len(x) for x in thermal.fes[0].node_inds_list])
    print("mechanical_dirichlet_node_counts:", [len(x) for x in mechanics.fes[0].node_inds_list])
    print("build_axis:", args.build_axis)
    print("base_side:", args.base_side)
    print("plane_axes:", [ID_TO_AXIS[i] for i in plane_axis_ids])
    print("beam_radius:", beam_radius)
    print("source_depth:", source_depth)
    print("scan_axis:", ID_TO_AXIS[scan_axis_id])
    print("scan_start:", scan_start)
    print("scan_end:", scan_end)
    print("scan_speed:", args.scan_speed)

    for step in range(args.steps):
        laser_center, laser_switch, laser_frac = laser_state(step, args, scan_start, scan_end)
        thermal.set_params([T_old, args.dt, laser_center, args.laser_power, beam_radius, source_depth, laser_switch])
        T_new = solver(thermal, solver_options={"newton": {"linear": {"spsolve_solver": {}}}})[0]

        if step % args.mechanics_every == 0:
            dT_quad = mechanics.fes[0].convert_from_dof_to_quad(T_new - args.ambient)
            mechanics.set_params([dT_quad])
            u_guess = solver(
                mechanics,
                solver_options={
                    "newton": {
                        "initial_guess": u_guess,
                        "linear": {"spsolve_solver": {}},
                        "tol": 1e-9,
                        "rel_tol": 1e-11,
                    }
                },
            )
            sigma_mean, vm = mechanics.compute_cell_stress(u_guess[0], dT_quad)
            vtk_path = os.path.join(args.output_dir, f"step_{step:04d}.vtu")
            save_sol(
                thermal.fes[0],
                T_new,
                vtk_path,
                point_infos=[("u", u_guess[0])],
                cell_infos=[
                    ("dT", np.mean(dT_quad[:, :, 0], axis=1)),
                    ("stress_xx", sigma_mean[:, 0, 0]),
                    ("von_mises", vm),
                ],
            )
            print(
                f"step={step} T_min={float(np.min(T_new)):.12g} "
                f"T_max={float(np.max(T_new)):.12g} "
                f"u_max={float(np.max(np.abs(u_guess[0]))):.12g} "
                f"vm_max={float(np.max(vm)):.12g} "
                f"laser_center={onp.asarray(laser_center)} "
                f"laser_frac={laser_frac:.6g} laser_switch={laser_switch:.6g} "
                f"vtk={vtk_path}"
            )

        T_old = T_new


if __name__ == "__main__":
    main()
