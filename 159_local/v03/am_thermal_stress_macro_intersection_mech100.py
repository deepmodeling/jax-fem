import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

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
MODE_TO_ID = {
    "scan": 1,
    "hatch_dwell": 2,
    "layer_dwell": 3,
    "recoat": 4,
    "cooling": 5,
    "path": 6,
    "release": 7,
    "jump": 8,
}

# Material/phase codes saved to ParaView and used by the thermal/mechanical
# property maps. Keep these values stable for post-processing scripts.
STATE_VOID = 0.0
STATE_POWDER = 1.0
STATE_SOLID = 2.0
STATE_MUSHY = 3.0
STATE_LIQUID = 4.0
STATE_SUBSTRATE = 5.0
STATE_SUPPORT = 6.0


@dataclass
class StepState:
    global_step: int
    mode: str
    layer_idx: int
    hatch_idx: int
    scan_idx: int
    laser_center: onp.ndarray
    laser_power: float
    laser_switch: float
    dt: float
    scan_frac: float
    hatch_frac: float
    front_coord: float
    layer_frac: float


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


class PropertyTable:
    def __init__(self, path):
        self.path = path
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "T" not in reader.fieldnames or "value" not in reader.fieldnames:
                raise ValueError(f"Property table must contain T,value columns: {path}")
            for row in reader:
                rows.append((float(row["T"]), float(row["value"])))
        if len(rows) < 2:
            raise ValueError(f"Property table needs at least two rows: {path}")
        rows.sort(key=lambda item: item[0])
        self.T = np.asarray([r[0] for r in rows])
        self.values = np.asarray([r[1] for r in rows])

    def eval(self, T):
        return np.interp(T, self.T, self.values)


def apply_thermal_mass_lumping(problem):
    """Switch the thermal volume quadrature to the TET4 vertex rule (lumping).

    With linear tets the stiffness integrand is constant per cell, so moving
    the quadrature points to the vertices (equal weights V/4) leaves
    conduction bitwise unchanged while making the capacitance matrix exactly
    the row-sum lumped diagonal: shape_vals becomes the identity, so the
    mass-map trial/test products N_i*N_j collapse to delta_ij. Source and
    loss terms in the mass map become nodal collocation, which is why the
    physical quad points must be relocated to the vertices as well.
    """
    fe = problem.fes[0]
    if fe.num_quads != fe.num_nodes:
        raise ValueError(
            "--thermal-mass-lumping requires num_quads == num_nodes "
            f"(TET4 with --quadrature-order 2); got {fe.num_quads} quads, "
            f"{fe.num_nodes} nodes")
    weights = onp.asarray(fe.quad_weights)
    if not onp.allclose(weights, weights[0]):
        raise ValueError(
            "--thermal-mass-lumping requires an equal-weight quadrature rule; "
            f"got weights {weights}")
    grads = onp.asarray(fe.shape_grads_ref)
    if not onp.allclose(grads, grads[0:1]):
        raise ValueError(
            "--thermal-mass-lumping requires constant shape gradients "
            "(linear elements); higher-order elements are not supported")
    fe.shape_vals = np.eye(fe.num_nodes)
    problem.physical_quad_points = fe.get_physical_quad_points()
    print("thermal mass lumping: TET4 vertex quadrature installed "
          "(diagonal capacitance, conduction unchanged)")
    return problem


class TransientThermal(Problem):
    def custom_init(
        self,
        convection_h,
        ambient,
        emissivity,
        stefan_boltzmann,
        build_axis_id,
        plane_axis0_id,
        plane_axis1_id,
        build_sign,
        num_surface_maps,
        front_surface_loss_h,
        front_surface_loss_thickness,
        front_surface_loss_radiation,
    ):
        self.convection_h = convection_h
        self.ambient = ambient
        self.emissivity = emissivity
        self.stefan_boltzmann = stefan_boltzmann
        self.build_axis_id = int(build_axis_id)
        self.plane_axis0_id = int(plane_axis0_id)
        self.plane_axis1_id = int(plane_axis1_id)
        self.build_sign = float(build_sign)
        self.num_surface_maps = int(num_surface_maps)
        # Optional volumetric approximation of convection/radiation loss on the
        # moving active/inactive build front. The true build front is an internal
        # interface of the full mesh, so it is not available to static boundary
        # face selectors. Set front_surface_loss_h=0.0 to disable this term.
        self.front_surface_loss_h = float(front_surface_loss_h)
        self.front_surface_loss_thickness = float(front_surface_loss_thickness)
        self.front_surface_loss_radiation = bool(front_surface_loss_radiation)

    def get_tensor_map(self):
        def heat_flux(
            T_grad,
            T_old,
            dt,
            laser_center,
            laser_power,
            beam_radius,
            source_depth,
            laser_switch,
            active,
            rho,
            cp,
            conductivity,
            latent_cp,
            cooling_only,
            old_layer_cooling_h,
        ):
            return conductivity[0] * T_grad

        return heat_flux

    def get_mass_map(self):
        def heat_capacity(
            T,
            x,
            T_old,
            dt,
            laser_center,
            laser_power,
            beam_radius,
            source_depth,
            laser_switch,
            active,
            rho,
            cp,
            conductivity,
            latent_cp,
            cooling_only,
            old_layer_cooling_h,
        ):
            r0 = x[self.plane_axis0_id] - laser_center[self.plane_axis0_id]
            r1 = x[self.plane_axis1_id] - laser_center[self.plane_axis1_id]
            r2 = r0**2 + r1**2
            depth = self.build_sign * (laser_center[self.build_axis_id] - x[self.build_axis_id])
            q_depth = np.where(depth >= 0.0, np.exp(-depth / source_depth[0]), 0.0)
            # The in-plane Gaussian integrates to pi*r_b^2/2 and the one-sided
            # exponential depth decay integrates to source_depth. The factor 2
            # therefore makes the volume integral equal the absorbed laser power.
            q_vol = (
                2.0
                * laser_power[0]
                / (np.pi * beam_radius[0] ** 2 * source_depth[0])
                * np.exp(-2.0 * r2 / beam_radius[0] ** 2)
                * q_depth
                * laser_switch[0]
                * active[0]
            )

            if self.front_surface_loss_h > 0.0 and self.front_surface_loss_thickness > 0.0:
                front_band = np.where(
                    depth >= 0.0,
                    np.exp(-(depth / self.front_surface_loss_thickness) ** 2),
                    0.0,
                ) * active[0]
                q_front_loss = (
                    self.front_surface_loss_h
                    / self.front_surface_loss_thickness
                    * (T_old[0] - self.ambient)
                    * front_band
                )
                if self.front_surface_loss_radiation:
                    q_front_loss = q_front_loss + (
                        self.emissivity
                        * self.stefan_boltzmann
                        / self.front_surface_loss_thickness
                        * (T_old[0] ** 4 - self.ambient**4)
                        * front_band
                    )
            else:
                q_front_loss = 0.0

            # Printed layers below the moving thermal window are optionally
            # cooled by a volumetric sink while their conductivity is reduced
            # in thermal_material_quads(). old_layer_cooling_h has units
            # W/(m^3*K). This term is explicit in T_old for robustness.
            q_old_layer_loss = old_layer_cooling_h[0] * cooling_only[0] * (T_old[0] - self.ambient)

            cp_eff = cp[0] + latent_cp[0]
            return np.array([rho[0] * cp_eff * (T[0] - T_old[0]) / dt[0] - q_vol + q_front_loss + q_old_layer_loss])

        return heat_capacity

    def get_surface_maps(self):
        # face_active masks the flux to faces owned by printed/real material.
        # Void (not yet spread) cells are not physical surfaces; applying
        # convection/radiation to their near-singular equations produces
        # unphysical temperatures. In legacy 'box' mode the mask is all-ones,
        # which is numerically identical to the historical behavior.
        def surface_flux(T, _point, face_active):
            q_conv = self.convection_h * (self.ambient - T[0])
            q_rad = self.emissivity * self.stefan_boltzmann * (self.ambient**4 - T[0] ** 4)
            return -np.array([(q_conv + q_rad) * face_active[0]])

        return [surface_flux for _ in range(self.num_surface_maps)]

    def set_params(self, params):
        (
            T_old,
            dt,
            laser_center,
            effective_laser_power,
            beam_radius,
            source_depth,
            laser_switch,
            active_quad,
            rho_quad,
            cp_quad,
            conductivity_quad,
            latent_cp_quad,
            cooling_only_quad,
            old_layer_cooling_h,
            surface_mask_quad,
        ) = params

        num_cells = self.fes[0].num_cells
        num_quads = self.fes[0].num_quads
        dt_quad = dt * np.ones((num_cells, num_quads, 1))
        laser_center_quad = laser_center[None, None, :] * np.ones((num_cells, num_quads))[:, :, None]
        laser_power_quad = effective_laser_power * np.ones((num_cells, num_quads, 1))
        beam_radius_quad = beam_radius * np.ones((num_cells, num_quads, 1))
        source_depth_quad = source_depth * np.ones((num_cells, num_quads, 1))
        laser_switch_quad = laser_switch * np.ones((num_cells, num_quads, 1))
        old_layer_cooling_h_quad = old_layer_cooling_h * np.ones((num_cells, num_quads, 1))

        self.internal_vars = [
            self.fes[0].convert_from_dof_to_quad(T_old),
            dt_quad,
            laser_center_quad,
            laser_power_quad,
            beam_radius_quad,
            source_depth_quad,
            laser_switch_quad,
            active_quad,
            rho_quad,
            cp_quad,
            conductivity_quad,
            latent_cp_quad,
            cooling_only_quad,
            old_layer_cooling_h_quad,
        ]
        # Per-face activity flags for the surface flux maps. surface_mask_quad
        # is a (num_cells, num_quads, 1) printed indicator (or all-ones in
        # legacy mode); each boundary face inherits the flag of its owner cell.
        num_face_quads = self.fes[0].num_face_quads
        cell_mask = surface_mask_quad[:, 0, 0]
        self.internal_vars_surfaces = []
        for boundary_inds in self.boundary_inds_list:
            if boundary_inds.shape[0] == 0:
                self.internal_vars_surfaces.append([])
                continue
            face_flags = cell_mask[boundary_inds[:, 0]]
            face_active = face_flags[:, None, None] * np.ones((1, num_face_quads, 1))
            self.internal_vars_surfaces.append([face_active])


class ThermoMechanical(Problem):
    def custom_init(self, mechanics_model, yield_saturation=None, foundation_stiffness=0.0,
                    powder_foundation_stiffness=0.0, powder_plane_axes=()):
        self.mechanics_model = mechanics_model
        # Cap on the hardened yield stress (~UTS). Linear isotropic hardening
        # extrapolated past its ~10% strain validity produced 2 GPa fictitious
        # von Mises at the bottom-clamp singularity; saturation bounds it.
        self.yield_saturation = yield_saturation
        # Elastic-foundation stiffness (Pa/m) for the bottom surface springs.
        # 0 disables the springs (rigid Dirichlet clamp is used instead).
        self.foundation_stiffness = float(foundation_stiffness)
        # Lateral powder-bed support (Winkler springs, Pa/m) on the exterior
        # side faces above the base. Powder resists horizontal motion of the
        # printed layers but offers essentially no shear resistance along the
        # build direction, so the springs act only on the plane-axis
        # components (powder_plane_axes). Faces are masked per step by the
        # printed state of their owner cell (set_powder_surface_mask); the
        # release problem is constructed without these surfaces, so release
        # keeps its de-powdered meaning.
        self.powder_foundation_stiffness = float(powder_foundation_stiffness or 0.0)
        if self.powder_foundation_stiffness > 0.0:
            axis_mask = onp.zeros((3,))
            axis_mask[list(powder_plane_axes)] = 1.0
            self.powder_axis_mask = np.asarray(axis_mask)
            # The powder side surface is always the LAST location fn (main()
            # appends it after the optional bottom foundation surface).
            self.powder_boundary_index = len(self.boundary_inds_list) - 1
            # No support until the first activation update of the run.
            self.set_powder_surface_mask(
                onp.zeros(len(self.fes[0].cells), dtype=bool)
            )
        self.internal_vars = [
            np.zeros((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
            np.zeros((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
            np.ones((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
            np.ones((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
            np.ones((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
            np.ones((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
            np.ones((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
            np.zeros((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
            np.zeros((len(self.fes[0].cells), self.fes[0].num_quads, 1)),
        ]

    def stress_fn(self, u_grad, T, dT, active_factor, young, alpha, poisson, yield_stress, hardening, eqp_old):
        eps = 0.5 * (u_grad + u_grad.T)
        nu = np.clip(poisson[0], -0.49, 0.49)
        E = young[0]
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        thermal_eps = alpha[0] * dT[0] * np.eye(self.dim)
        elastic_eps = eps - thermal_eps
        sigma_trial = lmbda * np.trace(elastic_eps) * np.eye(self.dim) + 2.0 * mu * elastic_eps

        if self.mechanics_model == "j2_plastic":
            # Radial return with linear isotropic hardening. Identical to the
            # legacy min(1, yield/seq) clip when hardening == 0, but with
            # hardening > 0 the yield surface expands consistently WITHIN the
            # Newton solve, giving a positive-definite consistent tangent
            # (the pure clip left the plastic direction with zero stiffness,
            # which made anchor-only release solves of fully yielded material
            # ill-posed). delta_eqp matches compute_eqp_update().
            hydro = np.trace(sigma_trial) / 3.0 * np.eye(self.dim)
            dev = sigma_trial - hydro
            seq = np.sqrt(1.5 * np.sum(dev * dev) + 1e-30)
            hardened_yield = yield_stress[0] + hardening[0] * eqp_old[0]
            if self.yield_saturation is not None and self.yield_saturation > 0.0:
                sat = self.yield_saturation
                current_yield = np.maximum(np.minimum(hardened_yield, sat), 1e-12)
                hardening_eff = np.where(hardened_yield < sat, hardening[0], 0.0)
            else:
                current_yield = np.maximum(hardened_yield, 1e-12)
                hardening_eff = hardening[0]
            delta_eqp = np.maximum(seq - current_yield, 0.0) / (3.0 * mu + hardening_eff + 1e-12)
            scale = 1.0 - 3.0 * mu * delta_eqp / seq
            sigma = hydro + scale * dev
        else:
            sigma = sigma_trial

        return active_factor[0] * sigma

    def get_tensor_map(self):
        return self.stress_fn

    def get_surface_maps(self):
        # Elastic foundation (Winkler springs) on the base surface: the build
        # plate has finite compliance; a rigid clamp concentrates fictitious
        # stress at the clamp edge. Traction on the body is t = -k_s * u; with
        # the same weak-form sign convention as the thermal surface flux, the
        # kernel returns +k_s * u. Only used when location_fns select the
        # bottom faces (bottom-mechanics-bc elastic).
        #
        # Map order must match the location_fns order built in main():
        # [bottom (if elastic), powder side (if enabled)].
        maps = []

        if self.foundation_stiffness > 0.0:
            def foundation_traction(u, _point):
                return self.foundation_stiffness * u

            maps.append(foundation_traction)

        if getattr(self, "powder_foundation_stiffness", 0.0) > 0.0:
            def powder_traction(u, _point, face_powder):
                # Horizontal-only Winkler support from the surrounding powder
                # bed; face_powder gates the springs to printed material.
                return (
                    self.powder_foundation_stiffness
                    * face_powder[0]
                    * self.powder_axis_mask
                    * u
                )

            maps.append(powder_traction)

        return maps

    def set_powder_surface_mask(self, printed_cell_mask):
        """Gate the lateral powder springs to faces of printed cells.

        Faces owned by unprinted (void/future) cells must not be anchored:
        their equations are near-singular (inactive_mechanics_factor) and a
        spring there would both pollute the solve and apply support to
        material that does not exist yet. Call once per mechanics solve with
        the current printed-cell mask; the release problem never has this
        surface, so powder support vanishes on release.
        """
        if getattr(self, "powder_foundation_stiffness", 0.0) <= 0.0:
            return
        boundary_inds = self.boundary_inds_list[self.powder_boundary_index]
        if boundary_inds.shape[0] == 0:
            self.internal_vars_surfaces[self.powder_boundary_index] = []
            return
        num_face_quads = self.fes[0].num_face_quads
        face_flags = onp.asarray(printed_cell_mask, dtype=onp.float64)[
            boundary_inds[:, 0]
        ]
        face_powder = np.asarray(
            face_flags[:, None, None] * onp.ones((1, num_face_quads, 1))
        )
        self.internal_vars_surfaces[self.powder_boundary_index] = [face_powder]

    def set_params(self, params):
        self.internal_vars = params

    def compute_cell_stress(self, sol, params):
        T_quad, dT_quad, active_factor_quad, young_quad, alpha_quad, poisson_quad, yield_quad, hardening_quad, eqp_old_quad = params
        u_grads = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)
        sigmas = jax.vmap(jax.vmap(self.stress_fn))(
            u_grads,
            T_quad,
            dT_quad,
            active_factor_quad,
            young_quad,
            alpha_quad,
            poisson_quad,
            yield_quad,
            hardening_quad,
            eqp_old_quad,
        )
        return {
            "stress_quad": sigmas,
            "vm_quad": von_mises_from_stress(sigmas),
        }

    def compute_eqp_update(self, sol, params):
        if self.mechanics_model != "j2_plastic":
            return params[-1]

        T_quad, dT_quad, active_factor_quad, young_quad, alpha_quad, poisson_quad, yield_quad, hardening_quad, eqp_old_quad = params
        u_grads = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)

        def one_quad(u_grad, T, dT, active_factor, young, alpha, poisson, yield_stress, hardening, eqp_old):
            eps = 0.5 * (u_grad + u_grad.T)
            nu = np.clip(poisson[0], -0.49, 0.49)
            E = young[0]
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            thermal_eps = alpha[0] * dT[0] * np.eye(self.dim)
            elastic_eps = eps - thermal_eps
            sigma_trial = lmbda * np.trace(elastic_eps) * np.eye(self.dim) + 2.0 * mu * elastic_eps
            hydro = np.trace(sigma_trial) / 3.0 * np.eye(self.dim)
            dev = sigma_trial - hydro
            seq = np.sqrt(1.5 * np.sum(dev * dev) + 1e-30)
            hardened_yield = yield_stress[0] + hardening[0] * eqp_old[0]
            if self.yield_saturation is not None and self.yield_saturation > 0.0:
                sat = self.yield_saturation
                current_yield = np.maximum(np.minimum(hardened_yield, sat), 1e-12)
                hardening_eff = np.where(hardened_yield < sat, hardening[0], 0.0)
            else:
                current_yield = np.maximum(hardened_yield, 1e-12)
                hardening_eff = hardening[0]
            delta_eqp = np.maximum(seq - current_yield, 0.0) / (3.0 * mu + hardening_eff + 1e-12)
            active = np.where(active_factor[0] > 0.5, 1.0, 0.0)
            return np.array([eqp_old[0] + active * delta_eqp])

        return jax.vmap(jax.vmap(one_quad))(
            u_grads,
            T_quad,
            dT_quad,
            active_factor_quad,
            young_quad,
            alpha_quad,
            poisson_quad,
            yield_quad,
            hardening_quad,
            eqp_old_quad,
        )


def parse_scalar(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    if text.lower() in ("true", "yes", "on"):
        return True
    if text.lower() in ("false", "no", "off"):
        return False
    if text.lower() in ("none", "null"):
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def read_config(path):
    if path is None:
        return {}
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("--config JSON must contain an object")
        return data

    data = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Only flat YAML key: value entries are supported, got: {line}")
        key, value = stripped.split(":", 1)
        data[key.strip().replace("-", "_")] = parse_scalar(value)
    return data


def cfg(config, key, default):
    return config.get(key, default)


def build_parser(config=None):
    config = config or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--inp", default=cfg(config, "inp", "/home/user/work/159/schema/0119_c3d4_only.inp"))
    parser.add_argument("--max-cells", type=int, default=cfg(config, "max_cells", 0))
    parser.add_argument("--mesh-length-scale", type=float, default=cfg(config, "mesh_length_scale", 1.0))
    parser.add_argument("--path-length-scale", type=float, default=cfg(config, "path_length_scale", None), help="Scale applied to x/y/z coordinates in --path-file. Defaults to --mesh-length-scale.")

    parser.add_argument("--layers", type=int, default=cfg(config, "layers", 50))
    parser.add_argument("--steps", type=int, default=cfg(config, "steps", None), help="Backward-compatible alias for --layers.")
    parser.add_argument("--layer-thickness", type=float, default=cfg(config, "layer_thickness", None), help="Physical layer thickness in scaled mesh units. If provided, it overrides --layers.")
    parser.add_argument("--max-print-layers", type=int, default=cfg(config, "max_print_layers", None), help="Limit printed layers after physical layer-thickness expansion. Useful for debugging full-size builds with realistic layer thickness.")
    parser.add_argument("--active-window-below-layers", type=int, default=cfg(config, "active_window_below_layers", 0), help="Moving thermal active window below current layer. 0 keeps cumulative activation. If 10, current layer L uses layers max(1,L-10)..L as the thermal window.")
    parser.add_argument("--old-layer-thermal-factor", type=float, default=cfg(config, "old_layer_thermal_factor", 1e-6), help="Conductivity multiplier for printed layers that fall below the moving thermal window.")
    parser.add_argument("--old-layer-cooling-h", type=float, default=cfg(config, "old_layer_cooling_h", 0.0), help="Optional volumetric cooling coefficient W/(m^3*K) for printed layers below the moving thermal window. 0 disables this sink.")
    parser.add_argument("--scan-steps-per-layer", type=int, default=cfg(config, "scan_steps_per_layer", 20))
    parser.add_argument("--hatch-lines-per-layer", type=int, default=cfg(config, "hatch_lines_per_layer", 1))
    parser.add_argument("--hatch-spacing", type=float, default=cfg(config, "hatch_spacing", None), help="Physical hatch spacing in the build plane. If provided, it overrides --hatch-lines-per-layer.")
    parser.add_argument("--dt", type=float, default=cfg(config, "dt", 1e-4))

    parser.add_argument("--ambient", type=float, default=cfg(config, "ambient", 300.0))
    parser.add_argument("--preheat-temperature", type=float, default=cfg(config, "preheat_temperature", None))
    parser.add_argument("--bottom-temperature", type=float, default=cfg(config, "bottom_temperature", None), help="Fixed bottom temperature. Defaults to preheat temperature when provided, otherwise ambient.")
    parser.add_argument("--rho", type=float, default=cfg(config, "rho", 7800.0))
    parser.add_argument("--cp", type=float, default=cfg(config, "cp", 500.0))
    parser.add_argument("--conductivity", type=float, default=cfg(config, "conductivity", 20.0))
    parser.add_argument("--rho-solid", type=float, default=cfg(config, "rho_solid", None))
    parser.add_argument("--cp-solid", type=float, default=cfg(config, "cp_solid", None))
    parser.add_argument("--conductivity-solid", type=float, default=cfg(config, "conductivity_solid", None))
    parser.add_argument("--rho-powder", type=float, default=cfg(config, "rho_powder", 3900.0))
    parser.add_argument("--cp-powder", type=float, default=cfg(config, "cp_powder", 500.0))
    parser.add_argument("--conductivity-powder", type=float, default=cfg(config, "conductivity_powder", 1.0))
    parser.add_argument("--rho-liquid", type=float, default=cfg(config, "rho_liquid", None), help="Liquid density. Defaults to solid density when omitted.")
    parser.add_argument("--cp-liquid", type=float, default=cfg(config, "cp_liquid", None), help="Liquid heat capacity. Defaults to solid heat capacity when omitted.")
    parser.add_argument("--conductivity-liquid", type=float, default=cfg(config, "conductivity_liquid", None), help="Liquid thermal conductivity. Defaults to solid conductivity when omitted.")
    parser.add_argument("--powder-mode", choices=("powder", "void"), default=cfg(config, "powder_mode", "powder"))
    parser.add_argument("--layer-activation-mode", choices=("front", "layer_on_scan"), default=cfg(config, "layer_activation_mode", "layer_on_scan"), help="Layer activation model. 'front' keeps the old front-coordinate activation; 'layer_on_scan' activates the whole current layer when laser scanning starts, mimicking recoating/powder spreading.")
    parser.add_argument("--future-layer-mode", choices=("void", "powder"), default=cfg(config, "future_layer_mode", "void"), help="Material treatment for future, not-yet-spread layers when --layer-activation-mode layer_on_scan is used. Use 'void' to make future layers inactive before spreading.")
    parser.add_argument("--layer-activation-geometry", choices=("centroid", "intersection"), default=cfg(config, "layer_activation_geometry", "intersection"), help="How cells are assigned to printed layers. 'centroid' uses cell centroid layer id; 'intersection' activates a cell if its vertex interval intersects the layer band. Use intersection for coarse tetra meshes and macro-layer runs.")
    parser.add_argument("--inactive-thermal-factor", type=float, default=cfg(config, "inactive_thermal_factor", 1e-6))
    parser.add_argument("--inactive-mass-factor", type=float, default=cfg(config, "inactive_mass_factor", None),
                        help="Density scaling for inactive/void cells. Legacy behavior (None) reuses "
                             "--inactive-thermal-factor, which scales k and rho by the SAME factor and therefore leaves "
                             "the void diffusivity k/(rho*cp) at solid-like values: temperature 'ghost-diffuses' 5-10 mm "
                             "into un-spread layers above the laser. Set 1.0 to keep full thermal mass in void (only k "
                             "is reduced), cutting void diffusivity by the k factor and anchoring the otherwise "
                             "near-singular void equations.")
    parser.add_argument("--inactive-mechanics-factor", type=float, default=cfg(config, "inactive_mechanics_factor", 1e-9))

    parser.add_argument("--k-table-solid", default=cfg(config, "k_table_solid", None))
    parser.add_argument("--cp-table-solid", default=cfg(config, "cp_table_solid", None))
    parser.add_argument("--k-table-powder", default=cfg(config, "k_table_powder", None))
    parser.add_argument("--cp-table-powder", default=cfg(config, "cp_table_powder", None))
    parser.add_argument("--k-table-liquid", default=cfg(config, "k_table_liquid", None))
    parser.add_argument("--cp-table-liquid", default=cfg(config, "cp_table_liquid", None))
    parser.add_argument("--solidus-temperature", type=float, default=cfg(config, "solidus_temperature", 0.0))
    parser.add_argument("--liquidus-temperature", type=float, default=cfg(config, "liquidus_temperature", 0.0))
    parser.add_argument("--latent-heat", type=float, default=cfg(config, "latent_heat", 0.0))

    parser.add_argument("--convection-h", type=float, default=cfg(config, "convection_h", 10.0))
    parser.add_argument("--emissivity", type=float, default=cfg(config, "emissivity", 0.0))
    parser.add_argument("--stefan-boltzmann", type=float, default=cfg(config, "stefan_boltzmann", 5.670374419e-8))
    parser.add_argument("--bottom-thermal-bc", choices=("fixed", "convection"), default=cfg(config, "bottom_thermal_bc", "fixed"))
    parser.add_argument("--surface-selection", choices=("box", "exterior"), default=cfg(config, "surface_selection", "box"),
                        help="How convection/radiation faces are selected. 'box' keeps the legacy bounding-box plane selectors "
                             "(only faces exactly on the box planes; zero faces for curved parts). 'exterior' applies the "
                             "surface losses to every mesh-exterior face above the base plane, matching a real part surrounded "
                             "by gas/powder.")
    parser.add_argument("--boundary-tol", type=float, default=cfg(config, "boundary_tol", None),
                        help="Absolute tolerance (mesh length units) for base/exposed/wall plane node selection. Defaults to "
                             "the legacy 1e-8 * bounding-box span, which can miss nodes on real CAD meshes whose base face "
                             "has sub-mm jitter.")
    parser.add_argument("--surface-active-mask", dest="surface_active_mask", action="store_true",
                        default=cfg(config, "surface_active_mask", None),
                        help="Restrict convection/radiation to faces owned by printed material. Void cells are not physical "
                             "surfaces and their near-singular equations produce unphysical temperatures under surface flux. "
                             "Defaults to on for --surface-selection exterior, off for legacy box mode.")
    parser.add_argument("--no-surface-active-mask", dest="surface_active_mask", action="store_false")
    parser.add_argument("--stress-relaxation-temperature", type=float,
                        default=cfg(config, "stress_relaxation_temperature", None),
                        help="Stress-free reference temperature written when material solidifies (macro calibration knob; "
                             "Ti64 typically 1073-1173 K). Without it, T_ref is the local temperature at solidification, "
                             "which in consolidation-on-activation mode equals the powder entry temperature and inverts the "
                             "residual stress sign.")
    parser.add_argument("--quadrature-order", type=int, default=cfg(config, "quadrature_order", None),
                        help="Quadrature order for both thermal and mechanical problems (they must match: phase/material "
                             "state arrays are shared per quadrature point). TET4 defaults to the legacy single-point rule, "
                             "whose rank-1 mass matrix produces large spurious temperature oscillations (observed +-1900K) "
                             "in low-conductivity powder at small Fourier numbers. Use 2 (4-point rule) for a full-rank "
                             "transient term.")
    parser.add_argument("--front-surface-loss-h", type=float, default=cfg(config, "front_surface_loss_h", 0.0), help="Optional volumetric approximation of convection on the moving build front. 0 disables it.")
    parser.add_argument("--front-surface-loss-thickness", type=float, default=cfg(config, "front_surface_loss_thickness", 0.0), help="Thickness for moving-front loss approximation. Defaults to source_depth when front loss is enabled.")
    parser.add_argument("--front-surface-loss-radiation", dest="front_surface_loss_radiation", action="store_true", default=cfg(config, "front_surface_loss_radiation", False))
    parser.add_argument("--no-front-surface-loss-radiation", dest="front_surface_loss_radiation", action="store_false")

    parser.add_argument("--young", type=float, default=cfg(config, "young", 2.0e11))
    parser.add_argument("--poisson", type=float, default=cfg(config, "poisson", 0.3))
    parser.add_argument("--alpha", type=float, default=cfg(config, "alpha", 1.2e-5))
    parser.add_argument("--E-table", dest="E_table", default=cfg(config, "E_table", None))
    parser.add_argument("--alpha-table", default=cfg(config, "alpha_table", None))
    parser.add_argument("--poisson-table", default=cfg(config, "poisson_table", None))
    parser.add_argument("--yield-table", default=cfg(config, "yield_table", None))
    parser.add_argument("--hardening-table", default=cfg(config, "hardening_table", None))
    parser.add_argument("--mechanics-model", choices=("linear_elastic", "j2_plastic"), default=cfg(config, "mechanics_model", "linear_elastic"))
    parser.add_argument("--yield-saturation-stress", type=float, default=cfg(config, "yield_saturation_stress", None),
                        help="Cap on the hardened yield stress (Pa), ~UTS (Ti64: ~1.15e9). Linear isotropic hardening "
                             "extrapolated past its ~10%% strain validity produced ~2 GPa fictitious von Mises at the "
                             "bottom-clamp region; the cap saturates hardening there. None keeps unbounded legacy hardening.")
    parser.add_argument("--bottom-mechanics-bc", choices=("fixed", "elastic"), default=cfg(config, "bottom_mechanics_bc", "fixed"),
                        help="'fixed' rigidly clamps the base nodes (legacy; models an infinitely stiff build plate and "
                             "concentrates fictitious stress at the clamp edge). 'elastic' replaces the clamp with a "
                             "Winkler elastic foundation on the base faces, giving the plate finite compliance.")
    parser.add_argument("--bottom-foundation-stiffness", type=float, default=cfg(config, "bottom_foundation_stiffness", 1.0e12),
                        help="Foundation modulus k_s (Pa/m) for --bottom-mechanics-bc elastic. Calibration knob for the "
                             "build-plate compliance; 1e12 approximates a ~25 mm steel plate, larger values approach the "
                             "rigid clamp.")
    parser.add_argument("--powder-mechanics-bc", choices=("none", "elastic"), default=cfg(config, "powder_mechanics_bc", "none"),
                        help="'none' keeps free lateral surfaces (legacy: the surrounding powder bed constrains nothing). "
                             "'elastic' adds horizontal-only Winkler springs on the exterior side faces of printed "
                             "material, modeling lateral support from the unmelted powder bed; springs follow the "
                             "printed state per step and are absent from the release solve (de-powdering). "
                             "Requires --surface-selection exterior.")
    parser.add_argument("--powder-foundation-stiffness", type=float, default=cfg(config, "powder_foundation_stiffness", 1.0e9),
                        help="Foundation modulus k_p (Pa/m) for --powder-mechanics-bc elastic. Calibration knob: "
                             "k_p ~ E_powder / L_embed with loose Ti64 powder E ~ 1-100 MPa and mm-scale embedment "
                             "gives ~1e8-1e11; default 1e9 is a soft support ~3 orders below the build plate.")
    parser.add_argument("--mushy-mechanics-factor", type=float, default=cfg(config, "mushy_mechanics_factor", 1e-2), help="Stress/stiffness scaling for mushy-zone material.")
    parser.add_argument("--liquid-mechanics-factor", type=float, default=cfg(config, "liquid_mechanics_factor", 1e-4), help="Stress/stiffness scaling for liquid material.")
    parser.add_argument("--reset-plastic-on-melt", dest="reset_plastic_on_melt", action="store_true", default=cfg(config, "reset_plastic_on_melt", True))
    parser.add_argument("--no-reset-plastic-on-melt", dest="reset_plastic_on_melt", action="store_false")

    parser.add_argument("--laser-power", type=float, default=cfg(config, "laser_power", 1.0))
    parser.add_argument("--absorptivity", type=float, default=cfg(config, "absorptivity", 0.35))
    parser.add_argument("--beam-radius", type=float, default=cfg(config, "beam_radius", 1.0e-4))
    parser.add_argument("--source-depth", type=float, default=cfg(config, "source_depth", 6.0e-5))

    parser.add_argument("--build-axis", choices=("x", "y", "z"), default=cfg(config, "build_axis", "x"))
    parser.add_argument("--base-side", choices=("min", "max"), default=cfg(config, "base_side", "min"))
    parser.add_argument("--scan-axis", choices=("auto", "x", "y", "z"), default=cfg(config, "scan_axis", "auto"))
    parser.add_argument("--scan-pattern", choices=("raster",), default=cfg(config, "scan_pattern", "raster"))
    parser.add_argument("--scan-rotation-per-layer", type=float, default=cfg(config, "scan_rotation_per_layer", 0.0), help="Rotation angle in degrees applied inside the build plane for each new layer.")
    parser.add_argument("--scan-start", type=float, default=cfg(config, "scan_start", None))
    parser.add_argument("--scan-end", type=float, default=cfg(config, "scan_end", None))
    parser.add_argument("--scan-start-frac", type=float, default=cfg(config, "scan_start_frac", 0.05))
    parser.add_argument("--scan-end-frac", type=float, default=cfg(config, "scan_end_frac", 0.95))
    parser.add_argument("--hatch-start", type=float, default=cfg(config, "hatch_start", None))
    parser.add_argument("--hatch-end", type=float, default=cfg(config, "hatch_end", None))
    parser.add_argument("--hatch-start-frac", type=float, default=cfg(config, "hatch_start_frac", 0.05))
    parser.add_argument("--hatch-end-frac", type=float, default=cfg(config, "hatch_end_frac", 0.95))
    parser.add_argument("--hatch-fixed", type=float, default=cfg(config, "hatch_fixed", None))
    parser.add_argument("--serpentine", dest="serpentine", action="store_true", default=cfg(config, "serpentine", True))
    parser.add_argument("--no-serpentine", dest="serpentine", action="store_false")
    parser.add_argument("--scan-speed", type=float, default=cfg(config, "scan_speed", 0.0))
    parser.add_argument("--auto-scan-steps-from-speed", dest="auto_scan_steps_from_speed", action="store_true", default=cfg(config, "auto_scan_steps_from_speed", False))
    parser.add_argument("--no-auto-scan-steps-from-speed", dest="auto_scan_steps_from_speed", action="store_false")
    parser.add_argument("--dwell-steps-between-layers", type=int, default=cfg(config, "dwell_steps_between_layers", 0))
    parser.add_argument("--dwell-steps-between-hatches", type=int, default=cfg(config, "dwell_steps_between_hatches", 0))
    parser.add_argument("--jump-speed", type=float, default=cfg(config, "jump_speed", 0.0), help="Laser-off travel speed between hatch lines. 0 disables generated jump states.")
    parser.add_argument("--recoat-time", type=float, default=cfg(config, "recoat_time", 0.0))
    parser.add_argument("--recoat-steps", type=int, default=cfg(config, "recoat_steps", 10),
                        help="Number of implicit time steps used to span each recoat interval (dt = recoat_time / recoat_steps). "
                             "Applies to both the raster generator and layer transitions in --path-file runs.")
    parser.add_argument("--path-file", default=cfg(config, "path_file", None))
    parser.add_argument("--path-output", default=cfg(config, "path_output", "path_used.csv"), help="CSV file name/path for the actual sampled path. Set empty string to disable.")

    parser.add_argument("--substrate-thickness", type=float, default=cfg(config, "substrate_thickness", 0.0))
    parser.add_argument("--support-thickness", type=float, default=cfg(config, "support_thickness", 0.0))
    parser.add_argument("--cooling-steps", type=int, default=cfg(config, "cooling_steps", 0))
    parser.add_argument("--cooling-dt", type=float, default=cfg(config, "cooling_dt", None),
                        help="Time step for the trailing cooling states. Defaults to --dt; a larger value lets the part "
                             "actually cool to ambient before release instead of only spanning cooling_steps * dt seconds.")
    parser.add_argument("--reset-activation-temperature", dest="reset_activation_temperature", action="store_true",
                        default=cfg(config, "reset_activation_temperature", False),
                        help="Reset nodal temperatures of newly activated cells (nodes not shared with previously active "
                             "material) to the preheat/ambient temperature, modeling freshly spread powder.")
    parser.add_argument("--no-reset-activation-temperature", dest="reset_activation_temperature", action="store_false")
    parser.add_argument("--activation-reset-temperature", type=float,
                        default=cfg(config, "activation_reset_temperature", None),
                        help="Temperature assigned to newly activated nodes when --reset-activation-temperature is on. "
                             "Defaults to the preheat/ambient temperature. In consolidation-on-activation (route B) runs, "
                             "set it to the stress relaxation temperature: the fresh layer enters hot and stress-free and "
                             "builds stress gradually while cooling, instead of receiving an instantaneous GPa-level "
                             "thermal load step that stalls the mechanics Newton solve.")
    parser.add_argument("--release-after-cooling", dest="release_after_cooling", action="store_true", default=cfg(config, "release_after_cooling", False))
    parser.add_argument("--no-release-after-cooling", dest="release_after_cooling", action="store_false")
    parser.add_argument("--release-anchor-mode", choices=("rigid_body", "box"),
                        default=cfg(config, "release_anchor_mode", "rigid_body"),
                        help="rigid_body (default): free-free release with 3-point rigid-body anchors (full removal "
                             "from the plate). box: clamp all nodes inside --release-anchor-box (u=0), modeling a "
                             "partial EDM/saw cut that leaves a root attachment (Kaess 2023 cantilever semantics).")
    parser.add_argument("--release-anchor-box", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        default=cfg(config, "release_anchor_box", None),
                        help="Axis-aligned box (mesh coordinates) whose nodes stay clamped during a box-mode release.")
    parser.add_argument("--release-cut-box", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        default=cfg(config, "release_cut_box", None),
                        help="Cells whose centroid lies inside this box (mesh coordinates) are deactivated in the "
                             "release solve (stiffness AND locked-in stress scaled by the inactive factor) - the "
                             "equivalent of deleting sawed-off support elements (Kaess 2023 Fig 7 semantics).")
    parser.add_argument("--final-cooldown-temperature", type=float,
                        default=cfg(config, "final_cooldown_temperature", None),
                        help="Ramp the fixed bottom temperature linearly to this value (K) across the final cooling "
                             "steps, modeling a build-plate cooldown to room temperature before release "
                             "(Kaess 2023 style). Requires --bottom-thermal-bc fixed; default keeps the bottom "
                             "temperature constant.")

    parser.add_argument("--mechanics-every", type=int, default=cfg(config, "mechanics_every", 1))
    parser.add_argument("--mechanics-tol", type=float, default=cfg(config, "mechanics_tol", None),
                        help="Absolute Newton residual tolerance for mechanics solves (default keeps legacy 1e-9).")
    parser.add_argument("--mechanics-rel-tol", type=float, default=cfg(config, "mechanics_rel_tol", None),
                        help="Relative Newton residual tolerance for mechanics solves (default keeps legacy 1e-11; "
                             "1e-6 is plenty for engineering stress accuracy and dramatically faster for j2 states).")
    parser.add_argument("--mechanics-max-iter", type=int, default=cfg(config, "mechanics_max_iter", None),
                        help="Newton iteration cap for mechanics solves (solver default 100).")
    parser.add_argument("--mechanics-line-search", dest="mechanics_line_search", action="store_true",
                        default=cfg(config, "mechanics_line_search", False),
                        help="Enable Newton line search for mechanics solves; stabilizes j2 yield-surface states.")
    parser.add_argument("--no-mechanics-line-search", dest="mechanics_line_search", action="store_false")
    parser.add_argument("--mechanics-temperature-floor", type=float,
                        default=cfg(config, "mechanics_temperature_floor", None),
                        help="Clamp the temperature seen by the mechanics chain (thermal strain and "
                             "material tables) to at least this value in K. Guard against activation "
                             "undershoot artifacts (G1) feeding sub-physical temperatures into full-stiffness "
                             "solid; a mitigation knob, not a fix — the thermal field itself stays unclamped.")
    parser.add_argument("--thermal-mass-lumping", action="store_true",
                        default=cfg(config, "thermal_mass_lumping", False),
                        help="Evaluate the thermal transient/source (mass-map) terms with the TET4 "
                             "vertex quadrature rule instead of the interior Gauss points. For linear "
                             "tets this is exactly row-sum capacitance lumping (Abaqus first-order "
                             "heat-transfer element behavior): the capacitance matrix becomes diagonal, "
                             "restoring the discrete maximum principle that suppresses activation "
                             "undershoot; conduction is unaffected (constant gradients). Requires "
                             "--quadrature-order 2 (4 equal-weight points).")
    parser.add_argument("--mechanics-max-cuts", type=int,
                        default=cfg(config, "mechanics_max_cuts", 0),
                        help="Abaqus-style automatic increment cutback for mechanics solves: on Newton "
                             "failure retry the thermal-load increment in 2,4,...,2**N equal substeps "
                             "(temperature interpolated from the last accepted mechanics state). Substeps "
                             "are pure Newton continuation - the final substep solves the exact original "
                             "problem and no plastic state is committed in between. 0 disables (legacy).")
    parser.add_argument("--thermal-output-every", type=int, default=cfg(config, "thermal_output_every", 0))
    parser.add_argument("--mechanics-output-every", type=int, default=cfg(config, "mechanics_output_every", 1))
    parser.add_argument("--summary-every", type=int, default=cfg(config, "summary_every", 1))
    parser.add_argument("--calibration-dir", default=cfg(config, "calibration_dir", None))
    parser.add_argument("--output-dir", default=cfg(config, "output_dir", "/home/user/work/159/output/inp_thermal_stress_oneway_layers"))
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = read_config(config_args.config)
    parser = build_parser(config)
    args = parser.parse_args()
    args.config = config_args.config
    return args


def make_box_locations(points, build_axis="x", base_side="min", tol_ratio=1e-8, abs_tol=None):
    pmin = onp.min(points, axis=0)
    pmax = onp.max(points, axis=0)
    span = max(float(onp.max(pmax - pmin)), 1.0)
    # Real CAD meshes rarely have their base-face nodes exactly coplanar; the
    # legacy 1e-8 relative tolerance can select only a handful of nodes (8 of
    # ~1225 on the 0119 part). Pass abs_tol (--boundary-tol) to widen it.
    atol = float(abs_tol) if abs_tol is not None and abs_tol > 0.0 else tol_ratio * span
    build_axis_id = AXIS_TO_ID[build_axis]
    plane_axis_ids = tuple(i for i in range(3) if i != build_axis_id)
    if base_side == "min":
        base_coord = pmin[build_axis_id]
        exposed_coord = pmax[build_axis_id]
    else:
        base_coord = pmax[build_axis_id]
        exposed_coord = pmin[build_axis_id]

    def bottom(point):
        return np.isclose(point[build_axis_id], base_coord, rtol=0.0, atol=atol)

    def exposed(point):
        return np.isclose(point[build_axis_id], exposed_coord, rtol=0.0, atol=atol)

    def walls(point):
        a0, a1 = plane_axis_ids
        return (
            np.isclose(point[a0], pmin[a0], rtol=0.0, atol=atol)
            | np.isclose(point[a0], pmax[a0], rtol=0.0, atol=atol)
            | np.isclose(point[a1], pmin[a1], rtol=0.0, atol=atol)
            | np.isclose(point[a1], pmax[a1], rtol=0.0, atol=atol)
        )

    return pmin, pmax, bottom, exposed, walls, build_axis_id, plane_axis_ids, float(base_coord), float(exposed_coord)


def make_part_build_box(pmin, pmax, build_axis_id, base_side, substrate_thickness, support_thickness):
    """Return the build bounds used for part layers and raster paths.

    If substrate/support are included in the mesh, layer fronts should advance
    through the printed part only, not through the substrate/support thickness.
    Thickness values are assumed to be in the same internal units as the scaled
    mesh coordinates.
    """
    part_pmin = onp.array(pmin, dtype=onp.float64).copy()
    part_pmax = onp.array(pmax, dtype=onp.float64).copy()
    total_offset = max(float(substrate_thickness), 0.0) + max(float(support_thickness), 0.0)
    build_span = float(pmax[build_axis_id] - pmin[build_axis_id])
    if total_offset >= build_span and build_span > 0.0:
        raise ValueError("substrate_thickness + support_thickness must be smaller than the build-axis span")
    if base_side == "min":
        part_pmin[build_axis_id] = pmin[build_axis_id] + total_offset
    else:
        part_pmax[build_axis_id] = pmax[build_axis_id] - total_offset
    return part_pmin, part_pmax


def coord_from_frac(pmin, pmax, axis_id, frac):
    return float(pmin[axis_id] + frac * (pmax[axis_id] - pmin[axis_id]))


def resolve_axis_range(pmin, pmax, axis_id, start_value, end_value, start_frac, end_frac):
    start = float(start_value) if start_value is not None else coord_from_frac(pmin, pmax, axis_id, start_frac)
    end = float(end_value) if end_value is not None else coord_from_frac(pmin, pmax, axis_id, end_frac)
    return start, end


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

def load_property_tables(args):
    tables = {}
    for key, path in [
        ("k_solid", args.k_table_solid),
        ("cp_solid", args.cp_table_solid),
        ("k_powder", args.k_table_powder),
        ("cp_powder", args.cp_table_powder),
        ("k_liquid", args.k_table_liquid),
        ("cp_liquid", args.cp_table_liquid),
        ("E", args.E_table),
        ("alpha", args.alpha_table),
        ("poisson", args.poisson_table),
        ("yield", args.yield_table),
        ("hardening", args.hardening_table),
    ]:
        tables[key] = PropertyTable(path) if path else None
    return tables


def eval_property(T_quad, table, default):
    if table is None:
        return default * np.ones_like(T_quad)
    return table.eval(T_quad)


def compute_cell_temperature(T_nodes, cells):
    return onp.mean(onp.asarray(T_nodes)[cells, 0], axis=1)


def reset_new_cell_nodal_temperature(T_old, cells, newly_printed_cell, previous_active_cell, value):
    """Reset nodal temperatures of freshly activated cells to the powder value.

    Freshly spread powder enters the build at the preheat/ambient temperature,
    but inactive (void) DOFs drift freely because their thermal mass is scaled
    by inactive_thermal_factor. Only nodes NOT shared with previously active
    material are reset, so the interface to already-printed layers keeps its
    conducted temperature history.
    """
    cells_arr = onp.asarray(cells)
    new_nodes = onp.unique(cells_arr[newly_printed_cell].reshape(-1))
    if new_nodes.size == 0:
        return T_old
    if previous_active_cell.any():
        old_nodes = onp.unique(cells_arr[previous_active_cell].reshape(-1))
    else:
        old_nodes = onp.empty(0, dtype=new_nodes.dtype)
    reset_nodes = onp.setdiff1d(new_nodes, old_nodes, assume_unique=True)
    if reset_nodes.size == 0:
        return T_old
    return np.asarray(T_old).at[reset_nodes, :].set(value)


def make_quad_scalar(cell_values, num_quads):
    arr = np.asarray(cell_values)[:, None, None]
    return arr * np.ones((len(cell_values), num_quads, 1))


def classify_cells(points, cells, build_axis_id, build_sign, base_coord, args):
    centroids = onp.mean(points[cells], axis=1)
    cell_build_coord = centroids[:, build_axis_id]
    dist_from_base = build_sign * (cell_build_coord - base_coord)
    substrate = dist_from_base <= args.substrate_thickness if args.substrate_thickness > 0 else onp.zeros(len(cells), dtype=bool)
    support = (
        (dist_from_base > args.substrate_thickness)
        & (dist_from_base <= args.substrate_thickness + args.support_thickness)
        if args.support_thickness > 0
        else onp.zeros(len(cells), dtype=bool)
    )
    return centroids, cell_build_coord, substrate, support


def compute_layer_id(cell_build_coord, build_axis_id, pmin, pmax, args):
    build_min = float(pmin[build_axis_id])
    build_max = float(pmax[build_axis_id])
    if args.base_side == "min":
        frac = (cell_build_coord - build_min) / max(build_max - build_min, 1e-15)
    else:
        frac = (build_max - cell_build_coord) / max(build_max - build_min, 1e-15)
    return onp.clip(onp.ceil(frac * args.layers), 1, args.layers).astype(onp.int32)


def compute_active_cell(state, cell_build_coord, substrate_cell, support_cell, build_sign, args):
    tol = 1e-12 * max(float(onp.max(onp.abs(cell_build_coord))), 1.0)
    part_active = build_sign * (state.front_coord - cell_build_coord) >= -tol
    return substrate_cell | support_cell | part_active


def compute_physical_layer_id_cell(cell_build_coord, build_axis_id, part_pmin, part_pmax, build_sign, args):
    """Return physical layer id for each part cell.

    Unlike compute_layer_id(), this function is based on layer_thickness when
    available and does not clip to --max-print-layers. Cells beyond the printed
    test window therefore keep layer ids larger than the simulated layer count.
    Fixture cells are assigned outside this function.
    """
    if args.layer_thickness is None or args.layer_thickness <= 0.0:
        return compute_layer_id(cell_build_coord, build_axis_id, part_pmin, part_pmax, args)

    if args.base_side == "min":
        part_base_coord = float(part_pmin[build_axis_id])
    else:
        part_base_coord = float(part_pmax[build_axis_id])

    dist_from_part_base = build_sign * (cell_build_coord - part_base_coord)
    layer_id = onp.ceil(dist_from_part_base / float(args.layer_thickness)).astype(onp.int32)
    return onp.maximum(layer_id, 1).astype(onp.int32)



def compute_cell_build_interval(points, cells, build_axis_id, build_sign, part_base_coord):
    """Return each tetra cell's build-direction distance interval from part base.

    The previous layer assignment used the cell centroid only. That can miss
    thin layers when no centroid lies inside the layer band. This interval
    representation marks a cell as intersecting a layer if any part of its
    vertex span crosses that layer band.
    """
    cell_axis = onp.asarray(points[cells, build_axis_id], dtype=onp.float64)
    cell_dist = float(build_sign) * (cell_axis - float(part_base_coord))
    cell_d_min = onp.min(cell_dist, axis=1)
    cell_d_max = onp.max(cell_dist, axis=1)
    return cell_d_min, cell_d_max


def cells_intersect_distance_band(cell_d_min, cell_d_max, lower, upper, tol=1e-12):
    """Boolean mask: cell build-axis interval intersects [lower, upper]."""
    return (cell_d_max >= float(lower) - tol) & (cell_d_min <= float(upper) + tol)


def compute_nominal_layer_id_from_interval(cell_d_min, cell_d_max, args):
    """Assign a representative layer id for visualization under interval activation.

    This is not used for activation. It is the first physical layer intersected
    by the cell, useful for ParaView coloring when tetrahedra span multiple
    thin layers.
    """
    if args.layer_thickness is None or args.layer_thickness <= 0.0:
        return onp.ones_like(cell_d_min, dtype=onp.int32)
    lt = float(args.layer_thickness)
    layer_id = onp.floor(onp.maximum(cell_d_min, 0.0) / lt).astype(onp.int32) + 1
    return onp.maximum(layer_id, 1).astype(onp.int32)


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


def initial_phase_cell(active_cell, substrate_cell, support_cell, args):
    if getattr(args, "layer_activation_mode", "front") == "layer_on_scan" and getattr(args, "future_layer_mode", "void") == "void":
        # Future layers have not been recoated yet, so they are void/inactive.
        # They will become powder when their layer is activated at laser scan start.
        inactive_code = STATE_VOID
    else:
        inactive_code = STATE_POWDER if args.powder_mode == "powder" else STATE_VOID
    phase = inactive_code * onp.ones(len(active_cell), dtype=onp.float64)
    # Active printed cells still start as powder; they only become solid after a
    # melt/solidification event. Fixture regions are treated as solid supports.
    phase[active_cell & (~substrate_cell) & (~support_cell)] = STATE_POWDER
    phase[substrate_cell] = STATE_SUBSTRATE
    phase[support_cell] = STATE_SUPPORT
    return phase


def phase_cell_from_quad(phase_quad):
    return onp.max(onp.asarray(phase_quad)[:, :, 0], axis=1)


def material_cell_state(active_cell, substrate_cell, support_cell, args, cell_temperature=None, phase_cell=None):
    # State encoding for ParaView and CSV post-processing:
    #   0 void/inactive weak material
    #   1 powder
    #   2 solid
    #   3 mushy zone
    #   4 liquid
    #   5 substrate
    #   6 support
    if phase_cell is not None:
        state = onp.asarray(phase_cell, dtype=onp.float64).copy()
    else:
        inactive_code = STATE_POWDER if args.powder_mode == "powder" else STATE_VOID
        state = inactive_code * onp.ones(len(active_cell), dtype=onp.float64)
        state[active_cell] = STATE_SOLID
        if cell_temperature is not None and args.liquidus_temperature > args.solidus_temperature:
            active_non_fixture = active_cell & (~substrate_cell) & (~support_cell)
            mushy = active_non_fixture & (cell_temperature >= args.solidus_temperature) & (cell_temperature < args.liquidus_temperature)
            liquid = active_non_fixture & (cell_temperature >= args.liquidus_temperature)
            state[mushy] = STATE_MUSHY
            state[liquid] = STATE_LIQUID
    state[substrate_cell] = STATE_SUBSTRATE
    state[support_cell] = STATE_SUPPORT
    return state


def update_phase_reference_and_eqp(T_quad, active_quad, phase_quad, T_ref_quad, eqp_quad, args):
    """Update quadrature-point material phase and stress-free reference temperature.

    The important modeling change is that T_ref is written when material
    solidifies from liquid/mushy state, not when a layer is merely activated.
    Re-melting moves the point back to a stress-free liquid/mushy state; when it
    solidifies again the reference temperature is overwritten.
    """
    active = active_quad > 0.5
    fixture = (phase_quad == STATE_SUBSTRATE) | (phase_quad == STATE_SUPPORT)
    non_fixture = active & (~fixture)
    phase_new = phase_quad

    newly_active_void = non_fixture & (phase_new == STATE_VOID)
    phase_new = np.where(newly_active_void, STATE_POWDER, phase_new)

    if args.liquidus_temperature > args.solidus_temperature:
        Ts = float(args.solidus_temperature)
        Tl = float(args.liquidus_temperature)
        hot_liquid = non_fixture & (T_quad >= Tl)
        mushy = non_fixture & (T_quad >= Ts) & (T_quad < Tl)
        cold = non_fixture & (T_quad < Ts)

        old_was_melted = (phase_quad == STATE_LIQUID) | (phase_quad == STATE_MUSHY)
        became_solid = cold & old_was_melted
        stayed_solid = cold & (phase_quad == STATE_SOLID)

        phase_new = np.where(hot_liquid, STATE_LIQUID, phase_new)
        phase_new = np.where(mushy, STATE_MUSHY, phase_new)
        phase_new = np.where(became_solid | stayed_solid, STATE_SOLID, phase_new)

        newly_solidified = became_solid
        entered_melted_state = (hot_liquid | mushy) & ((phase_quad == STATE_SOLID) | (phase_quad == STATE_MUSHY) | (phase_quad == STATE_LIQUID))
    else:
        # Compatibility / macro consolidation-on-activation mode when no
        # phase-change interval is provided: activated material solidifies
        # directly. This is the standard part-scale AM approach when the mesh
        # cannot resolve the melt pool (beam radius < element size) or the
        # lumped macro path does not carry melt-level energy density.
        became_solid = non_fixture & (phase_quad != STATE_SOLID)
        phase_new = np.where(non_fixture, STATE_SOLID, phase_new)
        newly_solidified = became_solid
        entered_melted_state = np.zeros_like(active, dtype=bool)

    relax_T = getattr(args, "stress_relaxation_temperature", None)
    if relax_T is not None and relax_T > 0.0:
        # Stress-free reference is the relaxation temperature: above it the
        # material is assumed to carry no stress (macro calibration knob,
        # Ti64 typically ~1073-1173 K). Residual stress then builds from the
        # constrained shrinkage between relax_T and the local temperature.
        T_ref_value = relax_T * np.ones_like(T_quad)
    else:
        T_ref_value = T_quad
    T_ref_new = np.where(newly_solidified, T_ref_value, T_ref_quad)
    if args.reset_plastic_on_melt:
        eqp_new = np.where(entered_melted_state, np.zeros_like(eqp_quad), eqp_quad)
    else:
        eqp_new = eqp_quad
    return phase_new, T_ref_new, eqp_new, newly_solidified, entered_melted_state


def thermal_material_quads(T_old_quad, active_quad, phase_quad, args, tables, printed_quad=None, cooling_only_quad=None):
    rho_solid = args.rho_solid if args.rho_solid is not None else args.rho
    cp_solid = args.cp_solid if args.cp_solid is not None else args.cp
    k_solid = args.conductivity_solid if args.conductivity_solid is not None else args.conductivity
    rho_liquid = args.rho_liquid if args.rho_liquid is not None else rho_solid
    cp_liquid = args.cp_liquid if args.cp_liquid is not None else cp_solid
    k_liquid = args.conductivity_liquid if args.conductivity_liquid is not None else k_solid

    cp_solid_quad = eval_property(T_old_quad, tables["cp_solid"], cp_solid)
    k_solid_quad = eval_property(T_old_quad, tables["k_solid"], k_solid)
    cp_powder_quad = eval_property(T_old_quad, tables["cp_powder"], args.cp_powder)
    k_powder_quad = eval_property(T_old_quad, tables["k_powder"], args.conductivity_powder)
    cp_liquid_quad = eval_property(T_old_quad, tables["cp_liquid"], cp_liquid)
    k_liquid_quad = eval_property(T_old_quad, tables["k_liquid"], k_liquid)

    if printed_quad is None:
        printed_quad = active_quad
    if cooling_only_quad is None:
        cooling_only_quad = np.zeros_like(active_quad)

    inactive_mass_factor = getattr(args, "inactive_mass_factor", None)
    if inactive_mass_factor is None:
        inactive_mass_factor = args.inactive_thermal_factor
    rho_void = rho_solid * inactive_mass_factor
    cp_void = cp_solid * np.ones_like(T_old_quad)
    k_void = k_solid * args.inactive_thermal_factor * np.ones_like(T_old_quad)

    if args.liquidus_temperature > args.solidus_temperature:
        mushy_frac = np.clip(
            (T_old_quad - args.solidus_temperature) / (args.liquidus_temperature - args.solidus_temperature),
            0.0,
            1.0,
        )
    else:
        mushy_frac = np.zeros_like(T_old_quad)

    rho_mushy = (1.0 - mushy_frac) * rho_solid + mushy_frac * rho_liquid
    cp_mushy = (1.0 - mushy_frac) * cp_solid_quad + mushy_frac * cp_liquid_quad
    k_mushy = (1.0 - mushy_frac) * k_solid_quad + mushy_frac * k_liquid_quad

    is_void = phase_quad == STATE_VOID
    is_powder = phase_quad == STATE_POWDER
    is_liquid = phase_quad == STATE_LIQUID
    is_mushy = phase_quad == STATE_MUSHY

    # Phase-based properties for cells that belong to the current thermal window.
    rho_phase = np.where(
        is_void,
        rho_void,
        np.where(is_powder, args.rho_powder, np.where(is_liquid, rho_liquid, np.where(is_mushy, rho_mushy, rho_solid))),
    )
    cp_phase = np.where(
        is_void,
        cp_void,
        np.where(is_powder, cp_powder_quad, np.where(is_liquid, cp_liquid_quad, np.where(is_mushy, cp_mushy, cp_solid_quad))),
    )
    k_phase = np.where(
        is_void,
        k_void,
        np.where(is_powder, k_powder_quad, np.where(is_liquid, k_liquid_quad, np.where(is_mushy, k_mushy, k_solid_quad))),
    )

    # Unprinted region: either powder bed or near-void material.
    # In layer_on_scan mode with future_layer_mode=void, future layers are not
    # physically spread yet, so they should not act as a powder heat sink before
    # their scan begins.
    future_layers_are_void = (
        getattr(args, "layer_activation_mode", "front") == "layer_on_scan"
        and getattr(args, "future_layer_mode", "void") == "void"
    )
    if future_layers_are_void or args.powder_mode == "void":
        rho_unprinted = rho_void
        cp_unprinted = cp_void
        k_unprinted = k_void
    else:
        rho_unprinted = args.rho_powder
        cp_unprinted = cp_powder_quad
        k_unprinted = k_powder_quad

    # Printed layers below the moving thermal window: keep heat capacity/history,
    # but strongly reduce conductivity so they no longer dominate local heat loss.
    rho_old = rho_solid * np.ones_like(T_old_quad)
    cp_old = cp_solid_quad
    k_old = k_solid_quad * args.old_layer_thermal_factor

    is_printed = printed_quad > 0.5
    is_window = active_quad > 0.5
    is_cooling_only = cooling_only_quad > 0.5

    rho_quad = np.where(is_window, rho_phase, np.where(is_cooling_only, rho_old, rho_unprinted))
    cp_quad = np.where(is_window, cp_phase, np.where(is_cooling_only, cp_old, cp_unprinted))
    conductivity_quad = np.where(is_window, k_phase, np.where(is_cooling_only, k_old, k_unprinted))

    # Safety: cells that have not been printed are always treated as unprinted,
    # even if a phase history entry is accidentally non-void.
    rho_quad = np.where(is_printed | is_window, rho_quad, rho_unprinted)
    cp_quad = np.where(is_printed | is_window, cp_quad, cp_unprinted)
    conductivity_quad = np.where(is_printed | is_window, conductivity_quad, k_unprinted)

    if args.latent_heat > 0.0 and args.liquidus_temperature > args.solidus_temperature:
        in_mushy = (T_old_quad >= args.solidus_temperature) & (T_old_quad <= args.liquidus_temperature) & is_window
        latent_cp = np.where(in_mushy, args.latent_heat / (args.liquidus_temperature - args.solidus_temperature), 0.0)
    else:
        latent_cp = np.zeros_like(T_old_quad)

    return rho_quad, cp_quad, conductivity_quad, latent_cp


def clamp_mechanics_temperature(T_quad, floor):
    """Floor the mechanics-side temperature field; floor=None keeps it untouched.

    Only the mechanics chain (dT/thermal strain and material-table lookups) sees
    the clamped field; the thermal solution and phase bookkeeping stay unclamped
    so the guard cannot mask undershoot in thermal diagnostics/audits.
    """
    if floor is None:
        return T_quad
    return np.maximum(T_quad, floor)


def mechanics_material_quads(T_quad, active_quad, phase_quad, args, tables):
    E_quad = eval_property(T_quad, tables["E"], args.young)
    alpha_base = eval_property(T_quad, tables["alpha"], args.alpha)
    poisson_quad = eval_property(T_quad, tables["poisson"], args.poisson)
    if args.mechanics_model == "j2_plastic":
        if tables["yield"] is None:
            raise ValueError("--mechanics-model j2_plastic requires --yield-table")
        yield_quad = eval_property(T_quad, tables["yield"], args.young)
        hardening_quad = eval_property(T_quad, tables["hardening"], 0.0)
    else:
        yield_quad = args.young * np.ones_like(T_quad)
        hardening_quad = np.zeros_like(T_quad)

    is_solid_like = (phase_quad == STATE_SOLID) | (phase_quad == STATE_SUBSTRATE) | (phase_quad == STATE_SUPPORT)
    is_mushy = phase_quad == STATE_MUSHY
    is_liquid = phase_quad == STATE_LIQUID

    active_factor_quad = np.where(
        is_solid_like,
        1.0,
        np.where(is_mushy, args.mushy_mechanics_factor, np.where(is_liquid, args.liquid_mechanics_factor, args.inactive_mechanics_factor)),
    )
    active_factor_quad = active_factor_quad * active_quad + args.inactive_mechanics_factor * (1.0 - active_quad)
    alpha_quad = np.where(is_solid_like, alpha_base, np.zeros_like(alpha_base))
    return active_factor_quad, E_quad, alpha_quad, poisson_quad, yield_quad, hardening_quad

def make_full_bottom_mechanics_bc(bottom):
    def zero(_point):
        return 0.0

    return [[bottom, bottom, bottom], [0, 1, 2], [zero, zero, zero]]


def make_anchor_mechanics_bc(points, candidate_node_ids=None):
    # Anchors must sit on load-bearing (printed) material: the mesh extremes
    # can land on void cells whose near-zero stiffness makes the release
    # system singular (observed NaN). Pass the printed-node subset.
    if candidate_node_ids is not None and len(candidate_node_ids) >= 3:
        candidates = points[onp.asarray(candidate_node_ids)]
    else:
        candidates = points
    span = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1.0)
    atol = 1e-8 * span
    anchor0_id = int(onp.argmin(candidates[:, 0]))
    anchor0 = candidates[anchor0_id]
    dist0 = onp.linalg.norm(candidates - anchor0, axis=1)
    anchor1_id = int(onp.argmax(dist0))
    anchor1 = candidates[anchor1_id]
    axis = anchor1 - anchor0
    axis_norm = max(float(onp.linalg.norm(axis)), 1e-12)
    cross_dist = onp.linalg.norm(onp.cross(candidates - anchor0, axis / axis_norm), axis=1)
    anchor2_id = int(onp.argmax(cross_dist))
    anchor2 = candidates[anchor2_id]

    def at_node(target):
        target = np.asarray(target)

        def location(point):
            return np.linalg.norm(point - target) <= atol

        return location

    def zero(_point):
        return 0.0

    return [
        [at_node(anchor0), at_node(anchor0), at_node(anchor0), at_node(anchor1), at_node(anchor1), at_node(anchor2)],
        [0, 1, 2, 1, 2, 2],
        [zero, zero, zero, zero, zero, zero],
    ]


def make_box_anchor_mechanics_bc(points, box):
    # Partial-cut release: nodes inside the axis-aligned box stay clamped
    # (u=0, all components), modeling the un-cut root attachment that an
    # EDM/saw cut leaves behind (e.g. the Kaess 2023 cantilever separation).
    # The box is given in mesh coordinates (after --mesh-length-scale).
    box = [float(v) for v in box]
    if len(box) != 6:
        raise ValueError("release anchor box needs 6 floats: xmin xmax ymin ymax zmin zmax")
    lo = onp.asarray(box[0::2])
    hi = onp.asarray(box[1::2])
    if onp.any(hi <= lo):
        raise ValueError(f"release anchor box is empty: min={lo}, max={hi}")
    span = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1e-30)
    atol = 1e-8 * span
    inside = onp.all((points >= lo - atol) & (points <= hi + atol), axis=1)
    count = int(onp.sum(inside))
    if count < 3:
        raise ValueError(
            f"release anchor box contains only {count} mesh nodes; "
            "a partial-cut anchor needs at least 3 to suppress rigid modes"
        )
    print(f"release anchor box: clamping {count} nodes in {box}")

    def location(point):
        return np.all((point >= lo - atol) & (point <= hi + atol))

    def zero(_point):
        return 0.0

    return [
        [location, location, location],
        [0, 1, 2],
        [zero, zero, zero],
    ]


def should_run_mechanics(global_step, args):
    return args.mechanics_every > 0 and global_step % args.mechanics_every == 0


def should_save_step(global_step, did_mechanics, is_last, args):
    if is_last:
        return True
    if did_mechanics and args.mechanics_output_every > 0 and global_step % args.mechanics_output_every == 0:
        return True
    if args.thermal_output_every > 0 and global_step % args.thermal_output_every == 0:
        return True
    return False


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
    print("heat_source_normalization:", "2*P/(pi*r_b^2*source_depth)")
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


def mechanics_newton_overrides_from_args(args):
    """Newton overrides for the mechanics solves, built from CLI options.

    The legacy defaults (tol=1e-9, rel_tol=1e-11) are far tighter than
    engineering stress accuracy requires and make yield-surface (j2) states
    crawl; line search stabilizes Newton when many quadrature points sit on
    the yield-surface kink.
    """
    overrides = {}
    if args.mechanics_tol is not None:
        overrides["tol"] = float(args.mechanics_tol)
    if args.mechanics_rel_tol is not None:
        overrides["rel_tol"] = float(args.mechanics_rel_tol)
    if args.mechanics_max_iter is not None:
        overrides["max_iter"] = int(args.mechanics_max_iter)
    if args.mechanics_line_search:
        overrides["line_search_flag"] = True
    return overrides


def main():
    args = parse_args()
    if args.steps is not None:
        print("WARNING: --steps is treated as an alias for --layers in this version.")
        args.layers = args.steps
    if args.layers < 1 or args.scan_steps_per_layer < 1 or args.hatch_lines_per_layer < 1:
        raise ValueError("--layers, --scan-steps-per-layer and --hatch-lines-per-layer must be >= 1")
    if args.mechanics_every < 0:
        raise ValueError("--mechanics-every must be >= 0; use 0 for thermal-only output")
    if args.release_after_cooling and args.cooling_steps < 1:
        print("WARNING: --release-after-cooling requested without cooling steps; release will run after printing.")

    raw_points, cells, selected_cells = read_tet4_inp(args.inp, args.max_cells)
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
    surface_active_mask_enabled = (
        args.surface_active_mask
        if args.surface_active_mask is not None
        else args.surface_selection == "exterior"
    )
    mechanics_newton_overrides = mechanics_newton_overrides_from_args(args)

    if args.surface_selection == "exterior":
        # Every mesh-exterior face above the base plane exchanges heat with the
        # environment. The base plane itself is handled by the bottom BC below.
        span_for_tol = max(float(onp.max(pmax - pmin)), 1.0)
        base_tol = (
            float(args.boundary_tol)
            if args.boundary_tol is not None and args.boundary_tol > 0.0
            else 1e-8 * span_for_tol
        )

        def exterior_above_base(point):
            return build_sign * (point[build_axis_id] - base_coord) > base_tol

        exterior_above_base.exterior_only = True
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

    mesh = Mesh(points, cells, ele_type="TET4")
    thermal = TransientThermal(
        mesh=mesh,
        vec=1,
        dim=3,
        ele_type="TET4",
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
        ),
    )
    if args.thermal_mass_lumping:
        apply_thermal_mass_lumping(thermal)
    if args.bottom_mechanics_bc == "elastic":
        mechanics_bc = None
        mechanics_location_fns = [bottom]
        mechanics_foundation = args.bottom_foundation_stiffness
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
        ele_type="TET4",
        quadrature_order=args.quadrature_order,
        dirichlet_bc_info=mechanics_bc,
        location_fns=mechanics_location_fns,
        additional_info=(
            args.mechanics_model,
            args.yield_saturation_stress,
            mechanics_foundation,
            powder_foundation,
            tuple(plane_axis_ids),
        ),
    )

    tables = load_property_tables(args)
    cell_centroids, cell_build_coord, substrate_cell, support_cell = classify_cells(points, cells, build_axis_id, build_sign, base_coord, args)
    layer_id_cell = compute_layer_id(cell_build_coord, build_axis_id, part_pmin, part_pmax, args)
    # Fixture cells are not printed part layers. Keep their layer id at 0 for
    # clearer ParaView interpretation.
    layer_id_cell = onp.asarray(layer_id_cell, dtype=onp.int32)
    layer_id_cell[substrate_cell | support_cell] = 0
    physical_layer_id_cell = compute_physical_layer_id_cell(
        cell_build_coord,
        build_axis_id,
        part_pmin,
        part_pmax,
        build_sign,
        args,
    )
    physical_layer_id_cell = onp.asarray(physical_layer_id_cell, dtype=onp.int32)
    physical_layer_id_cell[substrate_cell | support_cell] = 0
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
        layer_id_cell[substrate_cell | support_cell] = 0

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
    }
    write_used_config(args, args.output_dir, derived)
    print_startup(args, raw_pmin, raw_pmax, pmin, pmax, part_pmin, part_pmax, selected_cells, points, thermal, mechanics, derived)

    final_cooldown_enabled = (
        args.final_cooldown_temperature is not None
        and args.bottom_thermal_bc == "fixed"
    )
    if args.final_cooldown_temperature is not None and args.bottom_thermal_bc != "fixed":
        print("WARNING: --final-cooldown-temperature requires --bottom-thermal-bc fixed; ignoring.")
    n_cooling_steps = sum(1 for s in step_states if s.mode == "cooling")
    cooling_step_counter = 0

    quad_stress = None
    last_mechanics_step = -1
    last_active_cell = previous_active.copy()
    last_printed_cell = previous_active.copy()
    last_cooling_only_cell = onp.zeros_like(previous_active, dtype=bool)
    last_material_state = material_cell_state(last_active_cell, substrate_cell, support_cell, args, phase_cell=phase_cell_from_quad(phase_quad))
    last_dT_quad = np.zeros((len(cells), thermal.fes[0].num_quads, 1))
    last_T_mech_quad = None
    last_mechanical_active_quad = None

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

        active_quad = make_quad_scalar(active_cell.astype(onp.float64), thermal.fes[0].num_quads)
        printed_quad = make_quad_scalar(printed_cell.astype(onp.float64), thermal.fes[0].num_quads)
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
            printed_quad=printed_quad,
            cooling_only_quad=cooling_only_quad,
        )
        if final_cooldown_enabled and state.mode == "cooling" and n_cooling_steps > 0:
            cooling_step_counter += 1
            ramp = cooling_step_counter / float(n_cooling_steps)
            ramped_bottom_temperature = (
                bottom_temperature_effective
                + ramp * (float(args.final_cooldown_temperature) - bottom_temperature_effective)
            )

            def ramped_bottom_value(_point, _value=ramped_bottom_temperature):
                return _value

            thermal.fes[0].update_Dirichlet_boundary_conditions(
                [[bottom], [0], [ramped_bottom_value]]
            )

        effective_laser_power = args.absorptivity * state.laser_power
        if surface_active_mask_enabled:
            surface_mask_quad = printed_quad
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
        T_mech_quad = clamp_mechanics_temperature(T_quad, args.mechanics_temperature_floor)
        dT_quad = (T_mech_quad - T_ref_quad) * mechanical_active_quad
        active_factor_quad, E_quad, alpha_quad, poisson_quad, yield_quad, hardening_quad = mechanics_material_quads(T_mech_quad, mechanical_active_quad, phase_quad, args, tables)
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
        printed_node_ids = onp.unique(
            onp.asarray(cells)[onp.asarray(last_printed_cell, dtype=bool)].reshape(-1)
        )
        if args.release_anchor_mode == "box":
            if args.release_anchor_box is None:
                raise ValueError("--release-anchor-mode box requires --release-anchor-box")
            release_bc = make_box_anchor_mechanics_bc(points, args.release_anchor_box)
        else:
            release_bc = make_anchor_mechanics_bc(points, candidate_node_ids=printed_node_ids)
        if args.release_cut_box is not None:
            cut = [float(v) for v in args.release_cut_box]
            lo = onp.asarray(cut[0::2])
            hi = onp.asarray(cut[1::2])
            cut_cell = onp.all(
                (cell_centroids >= lo[None, :]) & (cell_centroids <= hi[None, :]),
                axis=1,
            )
            n_cut = int(cut_cell.sum())
            print(f"release cut box: deactivating {n_cut} cells in {cut}")
            if n_cut:
                cut_quad = make_quad_scalar(
                    cut_cell.astype(onp.float64),
                    mechanics.fes[0].num_quads,
                )
                # kill both stiffness and locked-in stress of sawed-off cells
                mechanics_params = list(mechanics_params)
                mechanics_params[2] = np.where(
                    cut_quad > 0.5,
                    args.inactive_mechanics_factor,
                    mechanics_params[2],
                )
        release_mechanics = ThermoMechanical(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type="TET4",
            quadrature_order=args.quadrature_order,
            dirichlet_bc_info=release_bc,
            additional_info=(args.mechanics_model, args.yield_saturation_stress, 0.0),
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
        )
        print(f"release_vtk={vtk_path} release_u_max={float(np.max(np.abs(u_release[0]))):.12g}")


if __name__ == "__main__":
    main()
