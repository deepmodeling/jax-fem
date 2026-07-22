"""Thermo-mechanical problem (J2 plasticity, B-bar universal kernel).

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import jax
import jax.flatten_util
import jax.numpy as np
import numpy as onp

from jax_fem.problem import Problem

from jax_fem_am.io.vtu import von_mises_from_stress


class ThermoMechanical(Problem):
    def custom_init(self, mechanics_model, yield_saturation=None, foundation_stiffness=0.0,
                    powder_foundation_stiffness=0.0, powder_plane_axes=(), bbar=False):
        # B-bar (element-average volumetric strain). jax-fem dispatches on
        # hasattr(get_tensor_map)/hasattr(get_universal_kernel) and SUMS all
        # kernels that exist, so exactly one of the two is bound per
        # instance (custom_init runs before pre_jit_fns).
        self.bbar = bool(bbar)
        if self.bbar:
            self.get_universal_kernel = self._make_bbar_universal_kernel
        else:
            self.get_tensor_map = self._tensor_map_getter
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

    def _tensor_map_getter(self):
        # Bound as instance attribute `get_tensor_map` when bbar is off.
        return self.stress_fn

    def _make_bbar_universal_kernel(self):
        """B-bar volume kernel: element-average volumetric strain on BOTH the
        trial and test sides (Hughes 1980; what Abaqus C3D8 does by default).

        Trial: sigma is evaluated at eps_bar = dev(eps) + (theta_bar/3) I with
        theta_bar the JxW-weighted element average of tr(grad u). Test: for a
        symmetric sigma, sigma : eps_bar(v) = dev(sigma) : grad(v) +
        p * theta_bar(v), so the deviatoric part contracts with the raw test
        gradients and the pressure with the element-average test dilatation.
        The consistent tangent comes free from jax-fem's jacfwd over this
        kernel. Cures TET4/HEX8 volumetric locking under J2 flow (checkerboard
        hydrostatic pressure, diagnosed 2026-07-21); an exact no-op wherever
        tr(grad u) is element-constant.
        """
        num_nodes = self.fes[0].num_nodes
        dim = self.dim
        tensor_map = self.stress_fn  # late-bound: subclasses' stress_fn wins

        def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                   cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            # shapes as in Problem.get_laplace_kernel / get_mass_kernel
            cell_sol = self.unflatten_fn_dof(cell_sol_flat)[0]      # (num_nodes, vec)
            shape_grads = cell_shape_grads[:, :num_nodes, :]        # (num_quads, num_nodes, dim)
            v_grads_JxW = cell_v_grads_JxW[:, :num_nodes, :, :]     # (num_quads, num_nodes, 1, dim)
            JxW = cell_JxW[0]                                       # (num_quads,)
            eye = np.eye(dim)

            u_grads = np.sum(cell_sol[None, :, :, None] * shape_grads[:, :, None, :],
                             axis=1)                                # (num_quads, vec, dim)
            vol = np.sum(JxW)
            theta = np.trace(u_grads, axis1=-2, axis2=-1)           # (num_quads,)
            theta_bar = np.sum(theta * JxW) / vol
            u_grads_bar = u_grads + ((theta_bar - theta) / dim)[:, None, None] * eye

            sigma = jax.vmap(tensor_map)(u_grads_bar, *cell_internal_vars)
            p = np.trace(sigma, axis1=-2, axis2=-1) / dim           # (num_quads,)
            sigma_dev = sigma - p[:, None, None] * eye

            # deviatoric stress against raw test gradients ...
            val = np.sum(sigma_dev[:, None, :, :] * v_grads_JxW, axis=(0, -1))
            # ... pressure against the element-average test dilatation
            avg_v_grads = np.sum(v_grads_JxW, axis=0)[:, 0, :] / vol  # (num_nodes, dim)
            val = val + np.sum(p * JxW) * avg_v_grads
            return jax.flatten_util.ravel_pytree(val)[0]

        return kernel

    def _u_grads(self, sol):
        """Displacement gradients at quad points, B-barred when enabled.

        Every consumer of strain outside the residual (stress output, eqp
        update, stress-free reference capture) must use the SAME strain
        measure as the residual, so the barring lives here.
        """
        u_grads = np.sum(
            np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None]
            * self.fes[0].shape_grads[:, :, :, None, :],
            axis=2)                                    # (num_cells, num_quads, vec, dim)
        if getattr(self, "bbar", False):
            JxW = self.fes[0].JxW                      # (num_cells, num_quads)
            theta = np.trace(u_grads, axis1=-2, axis2=-1)
            theta_bar = (np.sum(theta * JxW, axis=1, keepdims=True)
                         / np.sum(JxW, axis=1, keepdims=True))
            u_grads = u_grads + ((theta_bar - theta) / self.dim)[..., None, None] * np.eye(self.dim)
        return u_grads

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
        u_grads = self._u_grads(sol)
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
        u_grads = self._u_grads(sol)

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
