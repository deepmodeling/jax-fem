"""Thermo-mechanical problem (J2 plasticity, B-bar universal kernel).

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import jax
import jax.flatten_util
import jax.numpy as np
import numpy as onp

from jax_fem.problem import Problem

from jax_fem_am.io.vtu import von_mises_from_stress
from jax_fem_am.materials.j2 import PlasticState, radial_return


class ThermoMechanical(Problem):
    def custom_init(self, mechanics_model, yield_saturation=None, foundation_stiffness=0.0,
                    powder_foundation_stiffness=0.0, powder_plane_axes=(), bbar=False,
                    flow_curve=None):
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
        self.flow_curve = flow_curve
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
        if self.flow_curve is not None:
            self._flow_curve_active_mask = np.zeros_like(
                self.internal_vars[0]
            )
            self._flow_curve_mask_bound = False
            self.internal_vars.append(
                self._flow_curve_active_mask
            )
        else:
            self._flow_curve_active_mask = None
            self._flow_curve_mask_bound = True

    def _material_point_update(
        self,
        u_grad,
        T,
        dT,
        active_factor,
        young,
        alpha,
        poisson,
        yield_stress,
        hardening,
        eqp_old,
        eps_p_old=None,
        eps_ref_old=None,
        flow_curve_active=None,
    ):
        """Run the single canonical constitutive update used by every consumer."""
        if eps_p_old is None:
            eps_p_old = np.zeros((self.dim, self.dim))
        if eps_ref_old is None:
            eps_ref_old = np.zeros((self.dim, self.dim))
        strain = 0.5 * (u_grad + u_grad.T) - eps_ref_old
        nu = np.clip(poisson[0], -0.49, 0.49)
        is_plastic = self.mechanics_model == "j2_plastic"
        flow_curve = (
            getattr(self, "flow_curve", None)
            if is_plastic
            else None
        )
        if flow_curve is not None:
            if flow_curve_active is None:
                raise ValueError(
                    "flow-curve material update requires an explicit selector"
                )
            curve_selector = np.asarray(flow_curve_active)
            if curve_selector.ndim:
                curve_selector = curve_selector[0]
            curve_selector = curve_selector > 0.5
        else:
            curve_selector = False

        saturation = (
            self.yield_saturation
            if is_plastic
            and self.yield_saturation is not None
            and self.yield_saturation > 0.0
            else np.inf
        )
        return radial_return(
            strain=strain,
            thermal_strain=alpha[0] * dT[0] * np.eye(self.dim),
            state=PlasticState(
                eqp=eqp_old[0],
                eps_p=eps_p_old,
            ),
            young=young[0],
            poisson=nu,
            yield_stress=yield_stress[0] if is_plastic else np.inf,
            hardening=hardening[0],
            saturation=saturation,
            temperature=T[0],
            flow_curve=flow_curve,
            flow_curve_active=curve_selector,
        )

    def stress_fn(self, u_grad, T, dT, active_factor, young, alpha, poisson, yield_stress, hardening, eqp_old, flow_curve_active=None):
        update = ThermoMechanical._material_point_update(
            self,
            u_grad,
            T,
            dT,
            active_factor,
            young,
            alpha,
            poisson,
            yield_stress,
            hardening,
            eqp_old,
            flow_curve_active=flow_curve_active,
        )
        return active_factor[0] * update.stress

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

    def set_flow_curve_active_mask(self, selector):
        """Bind the per-quadrature solid-curve selector without changing params."""
        if self.flow_curve is None:
            if selector is not None:
                raise ValueError(
                    "flow-curve selector provided without a flow curve"
                )
            return
        selector = np.asarray(selector)
        if selector.shape != self.internal_vars[0].shape:
            raise ValueError(
                "flow-curve selector shape must match scalar quadrature "
                f"fields: {selector.shape} != {self.internal_vars[0].shape}"
            )
        self._flow_curve_active_mask = selector
        self._flow_curve_mask_bound = True

    def _require_flow_curve_mask_bound(self):
        if (
            getattr(self, "flow_curve", None) is not None
            and not getattr(self, "_flow_curve_mask_bound", False)
        ):
            raise ValueError(
                "flow-curve quadrature selector must be bound before "
                "using mechanics parameters or postprocessors"
            )

    def set_params(self, params):
        self._require_flow_curve_mask_bound()
        self.internal_vars = list(params)
        if self.flow_curve is not None:
            self.internal_vars.append(
                self._flow_curve_active_mask
            )

    def compute_cell_stress(self, sol, params):
        self._require_flow_curve_mask_bound()
        T_quad, dT_quad, active_factor_quad, young_quad, alpha_quad, poisson_quad, yield_quad, hardening_quad, eqp_old_quad = params
        u_grads = self._u_grads(sol)
        selector = (
            self._flow_curve_active_mask
            if self.flow_curve is not None
            else np.zeros_like(active_factor_quad[..., :1])
        )
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
            selector,
        )
        return {
            "stress_quad": sigmas,
            "vm_quad": von_mises_from_stress(sigmas),
        }

    def compute_eqp_update(self, sol, params):
        self._require_flow_curve_mask_bound()
        if self.mechanics_model != "j2_plastic":
            return params[-1]

        T_quad, dT_quad, active_factor_quad, young_quad, alpha_quad, poisson_quad, yield_quad, hardening_quad, eqp_old_quad = params
        u_grads = self._u_grads(sol)

        def one_quad(u_grad, T, dT, active_factor, young, alpha, poisson, yield_stress, hardening, eqp_old, flow_curve_active):
            update = ThermoMechanical._material_point_update(
                self,
                u_grad,
                T,
                dT,
                active_factor,
                young,
                alpha,
                poisson,
                yield_stress,
                hardening,
                eqp_old,
                flow_curve_active=flow_curve_active,
            )
            active = np.where(active_factor[0] > 0.5, 1.0, 0.0)
            return np.array(
                [eqp_old[0] + active * update.delta_eqp]
            )

        selector = (
            self._flow_curve_active_mask
            if self.flow_curve is not None
            else np.zeros_like(active_factor_quad[..., :1])
        )
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
            selector,
        )
