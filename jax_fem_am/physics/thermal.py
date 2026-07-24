"""Transient thermal problem with selectable moving volumetric laser source.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import jax.numpy as np

from jax_fem.problem import Problem


SOURCE_MODELS = frozenset(("legacy", "paper_hemispherical"))


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
        source_model="legacy",
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
        if source_model not in SOURCE_MODELS:
            raise ValueError(
                f"source_model must be one of {sorted(SOURCE_MODELS)}, "
                f"got {source_model!r}"
            )
        self.source_model = source_model

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
        source_model = getattr(self, "source_model", "legacy")
        if source_model not in SOURCE_MODELS:
            raise ValueError(
                f"source_model must be one of {sorted(SOURCE_MODELS)}, "
                f"got {source_model!r}"
            )

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
            if source_model == "paper_hemispherical":
                # Kaess et al. (2023), Equation (1): a spherically symmetric
                # three-dimensional Gaussian truncated to the material-side
                # half-space. The normalization integrates to P_abs over that
                # half-space; laser_power is already absorptivity * commanded
                # power, so no second efficiency factor belongs here.
                radius = beam_radius[0]
                q_shape = np.where(
                    depth >= 0.0,
                    np.exp(-3.0 * (r2 + depth**2) / radius**2),
                    0.0,
                )
                q_vol = (
                    6.0
                    * np.sqrt(3.0)
                    * laser_power[0]
                    / (np.pi * np.sqrt(np.pi) * radius**3)
                    * q_shape
                    * laser_switch[0]
                    * active[0]
                )
            else:
                q_depth = np.where(
                    depth >= 0.0,
                    np.exp(-depth / source_depth[0]),
                    0.0,
                )
                # The in-plane Gaussian integrates to pi*r_b^2/2 and the
                # one-sided exponential depth decay integrates to source_depth.
                # The factor 2 makes the integral equal the absorbed power.
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
