"""Transient thermal problem with selectable moving volumetric laser source.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import math
import numbers

import jax.numpy as np
import numpy as onp

from jax_fem.problem import Problem


SOURCE_MODELS = frozenset(("legacy", "paper_hemispherical"))


def _build_face_neighbor_cells(cells, face_inds):
    """Return the adjacent cell for every directed local face.

    Exterior faces receive ``-1``.  An interior face appears once for each
    owner cell, and the two entries point at one another's owner.  The topology
    is built once on the host; only the fixed-shape activity mask changes on
    the per-step JAX path.
    """

    cells = onp.asarray(cells)
    face_inds = onp.asarray(face_inds)
    face_nodes = cells[:, face_inds]
    num_cells, num_faces, num_face_nodes = face_nodes.shape
    keys = onp.sort(
        face_nodes.reshape(num_cells * num_faces, num_face_nodes),
        axis=1,
    )
    _, inverse, counts = onp.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if onp.any(counts > 2):
        count = int(onp.max(counts))
        raise ValueError(
            "active-domain exterior requires a manifold mesh; "
            f"a face has {count} owners"
        )

    neighbors = -onp.ones(num_cells * num_faces, dtype=onp.int32)
    order = onp.argsort(inverse, kind="stable")
    offsets = onp.concatenate(
        (onp.asarray([0], dtype=onp.int64), onp.cumsum(counts))
    )
    for group_id, count in enumerate(counts):
        if count != 2:
            continue
        first, second = order[offsets[group_id] : offsets[group_id + 1]]
        neighbors[first] = int(second // num_faces)
        neighbors[second] = int(first // num_faces)
    return neighbors.reshape(num_cells, num_faces)


def _concrete_scalar(value):
    """Return a host scalar only when ``value`` is already concrete on host.

    In particular, do not call ``asarray``/``float`` on JAX arrays or tracers:
    ``set_params`` is on the per-step path and materializing those values would
    introduce a device-to-host synchronization on GPU.
    """

    if isinstance(value, numbers.Real) and not isinstance(
        value, (bool, onp.bool_)
    ):
        return float(value)
    if isinstance(value, onp.ndarray) and value.size == 1:
        scalar = value.reshape(-1)[0]
        if isinstance(scalar, numbers.Real) and not isinstance(
            scalar, (bool, onp.bool_)
        ):
            return float(scalar)
    return None


def _validate_concrete_length_scale(value, name):
    concrete = _concrete_scalar(value)
    if concrete is None:
        return
    if not math.isfinite(concrete) or concrete <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


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
        self.process_ambient = ambient
        self.ambient = ambient
        self.emissivity = emissivity
        self.stefan_boltzmann = stefan_boltzmann
        self.build_axis_id = int(build_axis_id)
        self.plane_axis0_id = int(plane_axis0_id)
        self.plane_axis1_id = int(plane_axis1_id)
        self.build_sign = float(build_sign)
        self.num_surface_maps = int(num_surface_maps)
        # Optional legacy volumetric approximation of convection/radiation
        # loss near the moving build front. Dynamic face exchange is handled
        # separately when ``active_domain_exterior`` is enabled. Set
        # front_surface_loss_h=0.0 to disable this approximation.
        self.front_surface_loss_h = float(front_surface_loss_h)
        self.front_surface_loss_thickness = float(front_surface_loss_thickness)
        self.front_surface_loss_radiation = bool(front_surface_loss_radiation)
        if source_model not in SOURCE_MODELS:
            raise ValueError(
                f"source_model must be one of {sorted(SOURCE_MODELS)}, "
                f"got {source_model!r}"
            )
        self.source_model = source_model
        location_fns = self.location_fns or ()
        self._dynamic_surface_maps = tuple(
            bool(getattr(location_fn, "active_domain_exterior", False))
            for location_fn in location_fns
        )
        self._surface_neighbor_cells = [
            None for _ in range(len(self.boundary_inds_list))
        ]
        if any(self._dynamic_surface_maps):
            neighbor_cells = _build_face_neighbor_cells(
                self.fes[0].cells,
                self.fes[0].face_inds,
            )
            for index, (is_dynamic, boundary_inds) in enumerate(
                zip(self._dynamic_surface_maps, self.boundary_inds_list)
            ):
                if not is_dynamic:
                    continue
                boundary_inds_host = onp.asarray(boundary_inds)
                self._surface_neighbor_cells[index] = np.asarray(
                    neighbor_cells[
                        boundary_inds_host[:, 0],
                        boundary_inds_host[:, 1],
                    ]
                )

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
            _environment_ambient=None,
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
            environment_ambient=None,
        ):
            ambient = (
                self.ambient
                if environment_ambient is None
                else environment_ambient[0]
            )
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
                    * (T_old[0] - ambient)
                    * front_band
                )
                if self.front_surface_loss_radiation:
                    q_front_loss = q_front_loss + (
                        self.emissivity
                        * self.stefan_boltzmann
                        / self.front_surface_loss_thickness
                        * (T_old[0] ** 4 - ambient**4)
                        * front_band
                    )
            else:
                q_front_loss = 0.0

            # Printed layers below the moving thermal window are optionally
            # cooled by a volumetric sink while their conductivity is reduced
            # in thermal_material_quads(). old_layer_cooling_h has units
            # W/(m^3*K). This term is explicit in T_old for robustness.
            q_old_layer_loss = (
                old_layer_cooling_h[0]
                * cooling_only[0]
                * (T_old[0] - ambient)
            )

            cp_eff = cp[0] + latent_cp[0]
            return np.array([rho[0] * cp_eff * (T[0] - T_old[0]) / dt[0] - q_vol + q_front_loss + q_old_layer_loss])

        return heat_capacity

    def get_surface_maps(self):
        # face_active masks the flux to faces owned by printed/real material.
        # Void (not yet spread) cells are not physical surfaces; applying
        # convection/radiation to their near-singular equations produces
        # unphysical temperatures. In legacy 'box' mode the mask is all-ones,
        # which is numerically identical to the historical behavior.
        def surface_flux(
            T,
            _point,
            face_active,
            environment_ambient=None,
        ):
            ambient = (
                self.ambient
                if environment_ambient is None
                else environment_ambient[0]
            )
            q_conv = self.convection_h * (ambient - T[0])
            q_rad = (
                self.emissivity
                * self.stefan_boltzmann
                * (ambient**4 - T[0] ** 4)
            )
            return -np.array([(q_conv + q_rad) * face_active[0]])

        return [surface_flux for _ in range(self.num_surface_maps)]

    def set_params(self, params):
        if len(params) == 15:
            environment_ambient = getattr(
                self,
                "process_ambient",
                getattr(self, "ambient", 0.0),
            )
        elif len(params) == 16:
            environment_ambient = params[-1]
        else:
            raise ValueError(
                "TransientThermal.set_params expects 15 legacy values "
                "or 16 values including environment ambient"
            )
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
        ) = params[:15]
        # The configured runner supplies these two lengths as Python floats.
        # Validate such host-concrete values cheaply, but deliberately leave
        # JAX arrays/tracers on device: coercing them here would synchronize
        # every thermal step on GPU.
        _validate_concrete_length_scale(beam_radius, "beam_radius")
        if self.source_model == "legacy":
            _validate_concrete_length_scale(source_depth, "source_depth")
        concrete_ambient = _concrete_scalar(environment_ambient)
        if concrete_ambient is not None:
            self.ambient = concrete_ambient

        num_cells = self.fes[0].num_cells
        num_quads = self.fes[0].num_quads
        dt_quad = dt * np.ones((num_cells, num_quads, 1))
        laser_center_quad = laser_center[None, None, :] * np.ones((num_cells, num_quads))[:, :, None]
        laser_power_quad = effective_laser_power * np.ones((num_cells, num_quads, 1))
        beam_radius_quad = beam_radius * np.ones((num_cells, num_quads, 1))
        source_depth_quad = source_depth * np.ones((num_cells, num_quads, 1))
        laser_switch_quad = laser_switch * np.ones((num_cells, num_quads, 1))
        old_layer_cooling_h_quad = old_layer_cooling_h * np.ones((num_cells, num_quads, 1))
        environment_ambient_quad = environment_ambient * np.ones(
            (num_cells, num_quads, 1)
        )

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
            environment_ambient_quad,
        ]
        # Per-face activity flags for the surface flux maps. surface_mask_quad
        # is a (num_cells, num_quads, 1) physical-material indicator. Static
        # maps inherit their owner flag; active-domain exterior maps additionally
        # require an exterior face or an inactive neighbor.
        num_face_quads = self.fes[0].num_face_quads
        cell_mask = surface_mask_quad[:, 0, 0]
        self.internal_vars_surfaces = []
        for index, boundary_inds in enumerate(self.boundary_inds_list):
            if boundary_inds.shape[0] == 0:
                self.internal_vars_surfaces.append([])
                continue
            face_flags = cell_mask[boundary_inds[:, 0]]
            if (
                index < len(self._dynamic_surface_maps)
                and self._dynamic_surface_maps[index]
            ):
                neighbor_cells = self._surface_neighbor_cells[index]
                has_neighbor = neighbor_cells >= 0
                safe_neighbor_cells = np.maximum(neighbor_cells, 0)
                neighbor_flags = cell_mask[safe_neighbor_cells]
                face_flags = face_flags * np.where(
                    has_neighbor,
                    1.0 - neighbor_flags,
                    1.0,
                )
            face_active = face_flags[:, None, None] * np.ones((1, num_face_quads, 1))
            face_ambient = environment_ambient * np.ones(
                (boundary_inds.shape[0], num_face_quads, 1)
            )
            self.internal_vars_surfaces.append([face_active, face_ambient])
