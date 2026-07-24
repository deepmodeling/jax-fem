"""Solver-facing quadrature extraction for the v06 thermal energy ledger."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .thermal_balance import (
    check_temperature_invariants,
    compute_discrete_balance,
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _field(name, values, shape):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return values


def integrate_volume_terms(
    *,
    jxw,
    points,
    temperature_old,
    temperature_new,
    rho,
    cp,
    latent_cp,
    laser_center,
    effective_laser_power_w,
    beam_radius_m,
    source_depth_m,
    laser_switch,
    active,
    cooling_only,
    old_layer_cooling_h,
    ambient_k,
    dt_s,
    build_axis,
    plane_axes,
    build_sign,
    front_loss_h,
    front_loss_thickness_m,
    front_loss_radiation,
    emissivity,
    stefan_boltzmann,
    source_model="legacy",
):
    """Integrate the explicit volume terms used by the v03 weak form."""
    jxw = np.asarray(jxw, dtype=np.float64)
    if jxw.ndim != 2 or not np.all(np.isfinite(jxw)) or np.any(jxw <= 0.0):
        raise ValueError("jxw must be a positive finite (cell, quad) array")
    shape = jxw.shape
    points = np.asarray(points, dtype=np.float64)
    if points.shape != shape + (3,) or not np.all(np.isfinite(points)):
        raise ValueError("points must be finite with shape (cell, quad, 3)")
    old = _field("temperature_old", temperature_old, shape)
    new = _field("temperature_new", temperature_new, shape)
    rho = _field("rho", rho, shape)
    cp = _field("cp", cp, shape)
    latent_cp = _field("latent_cp", latent_cp, shape)
    active = _field("active", active, shape)
    cooling_only = _field("cooling_only", cooling_only, shape)
    if np.any(rho <= 0.0):
        raise ValueError("rho must be positive")
    if np.any(cp <= 0.0) or np.any(cp + latent_cp <= 0.0):
        raise ValueError("cp and cp + latent_cp must be positive")
    if np.any(latent_cp < 0.0):
        raise ValueError("latent_cp must be nonnegative")
    if not np.all(np.isin(active, (0.0, 1.0))) or not np.all(
        np.isin(cooling_only, (0.0, 1.0))
    ):
        raise ValueError("active and cooling_only must be binary")

    laser_center = np.asarray(laser_center, dtype=np.float64)
    scalars = {
        "effective_laser_power_w": effective_laser_power_w,
        "beam_radius_m": beam_radius_m,
        "source_depth_m": source_depth_m,
        "laser_switch": laser_switch,
        "old_layer_cooling_h": old_layer_cooling_h,
        "ambient_k": ambient_k,
        "dt_s": dt_s,
        "build_sign": build_sign,
        "front_loss_h": front_loss_h,
        "front_loss_thickness_m": front_loss_thickness_m,
        "emissivity": emissivity,
        "stefan_boltzmann": stefan_boltzmann,
    }
    scalars = {name: float(value) for name, value in scalars.items()}
    if laser_center.shape != (3,) or not np.all(np.isfinite(laser_center)):
        raise ValueError("laser_center must be a finite 3-vector")
    if not all(np.isfinite(value) for value in scalars.values()):
        raise ValueError("thermal ledger scalar inputs must be finite")
    if source_model not in ("legacy", "paper_hemispherical"):
        raise ValueError(
            "source_model must be 'legacy' or 'paper_hemispherical'"
        )
    if scalars["dt_s"] <= 0.0:
        raise ValueError("dt_s must be positive")
    if scalars["beam_radius_m"] <= 0.0:
        raise ValueError("beam_radius_m must be positive")
    if source_model == "legacy" and scalars["source_depth_m"] <= 0.0:
        raise ValueError("source_depth_m must be positive for legacy source")
    if scalars["effective_laser_power_w"] < 0.0:
        raise ValueError("effective_laser_power_w must be nonnegative")
    if not 0.0 <= scalars["laser_switch"] <= 1.0:
        raise ValueError("laser_switch must lie in [0, 1]")
    if scalars["old_layer_cooling_h"] < 0.0 or scalars["front_loss_h"] < 0.0:
        raise ValueError("loss coefficients must be nonnegative")
    if scalars["front_loss_h"] > 0.0 and scalars["front_loss_thickness_m"] <= 0.0:
        raise ValueError("front loss thickness must be positive when enabled")
    if not 0.0 <= scalars["emissivity"] <= 1.0:
        raise ValueError("emissivity must lie in [0, 1]")
    build_axis = int(build_axis)
    plane_axes = tuple(int(axis) for axis in plane_axes)
    if sorted((*plane_axes, build_axis)) != [0, 1, 2] or len(plane_axes) != 2:
        raise ValueError("build_axis and plane_axes must partition xyz")
    dt = scalars["dt_s"]
    storage = np.sum(jxw * rho * (cp + latent_cp) * (new - old))

    r0 = points[..., plane_axes[0]] - laser_center[plane_axes[0]]
    r1 = points[..., plane_axes[1]] - laser_center[plane_axes[1]]
    depth = scalars["build_sign"] * (
        laser_center[build_axis] - points[..., build_axis]
    )
    if source_model == "paper_hemispherical":
        radius = scalars["beam_radius_m"]
        q_shape = np.where(
            depth >= 0.0,
            np.exp(-3.0 * (r0**2 + r1**2 + depth**2) / radius**2),
            0.0,
        )
        q_laser = (
            6.0
            * np.sqrt(3.0)
            * scalars["effective_laser_power_w"]
            / (np.pi * np.sqrt(np.pi) * radius**3)
            * q_shape
            * scalars["laser_switch"]
            * active
        )
    else:
        q_depth = np.where(
            depth >= 0.0,
            np.exp(-depth / scalars["source_depth_m"]),
            0.0,
        )
        q_laser = (
            2.0
            * scalars["effective_laser_power_w"]
            / (
                np.pi
                * scalars["beam_radius_m"] ** 2
                * scalars["source_depth_m"]
            )
            * np.exp(-2.0 * (r0**2 + r1**2) / scalars["beam_radius_m"] ** 2)
            * q_depth
            * scalars["laser_switch"]
            * active
        )

    if scalars["front_loss_h"] > 0.0:
        front_band = np.where(
            depth >= 0.0,
            np.exp(-(depth / scalars["front_loss_thickness_m"]) ** 2),
            0.0,
        ) * active
        q_front = (
            scalars["front_loss_h"]
            / scalars["front_loss_thickness_m"]
            * (old - scalars["ambient_k"])
            * front_band
        )
        if front_loss_radiation:
            q_front += (
                scalars["emissivity"]
                * scalars["stefan_boltzmann"]
                / scalars["front_loss_thickness_m"]
                * (old**4 - scalars["ambient_k"] ** 4)
                * front_band
            )
    else:
        q_front = np.zeros(shape, dtype=np.float64)
    q_old = (
        scalars["old_layer_cooling_h"]
        * cooling_only
        * (old - scalars["ambient_k"])
    )
    return {
        "storage_j": float(storage),
        "laser_deposited_j": float(dt * np.sum(jxw * q_laser)),
        "front_loss_j": float(dt * np.sum(jxw * q_front)),
        "old_layer_loss_j": float(dt * np.sum(jxw * q_old)),
    }


def integrate_surface_exchange(
    *,
    temperature_face,
    surface_jxw,
    active,
    convection_h,
    ambient_k,
    emissivity,
    stefan_boltzmann,
    dt_s,
):
    """Integrate signed outward convection and radiation on boundary faces."""
    temperature = np.asarray(temperature_face, dtype=np.float64)
    jxw = np.asarray(surface_jxw, dtype=np.float64)
    active = np.asarray(active, dtype=np.float64)
    if (
        temperature.shape != jxw.shape
        or active.shape != jxw.shape
        or not np.all(np.isfinite(temperature))
        or not np.all(np.isfinite(jxw))
        or not np.all(np.isfinite(active))
        or np.any(jxw < 0.0)
    ):
        raise ValueError("surface arrays must be finite, nonnegative-weight, equal shape")
    if not np.all(np.isin(active, (0.0, 1.0))):
        raise ValueError("surface active flags must be binary")
    h = float(convection_h)
    ambient = float(ambient_k)
    emissivity = float(emissivity)
    sigma = float(stefan_boltzmann)
    dt = float(dt_s)
    if not all(np.isfinite(value) for value in (h, ambient, emissivity, sigma, dt)):
        raise ValueError("surface coefficients must be finite")
    if h < 0.0 or sigma < 0.0 or dt <= 0.0 or not 0.0 <= emissivity <= 1.0:
        raise ValueError("surface coefficients lie outside their physical domains")
    outward = active * (
        h * (temperature - ambient)
        + emissivity * sigma * (temperature**4 - ambient**4)
    )
    return float(dt * np.sum(jxw * outward))


def _uniform_scalar(name, values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    value = float(values[0])
    if not np.allclose(values, value, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError(f"{name} must be uniform over quadrature points")
    return value


def _surface_exchange_from_problem(problem, temperature_new, dt_s):
    fe = problem.fes[0]
    cells = np.asarray(fe.cells, dtype=np.int64)
    temperature_new = np.asarray(temperature_new, dtype=np.float64)
    total = 0.0
    for index, boundary_inds in enumerate(problem.boundary_inds_list):
        boundary_inds = np.asarray(boundary_inds, dtype=np.int64)
        if not len(boundary_inds):
            continue
        owner_cells = boundary_inds[:, 0]
        nodal = temperature_new[cells[owner_cells], 0]
        shape_values = np.asarray(
            problem.selected_face_shape_vals[index], dtype=np.float64
        )
        temperature_face = np.einsum("fn,fqn->fq", nodal, shape_values)
        surface_jxw = np.asarray(
            problem.nanson_scale[index], dtype=np.float64
        )[:, 0, :]
        surface_vars = problem.internal_vars_surfaces[index]
        if not surface_vars:
            raise ValueError("thermal surface is missing its active mask")
        active = np.asarray(surface_vars[0], dtype=np.float64)[..., 0]
        total += integrate_surface_exchange(
            temperature_face=temperature_face,
            surface_jxw=surface_jxw,
            active=active,
            convection_h=problem.convection_h,
            ambient_k=problem.ambient,
            emissivity=problem.emissivity,
            stefan_boltzmann=problem.stefan_boltzmann,
            dt_s=dt_s,
        )
    return float(total)


def _dirichlet_mask_and_values(fe, num_nodes):
    mask = np.zeros(num_nodes, dtype=bool)
    values = []
    for node_ids, vector_ids, prescribed in zip(
        getattr(fe, "node_inds_list", []),
        getattr(fe, "vec_inds_list", []),
        getattr(fe, "vals_list", []),
    ):
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        vector_ids = np.asarray(vector_ids, dtype=np.int64).reshape(-1)
        if vector_ids.size and np.any(vector_ids != 0):
            raise ValueError("thermal ledger expects scalar Dirichlet components")
        mask[node_ids] = True
        prescribed = np.asarray(prescribed, dtype=np.float64).reshape(-1)
        if prescribed.size == 1 and len(node_ids) != 1:
            prescribed = np.full(len(node_ids), prescribed[0])
        if prescribed.size != len(node_ids):
            raise ValueError("Dirichlet values do not match constrained nodes")
        values.extend(prescribed.tolist())
    return mask, np.asarray(values, dtype=np.float64)


def extract_solver_step(
    problem,
    solution,
    *,
    step_index,
    step_state,
    absorptivity,
    previous_solution,
    temperature_atol_k,
    solver_residual_tolerance_w=1.0e-6,
    relative_balance_tolerance=1.0e-6,
):
    """Extract one accepted thermal solve using the problem's current state."""
    if not isinstance(solution, (list, tuple)) or len(solution) != 1:
        raise ValueError("thermal solution must be a one-field solution list")
    temperature_new = np.asarray(solution[0], dtype=np.float64)
    if temperature_new.ndim != 2 or temperature_new.shape[1] != 1:
        raise ValueError("thermal solution must have shape (nodes, 1)")
    internal = list(problem.internal_vars)
    if len(internal) != 14:
        raise ValueError("unexpected TransientThermal internal variable contract")
    (
        temperature_old_quad,
        dt_quad,
        laser_center_quad,
        effective_power_quad,
        beam_radius_quad,
        source_depth_quad,
        switch_quad,
        active_quad,
        rho_quad,
        cp_quad,
        conductivity_quad,
        latent_cp_quad,
        cooling_only_quad,
        old_layer_h_quad,
    ) = internal
    dt_s = _uniform_scalar("dt", dt_quad)
    effective_power = _uniform_scalar("effective laser power", effective_power_quad)
    beam_radius = _uniform_scalar("beam radius", beam_radius_quad)
    source_depth = _uniform_scalar("source depth", source_depth_quad)
    laser_switch = _uniform_scalar("laser switch", switch_quad)
    old_layer_h = _uniform_scalar("old layer cooling coefficient", old_layer_h_quad)
    conductivity = np.asarray(conductivity_quad, dtype=np.float64)[..., 0]
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("thermal conductivity must be finite and positive")
    laser_center = np.asarray(laser_center_quad, dtype=np.float64)[0, 0]
    fe = problem.fes[0]
    temperature_new_quad = np.asarray(
        fe.convert_from_dof_to_quad(temperature_new), dtype=np.float64
    )[..., 0]
    volume = integrate_volume_terms(
        jxw=np.asarray(fe.JxW, dtype=np.float64),
        points=np.asarray(problem.physical_quad_points, dtype=np.float64),
        temperature_old=np.asarray(temperature_old_quad, dtype=np.float64)[..., 0],
        temperature_new=temperature_new_quad,
        rho=np.asarray(rho_quad, dtype=np.float64)[..., 0],
        cp=np.asarray(cp_quad, dtype=np.float64)[..., 0],
        latent_cp=np.asarray(latent_cp_quad, dtype=np.float64)[..., 0],
        laser_center=laser_center,
        effective_laser_power_w=effective_power,
        beam_radius_m=beam_radius,
        source_depth_m=source_depth,
        laser_switch=laser_switch,
        active=np.asarray(active_quad, dtype=np.float64)[..., 0],
        cooling_only=np.asarray(cooling_only_quad, dtype=np.float64)[..., 0],
        old_layer_cooling_h=old_layer_h,
        ambient_k=problem.ambient,
        dt_s=dt_s,
        build_axis=problem.build_axis_id,
        plane_axes=(problem.plane_axis0_id, problem.plane_axis1_id),
        build_sign=problem.build_sign,
        front_loss_h=problem.front_surface_loss_h,
        front_loss_thickness_m=problem.front_surface_loss_thickness,
        front_loss_radiation=problem.front_surface_loss_radiation,
        emissivity=problem.emissivity,
        stefan_boltzmann=problem.stefan_boltzmann,
        source_model=getattr(problem, "source_model", "legacy"),
    )
    surface_loss = _surface_exchange_from_problem(
        problem, temperature_new, dt_s
    )
    raw_residual = np.asarray(
        problem.compute_residual([solution[0]])[0], dtype=np.float64
    ).reshape(-1)
    if len(raw_residual) != len(temperature_new) or not np.all(
        np.isfinite(raw_residual)
    ):
        raise ValueError("unconstrained thermal residual must be finite per node")
    constrained, dirichlet_values = _dirichlet_mask_and_values(
        fe, len(temperature_new)
    )
    dirichlet_exchange = float(dt_s * np.sum(raw_residual[constrained]))
    free = raw_residual[~constrained]
    free_signed = float(dt_s * np.sum(free))
    explicit_total = (
        volume["storage_j"]
        - volume["laser_deposited_j"]
        + volume["front_loss_j"]
        + volume["old_layer_loss_j"]
        + surface_loss
    )
    residual_total = float(dt_s * np.sum(raw_residual))
    assembly_identity_signed = explicit_total - residual_total

    absorptivity = float(absorptivity)
    if not np.isfinite(absorptivity) or not 0.0 <= absorptivity <= 1.0:
        raise ValueError("absorptivity must lie in [0, 1]")
    absorbed_nominal = effective_power * laser_switch * dt_s
    if step_state is not None:
        commanded = (
            float(step_state.laser_power)
            * float(step_state.laser_switch)
            * dt_s
        )
    elif absorptivity > 0.0:
        commanded = absorbed_nominal / absorptivity
    else:
        commanded = 0.0
    balance = compute_discrete_balance(
        **volume,
        surface_loss_j=surface_loss,
        dirichlet_exchange_into_domain_j=dirichlet_exchange,
        assembly_identity_error_j=abs(assembly_identity_signed),
        free_residual_l1_j=float(dt_s * np.sum(np.abs(free))),
        free_residual_l2_j=float(dt_s * np.linalg.norm(free)),
        laser_commanded_j=commanded,
        laser_absorbed_nominal_j=absorbed_nominal,
    )
    solver_residual_tolerance_w = float(solver_residual_tolerance_w)
    relative_balance_tolerance = float(relative_balance_tolerance)
    if (
        not np.isfinite(solver_residual_tolerance_w)
        or solver_residual_tolerance_w < 0.0
        or not np.isfinite(relative_balance_tolerance)
        or relative_balance_tolerance < 0.0
    ):
        raise ValueError("balance tolerances must be finite and nonnegative")
    balance_scale = (
        abs(volume["storage_j"])
        + abs(volume["laser_deposited_j"])
        + abs(volume["front_loss_j"])
        + abs(volume["old_layer_loss_j"])
        + abs(surface_loss)
        + abs(dirichlet_exchange)
    )
    absolute_balance_tolerance = (
        np.sqrt(max(int(np.count_nonzero(~constrained)), 1))
        * dt_s
        * solver_residual_tolerance_w
    )
    balance_within_tolerance = bool(
        abs(balance.balance_error_j)
        <= absolute_balance_tolerance
        + relative_balance_tolerance * balance_scale
    )
    assembly_scale = max(
        abs(explicit_total),
        abs(residual_total),
        np.finfo(np.float64).tiny,
    )
    assembly_tolerance = 1.0e-12 + 1.0e-10 * assembly_scale
    assembly_within_tolerance = bool(
        abs(assembly_identity_signed) <= assembly_tolerance
    )
    invariants = check_temperature_invariants(
        np.asarray(temperature_old_quad, dtype=np.float64)[..., 0],
        temperature_new,
        ambient=problem.ambient,
        dirichlet_values=dirichlet_values,
        deposited_source_j=volume["laser_deposited_j"],
        coefficients_valid=True,
        atol_k=temperature_atol_k,
    )
    state_override = None
    if previous_solution is not None:
        previous_quad = np.asarray(
            fe.convert_from_dof_to_quad(previous_solution), dtype=np.float64
        )[..., 0]
        state_override = float(
            np.sum(
                np.asarray(fe.JxW, dtype=np.float64)
                * np.asarray(rho_quad, dtype=np.float64)[..., 0]
                * (
                    np.asarray(cp_quad, dtype=np.float64)[..., 0]
                    + np.asarray(latent_cp_quad, dtype=np.float64)[..., 0]
                )
                * (
                    np.asarray(temperature_old_quad, dtype=np.float64)[..., 0]
                    - previous_quad
                )
            )
        )
    state_override_tolerance = max(
        float(absolute_balance_tolerance), 1.0e-12
    )
    state_override_within_tolerance = bool(
        state_override is None
        or abs(state_override) <= state_override_tolerance
    )
    row = asdict(balance)
    row.update(
        {
            "schema_version": "v06.thermal-energy-ledger-step/1",
            "claim_level": "solver_discrete_weak_form_audit_only",
            "step_index": int(step_index),
            "assembly_identity_signed_j": float(assembly_identity_signed),
            "free_residual_signed_j": free_signed,
            "pre_solve_state_override_j": state_override,
            "state_override_tolerance_j": state_override_tolerance,
            "state_override_within_tolerance": (
                state_override_within_tolerance
            ),
            "balance_scale_j": float(balance_scale),
            "free_node_count": int(np.count_nonzero(~constrained)),
            "dt_s": float(dt_s),
            "solver_residual_tolerance_w": float(
                solver_residual_tolerance_w
            ),
            "absolute_balance_tolerance_j": float(
                absolute_balance_tolerance
            ),
            "relative_balance_tolerance": relative_balance_tolerance,
            "balance_within_solver_tolerance": balance_within_tolerance,
            "assembly_identity_tolerance_j": float(assembly_tolerance),
            "assembly_identity_within_tolerance": assembly_within_tolerance,
            "temperature_invariants_valid": bool(invariants["valid"]),
            "temperature_invariants": invariants,
        }
    )
    if step_state is not None:
        row["step_state"] = {
            name: getattr(step_state, name)
            for name in (
                "global_step",
                "mode",
                "layer_idx",
                "hatch_idx",
                "scan_idx",
            )
        }
    row["relative_balance_error"] = float(row["relative_balance_error"])
    return row


class EnergyLedgerRecorder:
    """Append-only JSONL recorder with a forensic completion summary."""

    def __init__(self, output_dir, *, expected_steps):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.output_dir / "thermal_energy_ledger.jsonl"
        self.summary_path = self.output_dir / "thermal_energy_ledger_summary.json"
        self.expected_steps = int(expected_steps)
        if self.expected_steps < 0:
            raise ValueError("expected_steps must be nonnegative")
        if self.ledger_path.exists() or self.summary_path.exists():
            raise FileExistsError("thermal energy ledger artifacts already exist")
        self.rows = []
        self._final_summary = None

    def append(self, row):
        if self._final_summary is not None:
            raise RuntimeError("cannot append to a finalized thermal ledger")
        row = dict(row)
        expected_index = len(self.rows)
        if row.get("step_index") != expected_index:
            raise ValueError(
                f"thermal ledger step_index must be sequential: {expected_index}"
            )
        encoded = json.dumps(
            row,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        self.rows.append(json.loads(encoded))

    def finalize(self, *, completed):
        if self._final_summary is not None:
            return self._final_summary
        relative = [float(row["relative_balance_error"]) for row in self.rows]
        assembly = [float(row["assembly_identity_error_j"]) for row in self.rows]
        invariants = [bool(row["temperature_invariants_valid"]) for row in self.rows]
        balances = [
            bool(row.get("balance_within_solver_tolerance", False))
            for row in self.rows
        ]
        identities = [
            bool(row.get("assembly_identity_within_tolerance", False))
            for row in self.rows
        ]
        state_overrides = [
            bool(row.get("state_override_within_tolerance", False))
            for row in self.rows
        ]
        state_override_values = [
            float(row["pre_solve_state_override_j"])
            for row in self.rows
            if row.get("pre_solve_state_override_j") is not None
        ]
        complete = bool(
            completed
            and len(self.rows) == self.expected_steps
            and all(invariants)
            and all(balances)
            and all(identities)
            and all(state_overrides)
        )
        summary = {
            "schema_version": "v06.thermal-energy-ledger-summary/1",
            "claim_level": "solver_discrete_weak_form_audit_only",
            "complete": complete,
            "solver_completed": bool(completed),
            "recorded_step_count": len(self.rows),
            "expected_step_count": self.expected_steps,
            "all_temperature_invariants_valid": bool(all(invariants)),
            "all_balance_steps_within_tolerance": bool(all(balances)),
            "all_assembly_identities_within_tolerance": bool(all(identities)),
            "all_pre_solve_state_overrides_within_tolerance": bool(
                all(state_overrides)
            ),
            "cumulative_pre_solve_state_override_j": float(
                sum(state_override_values)
            ),
            "maximum_relative_balance_error": max(relative) if relative else None,
            "maximum_absolute_balance_error_j": (
                max(abs(float(row.get("balance_error_j", 0.0))) for row in self.rows)
                if self.rows
                else None
            ),
            "maximum_assembly_identity_error_j": max(assembly) if assembly else None,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._final_summary = summary
        return summary
