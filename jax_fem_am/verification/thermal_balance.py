"""Discrete weak-form thermal balance and temperature invariant contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ThermalStepBalance:
    schema_version: str
    claim_level: str
    storage_j: float
    laser_deposited_j: float
    laser_commanded_j: Optional[float]
    laser_absorbed_nominal_j: float
    front_loss_j: float
    old_layer_loss_j: float
    surface_loss_j: float
    dirichlet_exchange_into_domain_j: float
    balance_error_j: float
    relative_balance_error: float
    assembly_identity_error_j: float
    free_residual_l1_j: float
    free_residual_l2_j: float
    source_capture_fraction: Optional[float]

    @property
    def laser_nominal_j(self) -> float:
        """Legacy alias: nominal means absorbed nominal energy, not command."""
        return self.laser_absorbed_nominal_j


def compute_discrete_balance(
    *,
    storage_j,
    laser_deposited_j,
    front_loss_j,
    old_layer_loss_j,
    surface_loss_j,
    dirichlet_exchange_into_domain_j,
    assembly_identity_error_j,
    free_residual_l1_j,
    free_residual_l2_j,
    laser_commanded_j=None,
    laser_absorbed_nominal_j=None,
    laser_nominal_j=None,
):
    """Assemble the signed balance used by the current thermal weak form.

    ``front_loss_j``, ``old_layer_loss_j`` and ``surface_loss_j`` are signed
    outward exchanges: positive values remove energy from the domain and
    negative values add energy to it. ``laser_commanded_j`` is the machine
    command before absorptivity, ``laser_absorbed_nominal_j`` is the nominal
    absorbed energy before mesh/quadrature truncation, and
    ``laser_deposited_j`` is the source actually integrated by the solver.

    ``laser_nominal_j`` remains accepted as a legacy alias for
    ``laser_absorbed_nominal_j``.
    """
    if laser_absorbed_nominal_j is None:
        if laser_nominal_j is None:
            raise ValueError(
                "laser_absorbed_nominal_j is required "
                "(legacy alias: laser_nominal_j)"
            )
        laser_absorbed_nominal_j = laser_nominal_j
    elif laser_nominal_j is not None and (
        float(laser_absorbed_nominal_j) != float(laser_nominal_j)
    ):
        raise ValueError(
            "laser_nominal_j and laser_absorbed_nominal_j must describe "
            "the same absorbed nominal energy"
        )

    names = {
        "storage_j": storage_j,
        "laser_deposited_j": laser_deposited_j,
        "laser_absorbed_nominal_j": laser_absorbed_nominal_j,
        "front_loss_j": front_loss_j,
        "old_layer_loss_j": old_layer_loss_j,
        "surface_loss_j": surface_loss_j,
        "dirichlet_exchange_into_domain_j": dirichlet_exchange_into_domain_j,
        "assembly_identity_error_j": assembly_identity_error_j,
        "free_residual_l1_j": free_residual_l1_j,
        "free_residual_l2_j": free_residual_l2_j,
    }
    values = {name: float(value) for name, value in names.items()}
    commanded = (
        None if laser_commanded_j is None else float(laser_commanded_j)
    )
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("all energy and residual terms must be finite")
    if commanded is not None and not np.isfinite(commanded):
        raise ValueError("laser_commanded_j must be finite when provided")
    for name in (
        "laser_deposited_j",
        "laser_absorbed_nominal_j",
        "assembly_identity_error_j",
        "free_residual_l1_j",
        "free_residual_l2_j",
    ):
        if values[name] < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if commanded is not None and commanded < 0.0:
        raise ValueError("laser_commanded_j must be nonnegative")
    absorbed = values["laser_absorbed_nominal_j"]
    if commanded is not None and absorbed > commanded and not np.isclose(
        absorbed, commanded, rtol=1.0e-12, atol=0.0
    ):
        raise ValueError(
            "laser_absorbed_nominal_j cannot exceed laser_commanded_j"
        )
    if absorbed == 0.0 and values["laser_deposited_j"] > 0.0:
        raise ValueError(
            "laser_deposited_j cannot be positive when absorbed nominal "
            "energy is zero"
        )

    error = (
        values["storage_j"]
        - values["laser_deposited_j"]
        + values["front_loss_j"]
        + values["old_layer_loss_j"]
        + values["surface_loss_j"]
        - values["dirichlet_exchange_into_domain_j"]
    )
    scale = max(
        abs(values["storage_j"])
        + abs(values["laser_deposited_j"])
        + abs(values["front_loss_j"])
        + abs(values["old_layer_loss_j"])
        + abs(values["surface_loss_j"])
        + abs(values["dirichlet_exchange_into_domain_j"]),
        np.finfo(np.float64).tiny,
    )
    capture = (
        values["laser_deposited_j"] / absorbed
        if absorbed > 0.0
        else None
    )
    return ThermalStepBalance(
        schema_version="v06.thermal_balance.v2",
        claim_level="discrete_weak_form_only",
        laser_commanded_j=commanded,
        balance_error_j=error,
        relative_balance_error=abs(error) / scale,
        source_capture_fraction=capture,
        **values,
    )


def check_temperature_invariants(
    T_old,
    T_new,
    *,
    ambient,
    dirichlet_values,
    deposited_source_j,
    coefficients_valid,
    atol_k,
):
    """Check physical bounds without claiming a discrete maximum principle."""
    T_old = np.asarray(T_old, dtype=np.float64).reshape(-1)
    T_new = np.asarray(T_new, dtype=np.float64).reshape(-1)
    dirichlet = np.asarray(dirichlet_values, dtype=np.float64).reshape(-1)
    ambient = float(ambient)
    deposited = float(deposited_source_j)
    atol = float(atol_k)
    if not len(T_old) or not len(T_new):
        raise ValueError("old and new temperatures must be nonempty")
    if not np.all(np.isfinite(T_old)) or not np.all(np.isfinite(dirichlet)):
        raise ValueError("old and Dirichlet temperatures must be finite")
    if not np.isfinite(ambient) or not np.isfinite(deposited) or deposited < 0.0:
        raise ValueError(
            "ambient and deposited source must be finite and source nonnegative"
        )
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("temperature tolerance must be finite and nonnegative")

    finite_new = np.isfinite(T_new)
    bounds = np.concatenate([T_old, dirichlet, np.asarray([ambient])])
    lower = float(bounds.min())
    upper = float(bounds.max())
    lower_count = int(np.count_nonzero(finite_new & (T_new < lower - atol)))
    source_free = deposited <= np.finfo(np.float64).eps
    upper_count = (
        int(np.count_nonzero(finite_new & (T_new > upper + atol)))
        if source_free
        else None
    )
    valid = bool(
        coefficients_valid
        and np.all(finite_new)
        and lower_count == 0
        and (upper_count in (None, 0))
    )
    return {
        "claim_level": "physical_temperature_invariant_diagnostic",
        "valid": valid,
        "coefficient_preconditions_valid": bool(coefficients_valid),
        "all_new_temperatures_finite": bool(np.all(finite_new)),
        "source_free": source_free,
        "lower_bound_k": lower,
        "upper_bound_k": upper if source_free else None,
        "lower_violation_count": lower_count,
        "upper_violation_count": upper_count,
        "atol_k": atol,
    }
