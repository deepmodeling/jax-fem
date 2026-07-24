"""Kaess (2023) material and phase-history contract tests.

The published model uses a temperature-dependent weak-solid powder, a
280 kJ/kg latent heat over 1370--1400 degC, and an irreversible field-variable
switch from powder to solid.  The public paper does not report an additional
liquid/mushy stiffness multiplier or a stress/plastic-history reset.  The
project therefore freezes those absent mechanisms as disabled assumptions.

Sources:
* ``cases/kaess_2023/references/cases/kaess_2023_fulltext.txt``, Section 2.5,
  Tables 1--2.
* ``cases/kaess_2023/inputs/assumptions.yaml``, ``A-PHASE-HISTORY``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem_am.domain.events import update_phase_reference_and_eqp
from jax_fem_am.materials.phases import (
    STATE_LIQUID,
    STATE_MUSHY,
    STATE_POWDER,
    STATE_SOLID,
    mechanics_material_quads,
    thermal_material_quads,
)
from jax_fem_am.materials.tables import PropertyTable


POWDER_TABLE_MIN_K = 293.15  # 20 degC
POWDER_TABLE_MAX_K = 1643.15  # 1370 degC
SOLIDUS_K = 1643.15
LIQUIDUS_K = 1673.15
LATENT_HEAT_J_KG = 280_000.0


class LinearTable:
    """Small in-memory equivalent of a frozen T,value property table."""

    def __init__(self, temperatures, values):
        self.T = jnp.asarray(temperatures)
        self.values = jnp.asarray(values)

    def eval(self, temperature):
        return jnp.interp(temperature, self.T, self.values)


def thermal_args(**overrides):
    values = {
        "rho": 8000.0,
        "rho_solid": 8000.0,
        "rho_liquid": 8000.0,
        "rho_powder": 4000.0,
        "cp": 500.0,
        "cp_solid": 500.0,
        "cp_liquid": 500.0,
        "cp_powder": 500.0,
        "conductivity": 20.0,
        "conductivity_solid": 20.0,
        "conductivity_liquid": 20.0,
        "conductivity_powder": 0.3,
        "inactive_mass_factor": None,
        "inactive_thermal_factor": 1.0e-6,
        "old_layer_thermal_factor": 1.0,
        "powder_mode": "powder",
        "layer_activation_mode": "layer_on_scan",
        "future_layer_mode": "void",
        "solidus_temperature": SOLIDUS_K,
        "liquidus_temperature": LIQUIDUS_K,
        "latent_heat": LATENT_HEAT_J_KG,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def thermal_tables(**overrides):
    values = {
        "k_solid": None,
        "cp_solid": None,
        "k_powder": None,
        "cp_powder": None,
        "k_liquid": None,
        "cp_liquid": None,
    }
    values.update(overrides)
    return values


def phase_args(**overrides):
    values = {
        "solidus_temperature": SOLIDUS_K,
        "liquidus_temperature": LIQUIDUS_K,
        "stress_relaxation_temperature": None,
        "reset_plastic_on_melt": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def mechanics_args(**overrides):
    values = {
        "young": 190.0e9,
        "alpha": 16.0e-6,
        "poisson": 0.3,
        "mechanics_model": "linear_elastic",
        "mushy_mechanics_factor": 1.0e-2,
        "liquid_mechanics_factor": 1.0e-4,
        "inactive_mechanics_factor": 0.0,
        "powder_solid_E": None,
        "layer_activation_mode": "layer_on_scan",
        "future_layer_mode": "void",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def mechanics_tables(**overrides):
    values = {
        "E": None,
        "alpha": None,
        "poisson": None,
        "yield": None,
        "hardening": None,
    }
    values.update(overrides)
    return values


def one_quad(value):
    return jnp.asarray([[[value]]], dtype=jnp.float64)


def advance_phase(temperature, phase, reference, eqp, args, *, active=True):
    return update_phase_reference_and_eqp(
        one_quad(temperature),
        one_quad(1.0 if active else 0.0),
        phase,
        reference,
        eqp,
        args,
    )


def test_powder_conductivity_uses_frozen_kaess_temperature_table(tmp_path):
    powder_table_path = tmp_path / "kaess_powder_k.csv"
    powder_table_path.write_text(
        "T,value\n"
        f"{POWDER_TABLE_MIN_K},0.15\n"
        f"{POWDER_TABLE_MAX_K},0.60\n",
        encoding="utf-8",
    )
    powder_table = PropertyTable(powder_table_path)
    temperatures = jnp.asarray(
        [
            250.0,
            POWDER_TABLE_MIN_K,
            0.5 * (POWDER_TABLE_MIN_K + POWDER_TABLE_MAX_K),
            POWDER_TABLE_MAX_K,
            1800.0,
        ],
        dtype=jnp.float64,
    ).reshape((-1, 1, 1))
    active = jnp.ones_like(temperatures)
    phase = STATE_POWDER * jnp.ones_like(temperatures)

    _, _, conductivity, _ = thermal_material_quads(
        temperatures,
        active,
        phase,
        thermal_args(),
        thermal_tables(k_powder=powder_table),
        printed_quad=active,
    )

    np.testing.assert_allclose(
        np.asarray(conductivity[:, 0, 0]),
        [0.15, 0.15, 0.375, 0.60, 0.60],
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_latent_heat_apparent_capacity_integrates_to_280_kj_per_kg():
    temperatures = jnp.linspace(
        SOLIDUS_K,
        LIQUIDUS_K,
        2001,
        dtype=jnp.float64,
    ).reshape((-1, 1, 1))
    active = jnp.ones_like(temperatures)
    phase = STATE_POWDER * jnp.ones_like(temperatures)

    _, _, _, latent_cp = thermal_material_quads(
        temperatures,
        active,
        phase,
        thermal_args(),
        thermal_tables(),
        printed_quad=active,
    )

    integrated = np.trapezoid(
        np.asarray(latent_cp[:, 0, 0]),
        np.asarray(temperatures[:, 0, 0]),
    )
    relative_error = abs(integrated - LATENT_HEAT_J_KG) / LATENT_HEAT_J_KG
    assert relative_error <= 0.005


def test_single_increment_crossing_melt_interval_cannot_skip_latent_heat():
    """The coefficient frozen at T_old must not lose a whole phase crossing.

    ``T_new_quad`` is an explicit extension point for the GREEN
    implementation.  Until it exists, this computes the energy represented by
    the current old-temperature coefficient, which exposes the documented
    zero-latent-heat crossing defect numerically rather than as an import error.
    """

    old_temperature = SOLIDUS_K - 20.0
    new_temperature = LIQUIDUS_K + 20.0
    old_quad = one_quad(old_temperature)
    new_quad = one_quad(new_temperature)
    active = one_quad(1.0)
    phase = one_quad(STATE_POWDER)
    kwargs = {"printed_quad": active}
    if "T_new_quad" in inspect.signature(thermal_material_quads).parameters:
        kwargs["T_new_quad"] = new_quad

    _, _, _, latent_cp = thermal_material_quads(
        old_quad,
        active,
        phase,
        thermal_args(),
        thermal_tables(),
        **kwargs,
    )

    represented_latent_heat = float(
        np.asarray(latent_cp)[0, 0, 0]
    ) * (new_temperature - old_temperature)
    assert represented_latent_heat == pytest.approx(
        LATENT_HEAT_J_KG,
        rel=0.005,
    )


def test_powder_to_solid_switch_is_irreversible_after_first_melt():
    args = phase_args()
    phase = one_quad(STATE_POWDER)
    reference = one_quad(300.0)
    eqp = one_quad(0.0)

    phase, reference, eqp, _, _ = advance_phase(
        LIQUIDUS_K + 10.0,
        phase,
        reference,
        eqp,
        args,
    )
    assert float(np.asarray(phase)[0, 0, 0]) == STATE_LIQUID

    phase, reference, eqp, _, _ = advance_phase(
        SOLIDUS_K - 10.0,
        phase,
        reference,
        eqp,
        args,
    )
    assert float(np.asarray(phase)[0, 0, 0]) == STATE_SOLID

    phase, reference, eqp, _, _ = advance_phase(
        300.0,
        phase,
        reference,
        eqp,
        args,
        active=False,
    )
    assert float(np.asarray(phase)[0, 0, 0]) == STATE_SOLID

    phase, _, _, _, _ = advance_phase(
        300.0,
        phase,
        reference,
        eqp,
        args,
        active=True,
    )
    assert float(np.asarray(phase)[0, 0, 0]) == STATE_SOLID


def test_remelt_cycle_preserves_first_stress_reference_and_eqp_history():
    args = phase_args(reset_plastic_on_melt=False)
    phase = one_quad(STATE_POWDER)
    reference = one_quad(300.0)
    eqp = one_quad(0.0)

    phase, reference, eqp, _, _ = advance_phase(
        LIQUIDUS_K + 10.0,
        phase,
        reference,
        eqp,
        args,
    )
    first_solidification_temperature = SOLIDUS_K - 10.0
    phase, reference, eqp, _, _ = advance_phase(
        first_solidification_temperature,
        phase,
        reference,
        eqp,
        args,
    )
    assert float(np.asarray(phase)[0, 0, 0]) == STATE_SOLID
    first_reference = np.asarray(reference).copy()

    eqp = one_quad(0.25)
    phase, reference, eqp, _, _ = advance_phase(
        LIQUIDUS_K + 20.0,
        phase,
        reference,
        eqp,
        args,
    )
    phase, reference, eqp, _, _ = advance_phase(
        SOLIDUS_K - 30.0,
        phase,
        reference,
        eqp,
        args,
    )

    assert float(np.asarray(phase)[0, 0, 0]) == STATE_SOLID
    np.testing.assert_allclose(np.asarray(reference), first_reference)
    np.testing.assert_allclose(np.asarray(eqp), [[[0.25]]])


def test_unapproved_eqp_reset_flag_cannot_erase_remelt_history():
    args = phase_args(reset_plastic_on_melt=True)
    phase = one_quad(STATE_SOLID)
    reference = one_quad(SOLIDUS_K - 10.0)
    eqp = one_quad(0.25)

    phase, reference, eqp, _, entered_melted = advance_phase(
        LIQUIDUS_K + 10.0,
        phase,
        reference,
        eqp,
        args,
    )

    assert bool(np.asarray(entered_melted)[0, 0, 0])
    np.testing.assert_allclose(np.asarray(eqp), [[[0.25]]])


def test_unapproved_relaxation_temperature_cannot_reset_stress_reference():
    args = phase_args(stress_relaxation_temperature=1100.0)
    phase = one_quad(STATE_LIQUID)
    reference = one_quad(300.0)
    eqp = one_quad(0.0)
    solidification_temperature = SOLIDUS_K - 10.0

    phase, reference, _, newly_solidified, _ = advance_phase(
        solidification_temperature,
        phase,
        reference,
        eqp,
        args,
    )

    assert bool(np.asarray(newly_solidified)[0, 0, 0])
    assert float(np.asarray(phase)[0, 0, 0]) == STATE_SOLID
    np.testing.assert_allclose(
        np.asarray(reference),
        [[[solidification_temperature]]],
    )


def test_high_temperature_curve_is_not_scaled_again_in_mushy_or_liquid_state():
    temperatures = jnp.asarray(
        [SOLIDUS_K + 10.0, LIQUIDUS_K + 10.0],
        dtype=jnp.float64,
    ).reshape((-1, 1, 1))
    phase = jnp.asarray(
        [STATE_MUSHY, STATE_LIQUID],
        dtype=jnp.float64,
    ).reshape((-1, 1, 1))
    active = jnp.ones_like(temperatures)
    high_temperature_E = LinearTable(
        [SOLIDUS_K, LIQUIDUS_K + 20.0],
        [2.0e9, 2.0e7],
    )

    active_factor, E_quad, *_ = mechanics_material_quads(
        temperatures,
        active,
        phase,
        mechanics_args(),
        mechanics_tables(E=high_temperature_E),
    )

    expected_E = np.asarray(high_temperature_E.eval(temperatures))
    np.testing.assert_allclose(np.asarray(E_quad), expected_E)
    np.testing.assert_allclose(np.asarray(active_factor), np.ones((2, 1, 1)))
    np.testing.assert_allclose(
        np.asarray(active_factor * E_quad),
        expected_E,
    )
