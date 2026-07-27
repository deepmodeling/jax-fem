"""CPU/JIT and lifecycle parity for the Kaess irreversible phase history."""

from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_fem_am.domain.events import update_phase_reference_and_eqp
from jax_fem_am.materials.phases import (
    STATE_LIQUID,
    STATE_MUSHY,
    STATE_POWDER,
    STATE_SOLID,
    STATE_SUBSTRATE,
    STATE_SUPPORT,
    STATE_VOID,
    mechanics_material_quads,
    thermal_material_quads,
)
from jax_fem_am.simulation import acceleration, runner


SOLIDUS_K = 1643.15
LIQUIDUS_K = 1673.15


def _one_quad(value):
    return jnp.asarray([[[value]]], dtype=jnp.float64)


def _phase_args():
    # These legacy switches are deliberately hostile. Paper-history behavior
    # must remain irreversible and history preserving even when an old config
    # still carries them.
    return SimpleNamespace(
        solidus_temperature=SOLIDUS_K,
        liquidus_temperature=LIQUIDUS_K,
        phase_history_model="paper_irreversible",
        stress_relaxation_temperature=1100.0,
        reset_plastic_on_melt=True,
    )


def _material_args():
    return SimpleNamespace(
        rho=8000.0,
        rho_solid=8000.0,
        rho_liquid=8000.0,
        rho_powder=4000.0,
        cp=500.0,
        cp_solid=500.0,
        cp_liquid=500.0,
        cp_powder=500.0,
        conductivity=20.0,
        conductivity_solid=20.0,
        conductivity_liquid=20.0,
        conductivity_powder=0.3,
        inactive_thermal_factor=1.0e-6,
        inactive_mass_factor=None,
        old_layer_thermal_factor=1.0,
        solidus_temperature=SOLIDUS_K,
        liquidus_temperature=LIQUIDUS_K,
        latent_heat=280_000.0,
        layer_activation_mode="layer_on_scan",
        future_layer_mode="void",
        powder_mode="powder",
        mechanics_model="linear_elastic",
        young=190.0e9,
        alpha=1.6e-5,
        poisson=0.3,
        mushy_mechanics_factor=1.0e-2,
        liquid_mechanics_factor=1.0e-4,
        inactive_mechanics_factor=0.0,
        powder_solid_E=10.0e9,
        powder_solid_yield=1.0e6,
        powder_solid_hardening=0.0,
        phase_history_model="paper_irreversible",
    )


def _base_module():
    return SimpleNamespace(
        jax=jax,
        np=jnp,
        STATE_VOID=STATE_VOID,
        STATE_POWDER=STATE_POWDER,
        STATE_SOLID=STATE_SOLID,
        STATE_MUSHY=STATE_MUSHY,
        STATE_LIQUID=STATE_LIQUID,
        STATE_SUBSTRATE=STATE_SUBSTRATE,
        STATE_SUPPORT=STATE_SUPPORT,
        thermal_material_quads=lambda *args, **kwargs: None,
        mechanics_material_quads=lambda *args, **kwargs: None,
        update_phase_reference_and_eqp=update_phase_reference_and_eqp,
    )


def _advance(function, temperature, state, args):
    phase, reference, eqp = state
    result = function(
        _one_quad(temperature),
        _one_quad(1.0),
        phase,
        reference,
        eqp,
        args,
    )
    return result, result[:3]


def test_jit_history_matches_cpu_through_first_melt_and_remelt():
    base = _base_module()
    cpu_update = base.update_phase_reference_and_eqp
    acceleration._LOOP_KERNEL_JIT_HISTORY_CACHE.clear()
    assert acceleration.install_loop_kernel_jit_patch(base, enabled=True)
    jit_update = base.update_phase_reference_and_eqp
    args = _phase_args()

    cpu_state = (_one_quad(STATE_POWDER), _one_quad(300.0), _one_quad(0.25))
    jit_state = tuple(field.copy() for field in cpu_state)
    expected = (
        (LIQUIDUS_K + 10.0, STATE_LIQUID, 300.0, False, False),
        (SOLIDUS_K - 10.0, STATE_SOLID, SOLIDUS_K - 10.0, True, False),
        (LIQUIDUS_K + 20.0, STATE_SOLID, SOLIDUS_K - 10.0, False, True),
        (SOLIDUS_K - 30.0, STATE_SOLID, SOLIDUS_K - 10.0, False, False),
    )

    for temperature, phase_expected, reference_expected, newly, entered in expected:
        cpu_result, cpu_state = _advance(
            cpu_update,
            temperature,
            cpu_state,
            args,
        )
        jit_result, jit_state = _advance(
            jit_update,
            temperature,
            jit_state,
            args,
        )

        for cpu_field, jit_field in zip(cpu_result, jit_result):
            np.testing.assert_array_equal(
                np.asarray(jit_field),
                np.asarray(cpu_field),
            )
        np.testing.assert_array_equal(
            np.asarray(cpu_result[0]),
            [[[phase_expected]]],
        )
        np.testing.assert_allclose(
            np.asarray(cpu_result[1]),
            [[[reference_expected]]],
        )
        np.testing.assert_allclose(np.asarray(cpu_result[2]), [[[0.25]]])
        assert bool(np.asarray(cpu_result[3])[0, 0, 0]) is newly
        assert bool(np.asarray(cpu_result[4])[0, 0, 0]) is entered


def test_lifecycle_ignores_legacy_reset_and_relaxation_side_effects():
    base = _base_module()
    runner.REGISTRY.reset()

    class BuildProblem:
        def __init__(self):
            self.eps_p = jnp.ones((1, 1, 3, 3), dtype=jnp.float64)
            self.reset_calls = 0

        def reset_remelted_state(self, remelted):
            self.reset_calls += 1
            self.eps_p = jnp.where(
                remelted[..., 0, None, None],
                0.0,
                self.eps_p,
            )

    build = BuildProblem()
    runner.REGISTRY.build_problem = build
    assert runner.install_phase_lifecycle_wrapper(base)
    args = _phase_args()
    state = (_one_quad(STATE_POWDER), _one_quad(300.0), _one_quad(0.25))

    first_melt, state = _advance(
        base.update_phase_reference_and_eqp,
        LIQUIDUS_K + 10.0,
        state,
        args,
    )
    assert not bool(np.asarray(runner.REGISTRY.pending_reference).any())

    first_solidification, state = _advance(
        base.update_phase_reference_and_eqp,
        SOLIDUS_K - 10.0,
        state,
        args,
    )
    assert bool(np.asarray(first_solidification[3]).all())
    assert bool(np.asarray(runner.REGISTRY.pending_reference).all())
    first_reference = np.asarray(first_solidification[1]).copy()

    runner.REGISTRY.pending_reference = jnp.zeros_like(
        runner.REGISTRY.pending_reference,
        dtype=bool,
    )
    remelt, state = _advance(
        base.update_phase_reference_and_eqp,
        LIQUIDUS_K + 20.0,
        state,
        args,
    )

    assert bool(np.asarray(remelt[4]).all())
    np.testing.assert_array_equal(np.asarray(remelt[0]), [[[STATE_SOLID]]])
    np.testing.assert_allclose(np.asarray(remelt[1]), first_reference)
    np.testing.assert_allclose(np.asarray(remelt[2]), [[[0.25]]])
    np.testing.assert_array_equal(
        np.asarray(build.eps_p),
        np.ones((1, 1, 3, 3)),
    )
    assert build.reset_calls == 0
    assert not bool(np.asarray(runner.REGISTRY.pending_reference).any())
    assert not bool(np.asarray(runner.REGISTRY.relaxation_mask).any())
    assert not bool(np.asarray(runner.REGISTRY.relaxation_hot).any())


def test_jit_thermal_wrapper_forwards_new_temperature_to_canonical_fallback():
    base = _base_module()
    calls = []

    def canonical_thermal(
        T_old_quad,
        active_quad,
        phase_quad,
        args,
        tables,
        printed_quad=None,
        cooling_only_quad=None,
        T_new_quad=None,
    ):
        calls.append(
            {
                "T_old_quad": T_old_quad,
                "printed_quad": printed_quad,
                "cooling_only_quad": cooling_only_quad,
                "T_new_quad": T_new_quad,
            }
        )
        return "canonical"

    base.thermal_material_quads = canonical_thermal
    report = acceleration.ProfilingReport("thermal-history-fallback")
    assert acceleration.install_loop_kernel_jit_patch(
        base,
        profiler=report,
        enabled=True,
    )

    old_temperature = _one_quad(SOLIDUS_K - 20.0)
    new_temperature = _one_quad(LIQUIDUS_K + 20.0)
    active = _one_quad(1.0)
    cooling_only = _one_quad(0.0)
    result = base.thermal_material_quads(
        old_temperature,
        active,
        _one_quad(STATE_POWDER),
        _phase_args(),
        {key: None for key in acceleration.THERMAL_TABLE_KEYS},
        printed_quad=active,
        cooling_only_quad=cooling_only,
        T_new_quad=new_temperature,
    )

    assert result == "canonical"
    assert len(calls) == 1
    assert calls[0]["T_old_quad"] is old_temperature
    assert calls[0]["printed_quad"] is active
    assert calls[0]["cooling_only_quad"] is cooling_only
    assert calls[0]["T_new_quad"] is new_temperature
    assert report.meta["loop_kernel_jit_thermal_fallbacks"] == 1


def test_empty_table_jit_material_fields_match_canonical_paper_semantics():
    base = _base_module()
    base.thermal_material_quads = thermal_material_quads
    base.mechanics_material_quads = mechanics_material_quads
    acceleration._LOOP_KERNEL_JIT_THERMAL_CACHE.clear()
    acceleration._LOOP_KERNEL_JIT_MECHANICS_CACHE.clear()
    args = _material_args()
    thermal_tables = {
        key: None for key in acceleration.THERMAL_TABLE_KEYS
    }
    mechanical_tables = {
        key: None for key in acceleration.MECHANICAL_TABLE_KEYS
    }
    temperature = _one_quad(0.5 * (SOLIDUS_K + LIQUIDUS_K))
    inactive = _one_quad(0.0)
    physical = _one_quad(1.0)
    cooling_only = _one_quad(1.0)
    solid = _one_quad(STATE_SOLID)
    powder = _one_quad(STATE_POWDER)

    expected_thermal = thermal_material_quads(
        temperature,
        inactive,
        solid,
        args,
        thermal_tables,
        printed_quad=physical,
        cooling_only_quad=cooling_only,
    )
    expected_mechanics = mechanics_material_quads(
        temperature,
        physical,
        powder,
        args,
        mechanical_tables,
    )
    assert acceleration.install_loop_kernel_jit_patch(base, enabled=True)
    actual_thermal = base.thermal_material_quads(
        temperature,
        inactive,
        solid,
        args,
        thermal_tables,
        printed_quad=physical,
        cooling_only_quad=cooling_only,
    )
    actual_mechanics = base.mechanics_material_quads(
        temperature,
        physical,
        powder,
        args,
        mechanical_tables,
    )

    for actual, expected in zip(actual_thermal, expected_thermal):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
        )
    for actual, expected in zip(actual_mechanics, expected_mechanics):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
        )
