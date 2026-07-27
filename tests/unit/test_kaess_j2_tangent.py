"""Kaess P0-J2 material-point and consistent-tangent contracts.

This is the T012 RED slice for T019.  It deliberately separates:

* capabilities already present in the canonical tensor J2 return map
  (path-dependent load cycling and an AD/finite-difference V-shaped valley);
* paper-parity gaps (multi-temperature nonlinear flow curves and the second,
  divergent J2 implementation in ``physics.mechanics``).

The failing assertions are physical/constitutive differences, not import or
runtime-environment failures.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import jax_fem_am.materials.j2 as j2_model
from jax_fem_am.materials.j2 import PlasticState, radial_return
from jax_fem_am.physics.mechanics import ThermoMechanical


jax.config.update("jax_enable_x64", True)

YOUNG = 120.0e9
POISSON = 0.30
YIELD0 = 500.0e6
HARDENING = 2.0e9
ZERO_STATE = PlasticState(eqp=jnp.asarray(0.0), eps_p=jnp.zeros((3, 3)))


def _uniaxial_strain(value):
    return jnp.diag(jnp.asarray([value, 0.0, 0.0]))


def _material_update(strain, state=ZERO_STATE, *, saturation=jnp.inf):
    return radial_return(
        strain=jnp.asarray(strain),
        thermal_strain=jnp.zeros((3, 3)),
        state=state,
        young=YOUNG,
        poisson=POISSON,
        yield_stress=YIELD0,
        hardening=HARDENING,
        saturation=saturation,
    )


def test_uniaxial_loading_unloading_reloading_preserves_committed_history():
    """Baseline GREEN: the canonical tensor return map has path memory."""
    peak_strain = 8.0e-3
    loaded = _material_update(_uniaxial_strain(peak_strain))
    unloaded = _material_update(_uniaxial_strain(0.0), loaded.state)
    reloaded = _material_update(_uniaxial_strain(peak_strain), unloaded.state)
    extended = _material_update(
        _uniaxial_strain(1.2 * peak_strain), unloaded.state
    )

    assert float(loaded.delta_eqp) > 0.0
    assert float(unloaded.delta_eqp) == 0.0
    assert float(unloaded.state.eqp) == float(loaded.state.eqp)
    assert float(unloaded.stress[0, 0]) < 0.0
    assert float(reloaded.delta_eqp) < 1.0e-12
    np.testing.assert_allclose(
        np.asarray(reloaded.stress),
        np.asarray(loaded.stress),
        rtol=1.0e-11,
        atol=1.0e-3,
    )
    assert float(extended.state.eqp) > float(unloaded.state.eqp)


def test_multitemperature_nonlinear_flow_curve_is_bilinearly_interpolated():
    """RED: the solver needs a generic multi-point flow-curve capability.

    Kaess Figure 4(b) exposes temperature-dependent flow curves, but not the
    original Abaqus table.  This synthetic nonlinear grid verifies solver
    capability without claiming that these particular points were published.
    """
    interpolate = getattr(j2_model, "flow_stress_from_curve", None)
    assert callable(interpolate), (
        "P0-J2 missing multi-point flow-curve kernel: expected "
        "jax_fem_am.materials.j2.flow_stress_from_curve; the current "
        "yield(T) + linear-hardening(T) pair cannot preserve multiple "
        "stress-versus-plastic-strain points at each temperature"
    )

    temperatures = jnp.asarray([300.0, 800.0, 1300.0])
    plastic_strains = jnp.asarray([0.0, 0.02, 0.10])
    flow_stresses = 1.0e6 * jnp.asarray(
        [
            [500.0, 560.0, 610.0],
            [350.0, 390.0, 420.0],
            [100.0, 110.0, 115.0],
        ]
    )

    # Exact tabulated points must be preserved at all three temperatures.
    actual_at_knots = np.asarray(
        [
            interpolate(
                temperature,
                0.02,
                temperatures,
                plastic_strains,
                flow_stresses,
            )
            for temperature in temperatures
        ]
    )
    np.testing.assert_allclose(
        actual_at_knots,
        np.asarray([560.0e6, 390.0e6, 110.0e6]),
        rtol=1.0e-13,
        atol=1.0e-6,
    )

    # Midway in both temperature and plastic strain: bilinear interpolation.
    midpoint = interpolate(
        550.0,
        0.01,
        temperatures,
        plastic_strains,
        flow_stresses,
    )
    np.testing.assert_allclose(
        np.asarray(midpoint),
        np.asarray(450.0e6),
        rtol=1.0e-13,
        atol=1.0e-6,
    )


def test_flow_curve_clamps_both_axes_and_remains_jittable():
    """RED: endpoint behavior must be explicit and usable inside JAX kernels."""
    interpolate = getattr(j2_model, "flow_stress_from_curve", None)
    assert callable(interpolate)

    temperatures = jnp.asarray([300.0, 800.0])
    plastic_strains = jnp.asarray([0.0, 0.10])
    flow_stresses = 1.0e6 * jnp.asarray(
        [
            [500.0, 600.0],
            [300.0, 340.0],
        ]
    )
    evaluate = jax.jit(
        lambda temperature, eqp: interpolate(
            temperature,
            eqp,
            temperatures,
            plastic_strains,
            flow_stresses,
        )
    )

    actual = np.asarray(
        [
            evaluate(100.0, -0.05),
            evaluate(100.0, 0.20),
            evaluate(1200.0, -0.05),
            evaluate(1200.0, 0.20),
        ]
    )
    np.testing.assert_allclose(
        actual,
        1.0e6 * np.asarray([500.0, 600.0, 300.0, 340.0]),
        rtol=1.0e-13,
        atol=1.0e-6,
    )


def test_mechanics_residual_and_canonical_return_map_share_one_j2_source():
    """RED: the duplicate mechanics formula mishandles a cap crossing.

    ``radial_return`` solves a within-increment crossing of the saturation cap
    exactly.  ``ThermoMechanical.stress_fn`` currently chooses the hardening
    modulus only from the old state, so its residual and its AD tangent remain
    linearly hardening after the same increment has crossed the cap.
    """
    mechanics = object.__new__(ThermoMechanical)
    mechanics.dim = 3
    mechanics.mechanics_model = "j2_plastic"
    mechanics.yield_saturation = 550.0e6
    direction = jnp.diag(jnp.asarray([1.0, -0.5, -0.5]))

    def mechanics_residual(amplitude):
        stress = mechanics.stress_fn(
            amplitude * direction,
            jnp.asarray([300.0]),
            jnp.asarray([0.0]),
            jnp.asarray([1.0]),
            jnp.asarray([YOUNG]),
            jnp.asarray([0.0]),
            jnp.asarray([POISSON]),
            jnp.asarray([YIELD0]),
            jnp.asarray([HARDENING]),
            jnp.asarray([0.0]),
        )
        return stress[0, 0]

    def canonical_residual(amplitude):
        update = _material_update(
            amplitude * direction,
            saturation=550.0e6,
        )
        return update.stress[0, 0]

    amplitude = 3.0e-2
    mechanics_stress = float(mechanics_residual(amplitude))
    canonical_stress = float(canonical_residual(amplitude))
    mechanics_tangent = float(jax.grad(mechanics_residual)(amplitude))
    canonical_tangent = float(jax.grad(canonical_residual)(amplitude))
    stress_relative_error = abs(mechanics_stress - canonical_stress) / abs(
        canonical_stress
    )
    tangent_relative_error = abs(
        mechanics_tangent - canonical_tangent
    ) / YOUNG

    assert stress_relative_error < 1.0e-10 and tangent_relative_error < 1.0e-10, (
        "P0-J2 residual/tangent source mismatch at a saturation crossing: "
        f"mechanics stress={mechanics_stress:.9e}, "
        f"canonical stress={canonical_stress:.9e}, "
        f"relative stress error={stress_relative_error:.3e}; "
        f"mechanics AD tangent={mechanics_tangent:.9e}, "
        f"canonical AD tangent={canonical_tangent:.9e}, "
        f"|delta tangent|/E={tangent_relative_error:.3e}"
    )


def test_canonical_j2_tangent_has_a_finite_difference_v_shaped_valley():
    """Baseline GREEN: AD and residual agree away from a branch transition."""
    strain = jnp.asarray(
        [
            [0.012, 0.003, 0.001],
            [0.003, -0.005, 0.002],
            [0.001, 0.002, -0.002],
        ]
    )
    direction = jnp.asarray(
        [
            [0.7, -0.2, 0.3],
            [-0.2, -0.1, 0.1],
            [0.3, 0.1, -0.6],
        ]
    )
    state = PlasticState(
        eqp=jnp.asarray(0.001),
        eps_p=jnp.asarray(
            [
                [0.001, 0.0001, 0.0],
                [0.0001, -0.0005, 0.0],
                [0.0, 0.0, -0.0005],
            ]
        ),
    )

    def residual(candidate_strain):
        return radial_return(
            strain=candidate_strain,
            thermal_strain=jnp.zeros((3, 3)),
            state=state,
            young=YOUNG,
            poisson=POISSON,
            yield_stress=YIELD0,
            hardening=HARDENING,
        ).stress

    tangent = jax.jacfwd(residual)(strain)
    tangent_action = jnp.einsum("ijkl,kl->ij", tangent, direction)
    step_sizes = np.logspace(-2, -11, 10)
    errors = []
    for step_size in step_sizes:
        finite_difference = (
            residual(strain + step_size * direction)
            - residual(strain - step_size * direction)
        ) / (2.0 * step_size)
        errors.append(
            float(
                jnp.linalg.norm(finite_difference - tangent_action)
                / jnp.linalg.norm(tangent_action)
            )
        )

    errors = np.asarray(errors)
    valley_index = int(np.argmin(errors))
    assert 1 < valley_index < len(errors) - 1, (
        f"finite-difference minimum is not an interior V-shaped valley: {errors}"
    )
    assert errors[valley_index] < 1.0e-8, (
        f"consistent-tangent valley floor is too high: {errors}"
    )
    assert errors[0] > 1.0e-3
    assert errors[-1] > 100.0 * errors[valley_index]
