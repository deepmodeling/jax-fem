import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem_am.physics.thermal import TransientThermal


ABSORBED_POWER_W = 125.0
BEAM_RADIUS_M = 50.0e-6


def _paper_center_density(power: float, radius: float) -> float:
    return 6.0 * math.sqrt(3.0) * power / (
        radius**3 * math.pi * math.sqrt(math.pi)
    )


def _source_density(point, center=(0.0, 0.0, 0.0)) -> float:
    problem = object.__new__(TransientThermal)
    problem.plane_axis0_id = 0
    problem.plane_axis1_id = 1
    problem.build_axis_id = 2
    problem.build_sign = 1.0
    problem.front_surface_loss_h = 0.0
    problem.front_surface_loss_thickness = 0.0
    problem.front_surface_loss_radiation = False
    problem.ambient = 423.15
    # The current implementation ignores this requested model and therefore
    # exposes the plane-Gaussian x exponential-depth mismatch as a RED test.
    problem.source_model = "paper_hemispherical"

    mass_map = problem.get_mass_map()
    temperature = jnp.asarray([423.15])
    zero = jnp.asarray([0.0])
    residual = mass_map(
        temperature,
        jnp.asarray(point),
        temperature,
        jnp.asarray([1.0]),
        jnp.asarray(center),
        jnp.asarray([ABSORBED_POWER_W]),
        jnp.asarray([BEAM_RADIUS_M]),
        jnp.asarray([BEAM_RADIUS_M]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        zero,
        zero,
        zero,
    )
    return -float(residual[0])


@pytest.mark.xfail(
    strict=True,
    reason="T015 pending: current source is plane Gaussian x exponential depth",
)
def test_paper_source_matches_equation_1_center_value():
    expected = _paper_center_density(ABSORBED_POWER_W, BEAM_RADIUS_M)

    assert math.isclose(
        _source_density((0.0, 0.0, 0.0)),
        expected,
        rel_tol=1e-12,
    )


@pytest.mark.xfail(
    strict=True,
    reason="T015 pending: current source has different radial and depth decay",
)
def test_paper_source_has_equal_radial_and_depth_decay():
    radial = _source_density((BEAM_RADIUS_M, 0.0, 0.0))
    depth = _source_density((0.0, 0.0, -BEAM_RADIUS_M))

    assert math.isclose(radial, depth, rel_tol=1e-12)
    assert math.isclose(
        radial / _source_density((0.0, 0.0, 0.0)),
        math.exp(-3.0),
        rel_tol=1e-12,
    )


def test_paper_source_is_zero_above_the_active_layer():
    assert math.isclose(
        _source_density((0.0, 0.0, BEAM_RADIUS_M)),
        0.0,
        abs_tol=1e-15,
    )


def test_paper_source_is_translation_invariant():
    center = (7.0e-4, 2.0e-4, 4.0e-4)
    offset = (0.5 * BEAM_RADIUS_M, 0.0, -0.25 * BEAM_RADIUS_M)

    translated = tuple(center[i] + offset[i] for i in range(3))
    assert math.isclose(
        _source_density(translated, center=center),
        _source_density(offset),
        rel_tol=1e-12,
    )


@pytest.mark.xfail(
    strict=True,
    reason="T015 pending: current exponential-depth tail misses the paper-source gate",
)
def test_paper_source_integrates_to_absorbed_power_within_half_percent():
    nodes, weights = np.polynomial.legendre.leggauss(24)
    radius = BEAM_RADIUS_M
    xy = 4.0 * radius * nodes
    z_depth = 2.0 * radius * (nodes + 1.0)
    xy_weights = 4.0 * radius * weights
    z_weights = 2.0 * radius * weights
    xx, yy, dd = np.meshgrid(xy, xy, z_depth, indexing="ij")
    wx, wy, wz = np.meshgrid(xy_weights, xy_weights, z_weights, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), -dd.ravel()))
    volume_weights = (wx * wy * wz).ravel()

    densities = np.asarray(
        jax.vmap(lambda point: jnp.asarray(_source_density_jax(point)))(
            jnp.asarray(points)
        )
    )
    integrated_power = float(np.dot(densities, volume_weights))

    assert math.isclose(
        integrated_power,
        ABSORBED_POWER_W,
        rel_tol=5e-3,
    )


def _source_density_jax(point):
    problem = object.__new__(TransientThermal)
    problem.plane_axis0_id = 0
    problem.plane_axis1_id = 1
    problem.build_axis_id = 2
    problem.build_sign = 1.0
    problem.front_surface_loss_h = 0.0
    problem.front_surface_loss_thickness = 0.0
    problem.front_surface_loss_radiation = False
    problem.ambient = 423.15
    problem.source_model = "paper_hemispherical"
    mass_map = problem.get_mass_map()
    temperature = jnp.asarray([423.15])
    zero = jnp.asarray([0.0])
    residual = mass_map(
        temperature,
        point,
        temperature,
        jnp.asarray([1.0]),
        jnp.zeros(3),
        jnp.asarray([ABSORBED_POWER_W]),
        jnp.asarray([BEAM_RADIUS_M]),
        jnp.asarray([BEAM_RADIUS_M]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        zero,
        zero,
        zero,
    )
    return -residual[0]
