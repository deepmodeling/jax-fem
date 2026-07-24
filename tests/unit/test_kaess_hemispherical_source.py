import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem_am.config.schema import build_parser
from jax_fem_am.physics.thermal import TransientThermal
from jax_fem_am.verification.thermal_ledger import integrate_volume_terms


ABSORBED_POWER_W = 125.0
BEAM_RADIUS_M = 50.0e-6


def _paper_center_density(power: float, radius: float) -> float:
    return 6.0 * math.sqrt(3.0) * power / (
        radius**3 * math.pi * math.sqrt(math.pi)
    )


def _source_density(
    point,
    center=(0.0, 0.0, 0.0),
    source_model="paper_hemispherical",
) -> float:
    problem = object.__new__(TransientThermal)
    problem.plane_axis0_id = 0
    problem.plane_axis1_id = 1
    problem.build_axis_id = 2
    problem.build_sign = 1.0
    problem.front_surface_loss_h = 0.0
    problem.front_surface_loss_thickness = 0.0
    problem.front_surface_loss_radiation = False
    problem.ambient = 423.15
    problem.source_model = source_model

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


def test_paper_source_matches_equation_1_center_value():
    expected = _paper_center_density(ABSORBED_POWER_W, BEAM_RADIUS_M)

    assert math.isclose(
        _source_density((0.0, 0.0, 0.0)),
        expected,
        rel_tol=1e-12,
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


def test_source_model_cli_keeps_legacy_default_and_requires_explicit_paper_mode():
    parser = build_parser()

    assert parser.parse_args([]).source_model == "legacy"
    assert (
        parser.parse_args(["--source-model", "paper_hemispherical"]).source_model
        == "paper_hemispherical"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-model", "unknown"])


def test_legacy_gaussian_exponential_source_remains_available():
    expected = 2.0 * ABSORBED_POWER_W / (
        math.pi * BEAM_RADIUS_M**3
    )

    assert math.isclose(
        _source_density((0.0, 0.0, 0.0), source_model="legacy"),
        expected,
        rel_tol=1e-12,
    )


def test_thermal_ledger_uses_the_selected_paper_source_equation():
    terms = integrate_volume_terms(
        jxw=np.ones((1, 1)),
        points=np.zeros((1, 1, 3)),
        temperature_old=np.full((1, 1), 423.15),
        temperature_new=np.full((1, 1), 423.15),
        rho=np.ones((1, 1)),
        cp=np.ones((1, 1)),
        latent_cp=np.zeros((1, 1)),
        laser_center=np.zeros(3),
        effective_laser_power_w=ABSORBED_POWER_W,
        beam_radius_m=BEAM_RADIUS_M,
        source_depth_m=BEAM_RADIUS_M,
        laser_switch=1.0,
        active=np.ones((1, 1)),
        cooling_only=np.zeros((1, 1)),
        old_layer_cooling_h=0.0,
        ambient_k=423.15,
        dt_s=1.0,
        build_axis=2,
        plane_axes=(0, 1),
        build_sign=1.0,
        front_loss_h=0.0,
        front_loss_thickness_m=0.0,
        front_loss_radiation=False,
        emissivity=0.0,
        stefan_boltzmann=0.0,
        source_model="paper_hemispherical",
    )

    assert math.isclose(
        terms["laser_deposited_j"],
        _paper_center_density(ABSORBED_POWER_W, BEAM_RADIUS_M),
        rel_tol=1e-12,
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
