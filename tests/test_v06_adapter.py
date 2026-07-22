import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
V03_PATH = (
    ROOT / "159_local" / "v03"
    / "am_thermal_stress_macro_intersection_mech100.py"
)
sys.path.insert(0, str(ROOT / "159_local"))
sys.path.insert(0, str(ROOT / "159_local" / "v01"))

from v06 import driver  # noqa: E402
from jax_fem_am.materials.j2 import equivalent_stress  # noqa: E402


def load_fresh_v03(name):
    spec = importlib.util.spec_from_file_location(name, V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class V06AdapterTest(unittest.TestCase):
    def setUp(self):
        driver.REGISTRY.reset()

    def test_installed_return_map_uses_exact_saturation_crossing(self):
        base = load_fresh_v03("v03_v06_saturation_test")
        driver.install_v06_adapter(base)
        cls = base.ThermoMechanical
        young = 120.0e9
        poisson = 0.3
        shear = young / (2.0 * (1.0 + poisson))
        eqp_old = 0.02
        saturation = 550.0e6
        trial_q = 1.4e9
        elastic_amplitude = trial_q / (3.0 * shear)
        eps_p_old = jnp.diag(jnp.asarray([eqp_old, -eqp_old / 2, -eqp_old / 2]))
        total_strain = jnp.diag(
            jnp.asarray([
                eqp_old + elastic_amplitude,
                -(eqp_old + elastic_amplitude) / 2,
                -(eqp_old + elastic_amplitude) / 2,
            ])
        )
        fake = SimpleNamespace(
            mechanics_model="j2_plastic",
            dim=3,
            yield_saturation=saturation,
        )

        stress, delta_eqp, _ = cls._return_map(
            fake,
            total_strain,
            jnp.asarray([0.0]),
            jnp.asarray([young]),
            jnp.asarray([0.0]),
            jnp.asarray([poisson]),
            jnp.asarray([500.0e6]),
            jnp.asarray([2.0e9]),
            jnp.asarray([eqp_old]),
            eps_p_old,
            jnp.zeros((3, 3)),
        )

        self.assertGreater(float(delta_eqp), 0.0)
        self.assertAlmostEqual(
            float(equivalent_stress(stress)) / saturation,
            1.0,
            places=10,
        )

    def test_linear_elastic_mode_ignores_plastic_saturation(self):
        base = load_fresh_v03("v03_v06_linear_elastic_test")
        driver.install_v06_adapter(base)
        cls = base.ThermoMechanical
        young = 120.0e9
        poisson = 0.3
        strain = jnp.diag(jnp.asarray([0.02, -0.01, -0.01]))
        fake = SimpleNamespace(
            mechanics_model="linear_elastic",
            dim=3,
            yield_saturation=1.15e9,
        )

        stress, delta_eqp, delta_eps_p = cls._return_map(
            fake,
            strain,
            jnp.asarray([0.0]),
            jnp.asarray([young]),
            jnp.asarray([0.0]),
            jnp.asarray([poisson]),
            jnp.asarray([500.0e6]),
            jnp.asarray([2.0e9]),
            jnp.asarray([0.0]),
            jnp.zeros((3, 3)),
            jnp.zeros((3, 3)),
        )

        expected_q = 3.0 * young / (2.0 * (1.0 + poisson)) * 0.02
        self.assertAlmostEqual(
            float(equivalent_stress(stress)) / expected_q,
            1.0,
            places=10,
        )
        self.assertEqual(float(delta_eqp), 0.0)
        np.testing.assert_array_equal(np.asarray(delta_eps_p), np.zeros((3, 3)))

    def test_birth_reference_removes_preexisting_configuration_strain(self):
        base = load_fresh_v03("v03_v06_birth_reference_test")
        driver.install_v06_adapter(base)
        cls = base.ThermoMechanical
        strain = jnp.diag(jnp.asarray([0.01, -0.005, -0.005]))
        fake = SimpleNamespace(
            mechanics_model="linear_elastic",
            dim=3,
            yield_saturation=1.15e9,
        )

        stress, delta_eqp, _ = cls._return_map(
            fake,
            strain,
            jnp.asarray([0.0]),
            jnp.asarray([110.0e9]),
            jnp.asarray([0.0]),
            jnp.asarray([0.3]),
            jnp.asarray([800.0e6]),
            jnp.asarray([1.0e9]),
            jnp.asarray([0.0]),
            jnp.zeros((3, 3)),
            strain,
        )

        np.testing.assert_allclose(np.asarray(stress), 0.0, atol=1.0e-7)
        self.assertEqual(float(delta_eqp), 0.0)

    def test_phase_update_resets_tensor_history_when_material_remelts(self):
        base = load_fresh_v03("v03_v06_remelt_test")
        driver.install_v06_adapter(base)

        class BuildProblem:
            def __init__(self):
                self._eps_p_state = jnp.ones((1, 1, 3, 3))

            def reset_remelted_state(self, remelted):
                self._eps_p_state = jnp.where(
                    remelted[..., 0, None, None],
                    0.0,
                    self._eps_p_state,
                )

        build = BuildProblem()
        driver.REGISTRY.build_problem = build
        driver.REGISTRY.eps_p = build._eps_p_state
        driver.REGISTRY.eqp = jnp.full((1, 1, 1), 0.25)
        args = SimpleNamespace(
            liquidus_temperature=1900.0,
            solidus_temperature=1800.0,
            stress_relaxation_temperature=1100.0,
            reset_plastic_on_melt=True,
        )

        result = base.update_phase_reference_and_eqp(
            jnp.asarray([[[1950.0]]]),
            jnp.asarray([[[1.0]]]),
            jnp.asarray([[[base.STATE_SOLID]]]),
            jnp.asarray([[[1100.0]]]),
            jnp.asarray([[[0.25]]]),
            args,
        )

        np.testing.assert_array_equal(np.asarray(result[2]), np.zeros((1, 1, 1)))
        np.testing.assert_array_equal(
            np.asarray(driver.REGISTRY.eqp),
            np.zeros((1, 1, 1)),
        )
        np.testing.assert_array_equal(
            np.asarray(build._eps_p_state),
            np.zeros((1, 1, 3, 3)),
        )
        self.assertTrue(bool(np.asarray(driver.REGISTRY.pending_reference).all()))
        self.assertTrue(bool(np.asarray(driver.REGISTRY.relaxation_mask).all()))

    def test_lifecycle_wrapper_is_restored_after_v04_replaces_phase_kernel(self):
        base = load_fresh_v03("v03_v06_phase_refresh_test")
        driver.install_v06_adapter(base)
        original = base._v06_original_phase_update
        called = {"jit": 0}

        def replacement(*args, **kwargs):
            called["jit"] += 1
            return original(*args, **kwargs)

        base.update_phase_reference_and_eqp = replacement
        self.assertTrue(driver.install_phase_lifecycle_wrapper(base))
        args = SimpleNamespace(
            liquidus_temperature=1900.0,
            solidus_temperature=1800.0,
            stress_relaxation_temperature=1100.0,
            reset_plastic_on_melt=True,
        )

        base.update_phase_reference_and_eqp(
            jnp.asarray([[[1950.0]]]),
            jnp.asarray([[[1.0]]]),
            jnp.asarray([[[base.STATE_SOLID]]]),
            jnp.asarray([[[1100.0]]]),
            jnp.asarray([[[0.0]]]),
            args,
        )

        self.assertEqual(called["jit"], 1)
        self.assertTrue(bool(np.asarray(driver.REGISTRY.pending_reference).all()))

    def test_reference_event_forces_mechanics_but_respects_disabled_mode(self):
        base = load_fresh_v03("v03_v06_event_cadence_test")
        driver.install_v06_adapter(base)
        driver.REGISTRY.pending_reference = jnp.ones((1, 1, 1), dtype=bool)

        self.assertTrue(
            base.should_run_mechanics(1, SimpleNamespace(mechanics_every=100))
        )
        self.assertFalse(
            base.should_run_mechanics(1, SimpleNamespace(mechanics_every=0))
        )

    def test_continuously_hot_material_forces_only_the_threshold_crossing(self):
        base = load_fresh_v03("v03_v06_hot_crossing_test")
        driver.install_v06_adapter(base)
        args = SimpleNamespace(
            liquidus_temperature=1900.0,
            solidus_temperature=1800.0,
            stress_relaxation_temperature=1100.0,
            reset_plastic_on_melt=False,
        )
        state = (
            jnp.asarray([[[1200.0]]]),
            jnp.asarray([[[1.0]]]),
            jnp.asarray([[[base.STATE_SOLID]]]),
            jnp.asarray([[[1100.0]]]),
            jnp.asarray([[[0.0]]]),
            args,
        )

        base.update_phase_reference_and_eqp(*state)
        self.assertTrue(bool(np.asarray(driver.REGISTRY.pending_reference).all()))
        driver.REGISTRY.pending_reference = jnp.zeros((1, 1, 1), dtype=bool)
        base.update_phase_reference_and_eqp(*state)

        self.assertFalse(bool(np.asarray(driver.REGISTRY.pending_reference).any()))
        self.assertTrue(bool(np.asarray(driver.REGISTRY.relaxation_hot).all()))

    def test_mechanics_event_wrapper_is_restored_after_v04_cache_patch(self):
        base = load_fresh_v03("v03_v06_mechanics_refresh_test")
        driver.install_v06_adapter(base)
        base.should_run_mechanics = lambda _step, _args: False

        self.assertTrue(driver.install_mechanics_event_wrapper(base))
        driver.REGISTRY.pending_reference = jnp.ones((1, 1, 1), dtype=bool)
        self.assertTrue(
            base.should_run_mechanics(7, SimpleNamespace(mechanics_every=20))
        )

    def test_pending_birth_event_is_captured_before_next_mechanics_solve(self):
        base = load_fresh_v03("v03_v06_pending_reference_test")
        base.run_mechanics = lambda _problem, u_guess, _params, _overrides=None: u_guess
        driver.install_v06_adapter(base)
        build = object.__new__(base.ThermoMechanical)
        build.mechanics_model = "linear_elastic"
        build.dim = 3
        build._eps_p_state = jnp.zeros((1, 1, 3, 3))
        build._eps_ref_state = jnp.zeros((1, 1, 3, 3))
        build._relaxation_mask = jnp.ones((1, 1, 1), dtype=bool)
        total_strain = jnp.diag(jnp.asarray([0.01, -0.005, -0.005]))
        build._u_grads = lambda _sol: total_strain[None, None, :, :]
        driver.REGISTRY.pending_reference = jnp.ones((1, 1, 1), dtype=bool)
        driver.REGISTRY.relaxation_mask = jnp.ones((1, 1, 1), dtype=bool)
        params = [jnp.zeros((1, 1, 1))] * 8 + [jnp.zeros((1, 1, 1))]

        base.run_mechanics(build, [jnp.zeros((1, 3))], params)

        np.testing.assert_allclose(
            np.asarray(build._eps_ref_state[0, 0]),
            np.asarray(total_strain),
        )
        self.assertFalse(bool(np.asarray(driver.REGISTRY.pending_reference).any()))

    def test_release_solve_adopts_and_commits_build_state(self):
        base = load_fresh_v03("v03_v06_release_test")
        base.run_mechanics = lambda _problem, u_guess, _params, _overrides=None: u_guess
        driver.install_v06_adapter(base)
        release = object.__new__(base.ThermoMechanical)
        release.mechanics_model = "j2_plastic"
        release._eps_p_state = jnp.zeros((1, 1, 3, 3))
        release._eps_ref_state = jnp.zeros((1, 1, 3, 3))
        release._relaxation_mask = jnp.zeros((1, 1, 1), dtype=bool)
        release.compute_eqp_update = (
            lambda _sol, params: jnp.full_like(params[-1], 0.25)
        )
        driver.REGISTRY.build_problem = object()
        driver.REGISTRY.eps_p = jnp.ones((1, 1, 3, 3))
        driver.REGISTRY.eps_ref = 2.0 * jnp.ones((1, 1, 3, 3))
        driver.REGISTRY.relaxation_mask = jnp.ones((1, 1, 1), dtype=bool)
        params = [None] * 8 + [jnp.zeros((1, 1, 1))]
        u_guess = [jnp.zeros((1, 3))]

        base.run_mechanics(release, u_guess, params)

        np.testing.assert_array_equal(
            np.asarray(release._eps_p_state),
            np.ones((1, 1, 3, 3)),
        )
        np.testing.assert_array_equal(
            np.asarray(release._eps_ref_state),
            2.0 * np.ones((1, 1, 3, 3)),
        )
        np.testing.assert_array_equal(
            np.asarray(params[-1]),
            np.full((1, 1, 1), 0.25),
        )
        np.testing.assert_array_equal(
            np.asarray(driver.REGISTRY.eqp),
            np.full((1, 1, 1), 0.25),
        )

    def test_release_rejects_tensor_state_shape_mismatch(self):
        base = load_fresh_v03("v03_v06_release_shape_test")
        base.run_mechanics = lambda _problem, u_guess, _params, _overrides=None: u_guess
        driver.install_v06_adapter(base)
        release = object.__new__(base.ThermoMechanical)
        release.mechanics_model = "linear_elastic"
        release._eps_p_state = jnp.zeros((1, 1, 3, 3))
        release._eps_ref_state = jnp.zeros((1, 1, 3, 3))
        release._relaxation_mask = jnp.zeros((1, 1, 1), dtype=bool)
        driver.REGISTRY.build_problem = object()
        driver.REGISTRY.eps_p = jnp.zeros((2, 1, 3, 3))
        driver.REGISTRY.eps_ref = jnp.zeros((1, 1, 3, 3))
        driver.REGISTRY.relaxation_mask = jnp.zeros((1, 1, 1), dtype=bool)
        params = [None] * 8 + [jnp.zeros((1, 1, 1))]

        with self.assertRaisesRegex(ValueError, "eps_p.*shape"):
            base.run_mechanics(release, [jnp.zeros((1, 3))], params)

    def test_vtu_cell_info_contains_elastic_plastic_and_reference_tensors(self):
        base = load_fresh_v03("v03_v06_output_state_test")
        driver.install_v06_adapter(base)
        driver.REGISTRY.eps_p = 2.0 * jnp.ones((1, 1, 3, 3))
        driver.REGISTRY.eps_ref = 3.0 * jnp.ones((1, 1, 3, 3))
        elastic = 4.0 * jnp.ones((1, 1, 3, 3))

        infos = dict(
            base.make_quad_stress_cell_infos(
                {
                    "stress_quad": jnp.zeros((1, 1, 3, 3)),
                    "vm_quad": jnp.zeros((1, 1)),
                    "elastic_strain_quad": elastic,
                }
            )
        )

        self.assertEqual(float(infos["elastic_strain_quad_xx"][0]), 4.0)
        self.assertEqual(float(infos["eps_p_quad_xy"][0]), 2.0)
        self.assertEqual(float(infos["eps_ref_quad_zz"][0]), 3.0)

    def test_vtu_output_rejects_mismatched_tensor_state(self):
        base = load_fresh_v03("v03_v06_output_shape_test")
        driver.install_v06_adapter(base)
        driver.REGISTRY.eps_p = jnp.zeros((2, 1, 3, 3))
        driver.REGISTRY.eps_ref = jnp.zeros((1, 1, 3, 3))

        with self.assertRaisesRegex(ValueError, "eps_p.*shape"):
            base.make_quad_stress_cell_infos(
                {
                    "stress_quad": jnp.zeros((1, 1, 3, 3)),
                    "vm_quad": jnp.zeros((1, 1)),
                    "elastic_strain_quad": jnp.zeros((1, 1, 3, 3)),
                }
            )


if __name__ == "__main__":
    unittest.main()
