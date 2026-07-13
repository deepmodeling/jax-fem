import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "159_local"))

from v06.mechanics.j2 import (  # noqa: E402
    PlasticState,
    elastic_strain_from_stress,
    equivalent_stress,
    radial_return,
    reset_on_remelt,
)


YOUNG = 120.0e9
POISSON = 0.30
YIELD0 = 500.0e6
HARDENING = 2.0e9


def deviatoric_uniaxial(amplitude):
    return jnp.diag(jnp.asarray([amplitude, -0.5 * amplitude, -0.5 * amplitude]))


class J2MaterialPointTest(unittest.TestCase):
    def setUp(self):
        self.zero_state = PlasticState(
            eqp=jnp.asarray(0.0),
            eps_p=jnp.zeros((3, 3)),
        )

    def update(self, strain, state=None, saturation=jnp.inf):
        return radial_return(
            strain=jnp.asarray(strain),
            thermal_strain=jnp.zeros((3, 3)),
            state=self.zero_state if state is None else state,
            young=YOUNG,
            poisson=POISSON,
            yield_stress=YIELD0,
            hardening=HARDENING,
            saturation=saturation,
        )

    def test_elastic_step_does_not_change_internal_state(self):
        update = self.update(deviatoric_uniaxial(1.0e-4))

        self.assertEqual(float(update.delta_eqp), 0.0)
        np.testing.assert_array_equal(np.asarray(update.state.eps_p), np.zeros((3, 3)))
        self.assertEqual(float(update.state.eqp), 0.0)

    def test_isotropic_compliance_recovers_elastic_strain_tensor(self):
        strain = jnp.asarray(
            [[0.001, 0.0002, 0.0], [0.0002, -0.0003, 0.0], [0.0, 0.0, 0.0001]]
        )
        update = self.update(strain)

        recovered = elastic_strain_from_stress(
            update.stress,
            YOUNG,
            POISSON,
        )

        np.testing.assert_allclose(
            np.asarray(recovered), np.asarray(strain), rtol=1.0e-12, atol=1.0e-15
        )

    def test_unbounded_hardening_return_lies_on_updated_yield_surface(self):
        update = self.update(deviatoric_uniaxial(0.02))

        expected_yield = YIELD0 + HARDENING * float(update.state.eqp)
        residual = abs(float(equivalent_stress(update.stress)) - expected_yield)
        self.assertGreater(float(update.delta_eqp), 0.0)
        self.assertLess(residual / expected_yield, 1.0e-10)

    def test_single_increment_crossing_saturation_cap_has_no_stress_overshoot(self):
        saturation = 550.0e6
        eqp_old = 0.020
        mu = YOUNG / (2.0 * (1.0 + POISSON))
        trial_equivalent_stress = 1.40e9
        elastic_amplitude = trial_equivalent_stress / (3.0 * mu)
        state = PlasticState(
            eqp=jnp.asarray(eqp_old),
            eps_p=deviatoric_uniaxial(eqp_old),
        )
        total_strain = deviatoric_uniaxial(eqp_old + elastic_amplitude)

        update = self.update(total_strain, state=state, saturation=saturation)

        self.assertGreater(float(update.state.eqp), (saturation - YIELD0) / HARDENING)
        self.assertAlmostEqual(
            float(equivalent_stress(update.stress)) / saturation,
            1.0,
            places=10,
        )

    def test_committed_state_is_idempotent_at_unchanged_total_strain(self):
        strain = deviatoric_uniaxial(0.02)
        first = self.update(strain)
        second = self.update(strain, state=first.state)

        self.assertGreater(float(first.delta_eqp), 0.0)
        self.assertLess(float(second.delta_eqp), 1.0e-12)
        np.testing.assert_allclose(
            np.asarray(second.stress),
            np.asarray(first.stress),
            rtol=1.0e-10,
            atol=1.0e-3,
        )


class PlasticStateLifecycleTest(unittest.TestCase):
    def test_remelt_resets_eqp_and_full_plastic_strain_tensor_together(self):
        state = PlasticState(
            eqp=jnp.asarray([[0.1, 0.2], [0.3, 0.4]]),
            eps_p=jnp.arange(36.0).reshape(2, 2, 3, 3),
        )
        remelted = jnp.asarray([[False, True], [True, False]])

        reset = reset_on_remelt(state, remelted)

        np.testing.assert_array_equal(
            np.asarray(reset.eqp),
            np.asarray([[0.1, 0.0], [0.0, 0.4]]),
        )
        np.testing.assert_array_equal(np.asarray(reset.eps_p[0, 1]), np.zeros((3, 3)))
        np.testing.assert_array_equal(np.asarray(reset.eps_p[1, 0]), np.zeros((3, 3)))
        np.testing.assert_array_equal(
            np.asarray(reset.eps_p[0, 0]),
            np.asarray(state.eps_p[0, 0]),
        )


if __name__ == "__main__":
    unittest.main()
