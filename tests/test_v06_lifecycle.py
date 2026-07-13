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

from v06.mechanics.j2 import PlasticState, radial_return  # noqa: E402
from v06.mechanics.lifecycle import (  # noqa: E402
    effective_thermal_increment,
    update_stress_free_reference,
)


class MechanicalReferenceLifecycleTest(unittest.TestCase):
    def test_relaxation_managed_material_carries_no_positive_hot_increment(self):
        dT = jnp.asarray([[[200.0]], [[-50.0]], [[200.0]]])
        managed = jnp.asarray([[[True]], [[True]], [[False]]])

        effective = effective_thermal_increment(dT, managed)

        np.testing.assert_array_equal(
            np.asarray(effective),
            np.asarray([[[0.0]], [[-50.0]], [[200.0]]]),
        )

    def test_birth_event_captures_total_configuration_as_stress_free(self):
        total_strain = jnp.asarray(
            [[[[0.012, 0.0, 0.0], [0.0, -0.006, 0.0], [0.0, 0.0, -0.006]]]]
        )
        eps_p = jnp.asarray(
            [[[[0.002, 0.0, 0.0], [0.0, -0.001, 0.0], [0.0, 0.0, -0.001]]]]
        )
        old_reference = jnp.zeros_like(total_strain)
        thermal_strain = 0.001 * jnp.eye(3)[None, None, :, :]
        event = jnp.asarray([[[True]]])

        reference = update_stress_free_reference(
            old_reference,
            total_strain,
            thermal_strain,
            eps_p,
            event,
        )
        update = radial_return(
            strain=total_strain[0, 0] - reference[0, 0],
            thermal_strain=thermal_strain[0, 0],
            state=PlasticState(eqp=jnp.asarray(0.01), eps_p=eps_p[0, 0]),
            young=110.0e9,
            poisson=0.3,
            yield_stress=800.0e6,
            hardening=1.0e9,
        )

        np.testing.assert_allclose(np.asarray(update.stress), 0.0, atol=1.0e-6)

    def test_no_event_preserves_existing_reference(self):
        old = jnp.ones((1, 1, 3, 3))
        updated = update_stress_free_reference(
            old,
            jnp.zeros_like(old),
            jnp.zeros_like(old),
            jnp.zeros_like(old),
            jnp.asarray([[[False]]]),
        )

        np.testing.assert_array_equal(np.asarray(updated), np.ones((1, 1, 3, 3)))


if __name__ == "__main__":
    unittest.main()
