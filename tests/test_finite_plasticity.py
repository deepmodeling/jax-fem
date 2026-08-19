import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy.testing as onptest

from applications.finite_plasticity.model import Plasticity
from jax_fem.generate_mesh import Mesh, box_mesh


class TestFinitePlasticity(unittest.TestCase):
    def setUp(self):
        meshio_mesh = box_mesh(1, 1, 1, 1., 1., 1.)
        mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
        self.problem = Plasticity(mesh, ele_type='HEX8', vec=3, dim=3)

    def test_trial_state_is_invariant_to_relative_volume_change(self):
        """A volumetric increment must not rescale an isochoric history."""
        _, update_int_vars, _ = self.problem.get_maps()
        identity = np.eye(3)
        be_bar_old = np.diag(np.array([np.exp(0.001), np.exp(-0.001), 1.]))
        F = 1.1*identity

        _, be_bar, alpha = update_int_vars(
            F - identity, identity, be_bar_old, np.array(0.)
        )

        onptest.assert_allclose(be_bar, be_bar_old, rtol=1e-12, atol=1e-12)
        onptest.assert_allclose(alpha, 0., atol=1e-12)

    def test_plastic_return_satisfies_yield_condition(self):
        """The corrected stress must lie on the hardened yield surface."""
        _, update_int_vars, compute_cauchy_stress = self.problem.get_maps()
        identity = np.eye(3)
        F = identity.at[0, 1].set(0.02)

        _, _, alpha = update_int_vars(
            F - identity, identity, identity, np.array(0.)
        )
        sigma = compute_cauchy_stress(
            F - identity, identity, identity, np.array(0.)
        )
        tau = np.linalg.det(F)*sigma
        s = tau - np.trace(tau)/3.*identity
        yield_residual = (
            np.linalg.norm(s) - np.sqrt(2./3.)*(400. + 18.*alpha)
        )

        self.assertGreater(float(alpha), 0.)
        onptest.assert_allclose(yield_residual, 0., atol=1e-10)


if __name__ == '__main__':
    unittest.main()
