import unittest

import jax
import jax.numpy as np
import numpy.testing as onptest

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import ad_wrapper, solver
from applications.phase_field_fracture.model import PhaseField, get_elasticity_maps


class TestAT2(unittest.TestCase):
    def setUp(self):
        self.lmbda, self.mu = 121.15384615384615, 80.76923076923077
        self.energy, self.stress = get_elasticity_maps(self.lmbda, self.mu)

    def test_zero_strain_and_compression(self):
        damage = np.array([0.6])
        onptest.assert_array_equal(self.stress(np.zeros((2, 2)), damage), 0.)
        epsilon = -1e-3*np.eye(2)
        expected = self.lmbda*np.trace(epsilon)*np.eye(2) + 2.*self.mu*epsilon
        onptest.assert_allclose(self.stress(epsilon, damage), expected, atol=1e-14)
        self.assertEqual(float(self.energy(epsilon)), 0.)

    def test_repeated_eigenvalue_tangents(self):
        damage = np.array([0.4])
        direction = np.array([[0.4, 0.2], [-0.1, -0.3]])
        for value in [0., 1e-3, -1e-3]:
            with self.subTest(principal_strain=value):
                epsilon = value*np.eye(2)
                _, tangent = jax.jvp(lambda e: self.stress(e, damage),
                                     (epsilon,), (direction,))
                self.assertTrue(bool(np.all(np.isfinite(tangent))))
                if value != 0.:  # The tension/compression split has a kink at zero.
                    h = 1e-7
                    fd = (self.stress(epsilon + h*direction, damage)
                          - self.stress(epsilon - h*direction, damage))/(2.*h)
                    onptest.assert_allclose(tangent, fd, rtol=1e-7, atol=1e-8)

    def test_damage_derivative(self):
        epsilon = 1e-3*np.eye(2)
        sigma_elastic = self.lmbda*np.trace(epsilon)*np.eye(2) + 2.*self.mu*epsilon
        for value in [0., 0.4, 1.]:
            with self.subTest(damage=value):
                damage = np.array([value])
                expected = -2.*(1. - value)*sigma_elastic
                derivative = jax.jacfwd(self.stress, argnums=1)(epsilon, damage)[..., 0]
                onptest.assert_allclose(derivative, expected, atol=1e-12)
                reverse = jax.grad(lambda d: np.trace(self.stress(epsilon, d)))(damage)
                onptest.assert_allclose(reverse[0], np.trace(expected), atol=1e-12)

        damage = np.array([1. - 1e-9])
        onptest.assert_allclose(self.stress(epsilon, damage),
                                (1. - damage[0])**2*sigma_elastic, rtol=1e-12, atol=0.)

    def test_homogeneous_damage_and_sensitivity(self):
        G_c, length_scale = 2.7e-3, 0.0075
        mesh = Mesh(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                    np.array([[0, 1, 2, 3]]))
        problem = PhaseField(mesh, vec=1, dim=2, ele_type='QUAD4',
                             additional_info=(G_c, length_scale))
        history_shape = (problem.num_cells, problem.fes[0].num_quads)
        for history in [0., 1e-8, 0.01, 0.1]:
            with self.subTest(history=history):
                problem.set_params(np.full(history_shape, history))
                sol = solver(problem, {'spsolve_solver': {}, 'tol': 1e-12})[0]
                expected = 2.*history/(G_c/length_scale + 2.*history)
                onptest.assert_allclose(sol, expected, atol=1e-12)

        forward = ad_wrapper(problem, {'spsolve_solver': {}}, {'spsolve_solver': {}})

        def mean_damage(history):
            return np.mean(forward(np.full(history_shape, history))[0])

        history = 0.04
        expected_grad = 2.*(G_c/length_scale)/(G_c/length_scale + 2.*history)**2
        onptest.assert_allclose(jax.grad(mean_damage)(history), expected_grad, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
