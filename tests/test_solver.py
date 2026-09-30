import unittest

import jax.numpy as np
import numpy.testing as onptest

from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver


class Reaction(Problem):
    def custom_init(self, residual):
        self.residual = residual

    def get_mass_map(self):
        return lambda u, x: self.residual(u)


class TestNewtonIterationLimit(unittest.TestCase):
    def make_problem(self, residual):
        mesh = Mesh(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                    np.array([[0, 1, 2, 3]]))
        return Reaction(mesh, vec=1, dim=2, ele_type='QUAD4',
                        additional_info=(residual,))

    def test_nonconvergent_problem_stops(self):
        # Newton's method cycles between u=0 and u=1 for this residual.
        for options, limit in [({'spsolve_solver': {}}, 100),
                               ({'spsolve_solver': {}, 'max_iters': 3}, 3),
                               ({'newton': {'linear': {'spsolve_solver': {}},
                                            'max_iters': 3}}, 3)]:
            with self.subTest(options=options):
                problem = self.make_problem(lambda u: u**3 - 2*u + 2)
                with self.assertRaisesRegex(RuntimeError,
                                            f'Newton solver did not converge in {limit} iterations'):
                    solver(problem, options)

    def test_convergence_on_last_iteration(self):
        for options in [{'spsolve_solver': {}, 'max_iters': 1},
                        {'newton': {'linear': {'spsolve_solver': {}}, 'max_iters': 1}}]:
            with self.subTest(options=options):
                problem = self.make_problem(lambda u: u - 1.)
                sol = solver(problem, options)[0]
                onptest.assert_allclose(sol, 1., rtol=0., atol=1e-12)

    def test_initially_converged_problem(self):
        problem = self.make_problem(lambda u: u)
        sol = solver(problem, {'max_iters': 0})[0]
        onptest.assert_array_equal(sol, 0.)

    def test_invalid_iteration_limit(self):
        for limit in [-1, 1.5, float('inf'), float('nan')]:
            with self.subTest(max_iters=limit):
                problem = self.make_problem(lambda u: u - 1.)
                with self.assertRaisesRegex(ValueError, 'max_iters must be a non-negative integer'):
                    solver(problem, {'max_iters': limit})


if __name__ == '__main__':
    unittest.main()
