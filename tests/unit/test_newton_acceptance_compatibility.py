"""Regression tests for the Abaqus-style Newton acceptance guard."""

import unittest
from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp
import numpy as onp

import jax_fem.solver as solver_module


class _ScalarProblem:
    """One-DOF problem whose residual drops below the strict tolerance at once."""

    def __init__(self):
        self.num_total_dofs_all_vars = 1
        self.num_vars = 1
        self.offset = [0]
        self.fes = [SimpleNamespace(
            vec=1,
            node_inds_list=[],
            vec_inds_list=[],
            vals_list=[],
        )]

    def unflatten_fn_sol_list(self, dofs):
        return [jnp.reshape(dofs, (1, 1))]

    @staticmethod
    def _residual(sol_list):
        dof = sol_list[0][0, 0]
        return [jnp.reshape(jnp.where(dof == 0.0, 1.0, 1.0e-8), (1, 1))]

    def newton_update(self, sol_list):
        return self._residual(sol_list)

    def compute_residual(self, sol_list):
        return self._residual(sol_list)


class AbaqusAcceptanceCompatibilityTest(unittest.TestCase):

    @staticmethod
    def _newton_options(**overrides):
        options = {
            "initial_guess": [jnp.zeros((1, 1))],
            "tol": 0.0,
            "rel_tol": 5.0e-5,
            "max_iter": 2,
            "residual_only_check": True,
            "acceptance": {
                "force_frac": 0.005,
                "disp_frac": 0.01,
                "fallback_frac": 0.02,
                "fallback_after": 9,
            },
        }
        options.update(overrides)
        return options

    def test_strict_residual_convergence_remains_an_acceptance_exit(self):
        """Abaqus fallback must not reject an already strictly converged state.

        The synthetic correction remains large relative to the total increment,
        so the displacement-correction criterion alone cannot pass.  This is the
        R4 failure shape: the configured residual tolerance is satisfied, but
        the optional Abaqus-style criteria otherwise run to ``max_iter``.
        """
        problem = _ScalarProblem()

        def fake_newton_step(problem, res_vec, A, dofs, cfg, timing):
            del problem, res_vec, A, cfg, timing
            return dofs + 1.0, 0.0

        newton = self._newton_options()

        with (
            mock.patch.object(solver_module, "get_A", return_value=onp.eye(1)),
            mock.patch.object(solver_module, "newton_step", side_effect=fake_newton_step),
        ):
            solution = solver_module.solver(
                problem, solver_options={"newton": newton})

        self.assertAlmostEqual(float(solution[0][0, 0]), 1.0)

    def test_failure_reports_force_and_displacement_acceptance_ratios(self):
        problem = _ScalarProblem()
        problem._residual = lambda sol_list: [jnp.ones((1, 1))]

        def fake_newton_step(problem, res_vec, A, dofs, cfg, timing):
            del problem, res_vec, A, cfg, timing
            return dofs + 1.0, 0.0

        with (
            mock.patch.object(solver_module, "get_A", return_value=onp.eye(1)),
            mock.patch.object(solver_module, "newton_step", side_effect=fake_newton_step),
            self.assertRaises(RuntimeError) as raised,
        ):
            solver_module.solver(
                problem,
                solver_options={"newton": self._newton_options(max_iter=1)},
            )

        message = str(raised.exception)
        self.assertIn("force_ratio=", message)
        self.assertIn("displacement_correction_ratio=", message)


if __name__ == "__main__":
    unittest.main()
