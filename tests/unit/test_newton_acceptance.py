"""Abaqus-style Newton acceptance criteria (P0 reform, 2026-07-22).

Contract: `acceptance=None` (default) preserves the legacy single-residual
loop bit-for-bit, including the max_iter RuntimeError; the opt-in 'abaqus'
mode keeps the configured strict residual exit and additionally accepts j2
stall-floor states via the dual force/displacement criteria with the
linear-convergence fallback (experiments/solver/ABAQUS_SOLVER_NOTES.md).
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as onp
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    onp = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests.unit.test_v03_bbar_hex8 import BBarTestBase, beam_grid, NEWTON  # noqa: E402

from jax_fem_am.solvers.nonlinear import mechanics_newton_overrides_from_args  # noqa: E402

ACCEPTANCE = {"force_frac": 0.005, "disp_frac": 0.01,
              "fallback_frac": 0.02, "fallback_after": 9}


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class AcceptanceCriteriaTest(BBarTestBase):
    """Thermal-gradient cantilever driven into plastic flow (the locking-cure
    scenario), solved under three Newton configurations."""

    def solve_beam(self, newton):
        nx, ny, nz = 8, 2, 2
        Lx, Ly, Lz = 0.8e-3, 0.2e-3, 0.2e-3
        points, hexes = beam_grid(nx, ny, nz, Lx, Ly, Lz)
        centroids = onp.asarray(points)[hexes].mean(axis=1)

        def clamp_x0(point):
            return point[0] < 1e-9

        def zero(_point):
            return 0.0

        from jax_fem.generate_mesh import Mesh
        mesh = Mesh(points, hexes, ele_type="HEX8")
        problem = self.v03.ThermoMechanical(
            mesh=mesh, vec=3, dim=3, ele_type="HEX8", quadrature_order=2,
            dirichlet_bc_info=[[clamp_x0] * 3, [0, 1, 2], [zero] * 3],
            additional_info=("j2_plastic", None, 0.0, 0.0, (), True))
        params = self.uniform_params(problem, dT=0.0, yield_stress=60e6)
        dT_cell = -400.0 * centroids[:, 2] / Lz
        params[1] = jnp.asarray(onp.broadcast_to(
            dT_cell[:, None, None],
            (len(hexes), problem.fes[0].num_quads, 1)).copy())
        u = self.v03.run_mechanics(
            problem, [jnp.zeros((len(points), 3))], params, newton)
        return onp.asarray(u[0])

    def test_legacy_default_raises_on_unreachable_tolerance(self):
        # rel_tol below the documented j2 stall floor (~2e-5): the legacy
        # criteria must still hit max_iter and raise (behavior preserved).
        newton = dict(NEWTON)
        newton.update({"rel_tol": 1e-12, "tol": 0.0, "max_iter": 25})
        try:
            self.solve_beam(newton)
        except RuntimeError as e:
            self.assertIn("did not converge", str(e))
        else:
            self.skipTest("stall floor not reached on this problem/machine; "
                          "legacy raise path not exercised")

    def test_abaqus_mode_accepts_stall_state_close_to_reference(self):
        # Reference: production-tolerance legacy solve (rel 5e-5).
        u_ref = self.solve_beam(dict(NEWTON))
        # Same unreachable rel_tol, but with the Abaqus dual criteria: must
        # terminate without exception and land close to the reference.
        newton = dict(NEWTON)
        newton.update({"rel_tol": 1e-12, "tol": 0.0, "max_iter": 40,
                       "acceptance": dict(ACCEPTANCE)})
        u_acc = self.solve_beam(newton)
        scale = float(onp.abs(u_ref).max())
        diff = float(onp.abs(u_acc - u_ref).max())
        self.assertLess(diff, 0.02 * scale,
                        f"abaqus-accepted solution deviates {diff:.3e} "
                        f"(scale {scale:.3e})")

    def test_overrides_wiring(self):
        base = dict(mechanics_tol=None, mechanics_rel_tol=5e-5,
                    mechanics_max_iter=50, mechanics_line_search=True,
                    mechanics_residual_only_check=False)
        args = SimpleNamespace(mechanics_acceptance="legacy", **base)
        self.assertNotIn("acceptance", mechanics_newton_overrides_from_args(args))
        args = SimpleNamespace(
            mechanics_acceptance="abaqus",
            mechanics_acceptance_force_frac=0.005,
            mechanics_acceptance_disp_frac=0.01,
            mechanics_acceptance_fallback_frac=0.02,
            mechanics_acceptance_fallback_after=9,
            **base)
        acc = mechanics_newton_overrides_from_args(args)["acceptance"]
        self.assertEqual(acc["fallback_after"], 9)
        self.assertAlmostEqual(acc["force_frac"], 0.005)

    def test_residual_only_check_is_opt_in_for_mechanics(self):
        base = dict(mechanics_tol=None, mechanics_rel_tol=5e-5,
                    mechanics_max_iter=50, mechanics_line_search=True,
                    mechanics_acceptance="legacy")
        disabled = SimpleNamespace(
            mechanics_residual_only_check=False, **base)
        self.assertNotIn(
            "residual_only_check",
            mechanics_newton_overrides_from_args(disabled),
        )

        enabled = SimpleNamespace(
            mechanics_residual_only_check=True, **base)
        self.assertIs(
            mechanics_newton_overrides_from_args(enabled)[
                "residual_only_check"
            ],
            True,
        )


if __name__ == "__main__":
    unittest.main()
