import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
XLA_WRAPPER_PATH = (
    REPO_ROOT
    / "159_local"
    / "v03"
    / "am_thermal_stress_macro_intersection_mech100_XLA.py"
)


def load_xla_wrapper():
    spec = importlib.util.spec_from_file_location("macro_mech100_xla", XLA_WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MacroMech100XlaWrapperTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_xla_wrapper()

    def test_linear_options_from_args_builds_jax_solver(self):
        args = SimpleNamespace(xla_linear_solver="jax", xla_jax_precond=False)

        options = self.wrapper.linear_options_from_args(args)

        self.assertEqual(options, {"jax_solver": {"precond": False}})

    def test_rewrites_nested_newton_linear_solver_without_losing_tolerances(self):
        original_options = {
            "newton": {
                "tol": 1e-9,
                "rel_tol": 1e-11,
                "linear": {"spsolve_solver": {}},
            }
        }
        replacement = {"jax_solver": {"precond": True}}

        rewritten = self.wrapper.rewrite_solver_options(original_options, replacement)

        self.assertEqual(rewritten["newton"]["tol"], 1e-9)
        self.assertEqual(rewritten["newton"]["rel_tol"], 1e-11)
        self.assertEqual(rewritten["newton"]["linear"], replacement)
        self.assertEqual(original_options["newton"]["linear"], {"spsolve_solver": {}})

    def test_rewrites_legacy_flat_linear_solver_options(self):
        original_options = {
            "tol": 1e-5,
            "spsolve_solver": {},
        }
        replacement = {"petsc_solver": {"ksp_type": "gmres", "pc_type": "jacobi"}}

        rewritten = self.wrapper.rewrite_solver_options(original_options, replacement)

        self.assertEqual(rewritten["tol"], 1e-5)
        self.assertNotIn("spsolve_solver", rewritten)
        self.assertEqual(rewritten["petsc_solver"], {"ksp_type": "gmres", "pc_type": "jacobi"})

    def test_preserve_solver_options_leaves_copy_unchanged(self):
        original_options = {"newton": {"linear": {"spsolve_solver": {}}}}

        rewritten = self.wrapper.rewrite_solver_options(original_options, None)

        self.assertEqual(rewritten, original_options)
        self.assertIsNot(rewritten, original_options)


if __name__ == "__main__":
    unittest.main()
