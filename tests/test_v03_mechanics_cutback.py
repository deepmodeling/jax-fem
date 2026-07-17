import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V01_DIR = REPO_ROOT / "159_local" / "v01"
V03_PATH = (
    REPO_ROOT
    / "159_local"
    / "v03"
    / "am_thermal_stress_macro_intersection_mech100.py"
)

try:
    import numpy as onp
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - depends on local runtime
    onp = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def load_v03():
    if str(V01_DIR) not in sys.path:
        sys.path.insert(0, str(V01_DIR))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("v03_cutback_test_base", V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_material_quads(T_quad, active_quad, phase_quad, args, tables):
    like = jnp.ones_like(T_quad)
    return like, 2.0 * like, 3.0 * like, 4.0 * like, 5.0 * like, 6.0 * like


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class MechanicsCutbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v03 = load_v03()

    def setUp(self):
        self.T_ref = jnp.full((3, 2, 1), 423.15)
        self.T_prev = jnp.full((3, 2, 1), 500.0)
        self.T_cur = jnp.full((3, 2, 1), 900.0)
        self.active = jnp.ones((3, 2, 1))
        self.eqp = jnp.zeros((3, 2, 1))
        self.params = [
            self.T_cur,
            (self.T_cur - self.T_ref) * self.active,
            self.active, self.active, self.active,
            self.active, self.active, self.active,
            self.eqp,
        ]
        self.args = SimpleNamespace(mechanics_max_cuts=3)
        self.u0 = ["u0"]

    def run_cutback(self, fake_run_mechanics, T_prev=None, active_prev=None, args=None):
        with mock.patch.object(self.v03, "run_mechanics", side_effect=fake_run_mechanics), \
             mock.patch.object(self.v03, "mechanics_material_quads",
                               side_effect=fake_material_quads):
            return self.v03.run_mechanics_with_cutback(
                "mechanics", self.u0, self.params, {"rel_tol": 5e-5},
                args or self.args,
                T_prev, active_prev, self.T_ref, self.active, "phase", "tables",
            )

    def test_converged_full_increment_is_single_call(self):
        calls = []

        def ok(mechanics, u_guess, params, overrides=None):
            calls.append(params)
            return ["u1"]

        out = self.run_cutback(ok, T_prev=self.T_prev, active_prev=self.active)
        self.assertEqual(out, ["u1"])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], self.params)

    def test_disabled_cutback_reraises(self):
        def always_fail(mechanics, u_guess, params, overrides=None):
            raise RuntimeError("Newton solver did not converge")

        args = SimpleNamespace(mechanics_max_cuts=0)
        with self.assertRaises(RuntimeError):
            self.run_cutback(always_fail, T_prev=self.T_prev,
                             active_prev=self.active, args=args)

    def test_two_substeps_interpolate_and_finish_on_original_params(self):
        calls = []

        def fail_full_then_ok(mechanics, u_guess, params, overrides=None):
            calls.append((u_guess, params))
            if len(calls) == 1:
                raise RuntimeError("Newton solver did not converge")
            return [f"u{len(calls)}"]

        out = self.run_cutback(fail_full_then_ok, T_prev=self.T_prev,
                               active_prev=self.active)
        # 1 failed full attempt + 2 substeps
        self.assertEqual(len(calls), 3)
        mid_params = calls[1][1]
        expected_T_mid = 0.5 * (self.T_prev + self.T_cur)
        onp.testing.assert_allclose(onp.asarray(mid_params[0]),
                                    onp.asarray(expected_T_mid))
        onp.testing.assert_allclose(
            onp.asarray(mid_params[1]),
            onp.asarray((expected_T_mid - self.T_ref) * self.active))
        # plastic state is held fixed across substeps
        self.assertIs(mid_params[-1], self.params[-1])
        # final substep must be the exact original problem
        self.assertIs(calls[2][1], self.params)
        # substeps chain their displacement guesses
        self.assertEqual(calls[2][0], ["u2"])
        self.assertEqual(out, ["u3"])

    def test_refines_to_four_substeps_and_gives_up_after_max(self):
        calls = []

        def fail_until_quarter_steps(mechanics, u_guess, params, overrides=None):
            calls.append(params)
            # succeed only when the temperature jump per solve is <= 1/4 of full
            T_target = onp.asarray(params[0])
            jump = onp.max(onp.abs(T_target - onp.asarray(fail_until_quarter_steps.T_state)))
            if jump > 100.0 + 1e-9:  # full jump is 400 K -> needs n=4
                raise RuntimeError("Newton solver did not converge")
            fail_until_quarter_steps.T_state = T_target
            return ["ok"]

        fail_until_quarter_steps.T_state = onp.asarray(self.T_prev)
        out = self.run_cutback(fail_until_quarter_steps, T_prev=self.T_prev,
                               active_prev=self.active)
        self.assertEqual(out, ["ok"])
        # 1 full fail + 1 first-of-two fail + 4 successful quarter substeps
        self.assertEqual(len(calls), 6)
        self.assertIs(calls[-1], self.params)

    def test_exhausted_cuts_reraise_last_error(self):
        def always_fail(mechanics, u_guess, params, overrides=None):
            raise RuntimeError("Newton solver did not converge")

        args = SimpleNamespace(mechanics_max_cuts=2)
        with self.assertRaises(RuntimeError):
            self.run_cutback(always_fail, T_prev=self.T_prev,
                             active_prev=self.active, args=args)

    def test_no_previous_state_ramps_from_reference(self):
        calls = []

        def fail_full_then_ok(mechanics, u_guess, params, overrides=None):
            calls.append(params)
            if len(calls) == 1:
                raise RuntimeError("Newton solver did not converge")
            return ["u"]

        self.run_cutback(fail_full_then_ok, T_prev=None, active_prev=None)
        mid_params = calls[1]
        expected_T_mid = 0.5 * (self.T_ref + self.T_cur)
        onp.testing.assert_allclose(onp.asarray(mid_params[0]),
                                    onp.asarray(expected_T_mid))


if __name__ == "__main__":
    unittest.main()
