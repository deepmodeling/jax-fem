import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
V05_PATH = (
    REPO_ROOT / "159_local" / "v05"
    / "am_thermal_stress_macro_intersection_mech100_v05.py"
)

try:
    import numpy as onp
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    onp = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def load_v05():
    if str(REPO_ROOT / "159_local" / "v01") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "159_local" / "v01"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("v05_wrapper_test", V05_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patched_base():
    """Real v03 base module with the v05 plastic-history patch installed."""
    v05 = load_v05()
    v04 = v05.load_v04_wrapper()
    base = v04.load_base_solver()
    v05.install_plastic_history_patch(base)
    return v05, base


ARGS = dict(
    T=jnp.asarray([300.0]), active_factor=jnp.asarray([1.0]),
    young=jnp.asarray([125.0e9]), alpha=jnp.asarray([9.0e-6]),
    poisson=jnp.asarray([0.3]), yield_stress=jnp.asarray([500.0e6]),
)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class IncrementalRadialReturnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v05, cls.base = patched_base()
        cls.cls_ = cls.base.ThermoMechanical

    def fake_self(self):
        return SimpleNamespace(
            mechanics_model="j2_plastic", dim=3, yield_saturation=None,
        )

    def return_map(self, u_grad, eqp_old, eps_p_old, hardening=2.0e9, dT=0.0):
        return self.cls_._return_map(
            self.fake_self(), jnp.asarray(u_grad), jnp.asarray([dT]),
            ARGS["young"], ARGS["alpha"], ARGS["poisson"],
            ARGS["yield_stress"], jnp.asarray([hardening]),
            jnp.asarray([eqp_old]), jnp.asarray(eps_p_old),
        )

    @staticmethod
    def seq(sigma):
        s = onp.asarray(sigma)
        dev = s - onp.trace(s) / 3.0 * onp.eye(3)
        return float(onp.sqrt(1.5 * onp.sum(dev * dev)))

    def test_stored_plastic_strain_produces_stress_at_zero_load(self):
        # zero displacement + zero dT but nonzero deviatoric eps_p -> stress
        eps_p = [1.0e-3, -0.5e-3, -0.5e-3, 0.0, 0.0, 0.0]
        sigma, delta_eqp, _ = self.return_map(onp.zeros((3, 3)), 0.0, eps_p)
        self.assertGreater(self.seq(sigma), 1.0e8)
        # this is exactly what the scalar-only v03/v04 model cannot represent

    def test_return_mapped_state_is_consistent(self):
        # load far beyond yield, update state, re-evaluate: elastic (no new flow)
        u_grad = onp.diag([0.02, -0.006, -0.006])
        sigma1, d1, deps1 = self.return_map(u_grad, 0.0, onp.zeros(6))
        self.assertGreater(float(d1), 0.0)
        eps_p_new = onp.asarray(deps1)
        eqp_new = float(d1)
        sigma2, d2, _ = self.return_map(u_grad, eqp_new, eps_p_new)
        # analytically zero; float64 leaves O(1e-9) residual at GPa stress scale
        self.assertLess(float(d2), 1e-6 * float(d1))
        # stress unchanged by the state update (consistency of the return map)
        self.assertAlmostEqual(self.seq(sigma1) / self.seq(sigma2), 1.0, places=6)
        # and the stress sits on the hardened yield surface
        yield_eff = 500.0e6 + 2.0e9 * eqp_new
        self.assertAlmostEqual(self.seq(sigma2) / yield_eff, 1.0, places=6)

    def test_zero_eps_p_matches_v03_stress(self):
        u_grad = onp.diag([0.004, -0.001, -0.001])
        sigma_v05, _, _ = self.return_map(u_grad, 0.0, onp.zeros(6), hardening=0.0)
        legacy_self = SimpleNamespace(
            mechanics_model="j2_plastic", dim=3, yield_saturation=None,
        )
        original = self.base._v05_original_thermo_mechanical
        sigma_v03 = original.stress_fn(
            legacy_self, jnp.asarray(u_grad), ARGS["T"], jnp.asarray([0.0]),
            ARGS["active_factor"], ARGS["young"], jnp.asarray([0.0]),
            ARGS["poisson"], ARGS["yield_stress"], jnp.asarray([0.0]),
            jnp.asarray([0.0]),
        )
        onp.testing.assert_allclose(
            onp.asarray(sigma_v05), onp.asarray(sigma_v03), rtol=1e-9
        )

    def test_registry_reset_between_runs(self):
        self.v05.REGISTRY.eps_p = "stale"
        self.v05.REGISTRY.build_problem = "stale"
        self.v05.REGISTRY.reset()
        self.assertIsNone(self.v05.REGISTRY.eps_p)
        self.assertIsNone(self.v05.REGISTRY.build_problem)


if __name__ == "__main__":
    unittest.main()
