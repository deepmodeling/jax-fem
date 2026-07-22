import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V01_DIR = REPO_ROOT / "159_local" / "v01"
V03_PATH = (
    REPO_ROOT
    / "jax_fem_am"
    / "simulation"
    / "stepper.py"
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
    spec = importlib.util.spec_from_file_location("v03_mech_floor_test_base", V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class MechanicsTemperatureFloorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v03 = load_v03()

    def test_none_floor_is_identity(self):
        T = jnp.array([[-897.0], [423.15], [5742.0]])
        out = self.v03.clamp_mechanics_temperature(T, None)
        self.assertIs(out, T)

    def test_floor_clamps_only_below(self):
        T = jnp.array([[-897.0], [100.0], [293.15], [423.15], [5742.0]])
        out = self.v03.clamp_mechanics_temperature(T, 293.15)
        onp.testing.assert_allclose(
            onp.asarray(out),
            onp.array([[293.15], [293.15], [293.15], [423.15], [5742.0]]),
        )

    def test_undershoot_dt_is_bounded_by_floor(self):
        # Activation undershoot (G1 artifact) at T=-897 K with reset reference
        # 423.15 K produces dT=-1320 K raw; the floor must cap the contraction
        # the mechanics chain sees to floor - T_ref.
        T_ref = 423.15
        floor = 293.15
        T = jnp.array([[-897.0]])
        dT = self.v03.clamp_mechanics_temperature(T, floor) - T_ref
        self.assertAlmostEqual(float(dT[0, 0]), floor - T_ref, places=9)

    def test_parser_default_off_and_parses_value(self):
        parser = self.v03.build_parser()
        args = parser.parse_args(["--inp", "dummy.inp"])
        self.assertIsNone(args.mechanics_temperature_floor)
        args = parser.parse_args(
            ["--inp", "dummy.inp", "--mechanics-temperature-floor", "293.15"]
        )
        self.assertAlmostEqual(args.mechanics_temperature_floor, 293.15)


if __name__ == "__main__":
    unittest.main()
