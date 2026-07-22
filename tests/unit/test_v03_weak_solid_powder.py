import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
V01_DIR = REPO_ROOT / "legacy" / "v01"
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
    spec = importlib.util.spec_from_file_location("v03_powder_test_base", V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TINY_INP = """*HEADING
tiny two-tet mesh with powder elset
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
5, 1.0, 1.0, 1.0
*ELEMENT, TYPE=C3D4, ELSET=PART
1, 1, 2, 3, 4
2, 2, 3, 4, 5
*ELSET, ELSET=POWDER
2
"""


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class ReadInpCellSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v03 = load_v03()

    def write_inp(self, content):
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".inp", delete=False, dir=tempfile.gettempdir())
        f.write(content)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return f.name

    def test_reads_powder_mask(self):
        path = self.write_inp(TINY_INP)
        mask = self.v03.read_inp_cell_set(path, "POWDER", 2)
        onp.testing.assert_array_equal(mask, [False, True])

    def test_missing_set_raises(self):
        path = self.write_inp(TINY_INP)
        with self.assertRaises(ValueError):
            self.v03.read_inp_cell_set(path, "NOPE", 2)

    def test_cell_count_mismatch_raises(self):
        path = self.write_inp(TINY_INP)
        with self.assertRaises(ValueError):
            self.v03.read_inp_cell_set(path, "POWDER", 3)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class WeakSolidPowderMaterialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v03 = load_v03()

    def make_args(self, powder_E=None):
        return SimpleNamespace(
            young=190e9,
            alpha=16e-6,
            poisson=0.3,
            mechanics_model="j2_plastic",
            mushy_mechanics_factor=0.01,
            liquid_mechanics_factor=1e-4,
            inactive_mechanics_factor=1e-8,
            powder_solid_E=powder_E,
            powder_solid_yield=1.0e6,
            powder_solid_hardening=1.0e7,
        )

    def make_inputs(self):
        v03 = self.v03
        T = jnp.full((4, 1, 1), 500.0)
        # cells: solid, powder, liquid, powder-but-mechanically-inactive
        phase = jnp.array(
            [[[v03.STATE_SOLID]], [[v03.STATE_POWDER]],
             [[v03.STATE_LIQUID]], [[v03.STATE_POWDER]]], dtype=jnp.int32)
        active = jnp.array([[[1.0]], [[1.0]], [[1.0]], [[0.0]]])
        def const_table(value):
            return SimpleNamespace(eval=lambda T: value * jnp.ones_like(T))

        tables = {"E": None, "alpha": None, "poisson": None,
                  "yield": const_table(500e6),
                  "hardening": const_table(2.5e8)}
        return T, active, phase, tables

    def test_weak_solid_overrides_powder_cells_only(self):
        v03 = self.v03
        T, active, phase, tables = self.make_inputs()
        args = self.make_args(powder_E=10e9)
        af, E, alpha, nu, yld, hard = v03.mechanics_material_quads(
            T, active, phase, args, tables)
        E = onp.asarray(E)[:, 0, 0]
        yld = onp.asarray(yld)[:, 0, 0]
        hard = onp.asarray(hard)[:, 0, 0]
        af = onp.asarray(af)[:, 0, 0]
        alpha = onp.asarray(alpha)[:, 0, 0]
        # solid untouched
        self.assertAlmostEqual(E[0], 190e9, delta=1e-3 * 190e9)
        self.assertGreater(yld[0], 1e8)
        self.assertAlmostEqual(af[0], 1.0)
        # active powder: weak solid, full active factor, no expansion
        self.assertEqual(E[1], 10e9)
        self.assertEqual(yld[1], 1.0e6)
        self.assertEqual(hard[1], 1.0e7)  # regularization hardening
        self.assertAlmostEqual(af[1], 1.0)
        self.assertEqual(alpha[1], 0.0)
        # liquid untouched
        self.assertAlmostEqual(af[2], args.liquid_mechanics_factor)
        # inactive powder gets the ersatz factor, not load-bearing stiffness
        self.assertAlmostEqual(af[3], args.inactive_mechanics_factor)

    def test_legacy_behavior_when_disabled(self):
        v03 = self.v03
        T, active, phase, tables = self.make_inputs()
        args = self.make_args(powder_E=None)
        af, E, alpha, nu, yld, hard = v03.mechanics_material_quads(
            T, active, phase, args, tables)
        af = onp.asarray(af)[:, 0, 0]
        # powder keeps the near-zero ersatz factor (carries no load)
        self.assertAlmostEqual(af[1], args.inactive_mechanics_factor, places=12)

    def test_parser_wiring(self):
        parser = self.v03.build_parser()
        args = parser.parse_args(["--inp", "x.inp"])
        self.assertIsNone(args.powder_elset)
        self.assertIsNone(args.powder_solid_E)
        args = parser.parse_args(
            ["--inp", "x.inp", "--powder-elset", "POWDER",
             "--powder-solid-E", "10e9", "--powder-solid-yield", "1e6"])
        self.assertEqual(args.powder_elset, "POWDER")
        self.assertEqual(args.powder_solid_E, 10e9)
        self.assertEqual(args.powder_solid_yield, 1e6)


if __name__ == "__main__":
    unittest.main()
