import importlib.util
import sys
import unittest
from types import SimpleNamespace
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
    from jax_fem.fe import FiniteElement
    from jax_fem.generate_mesh import Mesh
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
    spec = importlib.util.spec_from_file_location("v03_lumping_test_base", V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fe(quadrature_order=2):
    points = onp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [1.0, 1.0, 1.0]]
    )
    cells = onp.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    mesh = Mesh(points, cells, ele_type="TET4")
    return FiniteElement(
        mesh=mesh,
        vec=1,
        dim=3,
        ele_type="TET4",
        quadrature_rule=None,
        quadrature_order=quadrature_order,
        dirichlet_bc_info=None,
    ), points, cells


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class ThermalMassLumpingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v03 = load_v03()

    def apply(self, fe):
        problem = SimpleNamespace(fes=[fe], physical_quad_points=None)
        return self.v03.apply_thermal_mass_lumping(problem)

    def test_shape_vals_become_identity_and_mass_is_row_sum_lumped(self):
        fe, _, _ = make_fe()
        consistent_shape_vals = onp.asarray(fe.shape_vals).copy()
        weights = onp.asarray(fe.quad_weights).copy()
        # consistent mass on the reference element
        M_cons = onp.einsum("q,qi,qj->ij", weights, consistent_shape_vals,
                            consistent_shape_vals)
        self.apply(fe)
        lumped_shape_vals = onp.asarray(fe.shape_vals)
        onp.testing.assert_allclose(lumped_shape_vals, onp.eye(fe.num_nodes),
                                    atol=1e-14)
        M_lump = onp.einsum("q,qi,qj->ij", weights, lumped_shape_vals,
                            lumped_shape_vals)
        # diagonal, and each diagonal entry equals the consistent row sum
        onp.testing.assert_allclose(M_lump, onp.diag(onp.diag(M_lump)), atol=1e-14)
        onp.testing.assert_allclose(onp.diag(M_lump), M_cons.sum(axis=1),
                                    rtol=1e-12)

    def test_conduction_data_untouched(self):
        fe, _, _ = make_fe()
        grads_before = onp.asarray(fe.shape_grads_ref).copy()
        weights_before = onp.asarray(fe.quad_weights).copy()
        self.apply(fe)
        onp.testing.assert_allclose(onp.asarray(fe.shape_grads_ref), grads_before)
        onp.testing.assert_allclose(onp.asarray(fe.quad_weights), weights_before)

    def test_quad_points_relocated_to_vertices(self):
        fe, points, cells = make_fe()
        problem = self.apply(fe)
        pqp = onp.asarray(problem.physical_quad_points)
        self.assertEqual(pqp.shape, (len(cells), 4, 3))
        for c, cell in enumerate(cells):
            onp.testing.assert_allclose(pqp[c], points[cell], atol=1e-14)

    def test_rejects_single_point_rule(self):
        fe, _, _ = make_fe(quadrature_order=None)  # legacy 1-point rule
        with self.assertRaises(ValueError):
            self.apply(fe)

    def test_parser_default_off(self):
        parser = self.v03.build_parser()
        args = parser.parse_args(["--inp", "dummy.inp"])
        self.assertFalse(args.thermal_mass_lumping)
        args = parser.parse_args(["--inp", "dummy.inp", "--thermal-mass-lumping"])
        self.assertTrue(args.thermal_mass_lumping)


if __name__ == "__main__":
    unittest.main()
