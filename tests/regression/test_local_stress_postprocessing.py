import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as onp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "v01"))
sys.path.insert(0, str(ROOT / "examples_local"))

SCRIPT_PATH = ROOT / "legacy" / "v01" / "inp_thermal_stress_oneway_xbuild_p0p1_fixed.py"
POSTPROCESS_PATH = ROOT / "legacy" / "v01" / "postprocess_quad_stress.py"
HAS_REAL_JAX = importlib.util.find_spec("jax") is not None


def install_import_stubs_if_jax_is_missing():
    if HAS_REAL_JAX:
        return

    jax_stub = types.ModuleType("jax")
    jax_stub.vmap = lambda fn: fn
    sys.modules.setdefault("jax", jax_stub)
    sys.modules.setdefault("jax.numpy", onp)

    inp_stub = types.ModuleType("inp_initial_guess_smoke")
    inp_stub.read_tet4_inp = None
    sys.modules.setdefault("inp_initial_guess_smoke", inp_stub)

    jax_fem_stub = types.ModuleType("jax_fem")
    generate_mesh_stub = types.ModuleType("jax_fem.generate_mesh")
    problem_stub = types.ModuleType("jax_fem.problem")
    solver_stub = types.ModuleType("jax_fem.solver")
    utils_stub = types.ModuleType("jax_fem.utils")

    class Mesh:
        pass

    class Problem:
        pass

    generate_mesh_stub.Mesh = Mesh
    problem_stub.Problem = Problem
    solver_stub.solver = None
    utils_stub.save_sol = None

    sys.modules.setdefault("jax_fem", jax_fem_stub)
    sys.modules.setdefault("jax_fem.generate_mesh", generate_mesh_stub)
    sys.modules.setdefault("jax_fem.problem", problem_stub)
    sys.modules.setdefault("jax_fem.solver", solver_stub)
    sys.modules.setdefault("jax_fem.utils", utils_stub)


install_import_stubs_if_jax_is_missing()
SPEC = importlib.util.spec_from_file_location("inp_thermal_p0p1_fixed", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
POST_SPEC = importlib.util.spec_from_file_location("postprocess_quad_stress", POSTPROCESS_PATH)
POST = importlib.util.module_from_spec(POST_SPEC)
sys.modules[POST_SPEC.name] = POST
POST_SPEC.loader.exec_module(POST)


def make_oriented_tet_box(nx=2, ny=2, nz=3):
    xs = onp.linspace(0.0, 1.0, nx + 1)
    ys = onp.linspace(0.0, 1.0, ny + 1)
    zs = onp.linspace(0.0, 1.0, nz + 1)
    points = onp.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=onp.float64)

    def node(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n000 = node(i, j, k)
                n100 = node(i + 1, j, k)
                n010 = node(i, j + 1, k)
                n110 = node(i + 1, j + 1, k)
                n001 = node(i, j, k + 1)
                n101 = node(i + 1, j, k + 1)
                n011 = node(i, j + 1, k + 1)
                n111 = node(i + 1, j + 1, k + 1)
                cells.extend(
                    [
                        [n000, n100, n110, n111],
                        [n000, n110, n010, n111],
                        [n000, n010, n011, n111],
                        [n000, n011, n001, n111],
                        [n000, n001, n101, n111],
                        [n000, n101, n100, n111],
                    ]
                )

    oriented = []
    for tet in cells:
        tet = list(tet)
        p0, p1, p2, p3 = points[tet]
        det = onp.linalg.det(onp.stack([p1 - p0, p2 - p0, p3 - p0], axis=1))
        if det < 0.0:
            tet[1], tet[2] = tet[2], tet[1]
        oriented.append(tet)
    return points, onp.asarray(oriented, dtype=onp.int64)


class TestLocalStressPostprocessing(unittest.TestCase):
    def test_von_mises_matches_uniaxial_and_pure_shear(self):
        uniaxial = onp.zeros((3, 3))
        uniaxial[0, 0] = 123.0

        shear = onp.zeros((3, 3))
        shear[0, 1] = 7.0
        shear[1, 0] = 7.0

        self.assertAlmostEqual(float(MODULE.von_mises_from_stress(uniaxial)), 123.0)
        self.assertAlmostEqual(float(MODULE.von_mises_from_stress(shear)), onp.sqrt(3.0) * 7.0)

    def test_quad_level_von_mises_is_not_computed_from_cell_mean_stress(self):
        sigmas = onp.zeros((1, 2, 3, 3))
        sigmas[0, 0, 0, 0] = 100.0
        sigmas[0, 1, 0, 0] = -100.0
        jxw = onp.ones((1, 2))

        vm_quad = MODULE.von_mises_from_stress(sigmas)
        stats = POST.summarize_quad_stress(sigmas, vm_quad)

        vm_from_cell_mean = MODULE.von_mises_from_stress(stats["stress_mean"])
        self.assertAlmostEqual(float(vm_from_cell_mean[0]), 0.0)
        self.assertAlmostEqual(float(stats["von_mises_mean"][0]), 100.0)
        self.assertAlmostEqual(float(stats["von_mises_max"][0]), 100.0)
        self.assertAlmostEqual(float(stats["von_mises_p95"][0]), 100.0)

    def test_nodal_recovery_averages_adjacent_cell_data(self):
        cells = onp.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        cell_values = onp.array([10.0, 20.0])

        recovered = POST.recover_nodal_averaged_cell_data(cells, cell_values, num_nodes=5)

        onp.testing.assert_allclose(onp.asarray(recovered), onp.array([15.0, 15.0, 15.0, 10.0, 20.0]))

    def test_save_step_exposes_only_raw_quad_stress_fields(self):
        class FakeFe:
            num_cells = 2
            num_quads = 2
            points = onp.zeros((5, 3))
            cells = onp.array([[0, 1, 2, 3], [0, 1, 2, 4]])

        captured = {}

        def fake_save_sol(fe, sol, vtk_path, cell_infos=None, point_infos=None):
            captured["cell_infos"] = dict(cell_infos)
            captured["point_infos"] = dict(point_infos)

        old_save_sol = MODULE.save_sol
        MODULE.save_sol = fake_save_sol
        try:
            quad_stress = MODULE.empty_quad_stress(FakeFe.num_cells, FakeFe.num_quads)
            MODULE.save_step(
                FakeFe,
                onp.zeros((5, 1)),
                onp.zeros((5, 3)),
                "unused.vtu",
                onp.zeros((2, 2, 1)),
                quad_stress,
                onp.ones((2,)),
                onp.zeros((2,)),
                onp.zeros((2,)),
                onp.zeros((2,)),
                onp.zeros((2,)),
                onp.zeros((2,)),
                onp.zeros((2, 2, 1)),
                1.0,
                3,
                1,
            )
        finally:
            MODULE.save_sol = old_save_sol

        for name in [
            "stress_quad0_xx",
            "stress_quad0_yy",
            "stress_quad0_zz",
            "stress_quad0_xy",
            "stress_quad0_yz",
            "stress_quad0_xz",
            "vm_quad0",
            "stress_quad1_xx",
            "stress_quad1_yy",
            "stress_quad1_zz",
            "stress_quad1_xy",
            "stress_quad1_yz",
            "stress_quad1_xz",
            "vm_quad1",
        ]:
            self.assertEqual(onp.asarray(captured["cell_infos"][name]).shape, (FakeFe.num_cells,))

        for derived_name in [
            "von_mises",
            "von_mises_mean",
            "von_mises_max",
            "von_mises_p95",
            "stress_xx_mean",
            "stress_xx",
            "recovered_von_mises_mean",
        ]:
            self.assertNotIn(derived_name, captured["cell_infos"])
            self.assertNotIn(derived_name, captured["point_infos"])

    def test_postprocess_derives_summary_and_recovered_fields_from_raw_quad_fields(self):
        cells = onp.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        cell_data = {
            "stress_quad0_xx": onp.array([100.0, 20.0]),
            "stress_quad0_yy": onp.zeros(2),
            "stress_quad0_zz": onp.zeros(2),
            "stress_quad0_xy": onp.zeros(2),
            "stress_quad0_yz": onp.zeros(2),
            "stress_quad0_xz": onp.zeros(2),
            "vm_quad0": onp.array([100.0, 20.0]),
            "stress_quad1_xx": onp.array([-100.0, 40.0]),
            "stress_quad1_yy": onp.zeros(2),
            "stress_quad1_zz": onp.zeros(2),
            "stress_quad1_xy": onp.zeros(2),
            "stress_quad1_yz": onp.zeros(2),
            "stress_quad1_xz": onp.zeros(2),
            "vm_quad1": onp.array([100.0, 40.0]),
        }

        stress_quad, vm_quad = POST.extract_quad_stress(cell_data)
        stats = POST.summarize_quad_stress(stress_quad, vm_quad)
        derived_cell = POST.derived_cell_data(stats)
        derived_point = POST.derived_point_data(cells, 5, stats)

        onp.testing.assert_allclose(derived_cell["von_mises_mean"], onp.array([100.0, 30.0]))
        onp.testing.assert_allclose(derived_cell["von_mises_max"], onp.array([100.0, 40.0]))
        onp.testing.assert_allclose(derived_cell["von_mises_p95"], onp.array([100.0, 39.0]))
        onp.testing.assert_allclose(derived_cell["stress_xx_mean"], onp.array([0.0, 30.0]))
        self.assertIn("von_mises", derived_cell)
        self.assertEqual(len(derived_point["recovered_von_mises_mean"]), 5)

    def test_free_thermal_expansion_is_stress_free_in_constitutive_map(self):
        problem = object.__new__(MODULE.ThermoMechanical)
        problem.dim = 3
        problem.mechanics_model = "linear_elastic"
        alpha = 1.2e-5
        dT = 100.0
        u_grad = alpha * dT * onp.eye(3)

        sigma = problem.stress_fn(
            u_grad,
            onp.array([300.0]),
            onp.array([dT]),
            onp.array([1.0]),
            onp.array([200.0e9]),
            onp.array([alpha]),
            onp.array([0.3]),
            onp.array([1.0e9]),
            onp.array([0.0]),
            onp.array([0.0]),
        )

        onp.testing.assert_allclose(onp.asarray(sigma), onp.zeros((3, 3)), atol=1e-5)

    def test_constrained_heating_generates_nonzero_thermal_stress(self):
        problem = object.__new__(MODULE.ThermoMechanical)
        problem.dim = 3
        problem.mechanics_model = "linear_elastic"

        sigma = problem.stress_fn(
            onp.zeros((3, 3)),
            onp.array([300.0]),
            onp.array([100.0]),
            onp.array([1.0]),
            onp.array([200.0e9]),
            onp.array([1.2e-5]),
            onp.array([0.3]),
            onp.array([1.0e9]),
            onp.array([0.0]),
            onp.array([0.0]),
        )

        self.assertGreater(float(onp.linalg.norm(onp.asarray(sigma))), 0.0)
        self.assertLess(float(onp.trace(onp.asarray(sigma))), 0.0)

    @unittest.skipUnless(HAS_REAL_JAX, "requires real jax/jax-fem runtime")
    def test_bottom_fixed_thermal_expansion_concentrates_stress_near_base(self):
        points, cells = make_oriented_tet_box()

        def bottom(point):
            return MODULE.np.isclose(point[2], 0.0, rtol=0.0, atol=1e-10)

        def zero(_point):
            return 0.0

        mesh = MODULE.Mesh(points, cells, ele_type="TET4")
        problem = MODULE.ThermoMechanical(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type="TET4",
            dirichlet_bc_info=[[bottom, bottom, bottom], [0, 1, 2], [zero, zero, zero]],
            additional_info=("linear_elastic",),
        )

        num_cells = len(cells)
        num_quads = problem.fes[0].num_quads

        def quad(value):
            return MODULE.np.full((num_cells, num_quads, 1), value)

        params = [
            quad(400.0),
            quad(100.0),
            quad(1.0),
            quad(2.0e11),
            quad(1.2e-5),
            quad(0.3),
            quad(1.0e12),
            quad(0.0),
            quad(0.0),
        ]
        sol = MODULE.run_mechanics(problem, [MODULE.np.zeros((len(points), 3))], params)[0]
        quad_stress = problem.compute_cell_stress(sol, params)
        stress_stats = POST.summarize_quad_stress(quad_stress["stress_quad"], quad_stress["vm_quad"])

        vm = onp.asarray(stress_stats["von_mises_mean"])
        centroids = points[cells].mean(axis=1)
        bottom_cells = centroids[:, 2] < 0.35
        top_cells = centroids[:, 2] > 0.65

        self.assertGreater(float(vm.max()), 0.0)
        self.assertGreater(float(vm[bottom_cells].mean()), float(vm[top_cells].mean()))


if __name__ == "__main__":
    unittest.main()
