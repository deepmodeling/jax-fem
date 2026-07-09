import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    spec = importlib.util.spec_from_file_location("v03_physics_test_base", V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def two_tet_fe():
    """Two TET4 cells sharing one interior face; 6 exterior + 1 shared face."""
    points = onp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    cells = onp.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    mesh = Mesh(points, cells, ele_type="TET4")
    return FiniteElement(
        mesh=mesh,
        vec=1,
        dim=3,
        ele_type="TET4",
        quadrature_rule=None,
        quadrature_order=None,
        dirichlet_bc_info=None,
    )


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class ExteriorFaceSelectionTest(unittest.TestCase):
    def test_exterior_face_flags_mark_shared_face_as_interior(self):
        fe = two_tet_fe()
        flags = fe.get_exterior_face_flags()
        self.assertEqual(flags.shape, (2, 4))
        # 2 tets x 4 faces = 8 face slots; the shared face appears twice and
        # is interior in both cells.
        self.assertEqual(int(flags.sum()), 6)

    def test_exterior_only_location_fn_selects_only_exterior_faces(self):
        fe = two_tet_fe()

        def everywhere(_point):
            return True

        selected_all = fe.get_boundary_conditions_inds([everywhere])[0]
        self.assertEqual(len(selected_all), 8)  # legacy behavior: interior too

        everywhere_ext = lambda _point: True
        everywhere_ext.exterior_only = True
        selected_ext = fe.get_boundary_conditions_inds([everywhere_ext])[0]
        self.assertEqual(len(selected_ext), 6)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class BoxLocationToleranceTest(unittest.TestCase):
    def test_abs_tol_catches_jittered_base_nodes(self):
        base = load_v03()
        points = onp.array(
            [
                [0.0, 0.0, 0.0],
                [5.0e-5, 1.0, 0.0],  # base node with CAD jitter
                [8.0e-5, 0.0, 1.0],  # base node with CAD jitter
                [1.0, 0.5, 0.5],
            ]
        )
        _, _, bottom_legacy, *_ = base.make_box_locations(points, build_axis="x")
        legacy_hits = sum(bool(bottom_legacy(p)) for p in points)
        self.assertEqual(legacy_hits, 1)

        _, _, bottom_fixed, *_ = base.make_box_locations(
            points, build_axis="x", abs_tol=1.0e-4
        )
        fixed_hits = sum(bool(bottom_fixed(p)) for p in points)
        self.assertEqual(fixed_hits, 3)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class PathFileRecoatInsertionTest(unittest.TestCase):
    def make_args(self, path_file, **overrides):
        args = SimpleNamespace(
            dt=1.0e-4,
            layers=2,
            mesh_length_scale=1.0,
            path_length_scale=None,
            path_file=path_file,
            recoat_time=0.0,
            recoat_steps=10,
            cooling_steps=0,
            cooling_dt=None,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def write_path(self, rows):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.DictWriter(
            tmp,
            fieldnames=[
                "time", "x", "y", "z", "power", "laser_on", "layer",
                "hatch", "mode",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp.close()
        return tmp.name

    def two_layer_rows(self):
        rows = []
        t = 0.0
        for layer in (1, 1, 2, 2):
            t += 1.0e-4
            rows.append({
                "time": f"{t:.6f}", "x": "0.0", "y": "0.0", "z": "0.0",
                "power": "3000", "laser_on": "1", "layer": str(layer),
                "hatch": "1", "mode": "scan",
            })
        return rows

    def test_no_recoat_by_default(self):
        base = load_v03()
        path = self.write_path(self.two_layer_rows())
        args = self.make_args(path)
        states, _, _ = base.generate_path_file_step_states(
            args, onp.zeros(3), onp.ones(3), 0
        )
        self.assertEqual(len(states), 4)
        self.assertTrue(all(s.mode == "scan" for s in states))

    def test_recoat_states_inserted_at_layer_transition(self):
        base = load_v03()
        path = self.write_path(self.two_layer_rows())
        args = self.make_args(path, recoat_time=10.0, recoat_steps=5)
        states, _, _ = base.generate_path_file_step_states(
            args, onp.zeros(3), onp.ones(3), 0
        )
        self.assertEqual(len(states), 4 + 5)
        recoat = [s for s in states if s.mode == "recoat"]
        self.assertEqual(len(recoat), 5)
        for s in recoat:
            self.assertAlmostEqual(s.dt, 2.0)      # 10 s / 5 steps
            self.assertEqual(s.laser_switch, 0.0)  # laser off
            self.assertEqual(s.layer_idx, 0)       # dwell belongs to layer 1
        # recoat block sits between the layer-1 and layer-2 scan states
        modes = [s.mode for s in states]
        self.assertEqual(
            modes, ["scan", "scan"] + ["recoat"] * 5 + ["scan", "scan"]
        )
        # global steps stay contiguous after insertion
        self.assertEqual(
            [s.global_step for s in states], list(range(len(states)))
        )

    def test_cooling_dt_overrides_dt_for_trailing_cooling(self):
        base = load_v03()
        path = self.write_path(self.two_layer_rows())
        args = self.make_args(path, cooling_steps=3, cooling_dt=2.5)
        states, _, _ = base.generate_path_file_step_states(
            args, onp.zeros(3), onp.ones(3), 0
        )
        cooling = [s for s in states if s.mode == "cooling"]
        self.assertEqual(len(cooling), 3)
        for s in cooling:
            self.assertAlmostEqual(s.dt, 2.5)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class QuadratureOrderTest(unittest.TestCase):
    def make_fe(self, quadrature_order):
        points = onp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        cells = onp.array([[0, 1, 2, 3]])
        mesh = Mesh(points, cells, ele_type="TET4")
        return FiniteElement(
            mesh=mesh,
            vec=1,
            dim=3,
            ele_type="TET4",
            quadrature_rule=None,
            quadrature_order=quadrature_order,
            dirichlet_bc_info=None,
        )

    def test_legacy_default_is_single_point(self):
        fe = self.make_fe(None)
        self.assertEqual(fe.num_quads, 1)

    def test_order_two_gives_full_rank_mass_matrix(self):
        fe = self.make_fe(2)
        self.assertGreaterEqual(fe.num_quads, 4)
        # consistent mass from N quad points: M_ij = sum_q w_q phi_i phi_j
        shape_vals = onp.asarray(fe.shape_vals)  # (num_quads, num_nodes)
        weights = onp.ones(fe.num_quads)
        M = onp.einsum("q,qi,qj->ij", weights, shape_vals, shape_vals)
        self.assertEqual(onp.linalg.matrix_rank(M), 4)
        # the legacy single-point rule is rank-1 (the observed oscillation root cause)
        fe1 = self.make_fe(None)
        sv1 = onp.asarray(fe1.shape_vals)
        M1 = onp.einsum("qi,qj->ij", sv1, sv1)
        self.assertEqual(onp.linalg.matrix_rank(M1), 1)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class SurfaceFluxActiveMaskTest(unittest.TestCase):
    def test_surface_flux_masked_by_face_active_flag(self):
        base = load_v03()
        fake_self = SimpleNamespace(
            convection_h=10.0,
            ambient=300.0,
            emissivity=0.5,
            stefan_boltzmann=5.670374419e-8,
            num_surface_maps=1,
        )
        surface_flux = base.TransientThermal.get_surface_maps(fake_self)[0]
        T = jnp.asarray([500.0])
        point = jnp.zeros(3)

        active = onp.asarray(surface_flux(T, point, jnp.asarray([1.0])))
        masked = onp.asarray(surface_flux(T, point, jnp.asarray([0.0])))
        self.assertGreater(abs(float(active[0])), 0.0)
        self.assertEqual(float(masked[0]), 0.0)
        # all-ones mask reproduces the unmasked legacy flux exactly
        q_conv = 10.0 * (300.0 - 500.0)
        q_rad = 0.5 * 5.670374419e-8 * (300.0**4 - 500.0**4)
        self.assertAlmostEqual(float(active[0]), -(q_conv + q_rad), places=8)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class StressRelaxationReferenceTest(unittest.TestCase):
    def run_update(self, relax_T):
        base = load_v03()
        args = SimpleNamespace(
            solidus_temperature=0.0,
            liquidus_temperature=0.0,
            reset_plastic_on_melt=True,
            stress_relaxation_temperature=relax_T,
        )
        T_quad = 300.0 * jnp.ones((2, 1, 1))
        active_quad = jnp.ones((2, 1, 1))
        phase_quad = base.STATE_POWDER * jnp.ones((2, 1, 1))
        T_ref_quad = 300.0 * jnp.ones((2, 1, 1))
        eqp_quad = jnp.zeros((2, 1, 1))
        phase_new, T_ref_new, _, newly_solidified, _ = (
            base.update_phase_reference_and_eqp(
                T_quad, active_quad, phase_quad, T_ref_quad, eqp_quad, args
            )
        )
        return base, phase_new, T_ref_new, newly_solidified

    def test_consolidation_mode_writes_relaxation_reference(self):
        base, phase_new, T_ref_new, newly_solidified = self.run_update(1100.0)
        self.assertTrue(bool(onp.all(onp.asarray(newly_solidified))))
        self.assertTrue(
            bool(onp.all(onp.asarray(phase_new) == base.STATE_SOLID))
        )
        self.assertTrue(bool(onp.all(onp.asarray(T_ref_new) == 1100.0)))

    def test_without_relaxation_reference_keeps_local_temperature(self):
        _, _, T_ref_new, _ = self.run_update(None)
        self.assertTrue(bool(onp.all(onp.asarray(T_ref_new) == 300.0)))


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class InactiveMassFactorTest(unittest.TestCase):
    def material_args(self, **overrides):
        args = SimpleNamespace(
            rho=4000.0, cp=500.0, conductivity=10.0,
            rho_solid=4000.0, cp_solid=500.0, conductivity_solid=10.0,
            rho_liquid=None, cp_liquid=None, conductivity_liquid=None,
            rho_powder=2400.0, cp_powder=500.0, conductivity_powder=1.0,
            inactive_thermal_factor=1e-6,
            inactive_mass_factor=None,
            old_layer_thermal_factor=1e-6,
            solidus_temperature=0.0, liquidus_temperature=0.0,
            latent_heat=0.0,
            layer_activation_mode="layer_on_scan",
            future_layer_mode="void",
            powder_mode="powder",
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def void_props(self, args):
        base = load_v03()
        tables = {k: None for k in (
            "cp_solid", "k_solid", "cp_powder", "k_powder",
            "cp_liquid", "k_liquid",
        )}
        T = 300.0 * jnp.ones((1, 1, 1))
        active = jnp.zeros((1, 1, 1))
        phase = base.STATE_VOID * jnp.ones((1, 1, 1))
        printed = jnp.zeros((1, 1, 1))
        rho, cp, k, _ = base.thermal_material_quads(
            T, active, phase, args, tables, printed_quad=printed
        )
        return float(rho[0, 0, 0]), float(cp[0, 0, 0]), float(k[0, 0, 0])

    def test_legacy_void_diffusivity_is_solid_like(self):
        rho, cp, k = self.void_props(self.material_args())
        alpha_void = k / (rho * cp)
        alpha_solid = 10.0 / (4000.0 * 500.0)
        self.assertAlmostEqual(alpha_void / alpha_solid, 1.0, places=6)

    def test_full_mass_factor_kills_void_diffusivity(self):
        rho, cp, k = self.void_props(self.material_args(inactive_mass_factor=1.0))
        self.assertAlmostEqual(rho, 4000.0)
        alpha_void = k / (rho * cp)
        alpha_solid = 10.0 / (4000.0 * 500.0)
        self.assertLess(alpha_void / alpha_solid, 1e-5)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class YieldSaturationAndFoundationTest(unittest.TestCase):
    def seq_of(self, sigma):
        sigma = onp.asarray(sigma)
        hydro = onp.trace(sigma) / 3.0 * onp.eye(3)
        dev = sigma - hydro
        return onp.sqrt(1.5 * onp.sum(dev * dev))

    def eval_stress(self, yield_saturation, eqp_old):
        base = load_v03()
        fake_self = SimpleNamespace(
            mechanics_model="j2_plastic", dim=3,
            yield_saturation=yield_saturation,
        )
        u_grad = jnp.asarray(
            [[0.02, 0.0, 0.0], [0.0, -0.004, 0.0], [0.0, 0.0, -0.004]]
        )
        return base.ThermoMechanical.stress_fn(
            fake_self, u_grad,
            jnp.asarray([300.0]), jnp.asarray([0.0]), jnp.asarray([1.0]),
            jnp.asarray([125.0e9]), jnp.asarray([0.0]), jnp.asarray([0.3]),
            jnp.asarray([500.0e6]), jnp.asarray([2.0e9]), jnp.asarray([eqp_old]),
        )

    def test_saturation_caps_hardened_yield(self):
        # eqp_old = 0.5 -> hardened yield 500e6 + 2e9*0.5 = 1.5 GPa unbounded
        unbounded = self.seq_of(self.eval_stress(None, 0.5))
        capped = self.seq_of(self.eval_stress(1.1e9, 0.5))
        self.assertGreater(unbounded, 1.4e9)
        self.assertAlmostEqual(capped / 1.1e9, 1.0, places=6)

    def test_saturation_off_below_cap_matches_unbounded(self):
        a = self.seq_of(self.eval_stress(None, 0.01))
        b = self.seq_of(self.eval_stress(5.0e9, 0.01))
        self.assertAlmostEqual(a, b, delta=1.0)

    def test_foundation_surface_map_returns_spring_traction(self):
        base = load_v03()
        fake_self = SimpleNamespace(foundation_stiffness=1.0e12)
        maps = base.ThermoMechanical.get_surface_maps(fake_self)
        self.assertEqual(len(maps), 1)
        u = jnp.asarray([1.0e-3, -2.0e-3, 0.5e-3])
        t = onp.asarray(maps[0](u, jnp.zeros(3)))
        onp.testing.assert_allclose(t, [1.0e9, -2.0e9, 0.5e9])

        fake_off = SimpleNamespace(foundation_stiffness=0.0)
        self.assertEqual(base.ThermoMechanical.get_surface_maps(fake_off), [])


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class RadialReturnStressTest(unittest.TestCase):
    def eval_stress(self, hardening, eqp_old=0.0):
        base = load_v03()
        fake_self = SimpleNamespace(
            mechanics_model="j2_plastic", dim=3, yield_saturation=None
        )
        u_grad = jnp.asarray(
            [[0.01, 0.0, 0.0], [0.0, -0.002, 0.0], [0.0, 0.0, -0.002]]
        )
        return base.ThermoMechanical.stress_fn(
            fake_self,
            u_grad,
            jnp.asarray([300.0]),      # T
            jnp.asarray([0.0]),        # dT
            jnp.asarray([1.0]),        # active_factor
            jnp.asarray([125.0e9]),    # young
            jnp.asarray([0.0]),        # alpha
            jnp.asarray([0.3]),        # poisson
            jnp.asarray([500.0e6]),    # yield
            jnp.asarray([hardening]),
            jnp.asarray([eqp_old]),
        )

    def test_zero_hardening_matches_legacy_clip(self):
        sigma = onp.asarray(self.eval_stress(0.0))
        hydro = onp.trace(sigma) / 3.0 * onp.eye(3)
        dev = sigma - hydro
        seq = onp.sqrt(1.5 * onp.sum(dev * dev))
        # legacy clip pins seq exactly at the yield stress
        self.assertAlmostEqual(seq / 500.0e6, 1.0, places=9)

    def test_hardening_expands_yield_surface_within_solve(self):
        sigma = onp.asarray(self.eval_stress(2.0e9))
        hydro = onp.trace(sigma) / 3.0 * onp.eye(3)
        dev = sigma - hydro
        seq = onp.sqrt(1.5 * onp.sum(dev * dev))
        self.assertGreater(seq, 500.0e6)          # above initial yield
        self.assertLess(seq, 700.0e6)             # but well below trial


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class ActivationTemperatureResetTest(unittest.TestCase):
    def test_reset_only_touches_nodes_exclusive_to_new_cells(self):
        base = load_v03()
        cells = onp.array([[0, 1, 2, 3], [2, 3, 4, 5]])
        T_old = jnp.asarray(
            [[400.0], [410.0], [420.0], [430.0], [900.0], [950.0]]
        )
        newly_printed = onp.array([False, True])
        previous_active = onp.array([True, False])

        T_reset = base.reset_new_cell_nodal_temperature(
            T_old, cells, newly_printed, previous_active, 300.0
        )
        result = onp.asarray(T_reset)[:, 0]
        # nodes 2,3 shared with old material keep their temperature
        self.assertEqual(list(result[:4]), [400.0, 410.0, 420.0, 430.0])
        # nodes 4,5 exclusive to the new cell reset to powder temperature
        self.assertEqual(list(result[4:]), [300.0, 300.0])

    def test_no_previous_active_resets_all_new_nodes(self):
        base = load_v03()
        cells = onp.array([[0, 1, 2, 3]])
        T_old = jnp.asarray([[500.0], [500.0], [500.0], [500.0]])
        T_reset = base.reset_new_cell_nodal_temperature(
            T_old,
            cells,
            onp.array([True]),
            onp.array([False]),
            300.0,
        )
        self.assertEqual(list(onp.asarray(T_reset)[:, 0]), [300.0] * 4)


if __name__ == "__main__":
    unittest.main()
