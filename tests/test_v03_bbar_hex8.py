"""HEX8 + B-bar mechanics (volumetric-locking fix, 2026-07-21).

Three layers of evidence:
  1. consistency - the B-bar universal kernel reproduces the plain
     tensor-map residual wherever tr(grad u) is element-constant
     (TET4 always; HEX8 under an affine displacement field);
  2. cure - on a bottom-clamped HEX8 block cooled into plastic flow, the
     checkerboard hydrostatic-pressure oscillation (the diagnosed TET4+J2
     locking fingerprint) collapses when B-bar is on;
  3. wiring - --mechanics-bbar parsing, HEX8 inp reading, HEX8 ELSET masks.
"""

import importlib.util
import sys
import tempfile
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
    from jax_fem.generate_mesh import Mesh
except ImportError as exc:  # pragma: no cover - depends on local runtime
    onp = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

# Kuhn split (same table as make_kaess_mesh.py) - globally conforming on a
# structured grid when applied with identical orientation to every hex.
HEX_TO_TETS = (
    (0, 1, 2, 6),
    (0, 1, 5, 6),
    (0, 3, 2, 6),
    (0, 3, 7, 6),
    (0, 4, 5, 6),
    (0, 4, 7, 6),
)


def load_v03():
    if str(V01_DIR) not in sys.path:
        sys.path.insert(0, str(V01_DIR))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("v03_bbar_test_base", V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def box_grid(n, L=1.0e-3):
    """Structured (n+1)^3 nodes / n^3 hexes on [0, L]^3, VTK corner order."""
    xs = onp.linspace(0.0, L, n + 1)
    pts = onp.array([[xs[i], xs[j], xs[k]]
                     for k in range(n + 1)
                     for j in range(n + 1)
                     for i in range(n + 1)])
    nid = lambda i, j, k: (k * (n + 1) + j) * (n + 1) + i
    hexes = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                hexes.append((
                    nid(i, j, k), nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ))
    return pts, onp.array(hexes, dtype=onp.int64)


def kuhn_tets(hexes):
    tets = []
    for h in hexes:
        for t in HEX_TO_TETS:
            tets.append([h[c] for c in t])
    return onp.array(tets, dtype=onp.int64)


def orient_tets(points, tets):
    p = points[tets]
    signed = onp.einsum("ij,ij->i",
                        onp.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                        p[:, 3] - p[:, 0])
    flipped = tets.copy()
    neg = signed < 0
    flipped[neg] = flipped[neg][:, [0, 2, 1, 3]]
    return flipped


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class BBarTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v03 = load_v03()

    def build_mechanics(self, points, cells, ele_type, bbar,
                        mechanics_model="j2_plastic"):
        def bottom(point):
            return point[2] < 1e-9

        def zero(_point):
            return 0.0

        bc = [[bottom] * 3, [0, 1, 2], [zero] * 3]
        mesh = Mesh(points, cells, ele_type=ele_type)
        return self.v03.ThermoMechanical(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type=ele_type,
            quadrature_order=2,
            dirichlet_bc_info=bc,
            additional_info=(mechanics_model, None, 0.0, 0.0, (), bbar),
        )

    def uniform_params(self, problem, dT=-200.0, yield_stress=200e6):
        num_cells = len(problem.fes[0].cells)
        num_quads = problem.fes[0].num_quads
        full = lambda v: v * jnp.ones((num_cells, num_quads, 1))
        return [
            full(500.0),          # T
            full(dT),             # dT (uniform cooling)
            full(1.0),            # active_factor
            full(190e9),          # young
            full(16e-6),          # alpha
            full(0.3),            # poisson
            full(yield_stress),   # yield
            full(1e8),            # hardening
            full(0.0),            # eqp_old
        ]

    def residual_flat(self, problem, sol):
        problem.set_params(self.uniform_params(problem))
        res = problem.compute_residual([sol])
        return onp.asarray(res[0])


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class BBarConsistencyTest(BBarTestBase):
    def smooth_sol(self, points):
        L = 1.0e-3
        x = onp.asarray(points) / L
        u = onp.stack([
            1e-6 * onp.sin(x[:, 0]) * x[:, 1] ** 2,
            1e-6 * (x[:, 2] ** 2 - 0.3 * x[:, 0] * x[:, 1]),
            1e-6 * onp.cos(x[:, 1]) * x[:, 2],
        ], axis=1)
        return jnp.asarray(u)

    def test_tet4_bbar_is_exact_noop(self):
        # TET4 strain is element-constant: theta_bar == theta pointwise, so
        # the B-bar kernel must reproduce the tensor-map residual for ANY
        # displacement field.
        points, hexes = box_grid(2)
        tets = orient_tets(points, kuhn_tets(hexes))
        sol = self.smooth_sol(points)
        res = {}
        for bbar in (False, True):
            problem = self.build_mechanics(points, tets, "TET4", bbar)
            res[bbar] = self.residual_flat(problem, sol)
        scale = onp.abs(res[False]).max()
        onp.testing.assert_allclose(res[True], res[False], atol=1e-10 * scale)

    def test_hex8_bbar_matches_plain_on_affine_field(self):
        # Affine u -> uniform strain -> element-average dilatation equals the
        # pointwise one: B-bar must agree with the plain kernel (patch-test
        # level consistency of the implementation).
        points, hexes = box_grid(2)
        A = onp.array([[2.0, 0.3, -0.1],
                       [0.1, -1.0, 0.4],
                       [-0.2, 0.2, 0.5]]) * 1e-4
        sol = jnp.asarray(onp.asarray(points) @ A.T)
        res = {}
        for bbar in (False, True):
            problem = self.build_mechanics(points, hexes, "HEX8", bbar)
            res[bbar] = self.residual_flat(problem, sol)
        scale = onp.abs(res[False]).max()
        onp.testing.assert_allclose(res[True], res[False], atol=1e-10 * scale)

    def test_hex8_bbar_differs_on_nonuniform_field(self):
        # Sanity that the kernel switch is actually live: a non-affine field
        # must produce a different residual with B-bar on.
        points, hexes = box_grid(2)
        sol = self.smooth_sol(points)
        res = {}
        for bbar in (False, True):
            problem = self.build_mechanics(points, hexes, "HEX8", bbar)
            res[bbar] = self.residual_flat(problem, sol)
        diff = onp.abs(res[True] - res[False]).max()
        self.assertGreater(diff, 1e-8 * onp.abs(res[False]).max())


# production-grade Newton settings (run_kaess_phase2.sh): the default
# tol 1e-9 / rel_tol 1e-11 sits below the documented j2 tangent/residual
# stall floor (~2e-5 relative) and never terminates on heavily plastic
# problems.
NEWTON = {"rel_tol": 5e-5, "tol": 1e-3, "max_iter": 50, "line_search_flag": True}


def beam_grid(nx, ny, nz, Lx, Ly, Lz):
    xs = onp.linspace(0.0, Lx, nx + 1)
    ys = onp.linspace(0.0, Ly, ny + 1)
    zs = onp.linspace(0.0, Lz, nz + 1)
    pts = onp.array([[xs[i], ys[j], zs[k]]
                     for k in range(nz + 1)
                     for j in range(ny + 1)
                     for i in range(nx + 1)])
    nid = lambda i, j, k: (k * (ny + 1) + j) * (nx + 1) + i
    hexes = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                hexes.append((
                    nid(i, j, k), nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ))
    return pts, onp.array(hexes, dtype=onp.int64)


def demeaned_std(vals, group):
    out = vals.copy()
    for g in onp.unique(group):
        m = group == g
        out[m] -= out[m].mean()
    return float(onp.std(out))


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class BBarLockingCureTest(BBarTestBase):
    """Thermal-gradient cantilever (cool the top more than the bottom ->
    bimetallic curvature). Volumetric locking shows up as (a) an overstiff
    tip deflection and (b) x-slab-demeaned hydrostatic pressure scatter far
    above the deviatoric scale - the checkerboard fingerprint diagnosed on
    the Kaess TET4 runs (2026-07-21). Probe-measured margins (12x3x3 grid):
    elastic nu=0.49 tip +52% / p-std 6.9x; J2 tip +24% / p-std 4.7x."""

    NX, NY, NZ = 12, 3, 3
    LX, LY, LZ = 1.2e-3, 0.3e-3, 0.3e-3

    def solve_beam(self, bbar, model, nu, yield_stress):
        points, hexes = beam_grid(self.NX, self.NY, self.NZ,
                                  self.LX, self.LY, self.LZ)
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
            additional_info=(model, None, 0.0, 0.0, (), bbar))
        params = self.uniform_params(problem, dT=0.0, yield_stress=yield_stress)
        params[5] = nu * jnp.ones_like(params[5])
        dT_cell = -400.0 * centroids[:, 2] / self.LZ
        params[1] = jnp.asarray(onp.broadcast_to(
            dT_cell[:, None, None],
            (len(hexes), problem.fes[0].num_quads, 1)).copy())
        u = self.v03.run_mechanics(
            problem, [jnp.zeros((len(points), 3))], params, NEWTON)
        tips = onp.where(onp.asarray(points)[:, 0] > self.LX - 1e-9)[0]
        uz_tip = float(onp.asarray(u[0])[tips, 2].mean())
        qs = problem.compute_cell_stress(u[0], params)
        stress = onp.asarray(qs["stress_quad"])
        p = onp.trace(stress, axis1=-2, axis2=-1).mean(axis=1) / 3.0
        slab = onp.round(centroids[:, 0] / (self.LX / self.NX) - 0.5).astype(int)
        eqp = onp.asarray(problem.compute_eqp_update(u[0], params))
        return uz_tip, demeaned_std(p, slab), eqp

    def test_elastic_near_incompressible(self):
        uz_locked, pstd_locked, _ = self.solve_beam(
            False, "elastic", 0.49, 1e30)
        uz_bbar, pstd_bbar, _ = self.solve_beam(
            True, "elastic", 0.49, 1e30)
        # locking eats the bending deflection; B-bar releases it
        self.assertGreater(uz_bbar / uz_locked, 1.3,
                           f"tip uz locked {uz_locked:.3e} vs bbar {uz_bbar:.3e}")
        # and collapses the pressure checkerboard
        self.assertGreater(pstd_locked / pstd_bbar, 3.0,
                           f"p std locked {pstd_locked:.3e} vs bbar {pstd_bbar:.3e}")

    def test_j2_plastic_flow(self):
        uz_locked, pstd_locked, eqp_locked = self.solve_beam(
            False, "j2_plastic", 0.3, 60e6)
        uz_bbar, pstd_bbar, eqp_bbar = self.solve_beam(
            True, "j2_plastic", 0.3, 60e6)
        # the load must actually drive plastic flow, or the test is vacuous
        self.assertGreater(float(eqp_locked.max()), 1e-4)
        self.assertGreater(float(eqp_bbar.max()), 1e-4)
        self.assertGreater(uz_bbar / uz_locked, 1.15,
                           f"tip uz locked {uz_locked:.3e} vs bbar {uz_bbar:.3e}")
        self.assertGreater(pstd_locked / pstd_bbar, 3.0,
                           f"p std locked {pstd_locked:.3e} vs bbar {pstd_bbar:.3e}")


TINY_HEX_INP = """*HEADING
tiny two-hex mesh with powder elset
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
9, 0.0, 0.0, 2.0
10, 1.0, 0.0, 2.0
11, 1.0, 1.0, 2.0
12, 0.0, 1.0, 2.0
*ELEMENT, TYPE=C3D8, ELSET=PART
1, 1, 2, 3, 4, 5, 6, 7, 8
2, 5, 6, 7, 8, 9, 10, 11, 12
*ELSET, ELSET=POWDER
2
"""


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class Hex8WiringTest(BBarTestBase):
    def write_inp(self, content):
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".inp", delete=False, dir=tempfile.gettempdir())
        f.write(content)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return f.name

    def test_read_solid_inp_hex(self):
        path = self.write_inp(TINY_HEX_INP)
        points, cells, selected, ele_type = self.v03.read_solid_inp(path, 0)
        self.assertEqual(ele_type, "HEX8")
        self.assertEqual(cells.shape, (2, 8))
        self.assertEqual(len(points), 12)
        self.assertEqual(selected, 2)

    def test_read_solid_inp_hex_rejects_max_cells(self):
        path = self.write_inp(TINY_HEX_INP)
        with self.assertRaises(ValueError):
            self.v03.read_solid_inp(path, 1)

    def test_read_inp_cell_set_hex(self):
        path = self.write_inp(TINY_HEX_INP)
        mask = self.v03.read_inp_cell_set(path, "POWDER", 2)
        onp.testing.assert_array_equal(mask, [False, True])

    def test_parser_wiring(self):
        parser = self.v03.build_parser()
        args = parser.parse_args(["--inp", "x.inp"])
        self.assertEqual(args.mechanics_bbar, "auto")
        args = parser.parse_args(["--inp", "x.inp", "--mechanics-bbar", "off"])
        self.assertEqual(args.mechanics_bbar, "off")


if __name__ == "__main__":
    unittest.main()
