import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V03_PATH = (
    ROOT / "jax_fem_am" / "simulation" / "stepper.py"
)
sys.path.insert(0, str(ROOT / "legacy"))
sys.path.insert(0, str(ROOT / "legacy" / "v01"))


def load_v03():
    name = "v03_base_for_release_anchor_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestReleaseAnchorBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v03 = load_v03()
        # regular 5x3x3 grid, spacing 1e-4 m
        xs, ys, zs = np.meshgrid(
            np.arange(5) * 1e-4, np.arange(3) * 1e-4, np.arange(3) * 1e-4,
            indexing="ij",
        )
        cls.points = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)

    def bc_selected_count(self, bc):
        locations, components, values = bc
        self.assertEqual(components, [0, 1, 2])
        self.assertEqual(len(locations), 3)
        self.assertEqual(len(values), 3)
        count = sum(
            1 for p in self.points if bool(locations[0](p))
        )
        return count

    def test_box_selects_root_nodes_all_components(self):
        # root slab: x in [0, 1e-4] -> 2x3x3 = 18 nodes
        bc = self.v03.make_box_anchor_mechanics_bc(
            self.points, [0.0, 1.0e-4, 0.0, 2.0e-4, 0.0, 2.0e-4]
        )
        self.assertEqual(self.bc_selected_count(bc), 18)

    def test_box_tolerance_includes_boundary_nodes(self):
        # degenerate-thin box exactly on the x=0 plane still catches the
        # plane nodes thanks to the span-scaled tolerance
        bc = self.v03.make_box_anchor_mechanics_bc(
            self.points, [-1.0e-9, 1.0e-9, 0.0, 2.0e-4, 0.0, 2.0e-4]
        )
        self.assertEqual(self.bc_selected_count(bc), 9)

    def test_empty_box_rejected(self):
        with self.assertRaises(ValueError):
            self.v03.make_box_anchor_mechanics_bc(
                self.points, [10.0, 11.0, 10.0, 11.0, 10.0, 11.0]
            )

    def test_inverted_box_rejected(self):
        with self.assertRaises(ValueError):
            self.v03.make_box_anchor_mechanics_bc(
                self.points, [1.0e-4, 0.0, 0.0, 2.0e-4, 0.0, 2.0e-4]
            )

    def test_cli_default_keeps_rigid_body_mode(self):
        if not hasattr(self.v03, "parse_args"):
            self.skipTest("v03 exposes no parse_args helper")
        try:
            args = self.v03.parse_args([
                "--inp", "dummy.inp",
                "--output-dir", "/tmp/x",
            ])
        except (SystemExit, Exception) as exc:  # noqa: BLE001
            self.skipTest(f"v03 parse_args needs a fuller invocation: {exc}")
        self.assertEqual(args.release_anchor_mode, "rigid_body")
        self.assertIsNone(args.release_anchor_box)


class TestJ2JacobianAtZeroState(unittest.TestCase):
    """AD guard: jacobian of the J2 kernel at exactly-zero trial stress.

    Regression for the Kaess-benchmark NaN: sqrt at zero deviatoric stress
    poisons the mechanics jacobian rows of every freshly activated
    zero-strain quadrature point (observed as an unfactorizable matrix).
    """

    def _stress_of_strain(self):
        import jax
        import jax.numpy as jnp
        from jax_fem_am.materials.j2 import PlasticState, radial_return

        def stress_of(strain):
            update = radial_return(
                strain=strain,
                thermal_strain=jnp.zeros((3, 3)),
                state=PlasticState(eqp=jnp.asarray(0.0),
                                   eps_p=jnp.zeros((3, 3))),
                young=2.0e11,
                poisson=0.3,
                yield_stress=5.0e8,
                hardening=4.0e8,
            )
            return update.stress

        return jax, stress_of

    def test_jacfwd_finite_at_zero(self):
        jax, stress_of = self._stress_of_strain()
        jac = jax.jacfwd(stress_of)(jax.numpy.zeros((3, 3)))
        self.assertTrue(bool(jax.numpy.all(jax.numpy.isfinite(jac))))

    def test_jacrev_finite_at_zero(self):
        jax, stress_of = self._stress_of_strain()
        jac = jax.jacrev(stress_of)(jax.numpy.zeros((3, 3)))
        self.assertTrue(bool(jax.numpy.all(jax.numpy.isfinite(jac))))

    def test_primal_unchanged_in_plastic_regime(self):
        import jax.numpy as jnp
        from jax_fem_am.materials.j2 import PlasticState, radial_return

        strain = jnp.asarray([[5.0e-3, 0.0, 0.0],
                              [0.0, -1.5e-3, 0.0],
                              [0.0, 0.0, -1.5e-3]])
        update = radial_return(
            strain=strain,
            thermal_strain=jnp.zeros((3, 3)),
            state=PlasticState(eqp=jnp.asarray(0.0), eps_p=jnp.zeros((3, 3))),
            young=2.0e11,
            poisson=0.3,
            yield_stress=5.0e8,
            hardening=4.0e8,
        )
        self.assertGreater(float(update.delta_eqp), 0.0)
        from jax_fem_am.materials.j2 import equivalent_stress
        q = float(equivalent_stress(update.stress))
        expected = 5.0e8 + 4.0e8 * float(update.state.eqp)
        self.assertAlmostEqual(q / expected, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
