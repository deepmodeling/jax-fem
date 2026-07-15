"""Lateral powder-bed constraint (horizontal Winkler springs) unit tests.

The powder bed surrounding the printed layers supports their horizontal
motion. v03/v06 model it as face springs on the exterior side surfaces,
gated per step by the printed state and absent from the release problem.
These tests exercise the pure pieces (kernel, mask plumbing, option
defaults) without running a solve.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
V03_PATH = (
    ROOT / "159_local" / "v03"
    / "am_thermal_stress_macro_intersection_mech100.py"
)
sys.path.insert(0, str(ROOT / "159_local"))
sys.path.insert(0, str(ROOT / "159_local" / "v01"))


def load_fresh_v03(name):
    spec = importlib.util.spec_from_file_location(name, V03_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_bare_mechanics(base, **attrs):
    obj = base.ThermoMechanical.__new__(base.ThermoMechanical)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def fake_fes(num_cells=4, num_quads=1, num_face_quads=3):
    fe = SimpleNamespace(
        cells=np.zeros((num_cells, 4), dtype=int),
        num_quads=num_quads,
        num_face_quads=num_face_quads,
    )
    return [fe]


class PowderSurfaceMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_fresh_v03("v03_powder_constraint_test")

    def test_map_order_matches_location_fn_order(self):
        # [bottom, powder] when both foundations are enabled.
        both = make_bare_mechanics(
            self.base,
            foundation_stiffness=1.0e12,
            powder_foundation_stiffness=1.0e9,
            powder_axis_mask=jnp.array([0.0, 1.0, 1.0]),
        )
        self.assertEqual(len(both.get_surface_maps()), 2)

        bottom_only = make_bare_mechanics(
            self.base,
            foundation_stiffness=1.0e12,
            powder_foundation_stiffness=0.0,
        )
        self.assertEqual(len(bottom_only.get_surface_maps()), 1)

        powder_only = make_bare_mechanics(
            self.base,
            foundation_stiffness=0.0,
            powder_foundation_stiffness=1.0e9,
            powder_axis_mask=jnp.array([0.0, 1.0, 1.0]),
        )
        self.assertEqual(len(powder_only.get_surface_maps()), 1)

        neither = make_bare_mechanics(
            self.base,
            foundation_stiffness=0.0,
            powder_foundation_stiffness=0.0,
        )
        self.assertEqual(neither.get_surface_maps(), [])

    def test_powder_traction_is_horizontal_only_and_face_gated(self):
        k_p = 2.5e8
        # Build axis x -> plane axes (y, z) carry the springs.
        mech = make_bare_mechanics(
            self.base,
            foundation_stiffness=0.0,
            powder_foundation_stiffness=k_p,
            powder_axis_mask=jnp.array([0.0, 1.0, 1.0]),
        )
        (powder_traction,) = mech.get_surface_maps()
        u = jnp.array([1.0, 2.0, -3.0])
        point = jnp.zeros(3)

        active = np.asarray(powder_traction(u, point, jnp.array([1.0])))
        np.testing.assert_allclose(active, [0.0, k_p * 2.0, k_p * -3.0])

        gated = np.asarray(powder_traction(u, point, jnp.array([0.0])))
        np.testing.assert_allclose(gated, [0.0, 0.0, 0.0])

    def test_powder_traction_jacobian_is_symmetric_psd_diagonal(self):
        # d(traction)/du must be diag(0, k, k): no coupling into the build
        # axis, positive semi-definite so Newton stays well-posed.
        k_p = 1.0e9
        mech = make_bare_mechanics(
            self.base,
            foundation_stiffness=0.0,
            powder_foundation_stiffness=k_p,
            powder_axis_mask=jnp.array([0.0, 1.0, 1.0]),
        )
        (powder_traction,) = mech.get_surface_maps()
        jac = jax.jacfwd(
            lambda u: powder_traction(u, jnp.zeros(3), jnp.array([1.0]))
        )(jnp.array([0.5, -0.25, 4.0]))
        np.testing.assert_allclose(
            np.asarray(jac), np.diag([0.0, k_p, k_p])
        )


class PowderSurfaceMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_fresh_v03("v03_powder_mask_test")

    def make_masked_mechanics(self, boundary_inds_list, num_cells):
        return make_bare_mechanics(
            self.base,
            foundation_stiffness=0.0,
            powder_foundation_stiffness=1.0e9,
            powder_axis_mask=jnp.array([0.0, 1.0, 1.0]),
            powder_boundary_index=len(boundary_inds_list) - 1,
            boundary_inds_list=boundary_inds_list,
            fes=fake_fes(num_cells=num_cells),
            internal_vars_surfaces=[() for _ in boundary_inds_list],
        )

    def test_faces_inherit_owner_cell_printed_flag(self):
        # Faces owned by cells 0, 2, 3; only cell 2 printed.
        boundary_inds = np.array([[0, 1], [2, 0], [3, 2]])
        mech = self.make_masked_mechanics([boundary_inds], num_cells=4)
        mech.set_powder_surface_mask(
            np.array([False, False, True, False])
        )
        (face_powder,) = mech.internal_vars_surfaces[0]
        self.assertEqual(face_powder.shape, (3, 3, 1))
        np.testing.assert_allclose(
            np.asarray(face_powder)[:, 0, 0], [0.0, 1.0, 0.0]
        )

    def test_mask_targets_last_boundary_when_bottom_present(self):
        bottom_inds = np.array([[1, 3]])
        powder_inds = np.array([[0, 1], [3, 0]])
        mech = self.make_masked_mechanics(
            [bottom_inds, powder_inds], num_cells=4
        )
        mech.set_powder_surface_mask(np.array([True, False, False, True]))
        # Bottom boundary untouched (its kernel takes no surface vars).
        self.assertEqual(mech.internal_vars_surfaces[0], ())
        (face_powder,) = mech.internal_vars_surfaces[1]
        np.testing.assert_allclose(
            np.asarray(face_powder)[:, 0, 0], [1.0, 1.0]
        )

    def test_noop_when_powder_disabled(self):
        mech = make_bare_mechanics(
            self.base,
            foundation_stiffness=1.0e12,
            powder_foundation_stiffness=0.0,
            internal_vars_surfaces=[()],
        )
        mech.set_powder_surface_mask(np.array([True]))
        self.assertEqual(mech.internal_vars_surfaces, [()])

    def test_empty_powder_boundary_is_tolerated(self):
        mech = self.make_masked_mechanics(
            [np.zeros((0, 2), dtype=int)], num_cells=4
        )
        mech.set_powder_surface_mask(np.array([True, True, True, True]))
        self.assertEqual(mech.internal_vars_surfaces[0], [])


class PowderCliDefaultsTest(unittest.TestCase):
    def test_defaults_keep_legacy_free_sides(self):
        base = load_fresh_v03("v03_powder_cli_test")
        argv_backup = sys.argv
        try:
            sys.argv = ["prog"]
            args = base.parse_args()
        finally:
            sys.argv = argv_backup
        self.assertEqual(args.powder_mechanics_bc, "none")
        self.assertEqual(args.powder_foundation_stiffness, 1.0e9)


if __name__ == "__main__":
    unittest.main()
