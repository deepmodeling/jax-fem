import sys
import tempfile
import unittest
from pathlib import Path

import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "159_local"))

from v06.verification.run_audit import audit_run, audit_solution_fields  # noqa: E402


def regular_tetrahedron(offset):
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
        ]
    ) + np.asarray(offset)


class RunAuditTest(unittest.TestCase):
    def test_quality_filtered_stress_excludes_sliver_hotspot(self):
        regular = regular_tetrahedron([0.0, 0.0, 0.0])
        sliver = regular_tetrahedron([2.0, 0.0, 0.0])
        sliver[3, 2] = 1.0e-7
        points = np.vstack([regular, sliver])
        cells = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])
        result = audit_solution_fields(
            points=points,
            cells=cells,
            temperature=np.full(8, 300.0),
            displacement=np.zeros((8, 3)),
            vm_quad=np.asarray([[10.0], [1000.0]]),
            eqp=np.zeros(2),
            printed=np.ones(2, dtype=bool),
            mechanics_valid=np.ones(2),
            ambient=300.0,
            quality_threshold=0.05,
        )

        self.assertEqual(result["stress"]["diagnostic_global_max"], 1000.0)
        self.assertEqual(result["stress"]["diagnostic_global_max_cell"], 1)
        self.assertLess(
            result["stress"]["diagnostic_global_max_cell_quality"],
            0.05,
        )
        self.assertEqual(result["stress"]["quality_filtered_max"], 10.0)
        self.assertEqual(result["stress"]["quality_filtered_cell_count"], 1)

    def test_nonfinite_temperature_marks_solution_invalid(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        result = audit_solution_fields(
            points=points,
            cells=np.asarray([[0, 1, 2, 3]]),
            temperature=np.asarray([300.0, np.nan, 300.0, 300.0]),
            displacement=np.zeros((4, 3)),
            vm_quad=np.zeros((1, 1)),
            eqp=np.zeros(1),
            printed=np.ones(1, dtype=bool),
            mechanics_valid=np.ones(1),
            ambient=300.0,
        )

        self.assertFalse(result["valid"])
        self.assertFalse(result["temperature"]["all_finite"])

    def test_temperature_below_ambient_marks_solution_invalid(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        result = audit_solution_fields(
            points=points,
            cells=np.asarray([[0, 1, 2, 3]]),
            temperature=np.asarray([299.0, 300.0, 300.0, 300.0]),
            displacement=np.zeros((4, 3)),
            vm_quad=np.zeros((1, 1)),
            eqp=np.zeros(1),
            printed=np.ones(1, dtype=bool),
            mechanics_valid=np.ones(1),
            ambient=300.0,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["temperature"]["below_ambient_count"], 1)

    def test_temperature_gate_uses_explicit_solver_scale_tolerance(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        common = dict(
            points=points,
            cells=np.asarray([[0, 1, 2, 3]]),
            displacement=np.zeros((4, 3)),
            vm_quad=np.zeros((1, 1)),
            eqp=np.zeros(1),
            printed=np.ones(1),
            mechanics_valid=np.ones(1),
            ambient=300.0,
            temperature_atol_k=1.0e-3,
        )

        accepted = audit_solution_fields(
            temperature=np.asarray([299.9995, 300.0, 300.0, 300.0]),
            **common,
        )
        rejected = audit_solution_fields(
            temperature=np.asarray([299.99, 300.0, 300.0, 300.0]),
            **common,
        )

        self.assertTrue(accepted["valid"])
        self.assertFalse(rejected["valid"])
        self.assertEqual(accepted["temperature"]["absolute_tolerance_k"], 1.0e-3)

    def test_explicit_source_free_upper_bound_marks_overshoot_invalid(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        result = audit_solution_fields(
            points=points,
            cells=np.asarray([[0, 1, 2, 3]]),
            temperature=np.asarray([300.0, 1100.0, 1100.0, 1772.0]),
            displacement=np.zeros((4, 3)),
            vm_quad=np.zeros((1, 1)),
            eqp=np.zeros(1),
            printed=np.ones(1, dtype=bool),
            mechanics_valid=np.ones(1),
            ambient=300.0,
            source_free_upper_bound=1100.0,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["temperature"]["above_upper_bound_count"], 1)

    def test_no_cells_survive_quality_gate_marks_solution_invalid(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        points[3, 2] = 1.0e-7
        result = audit_solution_fields(
            points=points,
            cells=np.asarray([[0, 1, 2, 3]]),
            temperature=np.full(4, 300.0),
            displacement=np.zeros((4, 3)),
            vm_quad=np.zeros((1, 1)),
            eqp=np.zeros(1),
            printed=np.ones(1, dtype=bool),
            mechanics_valid=np.ones(1),
            ambient=300.0,
            quality_threshold=0.05,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["stress"]["quality_filtered_cell_count"], 0)

    def test_negative_von_mises_or_plastic_strain_is_rejected(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        common = dict(
            points=points,
            cells=np.asarray([[0, 1, 2, 3]]),
            temperature=np.full(4, 300.0),
            displacement=np.zeros((4, 3)),
            printed=np.ones(1, dtype=bool),
            mechanics_valid=np.ones(1),
            ambient=300.0,
        )

        self.assertFalse(
            audit_solution_fields(
                vm_quad=np.asarray([[-1.0]]), eqp=np.zeros(1), **common
            )["valid"]
        )
        self.assertFalse(
            audit_solution_fields(
                vm_quad=np.zeros((1, 1)), eqp=np.asarray([-1.0]), **common
            )["valid"]
        )

    def test_nonfinite_ambient_and_state_flags_are_rejected(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        common = dict(
            points=points,
            cells=np.asarray([[0, 1, 2, 3]]),
            temperature=np.full(4, 300.0),
            displacement=np.zeros((4, 3)),
            vm_quad=np.zeros((1, 1)),
            eqp=np.zeros(1),
            quality_threshold=0.05,
        )

        with self.assertRaisesRegex(ValueError, "ambient"):
            audit_solution_fields(
                printed=np.ones(1),
                mechanics_valid=np.ones(1),
                ambient=np.nan,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "state flags"):
            audit_solution_fields(
                printed=np.asarray([np.nan]),
                mechanics_valid=np.ones(1),
                ambient=300.0,
                **common,
            )

    def test_volume_weighted_stress_uses_quadrature_values_not_cell_max(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        result = audit_solution_fields(
            points=points,
            cells=np.asarray([[0, 1, 2, 3]]),
            temperature=np.full(4, 300.0),
            displacement=np.zeros((4, 3)),
            vm_quad=np.asarray([[0.0, 100.0]]),
            eqp=np.zeros(1),
            printed=np.ones(1, dtype=bool),
            mechanics_valid=np.ones(1),
            ambient=300.0,
        )

        self.assertEqual(
            result["stress"]["quality_filtered_volume_weighted_mean"],
            50.0,
        )

    def test_run_audit_rejects_bad_earlier_step_even_if_last_step_is_valid(self):
        points = regular_tetrahedron([0.0, 0.0, 0.0])
        cells = np.asarray([[0, 1, 2, 3]])

        def write_vtu(path, temperature):
            meshio.write(
                path,
                meshio.Mesh(
                    points=points,
                    cells=[("tetra", cells)],
                    point_data={
                        "T": np.asarray(temperature, dtype=float),
                        "u": np.zeros((4, 3)),
                    },
                    cell_data={
                        "vm_quad": [np.zeros(1)],
                        "eq_plastic_strain": [np.zeros(1)],
                        "printed": [np.ones(1)],
                        "mechanics_valid": [np.ones(1)],
                    },
                ),
            )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_vtu(run_dir / "step_000000_scan.vtu", [-1.0, 300, 300, 300])
            write_vtu(run_dir / "step_000001_cooling.vtu", [300, 300, 300, 300])
            write_vtu(run_dir / "release.vtu", [300, 300, 300, 300])

            result = audit_run(run_dir, ambient=300.0)

        self.assertFalse(result["transient"]["all_steps_valid"])
        self.assertEqual(result["transient"]["invalid_step_count"], 1)
        self.assertEqual(result["transient"]["minimum_temperature"], -1.0)


if __name__ == "__main__":
    unittest.main()
