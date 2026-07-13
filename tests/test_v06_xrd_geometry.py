import sys
import unittest
from pathlib import Path

import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "159_local"))

from v06.measurement.xrd import (  # noqa: E402
    compute_gauge_weights,
    predict_gauge_microstrain,
    tetra_box_intersection,
)


class TetBoxIntersectionTest(unittest.TestCase):
    def test_box_containing_tetra_recovers_exact_tetra_volume_and_centroid(self):
        tetra = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )

        result = tetra_box_intersection(
            tetra,
            center=np.asarray([0.5, 0.5, 0.5]),
            size=np.asarray([2.0, 2.0, 2.0]),
        )

        self.assertAlmostEqual(result.volume, 1.0 / 6.0, places=12)
        np.testing.assert_allclose(result.centroid, [0.25, 0.25, 0.25], atol=1e-12)

    def test_half_x_clipped_right_tetra_matches_analytic_volume_and_centroid(self):
        tetra = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )

        result = tetra_box_intersection(
            tetra,
            center=np.asarray([0.25, 0.5, 0.5]),
            size=np.asarray([0.5, 2.0, 2.0]),
        )

        self.assertAlmostEqual(result.volume, 7.0 / 48.0, places=12)
        np.testing.assert_allclose(
            result.centroid,
            [11.0 / 56.0, 15.0 / 56.0, 15.0 / 56.0],
            atol=1e-12,
        )

    def test_tetra_containing_box_recovers_box_volume(self):
        tetra = np.asarray(
            [
                [10.0, 10.0, 10.0],
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
            ]
        )

        result = tetra_box_intersection(
            tetra,
            center=np.zeros(3),
            size=np.ones(3),
        )

        self.assertAlmostEqual(result.volume, 1.0, places=12)
        np.testing.assert_allclose(result.centroid, np.zeros(3), atol=1e-12)

    def test_face_tangency_has_zero_intersection_volume(self):
        tetra = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )

        result = tetra_box_intersection(
            tetra,
            center=np.asarray([1.5, 0.0, 0.0]),
            size=np.asarray([1.0, 0.2, 0.2]),
        )

        self.assertEqual(result.volume, 0.0)


class GaugePredictionTest(unittest.TestCase):
    def test_six_tet_cube_has_full_coverage_and_recovers_constant_field(self):
        fixture = (
            ROOT
            / "159_local/v06/verification/fixtures/unit_cube_6tet.inp"
        )
        mesh = meshio.read(fixture)
        points = np.asarray(mesh.points)
        cells = np.asarray(mesh.cells_dict["tetra"])
        weights = compute_gauge_weights(
            points,
            cells,
            center=np.asarray([0.0005, 0.0005, 0.0005]),
            size=np.asarray([0.001, 0.001, 0.001]),
        )
        elastic = np.repeat(
            np.diag([1200.0e-6, -100.0e-6, 50.0e-6])[None, :, :],
            len(cells),
            axis=0,
        )

        prediction = predict_gauge_microstrain(
            elastic,
            weights,
            direction=np.asarray([1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(weights.material_fill_fraction, 1.0, places=10)
        self.assertEqual(weights.contributing_cell_count, 6)
        self.assertEqual(prediction["status"], "ok")
        self.assertAlmostEqual(prediction["predicted_microstrain"], 1200.0)
        self.assertEqual(prediction["input_unit"], "1")
        self.assertEqual(prediction["output_unit"], "microstrain")

    def test_partial_material_coverage_is_reported_and_not_silently_normalized(self):
        fixture = (
            ROOT
            / "159_local/v06/verification/fixtures/unit_cube_6tet.inp"
        )
        mesh = meshio.read(fixture)
        cells = np.asarray(mesh.cells_dict["tetra"])
        weights = compute_gauge_weights(
            np.asarray(mesh.points),
            cells,
            center=np.asarray([0.001, 0.0005, 0.0005]),
            size=np.asarray([0.001, 0.001, 0.001]),
        )
        elastic = np.repeat(np.eye(3)[None, :, :] * 1.0e-3, len(cells), axis=0)

        prediction = predict_gauge_microstrain(
            elastic,
            weights,
            direction=np.asarray([1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(weights.material_fill_fraction, 0.5, places=10)
        self.assertEqual(prediction["status"], "low_material_coverage")
        self.assertIsNone(prediction["predicted_microstrain"])

    def test_overlapping_tetrahedra_are_not_accepted_as_full_gauge_coverage(self):
        fixture = (
            ROOT
            / "159_local/v06/verification/fixtures/unit_cube_6tet.inp"
        )
        mesh = meshio.read(fixture)
        cells = np.asarray(mesh.cells_dict["tetra"])
        duplicate_cells = np.vstack([cells, cells])
        weights = compute_gauge_weights(
            np.asarray(mesh.points),
            duplicate_cells,
            center=np.asarray([0.0005, 0.0005, 0.0005]),
            size=np.asarray([0.001, 0.001, 0.001]),
        )
        elastic = np.repeat(
            np.diag([1.0e-3, 0.0, 0.0])[None, :, :],
            len(duplicate_cells),
            axis=0,
        )

        prediction = predict_gauge_microstrain(
            elastic,
            weights,
            direction=np.asarray([1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(weights.material_fill_fraction, 2.0, places=10)
        self.assertEqual(prediction["status"], "overlapping_material_coverage")
        self.assertIsNone(prediction["predicted_microstrain"])

    def test_nonfinite_gauge_center_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_gauge_weights(
                np.zeros((4, 3)),
                np.asarray([[0, 1, 2, 3]]),
                center=np.asarray([np.nan, 0.0, 0.0]),
                size=np.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
