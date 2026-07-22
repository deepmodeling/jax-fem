import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.xrd import (  # noqa: E402
    gauge_volume_average,
    project_normal_microstrain,
    project_normal_strain,
)
from jax_fem_am.verification.mesh_quality import audit_tet_mesh  # noqa: E402
from jax_fem_am.verification.weighted import weighted_mean, weighted_quantile  # noqa: E402


def regular_tetrahedron():
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
        ]
    )


class TetMeshQualityTest(unittest.TestCase):
    def test_regular_tetrahedron_has_unit_quality_and_edge_ratio(self):
        points = regular_tetrahedron()
        report = audit_tet_mesh(points, np.asarray([[0, 1, 2, 3]]))

        self.assertAlmostEqual(float(report.mean_ratio[0]), 1.0, places=12)
        self.assertAlmostEqual(float(report.edge_ratio[0]), 1.0, places=12)
        self.assertAlmostEqual(float(report.volume[0]), 1.0 / (6.0 * np.sqrt(2.0)))
        self.assertEqual(report.inverted_count, 0)
        self.assertEqual(report.degenerate_count, 0)

    def test_sliver_is_flagged_below_quality_threshold(self):
        points = regular_tetrahedron()
        points[3, 2] = 1.0e-7
        report = audit_tet_mesh(
            points,
            np.asarray([[0, 1, 2, 3]]),
            quality_threshold=0.05,
        )

        self.assertLess(float(report.mean_ratio[0]), 1.0e-6)
        self.assertEqual(report.below_threshold_count, 1)

    def test_inverted_and_degenerate_cells_are_counted_separately(self):
        points = np.vstack([regular_tetrahedron(), [[0.25, 0.25, 0.0]]])
        cells = np.asarray(
            [
                [0, 2, 1, 3],
                [0, 1, 2, 4],
            ]
        )

        report = audit_tet_mesh(points, cells)

        self.assertEqual(report.inverted_count, 1)
        self.assertEqual(report.degenerate_count, 1)

    def test_nonfinite_point_coordinates_are_rejected(self):
        points = regular_tetrahedron()
        points[2, 1] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            audit_tet_mesh(points, np.asarray([[0, 1, 2, 3]]))


class VolumeWeightedStatisticsTest(unittest.TestCase):
    def test_weighted_mean_is_not_cell_count_mean(self):
        values = np.asarray([1000.0, 10.0])
        volumes = np.asarray([1.0e-6, 1.0])

        result = weighted_mean(values, volumes)

        self.assertAlmostEqual(result, (1000.0e-6 + 10.0) / 1.000001)
        self.assertLess(result, 10.01)

    def test_weighted_quantile_uses_inverse_weighted_cdf(self):
        values = np.asarray([0.0, 10.0])
        weights = np.asarray([9.0, 1.0])

        self.assertEqual(weighted_quantile(values, weights, 0.50), 0.0)
        self.assertEqual(weighted_quantile(values, weights, 0.95), 10.0)


class XrdMeasurementOperatorTest(unittest.TestCase):
    def test_gauge_average_weights_quadrature_strain_by_physical_volume(self):
        strains = np.zeros((2, 3, 3))
        strains[0, 0, 0] = 1000.0e-6
        strains[1, 0, 0] = 100.0e-6

        average = gauge_volume_average(strains, np.asarray([1.0, 9.0]))

        self.assertAlmostEqual(float(average[0, 0]), 190.0e-6)

    def test_projection_normalizes_measurement_direction(self):
        strain = np.diag([1200.0e-6, -300.0e-6, 100.0e-6])

        projected = project_normal_strain(strain, np.asarray([2.0, 0.0, 0.0]))

        self.assertAlmostEqual(float(projected), 1200.0e-6)

    def test_microstrain_projection_applies_explicit_one_million_scale(self):
        strain = np.diag([1200.0e-6, 0.0, 0.0])

        projected = project_normal_microstrain(
            strain, np.asarray([1.0, 0.0, 0.0])
        )

        self.assertAlmostEqual(projected, 1200.0)

    def test_gauge_average_rejects_nonpositive_total_weight(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            gauge_volume_average(np.zeros((2, 3, 3)), np.zeros(2))


if __name__ == "__main__":
    unittest.main()
