import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.cases import load_case  # noqa: E402
from jax_fem_am.verification.metrics import evaluate_anchors, field_error_metrics  # noqa: E402


class ExperimentalMetricTest(unittest.TestCase):
    def test_field_metrics_include_uncertainty_weighted_chi_square(self):
        observed = np.asarray([0.0, 1.0, 2.0])
        predicted = np.asarray([0.1, 0.9, 2.2])
        sigma = np.asarray([0.1, 0.1, 0.2])

        metrics = field_error_metrics(observed, predicted, sigma)

        self.assertAlmostEqual(metrics["bias"], (0.1 - 0.1 + 0.2) / 3.0)
        self.assertAlmostEqual(metrics["reduced_chi_square"], 1.0)
        self.assertAlmostEqual(metrics["coverage_2sigma"], 1.0)
        self.assertAlmostEqual(
            metrics["nrmse_range"],
            np.sqrt(0.02) / 2.0,
        )

    def test_zero_observed_range_reports_null_nrmse_instead_of_dividing(self):
        metrics = field_error_metrics(
            np.asarray([2.0, 2.0]),
            np.asarray([1.0, 3.0]),
        )

        self.assertIsNone(metrics["nrmse_range"])

    def test_fitted_parameter_count_must_be_a_nonnegative_integer(self):
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            field_error_metrics(
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0]),
                np.asarray([0.1, 0.1]),
                fitted_parameters=-1,
            )


class StrantzaCaseContractTest(unittest.TestCase):
    def test_case_records_measurement_resolution_and_is_not_calibration_data(self):
        case = load_case("strantza_2018")

        self.assertEqual(case["specimen_id"], "C45")
        self.assertEqual(case["role"], "held_out_validation")
        self.assertEqual(case["material"], "Ti-6Al-4V")
        self.assertEqual(
            case["measurement"]["measurement_state"],
            "attached_to_build_plate_before_EDM",
        )
        self.assertEqual(
            case["measurement"]["gauge_definition"]["eps_zz"][
                "direction_specimen_xyz"
            ],
            [0.0, 0.0, 1.0],
        )
        self.assertEqual(
            case["measurement"]["operator_requirements"]["output_unit"],
            "microstrain",
        )
        self.assertEqual(case["process"]["reported_track_width_um"], 150.0)
        self.assertNotIn("hatch_spacing_um", case["process"])
        self.assertNotIn("laser_beam_diameter_d4sigma_um", case["process"])
        self.assertEqual(
            case["measurement"]["gauge_definition"]["eps_zz"][
                "geometry_model"
            ],
            "rhomboidal_diffraction_volume",
        )
        self.assertEqual(
            case["measurement"]["gauge_volume_mm"]["eps_xx_eps_zz"],
            [0.2, 1.5, 0.2],
        )
        self.assertEqual(
            case["measurement"]["gauge_volume_mm"]["eps_yy"],
            [1.5, 0.2, 0.2],
        )
        self.assertEqual(case["data_availability"]["status"], "paper_anchors_only")

    def test_reported_anchor_evaluator_distinguishes_target_and_range(self):
        case = load_case("strantza_2018")
        predictions = {
            "eps_zz_edge_tensile_max_microstrain": 7600.0,
            "eps_zz_interior_compressive_min_microstrain": -4900.0,
            "eps_yy_perimeter_tensile_microstrain": 3400.0,
        }

        report = evaluate_anchors(case["reported_anchors"], predictions)

        self.assertTrue(
            report["eps_zz_edge_tensile_max_microstrain"][
                "within_screening_band"
            ]
        )
        self.assertFalse(
            report["eps_zz_interior_compressive_min_microstrain"][
                "within_screening_band"
            ]
        )
        self.assertTrue(
            report["eps_yy_perimeter_tensile_microstrain"]["within_range"]
        )


if __name__ == "__main__":
    unittest.main()
