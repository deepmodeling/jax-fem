import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.cases import load_case  # noqa: E402
from jax_fem_am.verification.screening import (  # noqa: E402
    EvidenceLevelError,
    pointwise_field_comparison,
    screen_anchor_predictions,
)


class StrantzaScreeningContractTest(unittest.TestCase):
    def setUp(self):
        self.case = load_case("strantza_2018")

    def test_paper_anchors_generate_screening_report_without_accuracy_metric(self):
        report = screen_anchor_predictions(
            self.case,
            {
                "units": "microstrain",
                "predictions": {
                    "eps_zz_edge_tensile_max_microstrain": 7600.0,
                },
            },
        )

        self.assertEqual(
            report["evidence_level"], "manual_unverified_screening"
        )
        self.assertEqual(report["specimen_id"], "C45")
        self.assertEqual(report["units"], "microstrain")
        self.assertNotIn("nrmse", report)
        self.assertNotIn("reduced_chi_square", report)
        self.assertTrue(
            report["anchors"]["eps_zz_edge_tensile_max_microstrain"][
                "within_screening_band"
            ]
        )

    def test_screening_rejects_dimensionless_values_without_unit_conversion(self):
        with self.assertRaisesRegex(ValueError, "microstrain"):
            screen_anchor_predictions(
                self.case,
                {
                    "units": "strain",
                    "predictions": {
                        "eps_zz_edge_tensile_max_microstrain": 7600.0e-6,
                    },
                },
            )

    def test_paper_anchor_case_refuses_pointwise_accuracy_claim(self):
        with self.assertRaisesRegex(EvidenceLevelError, "pointwise"):
            pointwise_field_comparison(
                self.case,
                observed=[1.0],
                predicted=[1.0],
                uncertainty=[0.1],
                units="microstrain",
            )

    def test_pointwise_claim_stays_disabled_without_dataset_contract(self):
        case = deepcopy(self.case)
        case["data_availability"]["status"] = "pointwise_raw_available"

        with self.assertRaisesRegex(EvidenceLevelError, "dataset contract"):
            pointwise_field_comparison(
                case,
                observed=[1.0],
                predicted=[1.0],
                uncertainty=[0.1],
                units="microstrain",
            )


if __name__ == "__main__":
    unittest.main()
