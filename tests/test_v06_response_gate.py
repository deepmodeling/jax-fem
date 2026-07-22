import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.response_gate import evaluate_response_gate  # noqa: E402


class ManufacturedResponseGateTest(unittest.TestCase):
    def test_nonzero_run_requires_thermal_mechanical_and_xrd_response(self):
        report = evaluate_response_gate(
            laser_power_w=1.0,
            ambient_k=300.0,
            audit={
                "transient": {"maximum_temperature": 301.0},
                "latest_constrained": {
                    "stress": {"quality_filtered_max": 10.0}
                },
                "release": {
                    "displacement_norm": {"maximum": 1.0e-9},
                    "stress": {"quality_filtered_max": 5.0},
                },
            },
            xrd={"gauges": [{"status": "ok", "predicted_microstrain": 0.1}]},
            ledger_rows=[{"laser_deposited_j": 1.0e-3}],
        )

        self.assertTrue(report["required"])
        self.assertTrue(report["valid"])
        self.assertEqual(
            report["claim_level"], "manufactured_nonzero_response_smoke_only"
        )

    def test_nonzero_run_fails_if_mechanical_response_is_zero(self):
        report = evaluate_response_gate(
            laser_power_w=1.0,
            ambient_k=300.0,
            audit={
                "transient": {"maximum_temperature": 301.0},
                "latest_constrained": {
                    "stress": {"quality_filtered_max": 0.0}
                },
                "release": {
                    "displacement_norm": {"maximum": 0.0},
                    "stress": {"quality_filtered_max": 0.0},
                },
            },
            xrd={"gauges": [{"status": "ok", "predicted_microstrain": 0.0}]},
            ledger_rows=[{"laser_deposited_j": 1.0e-3}],
        )

        self.assertFalse(report["valid"])
        self.assertIn("constrained_stress", report["failed_checks"])

    def test_zero_power_run_records_that_nonzero_gate_is_not_required(self):
        report = evaluate_response_gate(
            laser_power_w=0.0,
            ambient_k=300.0,
            audit={},
            xrd={},
            ledger_rows=[],
        )

        self.assertFalse(report["required"])
        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "zero_input_invariant_smoke")


if __name__ == "__main__":
    unittest.main()
