import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.thermal_balance import (  # noqa: E402
    check_temperature_invariants,
    compute_discrete_balance,
)


class ThermalInvariantTest(unittest.TestCase):
    def test_source_free_step_enforces_old_dirichlet_ambient_upper_bound(self):
        result = check_temperature_invariants(
            T_old=np.asarray([300.0, 1100.0]),
            T_new=np.asarray([300.0, 1772.0]),
            ambient=300.0,
            dirichlet_values=np.asarray([300.0]),
            deposited_source_j=0.0,
            coefficients_valid=True,
            atol_k=1.0e-8,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["upper_violation_count"], 1)
        self.assertEqual(result["upper_bound_k"], 1100.0)

    def test_active_source_disables_only_upper_bound_check(self):
        result = check_temperature_invariants(
            T_old=np.asarray([300.0, 400.0]),
            T_new=np.asarray([300.0, 1200.0]),
            ambient=300.0,
            dirichlet_values=np.asarray([300.0]),
            deposited_source_j=1.0,
            coefficients_valid=True,
            atol_k=1.0e-8,
        )

        self.assertTrue(result["valid"])
        self.assertFalse(result["source_free"])
        self.assertIsNone(result["upper_violation_count"])

    def test_invalid_material_coefficients_fail_precondition(self):
        result = check_temperature_invariants(
            T_old=np.asarray([300.0]),
            T_new=np.asarray([300.0]),
            ambient=300.0,
            dirichlet_values=np.asarray([300.0]),
            deposited_source_j=0.0,
            coefficients_valid=False,
            atol_k=1.0e-8,
        )

        self.assertFalse(result["valid"])
        self.assertFalse(result["coefficient_preconditions_valid"])


class DiscreteEnergyBalanceTest(unittest.TestCase):
    def test_balance_uses_explicit_sign_convention_and_capture_fraction(self):
        balance = compute_discrete_balance(
            storage_j=8.0,
            laser_deposited_j=10.0,
            laser_commanded_j=25.0,
            laser_absorbed_nominal_j=20.0,
            front_loss_j=1.0,
            old_layer_loss_j=0.5,
            surface_loss_j=0.25,
            dirichlet_exchange_into_domain_j=-0.25,
            assembly_identity_error_j=0.0,
            free_residual_l1_j=0.0,
            free_residual_l2_j=0.0,
        )

        self.assertEqual(balance.claim_level, "discrete_weak_form_only")
        self.assertEqual(balance.schema_version, "v06.thermal_balance.v2")
        self.assertEqual(balance.balance_error_j, 0.0)
        self.assertEqual(balance.laser_commanded_j, 25.0)
        self.assertEqual(balance.laser_absorbed_nominal_j, 20.0)
        self.assertEqual(balance.laser_nominal_j, 20.0)
        self.assertEqual(balance.source_capture_fraction, 0.5)

    def test_signed_outward_exchanges_may_be_negative(self):
        balance = compute_discrete_balance(
            storage_j=5.0,
            laser_deposited_j=0.0,
            laser_commanded_j=0.0,
            laser_absorbed_nominal_j=0.0,
            front_loss_j=-2.0,
            old_layer_loss_j=-1.0,
            surface_loss_j=-1.0,
            dirichlet_exchange_into_domain_j=1.0,
            assembly_identity_error_j=0.0,
            free_residual_l1_j=0.0,
            free_residual_l2_j=0.0,
        )

        self.assertEqual(balance.balance_error_j, 0.0)
        self.assertIsNone(balance.source_capture_fraction)

    def test_relative_error_scale_uses_absolute_value_of_signed_terms(self):
        balance = compute_discrete_balance(
            storage_j=4.0,
            laser_deposited_j=1.0,
            laser_commanded_j=2.0,
            laser_absorbed_nominal_j=1.0,
            front_loss_j=-2.0,
            old_layer_loss_j=-3.0,
            surface_loss_j=-4.0,
            dirichlet_exchange_into_domain_j=-5.0,
            assembly_identity_error_j=0.0,
            free_residual_l1_j=0.0,
            free_residual_l2_j=0.0,
        )

        self.assertEqual(balance.balance_error_j, -1.0)
        self.assertAlmostEqual(balance.relative_balance_error, 1.0 / 19.0)

    def test_legacy_laser_nominal_argument_means_absorbed_nominal_energy(self):
        balance = compute_discrete_balance(
            storage_j=0.0,
            laser_deposited_j=2.0,
            laser_nominal_j=4.0,
            front_loss_j=0.0,
            old_layer_loss_j=0.0,
            surface_loss_j=0.0,
            dirichlet_exchange_into_domain_j=-2.0,
            assembly_identity_error_j=0.0,
            free_residual_l1_j=0.0,
            free_residual_l2_j=0.0,
        )

        self.assertIsNone(balance.laser_commanded_j)
        self.assertEqual(balance.laser_absorbed_nominal_j, 4.0)
        self.assertEqual(balance.laser_nominal_j, 4.0)
        self.assertEqual(balance.source_capture_fraction, 0.5)

    def test_conflicting_legacy_and_v2_absorbed_energy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same absorbed nominal"):
            compute_discrete_balance(
                storage_j=0.0,
                laser_deposited_j=0.0,
                laser_nominal_j=4.0,
                laser_absorbed_nominal_j=3.0,
                front_loss_j=0.0,
                old_layer_loss_j=0.0,
                surface_loss_j=0.0,
                dirichlet_exchange_into_domain_j=0.0,
                assembly_identity_error_j=0.0,
                free_residual_l1_j=0.0,
                free_residual_l2_j=0.0,
            )

    def test_invalid_laser_energy_relationships_are_rejected(self):
        common = {
            "storage_j": 0.0,
            "front_loss_j": 0.0,
            "old_layer_loss_j": 0.0,
            "surface_loss_j": 0.0,
            "dirichlet_exchange_into_domain_j": 0.0,
            "assembly_identity_error_j": 0.0,
            "free_residual_l1_j": 0.0,
            "free_residual_l2_j": 0.0,
        }
        invalid = (
            {
                "laser_deposited_j": 0.0,
            },
            {
                "laser_deposited_j": -1.0,
                "laser_absorbed_nominal_j": 1.0,
            },
            {
                "laser_deposited_j": 0.0,
                "laser_absorbed_nominal_j": -1.0,
            },
            {
                "laser_deposited_j": 0.0,
                "laser_commanded_j": -1.0,
                "laser_absorbed_nominal_j": 0.0,
            },
            {
                "laser_deposited_j": 0.0,
                "laser_commanded_j": 1.0,
                "laser_absorbed_nominal_j": 2.0,
            },
            {
                "laser_deposited_j": 1.0,
                "laser_absorbed_nominal_j": 0.0,
            },
        )
        for laser_terms in invalid:
            with self.subTest(laser_terms=laser_terms):
                with self.assertRaises(ValueError):
                    compute_discrete_balance(**common, **laser_terms)

    def test_nonfinite_energy_and_negative_error_norms_are_rejected(self):
        common = {
            "storage_j": 0.0,
            "laser_deposited_j": 0.0,
            "laser_absorbed_nominal_j": 0.0,
            "front_loss_j": 0.0,
            "old_layer_loss_j": 0.0,
            "surface_loss_j": 0.0,
            "dirichlet_exchange_into_domain_j": 0.0,
            "assembly_identity_error_j": 0.0,
            "free_residual_l1_j": 0.0,
            "free_residual_l2_j": 0.0,
        }
        invalid_overrides = (
            {"surface_loss_j": np.nan},
            {"assembly_identity_error_j": -1.0},
            {"free_residual_l1_j": -1.0},
            {"free_residual_l2_j": -1.0},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                inputs = dict(common)
                inputs.update(override)
                with self.assertRaises(ValueError):
                    compute_discrete_balance(**inputs)


if __name__ == "__main__":
    unittest.main()
