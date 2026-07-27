import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.thermal_ledger import (  # noqa: E402
    EnergyLedgerRecorder,
    extract_solver_step,
    integrate_surface_exchange,
    integrate_volume_terms,
)


class ThermalLedgerQuadratureTest(unittest.TestCase):
    def test_constant_single_quadrature_terms_match_analytic_integrals(self):
        terms = integrate_volume_terms(
            jxw=np.asarray([[2.0]]),
            points=np.asarray([[[0.0, 0.0, 0.0]]]),
            temperature_old=np.asarray([[290.0]]),
            temperature_new=np.asarray([[300.0]]),
            rho=np.asarray([[2.0]]),
            cp=np.asarray([[3.0]]),
            latent_cp=np.asarray([[1.0]]),
            laser_center=np.asarray([0.0, 0.0, 0.0]),
            effective_laser_power_w=4.0,
            beam_radius_m=2.0,
            source_depth_m=5.0,
            laser_switch=1.0,
            active=np.asarray([[1.0]]),
            cooling_only=np.asarray([[1.0]]),
            old_layer_cooling_h=5.0,
            ambient_k=300.0,
            dt_s=1.0,
            build_axis=2,
            plane_axes=(0, 1),
            build_sign=1.0,
            front_loss_h=0.0,
            front_loss_thickness_m=1.0,
            front_loss_radiation=False,
            emissivity=0.0,
            stefan_boltzmann=5.67e-8,
        )

        self.assertEqual(terms["storage_j"], 160.0)
        self.assertAlmostEqual(
            terms["laser_deposited_j"],
            2.0 * 2.0 * 4.0 / (np.pi * 2.0**2 * 5.0),
        )
        self.assertEqual(terms["old_layer_loss_j"], -100.0)
        self.assertEqual(terms["front_loss_j"], 0.0)

    def test_storage_uses_enthalpy_difference_across_melt_interval(self):
        terms = integrate_volume_terms(
            jxw=np.asarray([[1.0]]),
            points=np.zeros((1, 1, 3)),
            temperature_old=np.asarray([[1623.15]]),
            temperature_new=np.asarray([[1693.15]]),
            rho=np.asarray([[1.0]]),
            cp=np.asarray([[1.0]]),
            latent_cp=np.asarray([[0.0]]),
            laser_center=np.zeros(3),
            effective_laser_power_w=0.0,
            beam_radius_m=1.0,
            source_depth_m=1.0,
            laser_switch=0.0,
            active=np.ones((1, 1)),
            cooling_only=np.zeros((1, 1)),
            old_layer_cooling_h=0.0,
            ambient_k=300.0,
            dt_s=1.0,
            build_axis=2,
            plane_axes=(0, 1),
            build_sign=1.0,
            front_loss_h=0.0,
            front_loss_thickness_m=1.0,
            front_loss_radiation=False,
            emissivity=0.0,
            stefan_boltzmann=5.67e-8,
            solidus_temperature=1643.15,
            liquidus_temperature=1673.15,
            latent_heat=280_000.0,
        )

        self.assertEqual(terms["storage_j"], 280_070.0)

    def test_surface_exchange_is_signed_outward(self):
        heating = integrate_surface_exchange(
            temperature_face=np.asarray([[290.0]]),
            surface_jxw=np.asarray([[2.0]]),
            active=np.asarray([[1.0]]),
            convection_h=5.0,
            ambient_k=300.0,
            emissivity=0.0,
            stefan_boltzmann=5.67e-8,
            dt_s=2.0,
        )

        self.assertEqual(heating, -200.0)

    def test_invalid_quadrature_coefficients_fail_before_recording(self):
        with self.assertRaisesRegex(ValueError, "rho"):
            integrate_volume_terms(
                jxw=np.asarray([[1.0]]),
                points=np.zeros((1, 1, 3)),
                temperature_old=np.asarray([[300.0]]),
                temperature_new=np.asarray([[300.0]]),
                rho=np.asarray([[-1.0]]),
                cp=np.asarray([[1.0]]),
                latent_cp=np.asarray([[0.0]]),
                laser_center=np.zeros(3),
                effective_laser_power_w=0.0,
                beam_radius_m=1.0,
                source_depth_m=1.0,
                laser_switch=0.0,
                active=np.ones((1, 1)),
                cooling_only=np.zeros((1, 1)),
                old_layer_cooling_h=0.0,
                ambient_k=300.0,
                dt_s=1.0,
                build_axis=2,
                plane_axes=(0, 1),
                build_sign=1.0,
                front_loss_h=0.0,
                front_loss_thickness_m=1.0,
                front_loss_radiation=False,
                emissivity=0.0,
                stefan_boltzmann=5.67e-8,
            )


class ThermalLedgerRecorderTest(unittest.TestCase):
    def test_jsonl_and_summary_preserve_partial_run_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = EnergyLedgerRecorder(Path(temporary), expected_steps=2)
            recorder.append(
                {
                    "step_index": 0,
                    "relative_balance_error": np.float64(1.0e-9),
                    "assembly_identity_error_j": np.float64(1.0e-12),
                    "temperature_invariants_valid": np.bool_(True),
                }
            )
            summary = recorder.finalize(completed=False)
            rows = [
                json.loads(line)
                for line in (Path(temporary) / "thermal_energy_ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["recorded_step_count"], 1)
        self.assertEqual(summary["expected_step_count"], 2)


class ThermalLedgerSolverExtractionTest(unittest.TestCase):
    def test_exact_zero_future_void_is_excluded_from_coefficient_validation(self):
        class FE:
            JxW = np.ones((2, 1))
            cells = np.asarray([[0], [1]])
            node_inds_list = [np.asarray([0, 1])]
            vec_inds_list = [np.asarray([0, 0])]
            vals_list = [np.asarray([300.0, 300.0])]

            @staticmethod
            def convert_from_dof_to_quad(values):
                return np.asarray(values)[:, None, :]

        class Problem:
            fes = [FE()]
            physical_quad_points = np.zeros((2, 1, 3))
            internal_vars = [
                300.0 * np.ones((2, 1, 1)),
                np.ones((2, 1, 1)),
                np.zeros((2, 1, 3)),
                np.zeros((2, 1, 1)),
                np.ones((2, 1, 1)),
                np.ones((2, 1, 1)),
                np.zeros((2, 1, 1)),
                np.asarray([[[1.0]], [[0.0]]]),
                np.asarray([[[2.0]], [[0.0]]]),
                np.asarray([[[3.0]], [[0.0]]]),
                np.asarray([[[1.0]], [[0.0]]]),
                np.zeros((2, 1, 1)),
                np.zeros((2, 1, 1)),
                np.zeros((2, 1, 1)),
            ]
            boundary_inds_list = []
            selected_face_shape_vals = []
            nanson_scale = []
            internal_vars_surfaces = []
            convection_h = 0.0
            ambient = 300.0
            emissivity = 0.0
            stefan_boltzmann = 5.67e-8
            build_axis_id = 2
            plane_axis0_id = 0
            plane_axis1_id = 1
            build_sign = 1.0
            front_surface_loss_h = 0.0
            front_surface_loss_thickness = 1.0
            front_surface_loss_radiation = False

            @staticmethod
            def compute_residual(_solution):
                return [np.zeros((2, 1))]

        row = extract_solver_step(
            Problem(),
            [300.0 * np.ones((2, 1))],
            step_index=0,
            step_state=SimpleNamespace(
                global_step=0,
                mode="cooling",
                layer_idx=0,
                hatch_idx=0,
                scan_idx=0,
                laser_power=0.0,
                laser_switch=0.0,
            ),
            absorptivity=0.5,
            previous_solution=None,
            temperature_atol_k=1.0e-3,
        )

        self.assertEqual(row["storage_j"], 0.0)
        self.assertTrue(row["temperature_invariants_valid"])

    def test_dirichlet_reaction_closes_constant_storage_step(self):
        class FE:
            JxW = np.asarray([[2.0]])
            cells = np.asarray([[0]])
            shape_vals = np.asarray([[1.0]])
            node_inds_list = [np.asarray([0])]
            vec_inds_list = [np.asarray([0])]
            vals_list = [np.asarray([300.0])]

            @staticmethod
            def convert_from_dof_to_quad(values):
                return np.asarray(values)[None, :, :]

        class Problem:
            fes = [FE()]
            physical_quad_points = np.zeros((1, 1, 3))
            internal_vars = [
                np.asarray([[[290.0]]]),
                np.asarray([[[2.0]]]),
                np.zeros((1, 1, 3)),
                np.zeros((1, 1, 1)),
                np.ones((1, 1, 1)),
                np.ones((1, 1, 1)),
                np.zeros((1, 1, 1)),
                np.ones((1, 1, 1)),
                2.0 * np.ones((1, 1, 1)),
                3.0 * np.ones((1, 1, 1)),
                np.ones((1, 1, 1)),
                np.ones((1, 1, 1)),
                np.zeros((1, 1, 1)),
                np.zeros((1, 1, 1)),
            ]
            boundary_inds_list = []
            selected_face_shape_vals = []
            nanson_scale = []
            internal_vars_surfaces = []
            convection_h = 0.0
            ambient = 300.0
            emissivity = 0.0
            stefan_boltzmann = 5.67e-8
            build_axis_id = 2
            plane_axis0_id = 0
            plane_axis1_id = 1
            build_sign = 1.0
            front_surface_loss_h = 0.0
            front_surface_loss_thickness = 1.0
            front_surface_loss_radiation = False

            @staticmethod
            def compute_residual(_solution):
                return [np.asarray([[80.0]])]

        row = extract_solver_step(
            Problem(),
            [np.asarray([[300.0]])],
            step_index=0,
            step_state=SimpleNamespace(
                global_step=0,
                mode="cooling",
                layer_idx=0,
                hatch_idx=0,
                scan_idx=0,
                laser_power=0.0,
                laser_switch=0.0,
            ),
            absorptivity=0.5,
            previous_solution=None,
            temperature_atol_k=1.0e-3,
        )

        self.assertEqual(row["storage_j"], 160.0)
        self.assertEqual(row["dirichlet_exchange_into_domain_j"], 160.0)
        self.assertEqual(row["balance_error_j"], 0.0)
        self.assertTrue(row["balance_within_solver_tolerance"])
        self.assertTrue(row["assembly_identity_within_tolerance"])
        self.assertTrue(row["state_override_within_tolerance"])
        self.assertTrue(row["temperature_invariants_valid"])

    def test_completed_recorder_rejects_failed_balance_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = EnergyLedgerRecorder(Path(temporary), expected_steps=1)
            recorder.append(
                {
                    "step_index": 0,
                    "relative_balance_error": 1.0,
                    "balance_error_j": 1.0,
                    "assembly_identity_error_j": 0.0,
                    "balance_within_solver_tolerance": False,
                    "assembly_identity_within_tolerance": True,
                    "temperature_invariants_valid": True,
                }
            )

            summary = recorder.finalize(completed=True)

        self.assertFalse(summary["complete"])
        self.assertFalse(summary["all_balance_steps_within_tolerance"])

    def test_completed_recorder_rejects_unaccounted_state_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = EnergyLedgerRecorder(Path(temporary), expected_steps=1)
            recorder.append(
                {
                    "step_index": 0,
                    "relative_balance_error": 0.0,
                    "balance_error_j": 0.0,
                    "assembly_identity_error_j": 0.0,
                    "balance_within_solver_tolerance": True,
                    "assembly_identity_within_tolerance": True,
                    "pre_solve_state_override_j": 1.0,
                    "state_override_within_tolerance": False,
                    "temperature_invariants_valid": True,
                }
            )

            summary = recorder.finalize(completed=True)

        self.assertFalse(summary["complete"])
        self.assertFalse(
            summary["all_pre_solve_state_overrides_within_tolerance"]
        )
        self.assertEqual(summary["cumulative_pre_solve_state_override_j"], 1.0)


if __name__ == "__main__":
    unittest.main()
