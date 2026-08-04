import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_fem_am.materials.material_validation import validate_material_inputs  # noqa: E402


class Table:
    def __init__(self, temperatures, values):
        self.T = np.asarray(temperatures)
        self.values = np.asarray(values)


class FlowCurve:
    def __init__(self, temperatures, plastic_strains, stresses):
        self.temperatures = np.asarray(temperatures)
        self.plastic_strains = np.asarray(plastic_strains)
        self.stresses = np.asarray(stresses)


def valid_args(**overrides):
    values = {
        "mechanics_model": "j2_plastic",
        "young": 110.0e9,
        "alpha": 9.0e-6,
        "poisson": 0.32,
        "yield_saturation_stress": 1.15e9,
        "rho": 4420.0,
        "rho_solid": None,
        "rho_liquid": None,
        "rho_powder": 2500.0,
        "cp": 600.0,
        "cp_solid": None,
        "cp_liquid": None,
        "cp_powder": 500.0,
        "conductivity": 7.0,
        "conductivity_solid": None,
        "conductivity_liquid": None,
        "conductivity_powder": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def valid_tables(**overrides):
    tables = {
        "E": Table([300.0, 1200.0], [110.0e9, 70.0e9]),
        "alpha": Table([300.0, 1200.0], [9.0e-6, 12.0e-6]),
        "poisson": Table([300.0, 1200.0], [0.32, 0.34]),
        "yield": Table([300.0, 1200.0], [900.0e6, 100.0e6]),
        "hardening": Table([300.0, 1200.0], [2.0e9, 0.0]),
        "flow_curve": None,
        "k_solid": None,
        "cp_solid": None,
        "k_powder": None,
        "cp_powder": None,
        "k_liquid": None,
        "cp_liquid": None,
    }
    tables.update(overrides)
    return tables


class MaterialValidationTest(unittest.TestCase):
    def test_valid_material_tables_and_fallbacks_pass(self):
        self.assertTrue(validate_material_inputs(valid_args(), valid_tables()))

    def test_invalid_poisson_and_negative_hardening_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "poisson"):
            validate_material_inputs(
                valid_args(),
                valid_tables(poisson=Table([300.0, 1200.0], [0.32, 0.5])),
            )
        with self.assertRaisesRegex(ValueError, "hardening"):
            validate_material_inputs(
                valid_args(),
                valid_tables(hardening=Table([300.0, 1200.0], [1.0, -1.0])),
            )

    def test_nonfinite_or_duplicate_temperature_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_material_inputs(
                valid_args(),
                valid_tables(E=Table([300.0, 300.0], [1.0, 1.0])),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_material_inputs(
                valid_args(),
                valid_tables(E=Table([300.0, 1200.0], [1.0, np.nan])),
            )

    def test_nonpositive_thermal_defaults_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conductivity"):
            validate_material_inputs(
                valid_args(conductivity=0.0),
                valid_tables(),
            )

    def test_flow_curve_replaces_legacy_yield_and_hardening_tables(self):
        curve = FlowCurve(
            [300.0, 1200.0],
            [0.0, 0.1],
            [[900.0e6, 1.0e9], [100.0e6, 120.0e6]],
        )
        self.assertTrue(
            validate_material_inputs(
                valid_args(),
                valid_tables(
                    **{
                        "flow_curve": curve,
                        "yield": None,
                        "hardening": None,
                    }
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            validate_material_inputs(
                valid_args(),
                valid_tables(flow_curve=curve),
            )

    def test_flow_curve_axes_and_hardening_must_be_admissible(self):
        with self.assertRaisesRegex(ValueError, "start at zero"):
            validate_material_inputs(
                valid_args(),
                valid_tables(
                    **{
                        "flow_curve": FlowCurve(
                            [300.0, 1200.0],
                            [0.01, 0.1],
                            [[900.0e6, 1.0e9], [100.0e6, 120.0e6]],
                        ),
                        "yield": None,
                        "hardening": None,
                    }
                ),
            )
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            validate_material_inputs(
                valid_args(),
                valid_tables(
                    **{
                        "flow_curve": FlowCurve(
                            [300.0, 1200.0],
                            [0.0, 0.1],
                            [[900.0e6, 800.0e6], [100.0e6, 120.0e6]],
                        ),
                        "yield": None,
                        "hardening": None,
                    }
                ),
            )

    def test_short_flow_curve_warns_about_silent_tangent_clamping(self):
        """Past the last knot the tabulated tangent is exactly zero, silently."""
        with self.assertWarnsRegex(RuntimeWarning, "clamped to exactly zero"):
            validate_material_inputs(
                valid_args(),
                valid_tables(
                    **{
                        "flow_curve": FlowCurve(
                            [300.0, 1200.0],
                            [0.0, 0.2],
                            [[900.0e6, 1.2e9], [100.0e6, 200.0e6]],
                        ),
                        "yield": None,
                        "hardening": None,
                    }
                ),
            )

    def test_long_enough_flow_curve_does_not_warn_about_clamping(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_material_inputs(
                valid_args(),
                valid_tables(
                    **{
                        "flow_curve": FlowCurve(
                            [300.0, 1200.0],
                            [0.0, 2.0],
                            [[900.0e6, 3.0e9], [100.0e6, 400.0e6]],
                        ),
                        "yield": None,
                        "hardening": None,
                    }
                ),
            )
        self.assertEqual(
            [w for w in caught if "clamped to exactly zero" in str(w.message)], []
        )

    def test_row_that_is_effectively_perfectly_plastic_warns(self):
        """The V2 D-V2-19 failure, with its real numbers.

        Hot row: yield floored at 1 MPa with a fixed 1e7 Pa regularizing
        hardening, against E = 61.6 GPa at the solidus -- H/E = 1.6e-4. That is
        the shape that stalled every V2 mechanics arm at the same step, and the
        reason the regularizer had to be tied to E(T) instead of fixed.
        """
        with self.assertWarnsRegex(RuntimeWarning, "perfect plasticity"):
            validate_material_inputs(
                valid_args(),
                valid_tables(
                    **{
                        "E": Table([300.0, 1563.0], [171.0e9, 61.6e9]),
                        "flow_curve": FlowCurve(
                            [300.0, 1563.0],
                            [0.0, 5.0e-4, 2.0],
                            [
                                [650.0e6, 670.0e6, 3.0e9],
                                [1.0e6, 1.005e6, 21.0e6],
                            ],
                        ),
                        "yield": None,
                        "hardening": None,
                    }
                ),
            )

    def test_first_segment_stiffer_than_elastic_modulus_warns(self):
        """The V2 D-V2-22 shape: J-C tabulated as printed near eps_p -> 0.

        650 -> 1007 MPa across eps_p 0 -> 0.002 is H = 178.7 GPa against
        E = 171 GPa, i.e. a plastic branch stiffer than the elastic one.
        """
        with self.assertWarnsRegex(RuntimeWarning, "at or above"):
            validate_material_inputs(
                valid_args(),
                valid_tables(
                    **{
                        "E": Table([300.0, 1563.0], [171.0e9, 61.6e9]),
                        "flow_curve": FlowCurve(
                            [300.0, 1563.0],
                            [0.0, 0.002, 2.0],
                            [
                                [650.0e6, 1007.4e6, 3.0e9],
                                [200.0e6, 260.0e6, 900.0e6],
                            ],
                        ),
                        "yield": None,
                        "hardening": None,
                    }
                ),
            )


if __name__ == "__main__":
    unittest.main()
