import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_fem_am.materials.material_validation import validate_material_inputs  # noqa: E402


class Table:
    def __init__(self, temperatures, values):
        self.T = np.asarray(temperatures)
        self.values = np.asarray(values)


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


if __name__ == "__main__":
    unittest.main()
