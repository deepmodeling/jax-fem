import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
from jax_fem_am.simulation import runner as driver  # noqa: E402


class ThermalLedgerHookTest(unittest.TestCase):
    def setUp(self):
        driver.REGISTRY.reset()

    def test_outer_solver_hook_records_only_thermal_solve_once(self):
        class Thermal:
            pass

        calls = []

        def solver(problem, *_args, **_kwargs):
            calls.append(type(problem).__name__)
            return [np.asarray([[300.0]])]

        base = SimpleNamespace(solver=solver, TransientThermal=Thermal)
        original_extract = driver.extract_solver_step
        try:
            driver.extract_solver_step = lambda *_args, **kwargs: {
                "step_index": kwargs["step_index"],
                "relative_balance_error": 0.0,
                "assembly_identity_error_j": 0.0,
                "temperature_invariants_valid": True,
            }
            with tempfile.TemporaryDirectory() as temporary:
                driver.REGISTRY.args = SimpleNamespace(
                    output_dir=temporary,
                    absorptivity=0.5,
                )
                driver.REGISTRY.step_states = [SimpleNamespace()]
                self.assertTrue(driver.install_thermal_ledger_wrapper(base))

                base.solver(Thermal())
                base.solver(object())

                self.assertEqual(len(driver.REGISTRY.thermal_ledger.rows), 1)
                self.assertEqual(calls, ["Thermal", "object"])
        finally:
            driver.extract_solver_step = original_extract


if __name__ == "__main__":
    unittest.main()
