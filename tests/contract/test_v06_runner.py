import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V06 = ROOT / "cases"


class V06RunnerContractTest(unittest.TestCase):
    def test_kaess_r3_optimized_launcher_keeps_physics_and_enables_safe_optimizations(self):
        launcher = ROOT / "cases" / "kaess_2023" / "run_kaess_r3_optimized.sh"
        text = launcher.read_text(encoding="utf-8")

        self.assertIn('PATH_ARGS="${PATH_ARGS:---layers 2}"', text)
        self.assertIn('POWDER_SOLID="${POWDER_SOLID:-1}"', text)
        self.assertIn("--mechanics-residual-only-check", text)
        self.assertIn("--xla-pardiso-mode phase23", text)
        self.assertIn("run_kaess_phase2.sh", text)

    def test_smoke_runner_uses_v06_driver_fixture_and_audit(self):
        text = (V06 / "run_smoke.sh").read_text(encoding="utf-8")

        self.assertIn("jax_fem_am.simulation.runner", text)
        self.assertIn("unit_cube_6tet.inp", text)
        self.assertIn("jax_fem_am.verification.run_audit", text)
        self.assertIn("jax_fem_am.verification.xrd_vtu", text)
        self.assertIn("jax_fem_am.verification.provenance", text)
        self.assertIn("LASER_POWER_W", text)
        self.assertIn("refusing existing OUT_ROOT", text)
        self.assertNotIn("v05/am_thermal", text)

        nonzero = (V06 / "run_nonzero_smoke.sh").read_text(encoding="utf-8")
        self.assertIn('LASER_POWER_W="${LASER_POWER_W:-0.01}"', nonzero)
        self.assertIn('BEAM_RADIUS_M="${BEAM_RADIUS_M:-0.0005}"', nonzero)
        self.assertIn('SOURCE_DEPTH_M="${SOURCE_DEPTH_M:-0.001}"', nonzero)
        self.assertIn('SOLIDUS_TEMPERATURE_K="${SOLIDUS_TEMPERATURE_K:-0}"', nonzero)
        self.assertIn('LIQUIDUS_TEMPERATURE_K="${LIQUIDUS_TEMPERATURE_K:-0}"', nonzero)
        self.assertIn("run_smoke.sh", nonzero)

    def test_driver_dry_run_reaches_v04_runtime_boundary(self):
        env = dict(os.environ)
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["PYTHONPATH"] = str(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jax_fem_am.simulation.runner",
                    "--xla-dry-run",
                    "--xla-platform",
                    "cpu",
                    "--xla-linear-solver",
                    "pardiso",
                    "--xla-pardiso-mode",
                    "phase23",
                    "--mechanics-residual-only-check",
                    "--profile-json",
                    str(profile),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            profile_data = json.loads(profile.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("v06: paper-validation constitutive adapter installed", result.stdout)
        self.assertIn("full_loop_xla         = disabled", result.stdout)
        self.assertIn("v06_constitutive_model", profile_data["meta"])
        self.assertEqual(
            profile_data["meta"]["linear_solver_label"],
            "pardiso_v07(phase23)",
        )
        self.assertIs(
            profile_data["meta"]["mechanics_residual_only_check_enabled"],
            True,
        )


if __name__ == "__main__":
    unittest.main()
