import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V06 = ROOT / "159_local" / "v06"


class V06RunnerContractTest(unittest.TestCase):
    def test_smoke_runner_uses_v06_driver_fixture_and_audit(self):
        text = (V06 / "run_smoke.sh").read_text(encoding="utf-8")

        self.assertIn("v06/driver.py", text)
        self.assertIn("unit_cube_6tet.inp", text)
        self.assertIn("v06.verification.run_audit", text)
        self.assertIn("v06.measurement.xrd_vtu", text)
        self.assertIn("v06.provenance", text)
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
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(ROOT / "159_local" / "v01"),
                str(ROOT / "159_local"),
                str(ROOT),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(V06 / "driver.py"),
                    "--xla-dry-run",
                    "--xla-platform",
                    "cpu",
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
            profile_data = profile.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("v06: paper-validation constitutive adapter installed", result.stdout)
        self.assertIn("full_loop_xla         = disabled", result.stdout)
        self.assertIn('"v06_constitutive_model"', profile_data)


if __name__ == "__main__":
    unittest.main()
