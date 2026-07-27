import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jax_fem_am.process.scan_path import generate_path_file_step_states


ROOT = Path(__file__).resolve().parents[2]
V06 = ROOT / "cases"


class V06RunnerContractTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "requires a POSIX bash path")
    def test_kaess_medium_plan_rejects_inherited_physics_overrides(self):
        launcher = (
            ROOT
            / "cases"
            / "kaess_2023"
            / "run_kaess_medium_fullheight.sh"
        )
        env = dict(os.environ)
        env.update(
            {
                "PATH_ARGS": "--layers 2",
                "BUILD_LAYERS": "99",
                "ELEMENT_TYPE": "c3d4",
                "EXTRA_ARGS": "--layers 2",
            }
        )
        result = subprocess.run(
            ["bash", str(launcher), "--print-plan"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        plan = dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(plan["ELEMENT_TYPE"], "c3d8")
        self.assertEqual(plan["BUILD_LAYERS"], "3")
        self.assertEqual(plan["LAYER_THICKNESS"], "1.0e-4")
        self.assertEqual(plan["PATH_SAMPLE_STEP"], "5.0e-5")
        self.assertNotIn("--layers 2", plan["PATH_ARGS"])
        self.assertIn("--sample-step 5.0e-5", plan["PATH_ARGS"])
        self.assertIn("--xla-pardiso-mode phase23", plan["EXTRA_ARGS"])
        self.assertEqual(plan["EXPECTED_STEPS"], "384")

    def test_kaess_medium_fullheight_launcher_keeps_complete_height_and_budget(self):
        launcher = (
            ROOT
            / "cases"
            / "kaess_2023"
            / "run_kaess_medium_fullheight.sh"
        )
        text = launcher.read_text(encoding="utf-8")

        self.assertIn('BUILD_LAYERS="3"', text)
        self.assertIn('LAYER_THICKNESS="1.0e-4"', text)
        self.assertIn('PATH_SAMPLE_STEP="5.0e-5"', text)
        self.assertIn('PATH_ARGS="--power 250.0 --speed 0.850', text)
        self.assertNotIn('PATH_ARGS="${PATH_ARGS:-', text)
        self.assertIn('ELEMENT_TYPE="c3d8"', text)
        self.assertIn('MECH_EVERY="20"', text)
        self.assertIn('RECOAT_TIME="45.0"', text)
        self.assertIn("kaess-2023-medium-fullheight-macro", text)
        self.assertIn("--mechanics-residual-only-check", text)
        self.assertIn("--xla-pardiso-mode phase23", text)
        self.assertIn("--print-plan", text)
        self.assertIn("run_kaess_phase2.sh", text)

        phase2 = (
            ROOT / "cases" / "kaess_2023" / "run_kaess_phase2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('--layer-thickness "${LAYER_THICKNESS}"', phase2)
        self.assertIn('--layers "${BUILD_LAYERS}"', phase2)
        self.assertIn('RUN_LABEL="${RUN_LABEL:-kaess-2023-phase2-', phase2)
        self.assertIn('BUILD_LAYERS="${BUILD_LAYERS:-10}"', phase2)
        self.assertIn('LAYER_THICKNESS="${LAYER_THICKNESS:-3.0e-5}"', phase2)
        self.assertIn('RECOAT_TIME="${RECOAT_TIME:-10.0}"', phase2)
        path_generator_call = phase2[
            phase2.index("make_kaess_path.py") : phase2.index("XRD_PROTOCOL=")
        ]
        self.assertIn('--layers "${BUILD_LAYERS}"', path_generator_call)
        self.assertIn(
            '--layer-thickness "${LAYER_THICKNESS}"', path_generator_call
        )
        self.assertIn('--recoat-time "${RECOAT_TIME}"', phase2)
        self.assertIn('--recoat-steps "${RECOAT_STEPS}"', phase2)
        self.assertIn(
            '--cooling-steps "${COOLING_STEPS}" --cooling-dt "${COOLING_DT}"',
            phase2,
        )

    def test_kaess_medium_fullheight_path_has_384_step_budget(self):
        generator = ROOT / "cases" / "kaess_2023" / "make_kaess_path.py"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "medium_path.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    "--layers",
                    "3",
                    "--layer-thickness",
                    "1.0e-4",
                    "--sample-step",
                    "5.0e-5",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            states, _, _ = generate_path_file_step_states(
                SimpleNamespace(
                    mesh_length_scale=1.0,
                    path_length_scale=1.0,
                    path_file=str(output),
                    recoat_steps=10,
                    recoat_time=45.0,
                    dt=3.0e-5,
                    layers=3,
                    cooling_dt=1.0,
                    cooling_steps=30,
                ),
                [0.0, 0.0, 0.0],
                [1.0e-3, 0.5e-3, 0.6e-3],
                2,
            )

        self.assertEqual(len(rows), 334)
        self.assertEqual({int(row["layer"]) for row in rows}, {1, 2, 3})
        self.assertAlmostEqual(min(float(row["z"]) for row in rows), 4.0e-4)
        self.assertAlmostEqual(max(float(row["z"]) for row in rows), 6.0e-4)
        self.assertEqual(len(states), 384)
        recoat = [state for state in states if state.mode == "recoat"]
        cooling = [state for state in states if state.mode == "cooling"]
        self.assertEqual(len(recoat), 20)
        self.assertAlmostEqual(sum(state.dt for state in recoat), 90.0)
        self.assertEqual(len(cooling), 30)
        self.assertAlmostEqual(sum(state.dt for state in cooling), 30.0)
        self.assertEqual(states[-1].layer_idx, 2)

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
