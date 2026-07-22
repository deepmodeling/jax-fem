import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "legacy" / "v02" / "run_ti64_material.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_ti64_material", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ti64RunnerTest(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def write_table(self, path):
        path.write_text("T,value\n300,1\n1000,2\n", encoding="utf-8")

    def test_material_pack_resolves_project_relative_table_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            repo_root = project_root / "jax-fem"
            material_dir = project_root / "materials" / "Ti-6Al-4V"
            material_dir.mkdir(parents=True)
            for name in (
                "k_solid_table.csv",
                "cp_solid_table.csv",
                "E_table.csv",
                "alpha_table.csv",
                "yield_table.csv",
            ):
                self.write_table(material_dir / name)
            config_path = material_dir / "ti64_material_config_initial.json"
            config_path.write_text(
                json.dumps(
                    {
                        "k_table_solid": "materials/Ti-6Al-4V/k_solid_table.csv",
                        "cp_table_solid": "materials/Ti-6Al-4V/cp_solid_table.csv",
                        "E_table": "materials/Ti-6Al-4V/E_table.csv",
                        "alpha_table": "materials/Ti-6Al-4V/alpha_table.csv",
                        "yield_table": "materials/Ti-6Al-4V/yield_table.csv",
                        "absorptivity": 0.5,
                        "mechanics_model": "j2_plastic",
                        "reset_plastic_on_melt": True,
                    }
                ),
                encoding="utf-8",
            )

            config, tables = self.runner.load_material_pack(material_dir, config_path, repo_root)
            args = self.runner.material_args(config, tables)

            self.assertEqual(tables["k_table_solid"], (material_dir / "k_solid_table.csv").resolve())
            self.assertIn("--k-table-solid", args)
            self.assertIn(str((material_dir / "k_solid_table.csv").resolve()), args)
            self.assertIn("--absorptivity", args)
            self.assertIn("0.5", args)
            self.assertIn("--mechanics-model", args)
            self.assertIn("j2_plastic", args)
            self.assertIn("--reset-plastic-on-melt", args)

    def test_pythonpath_prepends_v01_and_repo_root(self):
        repo_root = Path("/tmp/example/jax-fem")
        value = self.runner.build_pythonpath(repo_root, {"PYTHONPATH": "existing"})
        parts = value.split(os.pathsep)
        self.assertEqual(parts[0], str(repo_root / "legacy" / "v01"))
        self.assertEqual(parts[1], str(repo_root))
        self.assertEqual(parts[2], "existing")

    def test_passthrough_separator_is_not_forwarded_to_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            repo_root = project_root / "jax-fem"
            solver = repo_root / "legacy" / "v02" / "am_thermal_stress_upgraded.py"
            material_dir = project_root / "materials" / "Ti-6Al-4V"
            solver.parent.mkdir(parents=True)
            material_dir.mkdir(parents=True)
            solver.write_text("# solver placeholder\n", encoding="utf-8")
            for name in (
                "k_solid_table.csv",
                "cp_solid_table.csv",
                "E_table.csv",
                "alpha_table.csv",
                "yield_table.csv",
            ):
                self.write_table(material_dir / name)
            config_path = material_dir / "ti64_material_config_initial.json"
            config_path.write_text(
                json.dumps(
                    {
                        "k_table_solid": "materials/Ti-6Al-4V/k_solid_table.csv",
                        "cp_table_solid": "materials/Ti-6Al-4V/cp_solid_table.csv",
                        "E_table": "materials/Ti-6Al-4V/E_table.csv",
                        "alpha_table": "materials/Ti-6Al-4V/alpha_table.csv",
                        "yield_table": "materials/Ti-6Al-4V/yield_table.csv",
                    }
                ),
                encoding="utf-8",
            )

            parser = self.runner.build_parser()
            args, passthrough = parser.parse_known_args(
                [
                    "--material-dir",
                    str(material_dir),
                    "--solver",
                    str(solver),
                    "--",
                    "--custom-solver-arg",
                    "value",
                ]
            )
            if passthrough and passthrough[0] == "--":
                passthrough = passthrough[1:]
            command, _env = self.runner.build_command(args, passthrough)

            self.assertIn("--custom-solver-arg", command)
            self.assertNotIn("-- --custom-solver-arg", " ".join(command))


if __name__ == "__main__":
    unittest.main()
