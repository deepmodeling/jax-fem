import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
XLA_WRAPPER_PATH = (
    REPO_ROOT
    / "jax_fem_am"
    / "simulation"
    / "acceleration.py"
)
BENCH_PATH = REPO_ROOT / "legacy" / "v04" / "bench_mech100_xla.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MacroMech100V04XlaWrapperTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_module(XLA_WRAPPER_PATH, "macro_mech100_v04_xla")

    def test_base_solver_path_points_to_v03_solver(self):
        self.assertEqual(
            self.wrapper.BASE_SOLVER_PATH,
            REPO_ROOT
            / "jax_fem_am"
            / "simulation"
            / "stepper.py",
        )
        self.assertTrue(self.wrapper.BASE_SOLVER_PATH.exists())

    def test_solver_patch_records_nonlinear_solve_overhead(self):
        def solver(_problem, solver_options=None):
            report.add(self.wrapper.STAGE_LOCAL_ASSEMBLY, 0.25)
            report.add(self.wrapper.STAGE_SOLVER, 0.5)
            return ["ok", solver_options]

        fake_base = SimpleNamespace(solver=solver)
        with mock.patch.object(
            self.wrapper.time,
            "perf_counter",
            side_effect=[0.0, 1.0, 2.0, 3.0],
        ):
            report = self.wrapper.ProfilingReport("solver-overhead")
            self.wrapper.install_solver_patch(
                fake_base,
                {"spsolve_solver": {}},
                fallback_to_spsolve=False,
                profiler=report,
                profile_solver_call=False,
            )
            result = fake_base.solver("problem", solver_options={})

        self.assertEqual(result[0], "ok")
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_NONLINEAR_SOLVE],
            1.0,
        )
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_NONLINEAR_SOLVE_OVERHEAD],
            0.25,
        )
        self.assertEqual(
            report.stage_calls[self.wrapper.STAGE_NONLINEAR_SOLVE],
            1,
        )
        self.assertEqual(
            report.stage_calls[self.wrapper.STAGE_NONLINEAR_SOLVE_OVERHEAD],
            1,
        )

    def test_dry_run_uses_v03_parser_without_requiring_run_interface(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_json = Path(tmp) / "profile.json"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                rc = self.wrapper.main(
                    [
                        "--xla-linear-solver",
                        "spsolve",
                        "--xla-dry-run",
                        "--profile-json",
                        str(profile_json),
                        "--profile-label",
                        "dry",
                    ]
                )

            report = json.loads(profile_json.read_text())

        self.assertEqual(rc, 0)
        fake_base.main.assert_not_called()
        self.assertIn("original_solver_module", stdout.getvalue())
        self.assertIn("full_loop_xla", stdout.getvalue())
        self.assertEqual(report["label"], "dry")
        self.assertEqual(report["meta"]["linear_solver"], "spsolve_solver")
        self.assertEqual(
            report["meta"]["linear_solver_label"],
            "spsolve_solver(cpu scipy baseline)",
        )
        self.assertEqual(report["meta"]["linear_solver_options"], {"spsolve_solver": {}})
        self.assertIn("thermal_only_mech_surrogate = True", stdout.getvalue())
        self.assertTrue(
            report["meta"]["thermal_only_mechanics_surrogate_enabled"]
        )

    def test_dry_run_records_jax_solver_controls_in_profile_meta(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_json = Path(tmp) / "profile.json"

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                rc = self.wrapper.main(
                    [
                        "--xla-linear-solver",
                        "jax",
                        "--xla-jax-method",
                        "cg",
                        "--xla-jax-tol",
                        "1e-8",
                        "--xla-jax-maxiter",
                        "20",
                        "--xla-dry-run",
                        "--profile-json",
                        str(profile_json),
                    ]
                )

            report = json.loads(profile_json.read_text())

        self.assertEqual(rc, 0)
        self.assertEqual(report["meta"]["linear_solver"], "jax_solver")
        self.assertEqual(
            report["meta"]["linear_solver_options"],
            {
                "jax_solver": {
                    "precond": False,
                    "method": "cg",
                    "tol": 1e-8,
                    "maxiter": 20,
                }
            },
        )
        self.assertEqual(
            report["meta"]["linear_solver_label"],
            "jax_solver(method=cg, precond=False, tol=1e-08, maxiter=20)",
        )

    def test_dry_run_records_jax_residual_check_skip_in_profile_meta(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_json = Path(tmp) / "profile.json"

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                rc = self.wrapper.main(
                    [
                        "--xla-linear-solver",
                        "jax",
                        "--xla-jax-method",
                        "cg",
                        "--xla-jax-skip-residual-check",
                        "--xla-dry-run",
                        "--profile-json",
                        str(profile_json),
                    ]
                )

            report = json.loads(profile_json.read_text())

        self.assertEqual(rc, 0)
        self.assertEqual(
            report["meta"]["linear_solver_options"],
            {
                "jax_solver": {
                    "precond": False,
                    "method": "cg",
                    "check_residual": False,
                }
            },
        )
        self.assertEqual(
            report["meta"]["linear_solver_label"],
            "jax_solver(method=cg, precond=False, check_residual=False)",
        )

    def test_dry_run_accepts_jax_sparse_direct_solver_method(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_json = Path(tmp) / "profile.json"

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                rc = self.wrapper.main(
                    [
                        "--xla-linear-solver",
                        "jax",
                        "--xla-jax-method",
                        "spsolve",
                        "--xla-jax-skip-residual-check",
                        "--xla-dry-run",
                        "--profile-json",
                        str(profile_json),
                    ]
                )

            report = json.loads(profile_json.read_text())

        self.assertEqual(rc, 0)
        self.assertEqual(
            report["meta"]["linear_solver_options"],
            {
                "jax_solver": {
                    "precond": False,
                    "method": "spsolve",
                    "check_residual": False,
                }
            },
        )
        self.assertEqual(
            report["meta"]["linear_solver_label"],
            "jax_solver(method=spsolve, precond=False, check_residual=False)",
        )

    def test_pardiso_phase23_mode_is_explicit_and_opt_in(self):
        parser = self.wrapper.build_arg_parser()
        base_args = parser.parse_args(["--xla-linear-solver", "pardiso"])
        base_solver = self.wrapper.linear_options_from_args(base_args)[
            "custom_solver"
        ]
        self.assertEqual(base_solver.label, "pardiso_solver(mkl multithreaded direct)")

        phase23_args = parser.parse_args(
            [
                "--xla-linear-solver",
                "pardiso",
                "--xla-pardiso-mode",
                "phase23",
            ]
        )
        phase23_solver = self.wrapper.linear_options_from_args(phase23_args)[
            "custom_solver"
        ]
        self.assertEqual(phase23_solver.label, "pardiso_v07(phase23)")

    def test_dry_run_records_cell_assembly_cut_override_in_profile_meta(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_json = Path(tmp) / "profile.json"
            stdout = io.StringIO()

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(stdout),
                redirect_stderr(io.StringIO()),
            ):
                rc = self.wrapper.main(
                    [
                        "--xla-dry-run",
                        "--xla-cell-num-cuts",
                        "1",
                        "--profile-json",
                        str(profile_json),
                    ]
                )

            report = json.loads(profile_json.read_text())

        self.assertEqual(rc, 0)
        self.assertIn("cell_assembly_cuts    = 1", stdout.getvalue())
        self.assertEqual(report["meta"]["cell_assembly_num_cuts"], 1)
        self.assertEqual(report["meta"]["cell_assembly_num_cuts_source"], "cli")
        self.assertIsNone(report["meta"]["cell_assembly_target_batch_size"])
        self.assertEqual(report["meta"]["cell_assembly_chunking"], "fixed_num_cuts")

    def test_dry_run_uses_auto_cell_assembly_chunking_by_default(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_json = Path(tmp) / "profile.json"
            stdout = io.StringIO()

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(stdout),
                redirect_stderr(io.StringIO()),
            ):
                rc = self.wrapper.main(
                    [
                        "--xla-dry-run",
                        "--profile-json",
                        str(profile_json),
                    ]
                )

            report = json.loads(profile_json.read_text())

        self.assertEqual(rc, 0)
        self.assertIn("cell_assembly_chunking = auto", stdout.getvalue())
        self.assertIsNone(report["meta"]["cell_assembly_num_cuts"])
        self.assertEqual(
            report["meta"]["cell_assembly_target_batch_size"],
            self.wrapper.DEFAULT_CELL_TARGET_BATCH_SIZE,
        )
        self.assertEqual(
            report["meta"]["cell_assembly_chunking"],
            "auto_target_batch_size",
        )

    def test_dry_run_records_loop_kernel_jit_default_and_override(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            default_json = Path(tmp) / "default.json"
            disabled_json = Path(tmp) / "disabled.json"
            default_stdout = io.StringIO()
            disabled_stdout = io.StringIO()

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(default_stdout),
                redirect_stderr(io.StringIO()),
            ):
                self.wrapper.main(
                    [
                        "--xla-dry-run",
                        "--profile-json",
                        str(default_json),
                    ]
                )

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(disabled_stdout),
                redirect_stderr(io.StringIO()),
            ):
                self.wrapper.main(
                    [
                        "--xla-dry-run",
                        "--no-xla-jit-loop-kernels",
                        "--no-xla-skip-unused-mechanics-material",
                        "--profile-json",
                        str(disabled_json),
                    ]
                )

            default_report = json.loads(default_json.read_text())
            disabled_report = json.loads(disabled_json.read_text())

        self.assertIn("loop_kernel_jit       = True", default_stdout.getvalue())
        self.assertIn("loop_kernel_jit       = False", disabled_stdout.getvalue())
        self.assertIn("skip_unused_mech_mat = True", default_stdout.getvalue())
        self.assertIn("skip_unused_mech_mat = False", disabled_stdout.getvalue())
        self.assertTrue(default_report["meta"]["loop_kernel_jit_enabled"])
        self.assertFalse(disabled_report["meta"]["loop_kernel_jit_enabled"])
        self.assertTrue(
            default_report["meta"]["skip_unused_mechanics_material_enabled"]
        )
        self.assertFalse(
            disabled_report["meta"]["skip_unused_mechanics_material_enabled"]
        )

    def test_dry_run_records_dof_to_quad_cache_default_and_override(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            default_json = Path(tmp) / "default.json"
            disabled_json = Path(tmp) / "disabled.json"
            default_stdout = io.StringIO()
            disabled_stdout = io.StringIO()

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(default_stdout),
                redirect_stderr(io.StringIO()),
            ):
                self.wrapper.main(
                    [
                        "--xla-dry-run",
                        "--profile-json",
                        str(default_json),
                    ]
                )

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(disabled_stdout),
                redirect_stderr(io.StringIO()),
            ):
                self.wrapper.main(
                    [
                        "--xla-dry-run",
                        "--no-xla-dof-to-quad-cache",
                        "--profile-json",
                        str(disabled_json),
                    ]
                )

            default_report = json.loads(default_json.read_text())
            disabled_report = json.loads(disabled_json.read_text())

        self.assertIn("dof_to_quad_cache    = True", default_stdout.getvalue())
        self.assertIn("dof_to_quad_cache    = False", disabled_stdout.getvalue())
        self.assertTrue(default_report["meta"]["dof_to_quad_cache_enabled"])
        self.assertEqual(
            default_report["meta"]["dof_to_quad_cache_max_entries"],
            self.wrapper.DOF_TO_QUAD_CACHE_MAX_ENTRIES,
        )
        self.assertFalse(disabled_report["meta"]["dof_to_quad_cache_enabled"])
        self.assertEqual(disabled_report["meta"]["dof_to_quad_cache_max_entries"], 0)

    def test_dry_run_records_quiet_jax_fem_logs_default_and_override(self):
        fake_base = SimpleNamespace(
            read_config=lambda path: {},
            build_parser=lambda config=None: self.wrapper.argparse.ArgumentParser(),
            main=mock.Mock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            default_json = Path(tmp) / "default.json"
            disabled_json = Path(tmp) / "disabled.json"
            default_stdout = io.StringIO()
            disabled_stdout = io.StringIO()

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(default_stdout),
                redirect_stderr(io.StringIO()),
            ):
                self.wrapper.main(
                    [
                        "--xla-dry-run",
                        "--profile-json",
                        str(default_json),
                    ]
                )

            with (
                mock.patch.object(self.wrapper, "load_base_solver", return_value=fake_base),
                redirect_stdout(disabled_stdout),
                redirect_stderr(io.StringIO()),
            ):
                self.wrapper.main(
                    [
                        "--xla-dry-run",
                        "--no-xla-quiet-jax-fem-logs",
                        "--profile-json",
                        str(disabled_json),
                    ]
                )

            default_report = json.loads(default_json.read_text())
            disabled_report = json.loads(disabled_json.read_text())

        self.assertIn("quiet_jax_fem_logs = True", default_stdout.getvalue())
        self.assertIn("quiet_jax_fem_logs = False", disabled_stdout.getvalue())
        self.assertTrue(default_report["meta"]["quiet_jax_fem_logs_enabled"])
        self.assertEqual(default_report["meta"]["jax_fem_log_level"], "WARNING")
        self.assertFalse(disabled_report["meta"]["quiet_jax_fem_logs_enabled"])
        self.assertEqual(disabled_report["meta"]["jax_fem_log_level"], "preserve")
        self.assertIn("lazy_output_postprocess = False", default_stdout.getvalue())
        self.assertFalse(default_report["meta"]["lazy_output_postprocess_enabled"])

    def test_loop_kernel_jit_matches_v03_no_table_kernels(self):
        try:
            import numpy as onp
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover - runtime dependent
            self.skipTest(f"jax/numpy unavailable: {exc}")

        base = self.wrapper.load_base_solver(
            module_name="macro_mech100_v03_loop_jit_test"
        )
        args = SimpleNamespace(
            rho=7800.0,
            cp=500.0,
            conductivity=20.0,
            rho_solid=None,
            cp_solid=None,
            conductivity_solid=None,
            rho_liquid=None,
            cp_liquid=None,
            conductivity_liquid=None,
            rho_powder=3900.0,
            cp_powder=460.0,
            conductivity_powder=1.2,
            inactive_thermal_factor=1e-6,
            old_layer_thermal_factor=1e-4,
            powder_mode="powder",
            layer_activation_mode="layer_on_scan",
            future_layer_mode="void",
            solidus_temperature=600.0,
            liquidus_temperature=900.0,
            latent_heat=2.7e5,
            young=2.0e11,
            alpha=1.2e-5,
            poisson=0.29,
            mechanics_model="linear_elastic",
            mushy_mechanics_factor=1e-2,
            liquid_mechanics_factor=1e-4,
            inactive_mechanics_factor=1e-9,
            reset_plastic_on_melt=True,
        )
        tables = {key: None for key in (
            *self.wrapper.THERMAL_TABLE_KEYS,
            *self.wrapper.MECHANICAL_TABLE_KEYS,
        )}
        T_old_quad = jnp.asarray(
            [
                [[300.0], [650.0]],
                [[950.0], [580.0]],
                [[750.0], [920.0]],
            ]
        )
        active_quad = jnp.asarray(
            [[[1.0], [1.0]], [[0.0], [0.0]], [[1.0], [1.0]]]
        )
        printed_quad = jnp.asarray(
            [[[1.0], [1.0]], [[0.0], [0.0]], [[1.0], [1.0]]]
        )
        cooling_only_quad = jnp.asarray(
            [[[0.0], [0.0]], [[0.0], [0.0]], [[1.0], [1.0]]]
        )
        phase_quad = jnp.asarray(
            [
                [[base.STATE_POWDER], [base.STATE_SOLID]],
                [[base.STATE_VOID], [base.STATE_VOID]],
                [[base.STATE_MUSHY], [base.STATE_LIQUID]],
            ]
        )
        T_ref_quad = 300.0 * jnp.ones_like(T_old_quad)
        eqp_quad = 0.2 * jnp.ones_like(T_old_quad)

        expected_thermal = base.thermal_material_quads(
            T_old_quad,
            active_quad,
            phase_quad,
            args,
            tables,
            printed_quad=printed_quad,
            cooling_only_quad=cooling_only_quad,
        )
        expected_mechanics = base.mechanics_material_quads(
            T_old_quad,
            active_quad,
            phase_quad,
            args,
            tables,
        )
        expected_history = base.update_phase_reference_and_eqp(
            T_old_quad,
            printed_quad,
            phase_quad,
            T_ref_quad,
            eqp_quad,
            args,
        )

        report = self.wrapper.ProfilingReport("jit-test")
        self.assertTrue(
            self.wrapper.install_loop_kernel_jit_patch(base, report, enabled=True)
        )
        actual_thermal = base.thermal_material_quads(
            T_old_quad,
            active_quad,
            phase_quad,
            args,
            tables,
            printed_quad=printed_quad,
            cooling_only_quad=cooling_only_quad,
        )
        actual_mechanics = base.mechanics_material_quads(
            T_old_quad,
            active_quad,
            phase_quad,
            args,
            tables,
        )
        actual_history = base.update_phase_reference_and_eqp(
            T_old_quad,
            printed_quad,
            phase_quad,
            T_ref_quad,
            eqp_quad,
            args,
        )

        for expected, actual in zip(expected_thermal, actual_thermal):
            onp.testing.assert_allclose(onp.asarray(actual), onp.asarray(expected))
        for expected, actual in zip(expected_mechanics, actual_mechanics):
            onp.testing.assert_allclose(onp.asarray(actual), onp.asarray(expected))
        for expected, actual in zip(expected_history, actual_history):
            onp.testing.assert_array_equal(onp.asarray(actual), onp.asarray(expected))
        self.assertEqual(report.meta["loop_kernel_jit_thermal_calls"], 1)
        self.assertEqual(report.meta["loop_kernel_jit_mechanics_calls"], 1)
        self.assertEqual(report.meta["loop_kernel_jit_history_calls"], 1)

    def test_loop_kernel_jit_skips_mechanics_material_when_mechanics_disabled(self):
        calls = []

        class FakeArray:
            shape = (2, 1, 1)

        T_quad = FakeArray()

        def mechanics_material_quads(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("mechanics material should be skipped")

        fake_base = SimpleNamespace(
            jax=object(),
            np=object(),
            thermal_material_quads=lambda *args, **kwargs: "thermal",
            mechanics_material_quads=mechanics_material_quads,
            update_phase_reference_and_eqp=lambda *args, **kwargs: "history",
        )
        args = SimpleNamespace(
            mechanics_every=0,
            release_after_cooling=False,
        )
        tables = {key: None for key in self.wrapper.MECHANICAL_TABLE_KEYS}
        tables["E"] = object()
        active_quad = FakeArray()
        phase_quad = FakeArray()
        report = self.wrapper.ProfilingReport("mechanics-disabled")

        self.assertTrue(
            self.wrapper.install_loop_kernel_jit_patch(fake_base, report, enabled=True)
        )
        result = fake_base.mechanics_material_quads(
            T_quad,
            active_quad,
            phase_quad,
            args,
            tables,
        )

        self.assertEqual(calls, [])
        self.assertEqual(len(result), 6)
        for value in result:
            self.assertEqual(value.shape, T_quad.shape)
        self.assertEqual(
            report.meta["loop_kernel_jit_mechanics_disabled_skips"],
            1,
        )

    def test_loop_kernel_jit_keeps_mechanics_material_for_release_run(self):
        calls = []

        class FakeArray:
            shape = (1, 1, 1)

        def mechanics_material_quads(*args, **kwargs):
            calls.append((args, kwargs))
            return ("original",) * 6

        fake_base = SimpleNamespace(
            jax=object(),
            np=object(),
            thermal_material_quads=lambda *args, **kwargs: "thermal",
            mechanics_material_quads=mechanics_material_quads,
            update_phase_reference_and_eqp=lambda *args, **kwargs: "history",
        )
        args = SimpleNamespace(
            mechanics_every=0,
            release_after_cooling=True,
        )
        tables = {key: None for key in self.wrapper.MECHANICAL_TABLE_KEYS}
        tables["E"] = object()
        report = self.wrapper.ProfilingReport("mechanics-release")

        self.assertTrue(
            self.wrapper.install_loop_kernel_jit_patch(fake_base, report, enabled=True)
        )
        result = fake_base.mechanics_material_quads(
            FakeArray(),
            FakeArray(),
            FakeArray(),
            args,
            tables,
        )

        self.assertEqual(result, ("original",) * 6)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report.meta["loop_kernel_jit_mechanics_fallbacks"], 1)

    def test_thermal_only_mechanics_surrogate_reuses_thermal_fe(self):
        calls = []
        thermal_fe = SimpleNamespace(
            points=[0, 1, 2],
            node_inds_list=["thermal-bc"],
            convert_from_dof_to_quad=lambda sol: ("quad", sol),
        )

        class FakeThermal:
            def __init__(self, *args, **kwargs):
                calls.append(("TransientThermal", args, kwargs))
                self.fes = [thermal_fe]

        class FakeMechanics:
            def __init__(self, *args, **kwargs):
                calls.append(("ThermoMechanical", args, kwargs))
                self.fes = [SimpleNamespace(node_inds_list=["mechanics-bc"])]
                self.num_total_dofs_all_vars = 99

        fake_base = SimpleNamespace(
            TransientThermal=FakeThermal,
            ThermoMechanical=FakeMechanics,
        )
        args = SimpleNamespace(mechanics_every=0, release_after_cooling=False)
        report = self.wrapper.ProfilingReport("thermal-only-mechanics")

        installed = self.wrapper.install_thermal_only_mechanics_surrogate_patch(
            fake_base,
            args,
            report,
            enabled=True,
        )
        thermal = fake_base.TransientThermal(mesh="mesh")
        mechanics = fake_base.ThermoMechanical(
            mesh="mesh",
            vec=3,
            dirichlet_bc_info=[[], [], []],
        )

        self.assertTrue(installed)
        self.assertEqual([entry[0] for entry in calls], ["TransientThermal"])
        self.assertIs(thermal.fes[0], thermal_fe)
        self.assertEqual(
            mechanics.fes[0].convert_from_dof_to_quad("T"),
            ("quad", "T"),
        )
        self.assertEqual(mechanics.num_total_dofs_all_vars, 9)
        self.assertEqual(
            report.meta["thermal_only_mechanics_surrogate_hits"],
            1,
        )

    def test_thermal_only_mechanics_surrogate_falls_back_without_bc_reuse(self):
        calls = []
        thermal_fe = SimpleNamespace(
            points=[0, 1],
            node_inds_list=[],
            convert_from_dof_to_quad=lambda sol: ("quad", sol),
        )

        class FakeThermal:
            def __init__(self, *args, **kwargs):
                self.fes = [thermal_fe]

        class FakeMechanics:
            def __init__(self, *args, **kwargs):
                calls.append(("ThermoMechanical", args, kwargs))
                self.fes = [SimpleNamespace(node_inds_list=["mechanics-bc"])]
                self.num_total_dofs_all_vars = 6

        fake_base = SimpleNamespace(
            TransientThermal=FakeThermal,
            ThermoMechanical=FakeMechanics,
        )
        args = SimpleNamespace(mechanics_every=0, release_after_cooling=False)
        report = self.wrapper.ProfilingReport("thermal-only-no-bc-reuse")

        self.assertTrue(
            self.wrapper.install_thermal_only_mechanics_surrogate_patch(
                fake_base,
                args,
                report,
                enabled=True,
            )
        )
        fake_base.TransientThermal(mesh="mesh")
        mechanics = fake_base.ThermoMechanical(
            mesh="mesh",
            vec=3,
            dirichlet_bc_info=[[lambda point: True], [0], [lambda point: 0.0]],
        )

        self.assertIsInstance(mechanics, FakeMechanics)
        self.assertEqual([entry[0] for entry in calls], ["ThermoMechanical"])
        self.assertEqual(
            report.meta["thermal_only_mechanics_surrogate_fallbacks"],
            1,
        )

    def test_thermal_only_mechanics_surrogate_keeps_release_path(self):
        calls = []

        class FakeThermal:
            def __init__(self, *args, **kwargs):
                self.fes = [SimpleNamespace(points=[0])]

        class FakeMechanics:
            def __init__(self, *args, **kwargs):
                calls.append(("ThermoMechanical", args, kwargs))
                self.fes = [SimpleNamespace(node_inds_list=["mechanics-bc"])]
                self.num_total_dofs_all_vars = 3

        fake_base = SimpleNamespace(
            TransientThermal=FakeThermal,
            ThermoMechanical=FakeMechanics,
        )
        args = SimpleNamespace(mechanics_every=0, release_after_cooling=True)
        report = self.wrapper.ProfilingReport("thermal-only-release")

        self.assertTrue(
            self.wrapper.install_thermal_only_mechanics_surrogate_patch(
                fake_base,
                args,
                report,
                enabled=True,
            )
        )
        fake_base.TransientThermal(mesh="mesh")
        mechanics = fake_base.ThermoMechanical(mesh="mesh", vec=3)

        self.assertIsInstance(mechanics, FakeMechanics)
        self.assertEqual([entry[0] for entry in calls], ["ThermoMechanical"])
        self.assertEqual(
            report.meta["thermal_only_mechanics_surrogate_fallbacks"],
            1,
        )

    def test_install_solver_patch_records_solver_stage_and_retries_spsolve(self):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            if len(calls) == 1:
                raise RuntimeError("gpu failed")
            return "ok"

        original_module = SimpleNamespace(solver=fake_solver)
        report = self.wrapper.ProfilingReport(label="unit")

        self.wrapper.install_solver_patch(
            original_module,
            {"jax_solver": {"precond": True}},
            fallback_to_spsolve=True,
            profiler=report,
        )
        result = original_module.solver("problem", solver_options={"newton": {}})

        self.assertEqual(result, "ok")
        self.assertEqual(calls[0]["newton"]["linear"], {"jax_solver": {"precond": True}})
        self.assertEqual(calls[1]["newton"]["linear"], {"spsolve_solver": {}})
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_SOLVER], 2)
        self.assertEqual(report.meta["solver_fallbacks"], 1)

    def test_install_solver_patch_does_not_retry_newton_stall(self):
        # A Newton stall reproduces bit-identically under spsolve (both direct
        # solvers), so the fallback must re-raise instead of burning a second
        # full Newton budget; increment cutback handles it upstream.
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            raise RuntimeError(
                "Newton solver did not converge within max_iter=50 iterations")

        original_module = SimpleNamespace(solver=fake_solver)
        report = self.wrapper.ProfilingReport(label="unit")

        self.wrapper.install_solver_patch(
            original_module,
            {"jax_solver": {"precond": True}},
            fallback_to_spsolve=True,
            profiler=report,
        )
        with self.assertRaises(RuntimeError):
            original_module.solver("problem", solver_options={"newton": {}})
        self.assertEqual(len(calls), 1)
        self.assertNotIn("solver_fallbacks", report.meta)

    def test_install_solver_patch_records_setup_once_before_first_solve(self):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            return "ok"

        original_module = SimpleNamespace(solver=fake_solver)
        report = self.wrapper.ProfilingReport(label="setup")
        report.wall_start = 10.0
        report.stage_seconds[self.wrapper.STAGE_IO] = 0.25
        report.stage_calls[self.wrapper.STAGE_IO] = 1

        self.wrapper.install_solver_patch(
            original_module,
            {"spsolve_solver": {}},
            fallback_to_spsolve=True,
            profiler=report,
            profile_solver_call=False,
        )
        with mock.patch.object(
            self.wrapper.time,
            "perf_counter",
            side_effect=[13.25, 20.0, 20.5, 21.0, 21.5],
        ):
            self.assertEqual(original_module.solver("problem"), "ok")
            self.assertEqual(original_module.solver("problem"), "ok")

        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(report.stage_seconds[self.wrapper.STAGE_SETUP], 3.0)
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_SETUP], 1)
        self.assertAlmostEqual(report.meta["setup_seconds_before_first_solve"], 3.0)
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_NONLINEAR_SOLVE],
            1.0,
        )
        self.assertEqual(
            report.stage_calls[self.wrapper.STAGE_NONLINEAR_SOLVE],
            2,
        )

    def test_setup_detail_timing_patch_records_init_substages_before_first_solve(self):
        calls = []

        class FakeMesh:
            def __init__(self, *args, **kwargs):
                calls.append(("Mesh", args, kwargs))

        class FakeThermal:
            def __init__(self, *args, **kwargs):
                calls.append(("TransientThermal", args, kwargs))

        class FakeMechanics:
            def __init__(self, *args, **kwargs):
                calls.append(("ThermoMechanical", args, kwargs))

        def read_tet4_inp(*args, **kwargs):
            calls.append(("read_tet4_inp", args, kwargs))
            return "points", "cells", "selected"

        def generate_raster_step_states(*args, **kwargs):
            calls.append(("generate_raster_step_states", args, kwargs))
            return ["state"], 1.0, 2.0

        fake_base = SimpleNamespace(
            read_tet4_inp=read_tet4_inp,
            generate_raster_step_states=generate_raster_step_states,
            Mesh=FakeMesh,
            TransientThermal=FakeThermal,
            ThermoMechanical=FakeMechanics,
        )
        report = self.wrapper.ProfilingReport(label="setup-detail")

        installed = self.wrapper.install_setup_detail_timing_patch(fake_base, report)

        with mock.patch.object(
            self.wrapper.time,
            "perf_counter",
            side_effect=[
                10.0, 10.25,
                20.0, 20.50,
                30.0, 30.10,
                40.0, 40.40,
                50.0, 50.80,
            ],
        ):
            self.assertEqual(
                fake_base.read_tet4_inp("mesh.inp", 10),
                ("points", "cells", "selected"),
            )
            self.assertEqual(
                fake_base.generate_raster_step_states("args"),
                (["state"], 1.0, 2.0),
            )
            self.assertIsInstance(fake_base.Mesh("points", "cells"), FakeMesh)
            self.assertIsInstance(fake_base.TransientThermal(mesh="mesh"), FakeThermal)
            self.assertIsInstance(fake_base.ThermoMechanical(mesh="mesh"), FakeMechanics)

        self.assertTrue(installed)
        self.assertEqual(
            [entry[0] for entry in calls],
            [
                "read_tet4_inp",
                "generate_raster_step_states",
                "Mesh",
                "TransientThermal",
                "ThermoMechanical",
            ],
        )
        self.assertAlmostEqual(report.meta["setup_detail_mesh_read_seconds"], 0.25)
        self.assertAlmostEqual(report.meta["setup_detail_path_generation_seconds"], 0.50)
        self.assertAlmostEqual(
            report.meta["setup_detail_mesh_construction_seconds"], 0.10
        )
        self.assertAlmostEqual(
            report.meta["setup_detail_thermal_problem_seconds"], 0.40
        )
        self.assertAlmostEqual(
            report.meta["setup_detail_mechanics_problem_seconds"], 0.80
        )
        self.assertEqual(report.meta["setup_detail_mesh_read_calls"], 1)
        self.assertEqual(report.meta["setup_detail_path_generation_calls"], 1)
        self.assertEqual(report.meta["setup_detail_mesh_construction_calls"], 1)
        self.assertEqual(report.meta["setup_detail_thermal_problem_calls"], 1)
        self.assertEqual(report.meta["setup_detail_mechanics_problem_calls"], 1)

    def test_setup_detail_timing_patch_skips_after_first_solve_boundary(self):
        def read_tet4_inp(*args, **kwargs):
            return "ok"

        fake_base = SimpleNamespace(read_tet4_inp=read_tet4_inp)
        report = self.wrapper.ProfilingReport(label="setup-detail")
        report.meta["setup_recorded_before_first_solve"] = True

        self.wrapper.install_setup_detail_timing_patch(fake_base, report)
        with mock.patch.object(
            self.wrapper.time,
            "perf_counter",
            side_effect=[10.0, 11.0],
        ):
            self.assertEqual(fake_base.read_tet4_inp(), "ok")

        self.assertNotIn("setup_detail_mesh_read_seconds", report.meta)

    def test_setup_boundary_records_unattributed_detail_seconds(self):
        report = self.wrapper.ProfilingReport(label="setup-detail")
        report.wall_start = 10.0
        report.meta["setup_detail_mesh_read_seconds"] = 0.25
        report.meta["setup_detail_thermal_problem_seconds"] = 0.75
        report.stage_seconds[self.wrapper.STAGE_IO] = 0.5

        with mock.patch.object(self.wrapper.time, "perf_counter", return_value=13.0):
            report.record_setup_before_first_solve()

        self.assertAlmostEqual(report.meta["setup_seconds_before_first_solve"], 2.5)
        self.assertAlmostEqual(report.meta["setup_detail_total_seconds"], 1.0)
        self.assertAlmostEqual(report.meta["setup_unattributed_seconds"], 1.5)

    def test_install_solver_patch_preserves_options_when_linear_override_is_none(self):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            return "ok"

        original_module = SimpleNamespace(solver=fake_solver)
        report = self.wrapper.ProfilingReport(label="keep")
        report.wall_start = 1.0
        original_options = {"newton": {"linear": {"spsolve_solver": {}}}}

        self.wrapper.install_solver_patch(
            original_module,
            None,
            fallback_to_spsolve=True,
            profiler=report,
            profile_solver_call=False,
        )
        with mock.patch.object(self.wrapper.time, "perf_counter", return_value=2.0):
            self.assertEqual(
                original_module.solver("problem", solver_options=original_options),
                "ok",
            )

        self.assertIs(calls[0], original_options)
        self.assertAlmostEqual(report.stage_seconds[self.wrapper.STAGE_SETUP], 1.0)

    def test_inject_newton_option_covers_nested_flat_and_explicit_layouts(self):
        nested = {"newton": {"linear": {"spsolve_solver": {}}}}
        injected = self.wrapper.inject_newton_option(
            nested, "residual_only_check", True
        )
        self.assertTrue(injected["newton"]["residual_only_check"])
        self.assertNotIn("residual_only_check", nested["newton"])

        explicit_off = {"newton": {"residual_only_check": False}}
        preserved = self.wrapper.inject_newton_option(
            explicit_off, "residual_only_check", True
        )
        self.assertFalse(preserved["newton"]["residual_only_check"])

        flat = {"spsolve_solver": {}}
        flat_injected = self.wrapper.inject_newton_option(
            flat, "residual_only_check", True
        )
        self.assertTrue(flat_injected["residual_only_check"])
        self.assertNotIn("residual_only_check", flat)

        arc_only = {"arc_length": {"control": "displacement"}}
        untouched = self.wrapper.inject_newton_option(
            arc_only, "residual_only_check", True
        )
        self.assertEqual(
            untouched, {"arc_length": {"control": "displacement"}}
        )

        self.assertTrue(
            self.wrapper.inject_newton_option(
                None, "residual_only_check", True
            )["residual_only_check"]
        )

    def test_install_solver_patch_injects_residual_only_check_for_thermal(self):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            return "ok"

        class FakeThermal:
            pass

        class FakeMechanics:
            pass

        original_module = SimpleNamespace(
            solver=fake_solver, TransientThermal=FakeThermal
        )
        original_options = {"newton": {"linear": {"spsolve_solver": {}}}}

        self.wrapper.install_solver_patch(
            original_module,
            None,
            fallback_to_spsolve=True,
            profiler=None,
            profile_solver_call=False,
            residual_only_check=True,
        )
        self.assertEqual(
            original_module.solver(
                FakeThermal(), solver_options=original_options
            ),
            "ok",
        )
        self.assertEqual(
            original_module.solver(
                FakeMechanics(), solver_options=original_options
            ),
            "ok",
        )

        self.assertTrue(calls[0]["newton"]["residual_only_check"])
        self.assertNotIn("residual_only_check", calls[1]["newton"])
        self.assertNotIn("residual_only_check", original_options["newton"])

    def test_residual_only_check_survives_wrapped_thermal_class(self):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            return "ok"

        class TransientThermal:
            pass

        # Simulate another v04 patch replacing the class with a factory fn.
        original_module = SimpleNamespace(
            solver=fake_solver,
            TransientThermal=lambda *a, **k: TransientThermal(),
        )

        self.wrapper.install_solver_patch(
            original_module,
            None,
            fallback_to_spsolve=True,
            profiler=None,
            profile_solver_call=False,
            residual_only_check=True,
        )
        self.assertEqual(
            original_module.solver(
                TransientThermal(),
                solver_options={"newton": {"linear": {"spsolve_solver": {}}}},
            ),
            "ok",
        )

        self.assertTrue(calls[0]["newton"]["residual_only_check"])

    def test_thermal_warm_start_patch_stashes_set_params_temperature(self):
        calls = []

        class FakeThermal:
            def set_params(self, params):
                calls.append(params)
                self.params_seen = params

        original_set_params = FakeThermal.set_params
        original_module = SimpleNamespace(TransientThermal=FakeThermal)
        report = self.wrapper.ProfilingReport(label="warm-start")
        initial_guess = SimpleNamespace(shape=(4, 1), dtype="float64")
        params = [initial_guess, "dt"]

        installed = self.wrapper.install_thermal_warm_start_patch(
            original_module,
            report,
        )
        thermal = FakeThermal()
        thermal.set_params(params)

        self.assertTrue(installed)
        self.assertIs(FakeThermal._v04_original_set_params, original_set_params)
        self.assertEqual(calls, [params])
        self.assertIs(thermal.params_seen, params)
        self.assertIs(thermal._v04_thermal_initial_guess, initial_guess)
        self.assertEqual(
            report.meta["thermal_warm_start_patch"],
            "TransientThermal.set_params[0]",
        )

    def test_install_solver_patch_injects_thermal_initial_guess(self):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            return "ok"

        original_module = SimpleNamespace(solver=fake_solver)
        report = self.wrapper.ProfilingReport(label="warm-start")
        initial_guess = SimpleNamespace(shape=(3, 1), dtype="float64")
        problem = SimpleNamespace(_v04_thermal_initial_guess=initial_guess)
        original_options = {"newton": {"linear": {"spsolve_solver": {}}}}

        self.wrapper.install_solver_patch(
            original_module,
            {"jax_solver": {"precond": False}},
            fallback_to_spsolve=True,
            profiler=report,
            profile_solver_call=False,
            thermal_warm_start=True,
        )
        self.assertEqual(
            original_module.solver(problem, solver_options=original_options),
            "ok",
        )

        self.assertIs(calls[0]["newton"]["initial_guess"], initial_guess)
        self.assertEqual(
            calls[0]["newton"]["linear"],
            {"jax_solver": {"precond": False}},
        )
        self.assertNotIn("initial_guess", original_options["newton"])
        self.assertEqual(report.meta["thermal_warm_start_injections"], 1)
        self.assertEqual(
            report.meta["thermal_warm_start_last_guess"],
            "shape=(3, 1), dtype=float64",
        )

    def test_thermal_initial_guess_does_not_override_explicit_guess(self):
        calls = []

        def fake_solver(problem, solver_options=None):
            calls.append(solver_options)
            return "ok"

        original_module = SimpleNamespace(solver=fake_solver)
        report = self.wrapper.ProfilingReport(label="warm-start")
        thermal_guess = SimpleNamespace(shape=(3, 1), dtype="float64")
        explicit_guess = "explicit-initial-guess"
        problem = SimpleNamespace(_v04_thermal_initial_guess=thermal_guess)
        original_options = {
            "newton": {
                "initial_guess": explicit_guess,
                "linear": {"spsolve_solver": {}},
            }
        }

        self.wrapper.install_solver_patch(
            original_module,
            {"jax_solver": {"precond": False}},
            fallback_to_spsolve=True,
            profiler=report,
            profile_solver_call=False,
            thermal_warm_start=True,
        )
        self.assertEqual(
            original_module.solver(problem, solver_options=original_options),
            "ok",
        )

        self.assertEqual(calls[0]["newton"]["initial_guess"], explicit_guess)
        self.assertNotIn("thermal_warm_start_injections", report.meta)

    def test_finish_excludes_setup_from_python_overhead(self):
        report = self.wrapper.ProfilingReport(label="finish")
        report.wall_start = 5.0
        report.stage_seconds[self.wrapper.STAGE_SETUP] = 2.0
        report.stage_seconds[self.wrapper.STAGE_SOLVER] = 1.0
        report.stage_seconds[self.wrapper.STAGE_IO] = 0.5

        with mock.patch.object(self.wrapper.time, "perf_counter", return_value=15.0):
            report.finish()

        self.assertAlmostEqual(report.wall_seconds, 10.0)
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_PYTHON],
            6.5,
        )

    def test_finish_excludes_explicit_loop_stages_from_python_overhead(self):
        report = self.wrapper.ProfilingReport(label="finish-loop")
        report.wall_start = 5.0
        report.stage_seconds[self.wrapper.STAGE_ACTIVATION] = 1.0
        report.stage_seconds[self.wrapper.STAGE_QUAD_STATE] = 2.0
        report.stage_seconds[self.wrapper.STAGE_MATERIAL] = 1.5
        report.stage_seconds[self.wrapper.STAGE_HISTORY] = 0.5
        report.stage_seconds[self.wrapper.STAGE_POSTPROCESS] = 1.0

        with mock.patch.object(self.wrapper.time, "perf_counter", return_value=15.0):
            report.finish()

        self.assertAlmostEqual(report.wall_seconds, 10.0)
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_PYTHON],
            4.0,
        )

    def test_finish_does_not_double_count_derived_assembly(self):
        report = self.wrapper.ProfilingReport(label="finish-assembly")
        report.wall_start = 5.0
        report.stage_seconds[self.wrapper.STAGE_LOCAL_ASSEMBLY] = 2.0
        report.stage_seconds[self.wrapper.STAGE_GLOBAL_MATRIX] = 1.0
        report.stage_seconds[self.wrapper.STAGE_ASSEMBLY] = 3.0
        report.stage_seconds[self.wrapper.STAGE_SOLVER] = 1.0

        with mock.patch.object(self.wrapper.time, "perf_counter", return_value=15.0):
            report.finish()

        self.assertAlmostEqual(report.wall_seconds, 10.0)
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_PYTHON],
            6.0,
        )

    def test_finish_does_not_double_count_local_assembly_detail_stages(self):
        report = self.wrapper.ProfilingReport(label="finish-local-details")
        report.wall_start = 5.0
        report.stage_seconds[self.wrapper.STAGE_LOCAL_ASSEMBLY] = 5.0
        report.stage_seconds[self.wrapper.STAGE_GLOBAL_MATRIX] = 1.0
        report.stage_seconds[self.wrapper.STAGE_ASSEMBLY] = 6.0
        report.stage_seconds[self.wrapper.STAGE_CELL_JACOBIAN] = 4.0
        report.stage_seconds[self.wrapper.STAGE_FACE_JACOBIAN] = 0.5
        report.stage_seconds[self.wrapper.STAGE_RESIDUAL_SCATTER] = 0.25

        with mock.patch.object(self.wrapper.time, "perf_counter", return_value=15.0):
            report.finish()

        self.assertAlmostEqual(report.wall_seconds, 10.0)
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_PYTHON],
            4.0,
        )

    def test_finish_does_not_double_count_residual_vector_detail_stages(self):
        report = self.wrapper.ProfilingReport(label="finish-residual-details")
        report.wall_start = 5.0
        report.stage_seconds[self.wrapper.STAGE_RESIDUAL_VECTOR] = 2.0
        report.stage_seconds[self.wrapper.STAGE_RESIDUAL_FLATTEN] = 0.25
        report.stage_seconds[self.wrapper.STAGE_RESIDUAL_BC] = 1.5
        report.stage_seconds[self.wrapper.STAGE_RESIDUAL_PROJECTION] = 0.25
        report.stage_seconds[self.wrapper.STAGE_SOLVER] = 1.0

        with mock.patch.object(self.wrapper.time, "perf_counter", return_value=15.0):
            report.finish()

        self.assertAlmostEqual(report.wall_seconds, 10.0)
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_PYTHON],
            7.0,
        )

    def test_explain_gpu_vs_cpu_does_not_claim_speedup_after_fallback(self):
        cpu = self.wrapper.ProfilingReport(label="spsolve")
        cpu.wall_seconds = 10.0
        gpu = self.wrapper.ProfilingReport(label="jax")
        gpu.wall_seconds = 1.0
        gpu.meta["solver_fallbacks"] = 1
        gpu.meta["last_solver_fallback"] = "TypeError: failed"

        verdict = self.wrapper.explain_gpu_vs_cpu(gpu, cpu)

        self.assertIn("fell back to spsolve", verdict)
        self.assertNotIn("faster", verdict)

    def test_explain_gpu_vs_cpu_flags_warm_cache_when_linear_solve_loses(self):
        cpu = self.wrapper.ProfilingReport(label="spsolve")
        cpu.wall_seconds = 10.0
        cpu.stage_seconds[self.wrapper.STAGE_SOLVER] = 0.01
        gpu = self.wrapper.ProfilingReport(label="jax")
        gpu.wall_seconds = 1.0
        gpu.stage_seconds[self.wrapper.STAGE_SOLVER] = 1.0

        verdict = self.wrapper.explain_gpu_vs_cpu(gpu, cpu)

        self.assertIn("overall wall time is lower", verdict)
        self.assertIn("linear solve is slower", verdict)
        self.assertNotIn("GPU path is 10.00x faster", verdict)

    def test_explain_gpu_vs_cpu_describes_jax_sparse_direct_solver(self):
        cpu = self.wrapper.ProfilingReport(label="spsolve")
        cpu.wall_seconds = 1.0
        cpu.stage_seconds[self.wrapper.STAGE_SOLVER] = 0.01
        gpu = self.wrapper.ProfilingReport(label="jax-spsolve")
        gpu.wall_seconds = 1.1
        gpu.stage_seconds[self.wrapper.STAGE_SOLVER] = 0.02
        gpu.meta["linear_solver_label"] = (
            "jax_solver(method=spsolve, precond=False, check_residual=False)"
        )
        gpu.meta["jax_spsolve_calls"] = 16

        verdict = self.wrapper.explain_gpu_vs_cpu(gpu, cpu)

        self.assertIn("sparse direct", verdict)
        self.assertNotIn("iterative solve", verdict)

    def test_timing_patch_maps_jax_internal_breakdown_without_double_counting(self):
        timing = {
            "local_assembly": 0.0,
            "global_matrix": 0.0,
            "linear": 0.0,
            "_last_linear_internal_breakdown": False,
        }

        def original_timing_record(parts, name, dt):
            parts[name] = parts.get(name, 0.0) + dt

        fake_solver_module = SimpleNamespace(_timing_record=original_timing_record)
        report = self.wrapper.ProfilingReport(label="timing")

        with mock.patch.dict(sys.modules, {"jax_fem.solver": fake_solver_module}):
            self.wrapper.install_jax_fem_timing_patch(report)
            fake_solver_module._timing_record(timing, "local_assembly", 0.4)
            fake_solver_module._timing_record(timing, "residual_vector", 0.06)
            fake_solver_module._timing_record(timing, "residual_flatten", 0.01)
            fake_solver_module._timing_record(timing, "residual_bc", 0.04)
            fake_solver_module._timing_record(timing, "residual_projection", 0.01)
            fake_solver_module._timing_record(timing, "global_matrix", 0.1)
            fake_solver_module._timing_record(timing, "bc_initial_guess", 0.05)
            fake_solver_module._timing_record(timing, "sparse_conversion", 0.2)
            timing["_last_linear_internal_breakdown"] = True
            fake_solver_module._timing_record(timing, "linear_kernel", 0.3)
            fake_solver_module._timing_record(timing, "linear", 0.7)

        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_LOCAL_ASSEMBLY],
            0.4,
        )
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_GLOBAL_MATRIX],
            0.1,
        )
        self.assertAlmostEqual(report.stage_seconds[self.wrapper.STAGE_ASSEMBLY], 0.5)
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_CONVERSION],
            0.2,
        )
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_BC_INITIAL_GUESS],
            0.05,
        )
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_RESIDUAL_VECTOR],
            0.06,
        )
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_RESIDUAL_FLATTEN],
            0.01,
        )
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_RESIDUAL_BC],
            0.04,
        )
        self.assertAlmostEqual(
            report.stage_seconds[self.wrapper.STAGE_RESIDUAL_PROJECTION],
            0.01,
        )
        self.assertAlmostEqual(report.stage_seconds[self.wrapper.STAGE_SOLVER], 0.3)
        self.assertAlmostEqual(timing["linear"], 0.7)

    def test_timing_patch_records_newton_iterations_from_log_table(self):
        calls = []

        def original_timing_record(parts, name, dt):
            parts[name] = parts.get(name, 0.0) + dt

        def original_log_timing_table(n_iters, parts, wall_s):
            calls.append((n_iters, parts, wall_s))

        fake_solver_module = SimpleNamespace(
            _timing_record=original_timing_record,
            _log_timing_table=original_log_timing_table,
        )
        report = self.wrapper.ProfilingReport(label="timing")

        with mock.patch.dict(sys.modules, {"jax_fem.solver": fake_solver_module}):
            self.wrapper.install_jax_fem_timing_patch(report)
            fake_solver_module._log_timing_table(2, {"linear": 1.0}, 3.5)
            fake_solver_module._log_timing_table(0, {}, 0.5)

        self.assertEqual(calls, [(2, {"linear": 1.0}, 3.5), (0, {}, 0.5)])
        self.assertEqual(report.linear_iterations, 2)
        self.assertEqual(report.meta["newton_solve_calls"], 2)
        self.assertEqual(report.meta["newton_zero_iter_solves"], 1)
        self.assertEqual(report.meta["last_newton_iterations"], 0)
        self.assertAlmostEqual(report.meta["newton_wall_seconds"], 4.0)

    def test_problem_local_assembly_timing_patch_records_detail_stages(self):
        calls = []

        class FakeProblem:
            def split_and_compute_cell(self, cells_sol_flat, np_version, jac_flag, internal_vars):
                calls.append(("cell", jac_flag, cells_sol_flat, np_version, internal_vars))
                return "cell"

            def compute_face(self, cells_sol_flat, np_version, jac_flag, internal_vars_surfaces):
                calls.append(("face", jac_flag, cells_sol_flat, np_version, internal_vars_surfaces))
                return "face"

            def compute_residual_vars_helper(self, weak_form_flat, weak_form_face_flat):
                calls.append(("scatter", weak_form_flat, weak_form_face_flat))
                return "scatter"

        fake_problem_module = SimpleNamespace(Problem=FakeProblem)
        report = self.wrapper.ProfilingReport(label="problem-timing")

        with mock.patch.dict(sys.modules, {"jax_fem.problem": fake_problem_module}):
            installed = self.wrapper.install_problem_local_assembly_timing_patch(report)
            problem = FakeProblem()
            self.assertEqual(
                problem.split_and_compute_cell("cells", "np", True, "vars"),
                "cell",
            )
            self.assertEqual(
                problem.split_and_compute_cell("cells", "np", False, "vars"),
                "cell",
            )
            self.assertEqual(
                problem.compute_face("cells", "np", True, "surface-vars"),
                "face",
            )
            self.assertEqual(
                problem.compute_face("cells", "np", False, "surface-vars"),
                "face",
            )
            self.assertEqual(
                problem.compute_residual_vars_helper("weak", "face-weak"),
                "scatter",
            )

        self.assertTrue(installed)
        self.assertEqual(
            calls,
            [
                ("cell", True, "cells", "np", "vars"),
                ("cell", False, "cells", "np", "vars"),
                ("face", True, "cells", "np", "surface-vars"),
                ("face", False, "cells", "np", "surface-vars"),
                ("scatter", "weak", "face-weak"),
            ],
        )
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_CELL_JACOBIAN], 1)
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_CELL_RESIDUAL], 1)
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_FACE_JACOBIAN], 1)
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_FACE_RESIDUAL], 1)
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_RESIDUAL_SCATTER], 1)
        self.assertEqual(
            report.meta["problem_timing_patch"],
            "jax_fem.problem.Problem local assembly methods",
        )

    def test_finite_element_dof_to_quad_timing_patch_records_stage(self):
        calls = []

        class FakeFiniteElement:
            def convert_from_dof_to_quad(self, sol):
                calls.append((self, sol))
                return f"quad:{sol}"

        fake_fe_module = SimpleNamespace(FiniteElement=FakeFiniteElement)
        report = self.wrapper.ProfilingReport(label="fe-timing")

        with mock.patch.dict(sys.modules, {"jax_fem.fe": fake_fe_module}):
            installed = self.wrapper.install_finite_element_timing_patch(report)
            fe = FakeFiniteElement()
            result = fe.convert_from_dof_to_quad("T")

        self.assertTrue(installed)
        self.assertEqual(result, "quad:T")
        self.assertEqual(calls, [(fe, "T")])
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_DOF_TO_QUAD], 1)
        self.assertEqual(
            report.meta["finite_element_timing_patch"],
            "jax_fem.fe.FiniteElement convert_from_dof_to_quad",
        )

    def test_finite_element_dof_to_quad_identity_cache_reuses_jax_array(self):
        calls = []

        class FakeJaxArray:
            __module__ = "jaxlib._jax"
            shape = (4, 1)
            dtype = "float64"

        class FakeFiniteElement:
            def __init__(self):
                self.cells = SimpleNamespace(shape=(2, 4))
                self.shape_vals = SimpleNamespace(shape=(1, 4))

            def convert_from_dof_to_quad(self, sol):
                calls.append((self, sol))
                return {"call": len(calls)}

        fake_fe_module = SimpleNamespace(FiniteElement=FakeFiniteElement)
        report = self.wrapper.ProfilingReport(label="fe-cache")

        with mock.patch.dict(sys.modules, {"jax_fem.fe": fake_fe_module}):
            installed = self.wrapper.install_finite_element_timing_patch(report)
            fe = FakeFiniteElement()
            sol = FakeJaxArray()
            first = fe.convert_from_dof_to_quad(sol)
            second = fe.convert_from_dof_to_quad(sol)

        self.assertTrue(installed)
        self.assertIs(second, first)
        self.assertEqual(calls, [(fe, sol)])
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_DOF_TO_QUAD], 1)
        self.assertEqual(report.meta["dof_to_quad_cache_hits"], 1)
        self.assertEqual(report.meta["dof_to_quad_cache_misses"], 1)
        self.assertEqual(report.meta["dof_to_quad_cache_entries"], 1)

    def test_finite_element_dof_to_quad_identity_cache_can_be_disabled(self):
        calls = []

        class FakeJaxArray:
            __module__ = "jaxlib._jax"
            shape = (4, 1)
            dtype = "float64"

        class FakeFiniteElement:
            def __init__(self):
                self.cells = SimpleNamespace(shape=(2, 4))
                self.shape_vals = SimpleNamespace(shape=(1, 4))

            def convert_from_dof_to_quad(self, sol):
                calls.append((self, sol))
                return {"call": len(calls)}

        fake_fe_module = SimpleNamespace(FiniteElement=FakeFiniteElement)
        report = self.wrapper.ProfilingReport(label="fe-cache-disabled")

        with mock.patch.dict(sys.modules, {"jax_fem.fe": fake_fe_module}):
            installed = self.wrapper.install_finite_element_timing_patch(
                report,
                cache_enabled=False,
            )
            fe = FakeFiniteElement()
            sol = FakeJaxArray()
            first = fe.convert_from_dof_to_quad(sol)
            second = fe.convert_from_dof_to_quad(sol)

        self.assertTrue(installed)
        self.assertIsNot(second, first)
        self.assertEqual(calls, [(fe, sol), (fe, sol)])
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_DOF_TO_QUAD], 2)
        self.assertFalse(report.meta["dof_to_quad_cache_enabled"])

    def test_dof_to_quad_identity_cache_does_not_cache_tracer_like_values(self):
        calls = []

        class FakeTracer:
            __module__ = "jax._src.interpreters.partial_eval"
            shape = (4, 1)
            dtype = "float64"

        class FakeFiniteElement:
            def __init__(self):
                self.cells = SimpleNamespace(shape=(2, 4))
                self.shape_vals = SimpleNamespace(shape=(1, 4))

            def convert_from_dof_to_quad(self, sol):
                calls.append((self, sol))
                return {"call": len(calls)}

        fake_fe_module = SimpleNamespace(FiniteElement=FakeFiniteElement)
        report = self.wrapper.ProfilingReport(label="fe-cache-tracer")

        with mock.patch.dict(sys.modules, {"jax_fem.fe": fake_fe_module}):
            installed = self.wrapper.install_finite_element_timing_patch(report)
            fe = FakeFiniteElement()
            sol = FakeTracer()
            first = fe.convert_from_dof_to_quad(sol)
            second = fe.convert_from_dof_to_quad(sol)

        self.assertTrue(installed)
        self.assertIsNot(second, first)
        self.assertEqual(calls, [(fe, sol), (fe, sol)])
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_DOF_TO_QUAD], 2)
        self.assertEqual(report.meta["dof_to_quad_cache_skipped_non_jax"], 2)

    def test_configure_problem_cell_assembly_num_cuts_sets_problem_class(self):
        class FakeProblem:
            cell_assembly_num_cuts = 20
            cell_assembly_target_batch_size = 2048

        fake_problem_module = SimpleNamespace(Problem=FakeProblem)
        report = self.wrapper.ProfilingReport(label="cuts")

        with mock.patch.dict(sys.modules, {"jax_fem.problem": fake_problem_module}):
            installed = self.wrapper.configure_problem_cell_assembly_num_cuts(
                3,
                report,
                target_batch_size=self.wrapper.DEFAULT_CELL_TARGET_BATCH_SIZE,
            )

        self.assertTrue(installed)
        self.assertEqual(FakeProblem.cell_assembly_num_cuts, 3)
        self.assertIsNone(FakeProblem.cell_assembly_target_batch_size)
        self.assertEqual(report.meta["cell_assembly_num_cuts"], 3)
        self.assertEqual(report.meta["cell_assembly_num_cuts_source"], "cli")
        self.assertIsNone(report.meta["cell_assembly_target_batch_size"])
        self.assertEqual(report.meta["cell_assembly_chunking"], "fixed_num_cuts")

    def test_configure_problem_cell_assembly_num_cuts_resets_default_auto(self):
        class FakeProblem:
            cell_assembly_num_cuts = 1
            cell_assembly_target_batch_size = None

        fake_problem_module = SimpleNamespace(Problem=FakeProblem)
        report = self.wrapper.ProfilingReport(label="cuts")

        with mock.patch.dict(sys.modules, {"jax_fem.problem": fake_problem_module}):
            installed = self.wrapper.configure_problem_cell_assembly_num_cuts(
                None,
                report,
                target_batch_size=self.wrapper.DEFAULT_CELL_TARGET_BATCH_SIZE,
            )

        self.assertTrue(installed)
        self.assertEqual(FakeProblem.cell_assembly_num_cuts, 20)
        self.assertEqual(
            FakeProblem.cell_assembly_target_batch_size,
            self.wrapper.DEFAULT_CELL_TARGET_BATCH_SIZE,
        )
        self.assertIsNone(report.meta["cell_assembly_num_cuts"])
        self.assertEqual(report.meta["cell_assembly_num_cuts_source"], "auto")
        self.assertEqual(
            report.meta["cell_assembly_target_batch_size"],
            self.wrapper.DEFAULT_CELL_TARGET_BATCH_SIZE,
        )
        self.assertEqual(
            report.meta["cell_assembly_chunking"],
            "auto_target_batch_size",
        )

    def test_timing_patch_records_jax_sparse_cache_counters_in_meta(self):
        def original_timing_record(parts, name, dt):
            parts[name] = parts.get(name, 0.0) + dt

        def original_counter_record(parts, name, count=1):
            parts[name] = int(parts.get(name, 0)) + int(count)

        fake_solver_module = SimpleNamespace(
            _timing_record=original_timing_record,
            _counter_record=original_counter_record,
        )
        report = self.wrapper.ProfilingReport(label="timing")
        timing = {}

        with mock.patch.dict(sys.modules, {"jax_fem.solver": fake_solver_module}):
            self.wrapper.install_jax_fem_timing_patch(report)
            fake_solver_module._counter_record(timing, "bcoo_cache_misses")
            fake_solver_module._counter_record(timing, "bcoo_cache_hits", 2)

        self.assertEqual(timing["bcoo_cache_misses"], 1)
        self.assertEqual(timing["bcoo_cache_hits"], 2)
        self.assertEqual(report.meta["jax_bcoo_cache_misses"], 1)
        self.assertEqual(report.meta["jax_bcoo_cache_hits"], 2)

    def test_profiling_patch_records_cached_activation_call_overhead(self):
        calls = []

        def compute_layer_on_scan_cells(
            highest_printed_layer,
            physical_layer_id_cell,
            substrate_cell,
            support_cell,
            args,
        ):
            calls.append(highest_printed_layer)
            return (object(), object(), object())

        fake_base = SimpleNamespace(
            compute_layer_on_scan_cells=compute_layer_on_scan_cells
        )
        report = self.wrapper.ProfilingReport(label="activation-profile")
        args = SimpleNamespace(active_window_below_layers=2, layers=5)
        physical = [1, 2, 3]
        substrate = [False, False, False]
        support = [False, False, False]

        self.wrapper.install_activation_cache_patch(fake_base, report)
        self.wrapper.install_profiling_patches(fake_base, report)
        first = fake_base.compute_layer_on_scan_cells(
            3, physical, substrate, support, args
        )
        second = fake_base.compute_layer_on_scan_cells(
            3, physical, substrate, support, args
        )

        self.assertIs(first, second)
        self.assertEqual(calls, [3])
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_ACTIVATION], 2)
        self.assertEqual(report.meta["activation_cache_misses"], 1)
        self.assertEqual(report.meta["activation_cache_hits"], 1)

    def test_profiling_patch_records_loop_hot_function_stages(self):
        fake_base = SimpleNamespace(
            make_quad_scalar=lambda *args, **kwargs: "quad",
            thermal_material_quads=lambda *args, **kwargs: "thermal",
            mechanics_material_quads=lambda *args, **kwargs: "mechanics",
            update_phase_reference_and_eqp=lambda *args, **kwargs: "history",
            compute_cell_temperature=lambda *args, **kwargs: "temperature",
            material_cell_state=lambda *args, **kwargs: "state",
        )
        report = self.wrapper.ProfilingReport(label="loop-stages")

        self.wrapper.install_profiling_patches(fake_base, report)

        self.assertEqual(fake_base.make_quad_scalar(), "quad")
        self.assertEqual(fake_base.thermal_material_quads(), "thermal")
        self.assertEqual(fake_base.mechanics_material_quads(), "mechanics")
        self.assertEqual(fake_base.update_phase_reference_and_eqp(), "history")
        self.assertEqual(fake_base.compute_cell_temperature(), "temperature")
        self.assertEqual(fake_base.material_cell_state(), "state")

        self.assertEqual(report.stage_calls[self.wrapper.STAGE_QUAD_STATE], 1)
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_MATERIAL], 2)
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_HISTORY], 1)
        self.assertEqual(report.stage_calls[self.wrapper.STAGE_POSTPROCESS], 2)

    def test_step_predicate_cache_reuses_generated_step_predicates(self):
        args = SimpleNamespace(
            mechanics_every=2,
            mechanics_output_every=2,
            thermal_output_every=3,
        )
        states = [
            SimpleNamespace(global_step=0, mode="scan", laser_switch=1.0),
            SimpleNamespace(global_step=1, mode="hatch_dwell", laser_switch=0.0),
            SimpleNamespace(global_step=2, mode="scan", laser_switch=1.0),
        ]
        calls = {"activate": [], "mechanics": [], "save": []}

        def generate_raster_step_states(*_args, **_kwargs):
            return states, 1.0, 1.0

        def should_activate_layer_for_state(state):
            calls["activate"].append(state.global_step)
            return state.mode == "scan" and state.laser_switch > 0.5

        def should_run_mechanics(global_step, _args):
            calls["mechanics"].append(global_step)
            return _args.mechanics_every > 0 and global_step % _args.mechanics_every == 0

        def should_save_step(global_step, did_mechanics, is_last, _args):
            calls["save"].append(global_step)
            return (
                is_last
                or (
                    did_mechanics
                    and _args.mechanics_output_every > 0
                    and global_step % _args.mechanics_output_every == 0
                )
                or (
                    _args.thermal_output_every > 0
                    and global_step % _args.thermal_output_every == 0
                )
            )

        fake_base = SimpleNamespace(
            generate_raster_step_states=generate_raster_step_states,
            should_activate_layer_for_state=should_activate_layer_for_state,
            should_run_mechanics=should_run_mechanics,
            should_save_step=should_save_step,
        )
        report = self.wrapper.ProfilingReport(label="step-predicate-cache")

        installed = self.wrapper.install_step_predicate_cache_patch(
            fake_base,
            report,
            enabled=True,
        )
        result_states, _, _ = fake_base.generate_raster_step_states(args, None)
        calls_after_build = {key: list(value) for key, value in calls.items()}

        self.assertTrue(installed)
        self.assertIs(result_states, states)
        self.assertTrue(fake_base.should_activate_layer_for_state(states[0]))
        self.assertFalse(fake_base.should_activate_layer_for_state(states[1]))
        self.assertTrue(fake_base.should_run_mechanics(2, args))
        self.assertTrue(fake_base.should_save_step(2, True, False, args))
        self.assertTrue(fake_base.should_save_step(2, False, True, args))
        self.assertEqual(calls, calls_after_build)
        self.assertEqual(report.meta["step_predicate_cache_entries"], 3)
        self.assertGreaterEqual(report.meta["step_predicate_cache_hits"], 5)
        self.assertEqual(report.meta.get("step_predicate_cache_misses", 0), 0)

    def test_step_predicate_cache_preserves_setup_path_generation_timing(self):
        args = SimpleNamespace(
            mechanics_every=2,
            mechanics_output_every=2,
            thermal_output_every=3,
        )
        states = [SimpleNamespace(global_step=0, mode="scan", laser_switch=1.0)]

        def generate_raster_step_states(*_args, **_kwargs):
            return states, 1.0, 2.0

        fake_base = SimpleNamespace(
            generate_raster_step_states=generate_raster_step_states,
            should_activate_layer_for_state=lambda state: True,
            should_run_mechanics=lambda global_step, args: True,
            should_save_step=lambda global_step, did_mechanics, is_last, args: False,
        )
        report = self.wrapper.ProfilingReport(label="setup-step-cache")

        self.wrapper.install_setup_detail_timing_patch(fake_base, report)
        self.wrapper.install_step_predicate_cache_patch(fake_base, report)

        with mock.patch.object(
            self.wrapper.time,
            "perf_counter",
            side_effect=[10.0, 10.25],
        ):
            result = fake_base.generate_raster_step_states(args)

        self.assertEqual(result, (states, 1.0, 2.0))
        self.assertAlmostEqual(
            report.meta["setup_detail_path_generation_seconds"], 0.25
        )
        self.assertEqual(report.meta["setup_detail_path_generation_calls"], 1)
        self.assertEqual(report.meta["step_predicate_cache_entries"], 1)

    def test_step_predicate_cache_falls_back_for_different_args_object(self):
        args = SimpleNamespace(
            mechanics_every=2,
            mechanics_output_every=2,
            thermal_output_every=0,
        )
        stale_args = SimpleNamespace(
            mechanics_every=3,
            mechanics_output_every=3,
            thermal_output_every=0,
        )
        states = [SimpleNamespace(global_step=3, mode="scan", laser_switch=1.0)]
        calls = {"mechanics": []}

        def generate_raster_step_states(*_args, **_kwargs):
            return states, 1.0, 1.0

        def should_activate_layer_for_state(state):
            return True

        def should_run_mechanics(global_step, _args):
            calls["mechanics"].append((global_step, _args.mechanics_every))
            return _args.mechanics_every > 0 and global_step % _args.mechanics_every == 0

        def should_save_step(global_step, did_mechanics, is_last, _args):
            return is_last or did_mechanics

        fake_base = SimpleNamespace(
            generate_raster_step_states=generate_raster_step_states,
            should_activate_layer_for_state=should_activate_layer_for_state,
            should_run_mechanics=should_run_mechanics,
            should_save_step=should_save_step,
        )
        report = self.wrapper.ProfilingReport(label="step-predicate-cache")

        self.wrapper.install_step_predicate_cache_patch(
            fake_base,
            report,
            enabled=True,
        )
        fake_base.generate_raster_step_states(args, None)

        self.assertTrue(fake_base.should_run_mechanics(3, stale_args))
        self.assertIn((3, 3), calls["mechanics"])
        self.assertEqual(report.meta["step_predicate_cache_misses"], 1)

    def test_lazy_output_postprocess_skips_material_state_until_save_step(self):
        args = SimpleNamespace(
            mechanics_every=0,
            mechanics_output_every=0,
            thermal_output_every=0,
        )
        states = [
            SimpleNamespace(global_step=0, mode="scan", laser_switch=1.0),
            SimpleNamespace(global_step=1, mode="scan", laser_switch=1.0),
            SimpleNamespace(global_step=2, mode="scan", laser_switch=1.0),
        ]
        calls = {"phase": [], "material": []}

        def generate_raster_step_states(*_args, **_kwargs):
            return states, 1.0, 1.0

        def phase_cell_from_quad(phase_quad):
            calls["phase"].append(phase_quad)
            return f"phase-{phase_quad}"

        def material_cell_state(
            active_cell,
            substrate_cell,
            support_cell,
            _args,
            cell_temperature=None,
            phase_cell=None,
        ):
            calls["material"].append((active_cell, phase_cell))
            return f"state-{active_cell}-{phase_cell}"

        fake_base = SimpleNamespace(
            generate_raster_step_states=generate_raster_step_states,
            should_activate_layer_for_state=lambda state: True,
            should_run_mechanics=lambda global_step, _args: False,
            should_save_step=lambda global_step, did_mechanics, is_last, _args: is_last,
            phase_cell_from_quad=phase_cell_from_quad,
            material_cell_state=material_cell_state,
        )
        report = self.wrapper.ProfilingReport(label="lazy-postprocess")

        self.wrapper.install_step_predicate_cache_patch(fake_base, report)
        installed = self.wrapper.install_lazy_output_postprocess_patch(
            fake_base,
            report,
            enabled=True,
        )
        fake_base.generate_raster_step_states(args)

        initial = fake_base.material_cell_state(
            "initial-active",
            "substrate",
            "support",
            args,
            phase_cell=fake_base.phase_cell_from_quad("initial-phase"),
        )
        fake_base.should_run_mechanics(1, args)
        skipped = fake_base.material_cell_state(
            "step1-active",
            "substrate",
            "support",
            args,
            phase_cell=fake_base.phase_cell_from_quad("step1-phase"),
        )
        fake_base.should_run_mechanics(2, args)
        saved = fake_base.material_cell_state(
            "step2-active",
            "substrate",
            "support",
            args,
            phase_cell=fake_base.phase_cell_from_quad("step2-phase"),
        )

        self.assertTrue(installed)
        self.assertEqual(initial, "state-initial-active-phase-initial-phase")
        self.assertIs(skipped, initial)
        self.assertEqual(saved, "state-step2-active-phase-step2-phase")
        self.assertEqual(calls["phase"], ["initial-phase", "step2-phase"])
        self.assertEqual(
            calls["material"],
            [
                ("initial-active", "phase-initial-phase"),
                ("step2-active", "phase-step2-phase"),
            ],
        )
        self.assertEqual(report.meta["lazy_output_postprocess_skips"], 1)
        self.assertEqual(report.meta["lazy_output_postprocess_computes"], 2)

    def test_quad_scalar_fast_path_reshapes_single_quad_float_without_original(self):
        try:
            import numpy as onp
        except ImportError as exc:  # pragma: no cover - test environment issue
            self.skipTest(str(exc))

        def original(_cell_values, _num_quads):
            raise AssertionError("single-quad float path should not call original")

        fake_base = SimpleNamespace(np=onp, make_quad_scalar=original)
        report = self.wrapper.ProfilingReport(label="quad-fast")

        installed = self.wrapper.install_quad_scalar_fast_path_patch(fake_base, report)
        cell_values = onp.asarray([1.0, 2.0])
        result = fake_base.make_quad_scalar(cell_values, 1)

        self.assertTrue(installed)
        onp.testing.assert_array_equal(result, onp.asarray([[[1.0]], [[2.0]]]))
        result[0, 0, 0] = 99.0
        self.assertEqual(cell_values[0], 1.0)
        self.assertEqual(report.meta["quad_scalar_fast_path_calls"], 1)

    def test_quad_scalar_fast_path_falls_back_for_multi_quad_and_handles_int(self):
        try:
            import numpy as onp
        except ImportError as exc:  # pragma: no cover - test environment issue
            self.skipTest(str(exc))

        calls = []

        def original(cell_values, num_quads):
            calls.append(num_quads)
            arr = onp.asarray(cell_values)[:, None, None]
            return arr * onp.ones((len(cell_values), num_quads, 1))

        fake_base = SimpleNamespace(np=onp, make_quad_scalar=original)
        report = self.wrapper.ProfilingReport(label="quad-fast")

        installed = self.wrapper.install_quad_scalar_fast_path_patch(fake_base, report)
        multi = fake_base.make_quad_scalar(onp.asarray([1.0, 2.0]), 2)
        ints = fake_base.make_quad_scalar(onp.asarray([1, 2]), 1)

        self.assertTrue(installed)
        self.assertEqual(calls, [2])
        onp.testing.assert_array_equal(multi, original(onp.asarray([1.0, 2.0]), 2))
        onp.testing.assert_array_equal(ints, original(onp.asarray([1, 2]), 1))
        self.assertEqual(report.meta["quad_scalar_fast_path_calls"], 1)
        self.assertEqual(report.meta["quad_scalar_fast_path_fallbacks"], 1)

    def test_activation_cache_reuses_layer_on_scan_masks_for_same_layer(self):
        calls = []

        def compute_layer_on_scan_cells(
            highest_printed_layer,
            physical_layer_id_cell,
            substrate_cell,
            support_cell,
            args,
        ):
            calls.append(highest_printed_layer)
            return (object(), object(), object())

        fake_base = SimpleNamespace(
            compute_layer_on_scan_cells=compute_layer_on_scan_cells
        )
        report = self.wrapper.ProfilingReport(label="activation-cache")
        args = SimpleNamespace(active_window_below_layers=2, layers=5)
        physical = [1, 2, 3]
        substrate = [False, False, False]
        support = [False, False, False]

        self.wrapper.install_activation_cache_patch(fake_base, report)
        first = fake_base.compute_layer_on_scan_cells(
            3, physical, substrate, support, args
        )
        second = fake_base.compute_layer_on_scan_cells(
            3, physical, substrate, support, args
        )
        third = fake_base.compute_layer_on_scan_cells(
            4, physical, substrate, support, args
        )

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(calls, [3, 4])
        self.assertEqual(report.meta["activation_cache_hits"], 1)
        self.assertEqual(report.meta["activation_cache_misses"], 2)
        self.assertEqual(report.meta["activation_cache_entries"], 2)

    def test_activation_cache_keys_include_window_and_intersection_thickness(self):
        calls = []

        def compute_intersection(
            highest_printed_layer,
            cell_d_min,
            cell_d_max,
            substrate_cell,
            support_cell,
            args,
        ):
            calls.append(
                (highest_printed_layer, args.active_window_below_layers,
                 args.layer_thickness)
            )
            return (object(), object(), object())

        fake_base = SimpleNamespace(
            compute_layer_on_scan_cells_by_intersection=compute_intersection
        )
        report = self.wrapper.ProfilingReport(label="activation-cache")
        args_a = SimpleNamespace(
            active_window_below_layers=2,
            layers=5,
            layer_thickness=1.0e-3,
        )
        args_b = SimpleNamespace(
            active_window_below_layers=3,
            layers=5,
            layer_thickness=1.0e-3,
        )
        d_min = [0.0, 0.5]
        d_max = [0.2, 0.7]
        substrate = [False, False]
        support = [False, False]

        self.wrapper.install_activation_cache_patch(fake_base, report)
        first = fake_base.compute_layer_on_scan_cells_by_intersection(
            2, d_min, d_max, substrate, support, args_a
        )
        second = fake_base.compute_layer_on_scan_cells_by_intersection(
            2, d_min, d_max, substrate, support, args_a
        )
        third = fake_base.compute_layer_on_scan_cells_by_intersection(
            2, d_min, d_max, substrate, support, args_b
        )

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(
            calls,
            [(2, 2, 1.0e-3), (2, 3, 1.0e-3)],
        )
        self.assertEqual(report.meta["activation_cache_hits"], 1)
        self.assertEqual(report.meta["activation_cache_misses"], 2)

    def test_activation_cache_reuses_moving_window_masks_by_state_layer(self):
        calls = []

        def compute_moving_window_cells(
            state,
            physical_layer_id_cell,
            substrate_cell,
            support_cell,
            args,
        ):
            calls.append(state.layer_idx)
            return (object(), object(), object())

        fake_base = SimpleNamespace(
            compute_moving_window_cells=compute_moving_window_cells
        )
        report = self.wrapper.ProfilingReport(label="activation-cache")
        args = SimpleNamespace(active_window_below_layers=4, layers=8)
        physical = [1, 2, 3]
        substrate = [False, False, False]
        support = [False, False, False]

        self.wrapper.install_activation_cache_patch(fake_base, report)
        first = fake_base.compute_moving_window_cells(
            SimpleNamespace(layer_idx=1), physical, substrate, support, args
        )
        second = fake_base.compute_moving_window_cells(
            SimpleNamespace(layer_idx=1), physical, substrate, support, args
        )
        third = fake_base.compute_moving_window_cells(
            SimpleNamespace(layer_idx=2), physical, substrate, support, args
        )

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(report.meta["activation_cache_hits"], 1)
        self.assertEqual(report.meta["activation_cache_misses"], 2)

    def test_linear_options_from_args_forwards_jax_solver_controls(self):
        args = SimpleNamespace(
            xla_linear_solver="jax",
            xla_jax_precond=True,
            xla_jax_method="gmres",
            xla_jax_tol=1e-7,
            xla_jax_atol=1e-8,
            xla_jax_maxiter=50,
            xla_jax_gmres_restart=10,
            xla_jax_gmres_solve_method="incremental",
        )

        options = self.wrapper.linear_options_from_args(args)

        self.assertEqual(
            options,
            {
                "jax_solver": {
                    "precond": True,
                    "method": "gmres",
                    "tol": 1e-7,
                    "atol": 1e-8,
                    "maxiter": 50,
                    "restart": 10,
                    "solve_method": "incremental",
                }
            },
        )


class MacroMech100V04BenchTest(unittest.TestCase):
    def test_benchmark_loads_current_v04_wrapper(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench")

        self.assertEqual(bench.WRAPPER, XLA_WRAPPER_PATH)

    def test_small_loop_tier_is_bounded_multi_step_without_mechanics(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_tiers")
        tier_args = bench.TIERS["small-loop"]
        values = dict(zip(tier_args[::2], tier_args[1::2]))

        self.assertGreater(int(values["--layers"]), 1)
        self.assertGreater(int(values["--scan-steps-per-layer"]), 1)
        self.assertLess(int(values["--max-cells"]), 500)
        self.assertEqual(values["--mechanics-every"], "0")
        self.assertEqual(values["--summary-every"], "999999")

    def test_benchmark_exposes_jax_sparse_direct_solver_candidate(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_spsolve")

        self.assertEqual(
            bench.SOLVER_FLAGS["jax-spsolve"],
            [
                "--xla-linear-solver",
                "jax",
                "--xla-jax-method",
                "spsolve",
                "--xla-jax-skip-residual-check",
            ],
        )

    def test_run_one_passes_profile_arguments_to_wrapper(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_args")
        captured = {}

        class FakeWrapper:
            ProfilingReport = self.wrapper_report_class()

            @staticmethod
            def main(argv):
                captured["argv"] = argv
                profile_path = Path(argv[argv.index("--profile-json") + 1])
                profile_path.write_text(
                    json.dumps(
                        {
                            "wall_seconds": 1.0,
                            "steps": 1,
                            "stage_seconds": {},
                        }
                    )
                )
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            report = bench.run_one(FakeWrapper, "tiny", "spsolve", Path(tmp))

        self.assertEqual(report["wall_seconds"], 1.0)
        self.assertIn("--xla-linear-solver", captured["argv"])
        self.assertIn("--profile-json", captured["argv"])
        self.assertNotIn("--xla-thermal-warm-start", captured["argv"])

    def test_run_one_can_enable_thermal_warm_start(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_warm_args")
        captured = {}

        class FakeWrapper:
            ProfilingReport = self.wrapper_report_class()

            @staticmethod
            def main(argv):
                captured["argv"] = argv
                profile_path = Path(argv[argv.index("--profile-json") + 1])
                profile_path.write_text(
                    json.dumps(
                        {
                            "wall_seconds": 1.0,
                            "steps": 1,
                            "stage_seconds": {},
                        }
                    )
                )
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            report = bench.run_one(
                FakeWrapper,
                "tiny",
                "spsolve",
                Path(tmp),
                thermal_warm_start=True,
            )

        self.assertEqual(report["wall_seconds"], 1.0)
        self.assertIn("--xla-thermal-warm-start", captured["argv"])

    def test_run_one_can_disable_step_predicate_cache(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_step_cache_args")
        captured = {}

        class FakeWrapper:
            ProfilingReport = self.wrapper_report_class()

            @staticmethod
            def main(argv):
                captured["argv"] = argv
                profile_path = Path(argv[argv.index("--profile-json") + 1])
                profile_path.write_text(
                    json.dumps(
                        {
                            "wall_seconds": 1.0,
                            "steps": 1,
                            "stage_seconds": {},
                        }
                    )
                )
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            report = bench.run_one(
                FakeWrapper,
                "tiny",
                "spsolve",
                Path(tmp),
                step_predicate_cache=False,
            )

        self.assertEqual(report["wall_seconds"], 1.0)
        self.assertIn("--no-xla-step-predicate-cache", captured["argv"])

    def test_run_one_can_disable_unused_mechanics_material_skip(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_mech_skip_args")
        captured = {}

        class FakeWrapper:
            ProfilingReport = self.wrapper_report_class()

            @staticmethod
            def main(argv):
                captured["argv"] = argv
                profile_path = Path(argv[argv.index("--profile-json") + 1])
                profile_path.write_text(
                    json.dumps(
                        {
                            "wall_seconds": 1.0,
                            "steps": 1,
                            "stage_seconds": {},
                        }
                    )
                )
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            report = bench.run_one(
                FakeWrapper,
                "tiny",
                "spsolve",
                Path(tmp),
                skip_unused_mechanics_material=False,
            )

        self.assertEqual(report["wall_seconds"], 1.0)
        self.assertIn(
            "--no-xla-skip-unused-mechanics-material",
            captured["argv"],
        )

    def test_run_one_can_disable_thermal_only_mechanics_surrogate(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_surrogate_args")
        captured = {}

        class FakeWrapper:
            ProfilingReport = self.wrapper_report_class()

            @staticmethod
            def main(argv):
                captured["argv"] = argv
                profile_path = Path(argv[argv.index("--profile-json") + 1])
                profile_path.write_text(
                    json.dumps(
                        {
                            "wall_seconds": 1.0,
                            "steps": 1,
                            "stage_seconds": {},
                        }
                    )
                )
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            report = bench.run_one(
                FakeWrapper,
                "tiny",
                "spsolve",
                Path(tmp),
                thermal_only_mechanics_surrogate=False,
            )

        self.assertEqual(report["wall_seconds"], 1.0)
        self.assertIn(
            "--no-xla-thermal-only-mechanics-surrogate",
            captured["argv"],
        )

    def test_run_one_can_disable_quiet_jax_fem_logs(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_log_args")
        captured = {}

        class FakeWrapper:
            ProfilingReport = self.wrapper_report_class()

            @staticmethod
            def main(argv):
                captured["argv"] = argv
                profile_path = Path(argv[argv.index("--profile-json") + 1])
                profile_path.write_text(
                    json.dumps(
                        {
                            "wall_seconds": 1.0,
                            "steps": 1,
                            "stage_seconds": {},
                        }
                    )
                )
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            report = bench.run_one(
                FakeWrapper,
                "tiny",
                "spsolve",
                Path(tmp),
                quiet_jax_fem_logs=False,
            )

        self.assertEqual(report["wall_seconds"], 1.0)
        self.assertIn("--no-xla-quiet-jax-fem-logs", captured["argv"])

    def test_run_one_can_enable_lazy_output_postprocess(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_postprocess_args")
        captured = {}

        class FakeWrapper:
            ProfilingReport = self.wrapper_report_class()

            @staticmethod
            def main(argv):
                captured["argv"] = argv
                profile_path = Path(argv[argv.index("--profile-json") + 1])
                profile_path.write_text(
                    json.dumps(
                        {
                            "wall_seconds": 1.0,
                            "steps": 1,
                            "stage_seconds": {},
                        }
                    )
                )
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            report = bench.run_one(
                FakeWrapper,
                "tiny",
                "spsolve",
                Path(tmp),
                lazy_output_postprocess=True,
            )

        self.assertEqual(report["wall_seconds"], 1.0)
        self.assertIn("--xla-lazy-output-postprocess", captured["argv"])

    def test_summarize_runs_discards_first_and_averages_samples(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_summary")
        runs = [
            {
                "label": "tiny/jax/r0",
                "wall_seconds": 10.0,
                "steps": 2,
                "linear_iterations": 1,
                "stage_seconds": {"solver": 8.0, "assembly": 1.0},
                "stage_calls": {"solver": 2, "assembly": 2},
                "meta": {"linear_solver_label": "cold"},
            },
            {
                "label": "tiny/jax/r1",
                "wall_seconds": 4.0,
                "steps": 2,
                "linear_iterations": 1,
                "stage_seconds": {"solver": 2.0, "assembly": 1.0},
                "stage_calls": {"solver": 2, "assembly": 2},
                "meta": {"linear_solver_label": "warm"},
            },
            {
                "label": "tiny/jax/r2",
                "wall_seconds": 2.0,
                "steps": 2,
                "linear_iterations": 1,
                "stage_seconds": {"solver": 1.0, "assembly": 1.0},
                "stage_calls": {"solver": 2, "assembly": 2},
                "meta": {"linear_solver_label": "warm"},
            },
        ]

        summary = bench.summarize_runs("tiny", "jax", runs, discard_first=1)

        self.assertEqual(summary["label"], "tiny/jax/mean")
        self.assertEqual(summary["wall_seconds"], 3.0)
        self.assertEqual(summary["stage_seconds"]["solver"], 1.5)
        self.assertEqual(summary["stage_seconds"]["assembly"], 1.0)
        self.assertEqual(summary["meta"]["benchmark_repeat"], 3)
        self.assertEqual(summary["meta"]["benchmark_discard_first"], 1)
        self.assertEqual(summary["meta"]["benchmark_samples"], 2)
        self.assertEqual(
            summary["meta"]["benchmark_sample_wall_seconds"],
            [4.0, 2.0],
        )
        self.assertEqual(len(summary["meta"]["benchmark_raw_runs"]), 3)

    def test_summarize_runs_averages_numeric_counter_meta(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_meta")
        runs = [
            {
                "label": "small/jax/r0",
                "wall_seconds": 10.0,
                "steps": 2,
                "linear_iterations": 1,
                "stage_seconds": {},
                "stage_calls": {},
                "meta": {
                    "activation_cache_misses": 2,
                    "activation_cache_hits": 14,
                    "step_predicate_cache_hits": 40,
                    "step_predicate_cache_misses": 0,
                    "loop_kernel_jit_mechanics_disabled_skips": 12,
                    "jax_bcoo_cache_misses": 1,
                    "jax_bcoo_cache_hits": 15,
                },
            },
            {
                "label": "small/jax/r1",
                "wall_seconds": 8.0,
                "steps": 2,
                "linear_iterations": 1,
                "stage_seconds": {},
                "stage_calls": {},
                "meta": {
                    "activation_cache_misses": 2,
                    "activation_cache_hits": 14,
                    "step_predicate_cache_hits": 48,
                    "step_predicate_cache_misses": 0,
                    "loop_kernel_jit_mechanics_disabled_skips": 16,
                    "jax_bcoo_cache_hits": 16,
                },
            },
        ]

        summary = bench.summarize_runs("small", "jax", runs, discard_first=1)

        self.assertEqual(summary["meta"]["activation_cache_misses"], 2)
        self.assertEqual(summary["meta"]["activation_cache_hits"], 14)
        self.assertEqual(summary["meta"]["step_predicate_cache_hits"], 48)
        self.assertEqual(summary["meta"]["step_predicate_cache_misses"], 0)
        self.assertEqual(
            summary["meta"]["loop_kernel_jit_mechanics_disabled_skips"],
            16,
        )
        self.assertEqual(summary["meta"]["jax_bcoo_cache_misses"], 0)
        self.assertEqual(summary["meta"]["jax_bcoo_cache_hits"], 16)

    def test_summarize_runs_averages_setup_detail_meta(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_setup_meta")
        runs = [
            {
                "label": "small/spsolve/r0",
                "wall_seconds": 10.0,
                "steps": 2,
                "linear_iterations": 1,
                "stage_seconds": {},
                "stage_calls": {},
                "meta": {
                    "setup_seconds_before_first_solve": 10.0,
                    "setup_detail_mesh_read_seconds": 1.0,
                    "setup_detail_mesh_read_calls": 1,
                    "setup_detail_total_seconds": 2.0,
                    "setup_unattributed_seconds": 8.0,
                },
            },
            {
                "label": "small/spsolve/r1",
                "wall_seconds": 6.0,
                "steps": 2,
                "linear_iterations": 1,
                "stage_seconds": {},
                "stage_calls": {},
                "meta": {
                    "setup_seconds_before_first_solve": 6.0,
                    "setup_detail_mesh_read_seconds": 0.5,
                    "setup_detail_mesh_read_calls": 1,
                    "setup_detail_total_seconds": 1.5,
                    "setup_unattributed_seconds": 4.5,
                },
            },
            {
                "label": "small/spsolve/r2",
                "wall_seconds": 4.0,
                "steps": 2,
                "linear_iterations": 1,
                "stage_seconds": {},
                "stage_calls": {},
                "meta": {
                    "setup_seconds_before_first_solve": 4.0,
                    "setup_detail_mesh_read_seconds": 0.25,
                    "setup_detail_mesh_read_calls": 1,
                    "setup_detail_total_seconds": 1.25,
                    "setup_unattributed_seconds": 2.75,
                },
            },
        ]

        summary = bench.summarize_runs("small", "spsolve", runs, discard_first=1)

        self.assertAlmostEqual(
            summary["meta"]["setup_seconds_before_first_solve"], 5.0
        )
        self.assertAlmostEqual(
            summary["meta"]["setup_detail_mesh_read_seconds"], 0.375
        )
        self.assertEqual(summary["meta"]["setup_detail_mesh_read_calls"], 1)
        self.assertAlmostEqual(summary["meta"]["setup_detail_total_seconds"], 1.375)
        self.assertAlmostEqual(summary["meta"]["setup_unattributed_seconds"], 3.625)

    def test_print_table_includes_setup_and_nonlinear_solve_stages(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_table")
        stdout = io.StringIO()
        results = {
            "spsolve": {
                "wall_seconds": 4.0,
                "steps": 2,
                "stage_seconds": {
                    "setup": 1.0,
                    "activation": 0.1,
                    "quad_state": 0.2,
                    "material": 0.3,
                    "history": 0.4,
                    "postprocess": 0.5,
                    "solver": 0.5,
                    "nonlinear_solve": 1.5,
                    "nonlinear_solve_overhead": 0.25,
                    "conversion": 0.0,
                    "transfer": 0.0,
                    "assembly": 2.0,
                    "io": 0.0,
                    "python_overhead": 0.5,
                },
            }
        }

        with redirect_stdout(stdout):
            bench.print_table(results)

        output = stdout.getvalue()
        header = output.splitlines()[0]
        self.assertIn("setup", header)
        self.assertIn("nonlinear_solve", header)
        self.assertIn("nonlinear_solve_overhead", header)
        self.assertIn("1.00", output)
        self.assertIn("1.50", output)
        self.assertIn("0.25", output)

    def test_run_repeated_writes_distinct_profiles_and_summarizes(self):
        bench = load_module(BENCH_PATH, "macro_mech100_v04_bench_repeat")
        captured = []

        class FakeWrapper:
            @staticmethod
            def main(argv):
                captured.append(argv)
                profile_path = Path(argv[argv.index("--profile-json") + 1])
                run_id = len(captured) - 1
                profile_path.write_text(
                    json.dumps(
                        {
                            "label": f"tiny/spsolve/r{run_id}",
                            "wall_seconds": float(10 - run_id * 2),
                            "steps": 1,
                            "linear_iterations": 0,
                            "stage_seconds": {"solver": float(5 - run_id)},
                            "stage_calls": {"solver": 1},
                            "meta": {},
                        }
                    )
                )
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            summary = bench.run_repeated(
                FakeWrapper,
                "tiny",
                "spsolve",
                Path(tmp),
                repeat=3,
                discard_first=1,
                thermal_warm_start=True,
                loop_kernel_jit=False,
                cell_num_cuts=3,
                cell_target_batch_size=2048,
            )

        self.assertEqual(len(captured), 3)
        self.assertTrue(
            all("--xla-thermal-warm-start" in argv for argv in captured)
        )
        self.assertTrue(
            all("--no-xla-jit-loop-kernels" in argv for argv in captured)
        )
        self.assertTrue(
            all("--xla-cell-num-cuts" in argv for argv in captured)
        )
        self.assertTrue(
            all(argv[argv.index("--xla-cell-num-cuts") + 1] == "3" for argv in captured)
        )
        self.assertTrue(
            all("--xla-cell-target-batch-size" in argv for argv in captured)
        )
        self.assertTrue(
            all(
                argv[argv.index("--xla-cell-target-batch-size") + 1] == "2048"
                for argv in captured
            )
        )
        profile_paths = [Path(argv[argv.index("--profile-json") + 1]) for argv in captured]
        self.assertEqual(len(set(profile_paths)), 3)
        output_dirs = [Path(argv[argv.index("--output-dir") + 1]) for argv in captured]
        self.assertEqual(len(set(output_dirs)), 3)
        self.assertEqual(summary["wall_seconds"], 7.0)
        self.assertEqual(summary["stage_seconds"]["solver"], 3.5)

    @staticmethod
    def wrapper_report_class():
        wrapper = load_module(XLA_WRAPPER_PATH, "macro_mech100_v04_xla_for_bench")
        return wrapper.ProfilingReport


if __name__ == "__main__":
    unittest.main()
