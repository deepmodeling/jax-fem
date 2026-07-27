"""Production-order integration coverage for Kaess phase-history state.

The lower-level history, JIT, lifecycle, and mechanics-state tests deliberately
exercise their own boundaries.  This module covers the missing composition:
the formal launcher contract feeds ``runner.main()``, acceleration replaces the
loop kernels and predicates, the v06 wrappers are refreshed, and a first
solidification event reaches the mechanics state seen by the solver.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem_am.simulation import acceleration, runner


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
STEPPER_PATH = ROOT / "jax_fem_am" / "simulation" / "stepper.py"
FORMAL_LAUNCHERS = (
    ROOT / "cases" / "kaess_2023" / "run_kaess_phase1.sh",
    ROOT / "cases" / "kaess_2023" / "run_kaess_phase2.sh",
)

# These patches do not participate in phase-history or mechanics-state
# composition.  Keeping them out makes this a three-second integration test
# without replacing the production runner, parser, JIT/history patch,
# profiling wrapper, predicate cache, lifecycle wrapper, or mechanics wrapper.
UNRELATED_ACCELERATION_PATCHES = (
    "configure_problem_cell_assembly_num_cuts",
    "install_jax_fem_timing_patch",
    "install_problem_local_assembly_timing_patch",
    "install_finite_element_timing_patch",
    "install_thermal_only_mechanics_surrogate_patch",
    "install_setup_detail_timing_patch",
    "install_lazy_output_postprocess_patch",
    "install_activation_cache_patch",
    "install_quad_scalar_fast_path_patch",
    "install_solver_patch",
)


@pytest.mark.parametrize("launcher", FORMAL_LAUNCHERS)
def test_formal_launcher_selects_non_dry_run_paper_history(launcher):
    text = launcher.read_text(encoding="utf-8")
    solver_start = text.index("SOLVER_CMD=(")
    solver_end = text.index("\n)", solver_start)
    solver_command = text[solver_start:solver_end]

    assert '"${PYTHON_BIN}" -m jax_fem_am.simulation.runner' in solver_command
    assert "--phase-history-model paper_irreversible" in solver_command
    assert "--no-reset-plastic-on-melt" in solver_command
    assert "--xla-dry-run" not in solver_command
    assert (
        '"${SOLVER_CMD[@]}" "${KAESS_EXTRA_ARGV[@]}"'
        in text[solver_end:]
    )


def _load_fresh_stepper(monkeypatch):
    module_name = "v03_kaess_production_history_chain"
    spec = importlib.util.spec_from_file_location(module_name, STEPPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load production stepper: {STEPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _array_has_any(value) -> bool:
    return value is not None and bool(np.asarray(value).any())


def test_first_solidification_reference_survives_production_patch_order(
    monkeypatch,
):
    base = _load_fresh_stepper(monkeypatch)
    observations = {"solver_calls": 0}
    reports = []

    original_report = acceleration.ProfilingReport

    class CapturingReport(original_report):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            reports.append(self)

    def no_op_patch(*_args, **_kwargs):
        return False

    for name in UNRELATED_ACCELERATION_PATCHES:
        monkeypatch.setattr(acceleration, name, no_op_patch)

    # Isolate compilation caches without mutating process-global dictionaries
    # used by other tests.
    monkeypatch.setattr(acceleration, "_LOOP_KERNEL_JIT_THERMAL_CACHE", {})
    monkeypatch.setattr(acceleration, "_LOOP_KERNEL_JIT_MECHANICS_CACHE", {})
    monkeypatch.setattr(acceleration, "_LOOP_KERNEL_JIT_HISTORY_CACHE", {})
    monkeypatch.setattr(acceleration, "ProfilingReport", CapturingReport)
    monkeypatch.setattr(acceleration, "load_base_solver", lambda: base)

    def solver_boundary(mechanics, u_guess, _params, _overrides=None):
        observations["solver_calls"] += 1
        observations["eps_ref_seen_by_solver"] = np.asarray(
            mechanics._eps_ref_state
        ).copy()
        return u_guess

    # install_v06_adapter captures this function as the expensive solve
    # boundary, then installs the real mechanics-state wrapper around it.
    base.run_mechanics = solver_boundary

    def two_increment_production_loop():
        args = base.parse_args()
        observations["args"] = args
        observations["phase_wrapper_outermost"] = bool(
            getattr(
                base.update_phase_reference_and_eqp,
                "_v06_phase_lifecycle_wrapper",
                False,
            )
        )
        observations["mechanics_wrapper_outermost"] = bool(
            getattr(
                base.should_run_mechanics,
                "_v06_mechanics_event_wrapper",
                False,
            )
        )

        mechanics = object.__new__(base.ThermoMechanical)
        mechanics.mechanics_model = "j2_plastic"
        mechanics.dim = 3
        mechanics._eps_p_state = jnp.zeros(
            (1, 1, 3, 3), dtype=jnp.float64
        )
        mechanics._eps_ref_state = jnp.zeros(
            (1, 1, 3, 3), dtype=jnp.float64
        )
        mechanics._relaxation_mask = jnp.zeros(
            (1, 1, 1), dtype=bool
        )
        total_strain = jnp.diag(
            jnp.asarray([0.01, -0.005, -0.005], dtype=jnp.float64)
        )
        mechanics._u_grads = (
            lambda _solution: total_strain[None, None, :, :]
        )

        active = jnp.ones((1, 1, 1), dtype=jnp.float64)
        phase = jnp.full(
            (1, 1, 1), base.STATE_POWDER, dtype=jnp.float64
        )
        reference = jnp.full((1, 1, 1), 300.0, dtype=jnp.float64)
        eqp = jnp.full((1, 1, 1), 0.25, dtype=jnp.float64)

        # First melting is not a stress-free-reference event.
        hot = jnp.full((1, 1, 1), 1700.0, dtype=jnp.float64)
        phase, reference, eqp, newly, _ = (
            base.update_phase_reference_and_eqp(
                hot,
                active,
                phase,
                reference,
                eqp,
                args,
            )
        )
        observations["phase_after_melt"] = np.asarray(phase).copy()
        observations["reference_after_melt"] = np.asarray(reference).copy()
        observations["newly_after_melt"] = bool(np.asarray(newly).any())
        observations["pending_after_melt"] = _array_has_any(
            runner.REGISTRY.pending_reference
        )

        # Cooling through solidus must latch the local temperature exactly once
        # and generate a pending mechanics-reference event.
        cold = jnp.full((1, 1, 1), 1600.0, dtype=jnp.float64)
        phase, reference, eqp, newly, entered_melted = (
            base.update_phase_reference_and_eqp(
                cold,
                active,
                phase,
                reference,
                eqp,
                args,
            )
        )
        observations["phase_after_solidification"] = np.asarray(phase).copy()
        observations["reference_after_solidification"] = np.asarray(
            reference
        ).copy()
        observations["eqp_after_solidification"] = np.asarray(eqp).copy()
        observations["newly_solidified"] = bool(np.asarray(newly).all())
        observations["entered_melted_on_cooling"] = bool(
            np.asarray(entered_melted).any()
        )
        observations["pending_before_mechanics"] = _array_has_any(
            runner.REGISTRY.pending_reference
        )
        observations["registry_eqp_before_mechanics"] = (
            None
            if runner.REGISTRY.eqp is None
            else np.asarray(runner.REGISTRY.eqp).copy()
        )

        # Step 1 is deliberately off the normal 20-step cadence.  Only the
        # refreshed mechanics-event wrapper can make this solve due.
        mechanics_due = bool(base.should_run_mechanics(1, args))
        observations["mechanics_due"] = mechanics_due
        if mechanics_due:
            mechanics_params = [
                cold,
                (cold - reference) * active,
                active,
                jnp.full_like(cold, 190.0e9),
                jnp.full_like(cold, 1.6e-5),
                jnp.full_like(cold, 0.3),
                jnp.full_like(cold, 500.0e6),
                jnp.zeros_like(cold),
                eqp,
            ]
            base.run_mechanics(
                mechanics,
                [jnp.zeros((1, 3), dtype=jnp.float64)],
                mechanics_params,
            )

        observations["pending_after_mechanics"] = _array_has_any(
            runner.REGISTRY.pending_reference
        )
        observations["eps_ref_after_mechanics"] = np.asarray(
            mechanics._eps_ref_state
        ).copy()
        observations["registry_eps_ref_after_mechanics"] = (
            None
            if runner.REGISTRY.eps_ref is None
            else np.asarray(runner.REGISTRY.eps_ref).copy()
        )
        return 0

    base.main = two_increment_production_loop

    runner.REGISTRY.reset()
    try:
        result = runner.main(
            [
                "--phase-history-model",
                "paper_irreversible",
                "--no-reset-plastic-on-melt",
                "--solidus-temperature",
                "1643.15",
                "--liquidus-temperature",
                "1673.15",
                "--stress-relaxation-temperature",
                "0",
                "--mechanics-model",
                "j2_plastic",
                "--mechanics-every",
                "20",
                "--xla-platform",
                "cpu",
                "--xla-linear-solver",
                "keep",
                "--xla-jit-loop-kernels",
                "--no-xla-quiet-jax-fem-logs",
                "--profile-label",
                "production-history-chain-test",
            ]
        )

        assert result == 0
        assert len(reports) == 1
        args = observations["args"]
        assert args.xla_dry_run is False
        assert args.phase_history_model == "paper_irreversible"
        assert args.reset_plastic_on_melt is False
        assert args.xla_jit_loop_kernels is True

        assert observations["phase_wrapper_outermost"] is True
        assert observations["mechanics_wrapper_outermost"] is True
        assert reports[0].meta["loop_kernel_jit_history_calls"] == 2
        assert reports[0].meta["loop_kernel_jit_history_cache_entries"] == 1
        assert reports[0].meta["step_predicate_cache_misses"] == 1

        np.testing.assert_array_equal(
            observations["phase_after_melt"],
            np.full((1, 1, 1), base.STATE_LIQUID),
        )
        np.testing.assert_allclose(
            observations["reference_after_melt"],
            np.full((1, 1, 1), 300.0),
        )
        assert observations["newly_after_melt"] is False
        assert observations["pending_after_melt"] is False

        np.testing.assert_array_equal(
            observations["phase_after_solidification"],
            np.full((1, 1, 1), base.STATE_SOLID),
        )
        np.testing.assert_allclose(
            observations["reference_after_solidification"],
            np.full((1, 1, 1), 1600.0),
        )
        np.testing.assert_allclose(
            observations["eqp_after_solidification"],
            np.full((1, 1, 1), 0.25),
        )
        np.testing.assert_allclose(
            observations["registry_eqp_before_mechanics"],
            np.full((1, 1, 1), 0.25),
        )
        assert observations["newly_solidified"] is True
        assert observations["entered_melted_on_cooling"] is False
        assert observations["pending_before_mechanics"] is True
        assert observations["mechanics_due"] is True

        expected_reference = np.asarray(
            [[[[0.01, 0.0, 0.0],
               [0.0, -0.005, 0.0],
               [0.0, 0.0, -0.005]]]]
        )
        assert observations["solver_calls"] == 1
        np.testing.assert_allclose(
            observations["eps_ref_seen_by_solver"],
            expected_reference,
        )
        np.testing.assert_allclose(
            observations["eps_ref_after_mechanics"],
            expected_reference,
        )
        np.testing.assert_allclose(
            observations["registry_eps_ref_after_mechanics"],
            expected_reference,
        )
        assert observations["pending_after_mechanics"] is False
    finally:
        runner.REGISTRY.reset()
