#!/usr/bin/env python3
"""Phase 1 profiling harness for the mech100 XLA wrapper.

Runs the wrapped solver at bounded problem tiers and across the requested
linear solvers, then prints a comparison table separating:

    wall time / setup / solver time / matrix conversion / transfer / io / python

Setup sub-step timing lives in the JSON `meta` block; the printed table keeps
`setup` as the first-solve-boundary total stage.

Usage:
    python 159_local/v04/bench_mech100_xla.py --tier tiny --solvers spsolve jax
    python 159_local/v04/bench_mech100_xla.py --tier representative \
        --solvers spsolve jax petsc amgx --json out/bench.json

Acceptance rules enforced here:
  * every comparison is a REAL run of the physics driver (no dry runs);
  * per-stage numbers come from the ProfilingReport injected into the loop;
  * if a GPU solver loses to spsolve, the harness prints the dominant
    GPU-side cost (conversion vs solve vs transfer) instead of hiding it.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

WRAPPER = Path(__file__).resolve().with_name(
    "am_thermal_stress_macro_intersection_mech100_XLA.py"
)

# Tier definitions. `case_args` are passed through to the base solver;
# adjust the mesh/path knobs to match the real driver's CLI once wired in.
#   tiny            -- seconds; CI smoke + kernel correctness
#   small-loop      -- bounded multi-step loop; catches per-step overhead and
#                      activation-cache regressions without mechanics cost
#   medium          -- minutes; catches coupled thermal/mechanics regressions
#   representative  -- the real 826k-scan-step workload shape (subsampled
#                      path but same mesh density and output cadence)
TIERS = {
    "tiny": [
        "--max-cells", "20",
        "--layers", "1",
        "--scan-steps-per-layer", "1",
        "--hatch-lines-per-layer", "1",
        "--layer-activation-geometry", "centroid",
        "--mechanics-every", "0",
        "--summary-every", "999999",
    ],
    "small-loop": [
        "--max-cells", "80",
        "--layers", "2",
        "--scan-steps-per-layer", "8",
        "--hatch-lines-per-layer", "1",
        "--layer-activation-geometry", "centroid",
        "--mechanics-every", "0",
        "--summary-every", "999999",
    ],
    "medium": [
        "--max-cells", "500",
        "--layers", "5",
        "--scan-steps-per-layer", "20",
        "--hatch-lines-per-layer", "2",
        "--layer-activation-geometry", "centroid",
        "--mechanics-every", "5",
        "--summary-every", "100",
    ],
    "real-slice": [
        "--max-cells", "0",
        "--layers", "2",
        "--scan-steps-per-layer", "4",
        "--hatch-lines-per-layer", "1",
        "--layer-activation-geometry", "centroid",
        "--mechanics-every", "0",
        "--summary-every", "999999",
    ],
    "real-slice-mech": [
        "--max-cells", "0",
        "--layers", "2",
        "--scan-steps-per-layer", "4",
        "--hatch-lines-per-layer", "1",
        "--layer-activation-geometry", "centroid",
        "--mechanics-every", "4",
        "--summary-every", "999999",
    ],
    "representative": [
        "--max-cells", "0",
        "--layers", "50",
        "--scan-steps-per-layer", "20",
        "--hatch-lines-per-layer", "1",
        "--layer-activation-geometry", "centroid",
        "--mechanics-every", "1",
        "--summary-every", "500",
    ],
}

SOLVER_FLAGS = {
    "spsolve": ["--xla-linear-solver", "spsolve"],
    "jax": ["--xla-linear-solver", "jax"],
    "jax-precond": ["--xla-linear-solver", "jax", "--xla-jax-precond"],
    "jax-cg": ["--xla-linear-solver", "jax", "--xla-jax-method", "cg"],
    "jax-cg-no-check": [
        "--xla-linear-solver", "jax",
        "--xla-jax-method", "cg",
        "--xla-jax-skip-residual-check",
    ],
    "jax-gmres": ["--xla-linear-solver", "jax", "--xla-jax-method", "gmres"],
    "jax-spsolve": [
        "--xla-linear-solver", "jax",
        "--xla-jax-method", "spsolve",
        "--xla-jax-skip-residual-check",
    ],
    "petsc": ["--xla-linear-solver", "petsc"],
    "petsc-gpu": ["--xla-linear-solver", "petsc", "--xla-petsc-gpu"],
    "amgx": ["--xla-linear-solver", "amgx"],
    "pardiso": ["--xla-linear-solver", "pardiso"],
}

COUNTER_META_KEYS = (
    "activation_cache_hits",
    "activation_cache_misses",
    "activation_cache_entries",
    "dof_to_quad_cache_hits",
    "dof_to_quad_cache_misses",
    "dof_to_quad_cache_entries",
    "dof_to_quad_cache_skipped_non_jax",
    "step_predicate_cache_hits",
    "step_predicate_cache_misses",
    "step_predicate_cache_entries",
    "step_predicate_cache_builds",
    "loop_kernel_jit_mechanics_disabled_skips",
    "thermal_only_mechanics_surrogate_hits",
    "thermal_only_mechanics_surrogate_fallbacks",
    "lazy_output_postprocess_skips",
    "lazy_output_postprocess_computes",
    "lazy_output_postprocess_forced_computes",
    "lazy_output_postprocess_phase_deferred",
    "jax_bcoo_cache_hits",
    "jax_bcoo_cache_misses",
    "jax_spsolve_calls",
    "solver_fallbacks",
)

PRINT_STAGE_COLUMNS = [
    "setup",
    "activation",
    "quad_state",
    "material",
    "history",
    "postprocess",
    "dof_to_quad",
    "nonlinear_solve",
    "nonlinear_solve_overhead",
    "bc_initial_guess",
    "residual_vector",
    "residual_flatten",
    "residual_bc",
    "residual_projection",
    "solver",
    "conversion",
    "transfer",
    "local_assembly",
    "global_matrix",
    "assembly",
    "cell_jacobian",
    "cell_residual",
    "face_jacobian",
    "face_residual",
    "residual_scatter",
    "io",
    "python_overhead",
]


def load_wrapper():
    spec = importlib.util.spec_from_file_location("macro_mech100_xla", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _is_setup_detail_seconds_meta(key: str) -> bool:
    return (
        key == "setup_seconds_before_first_solve"
        or key == "setup_unattributed_seconds"
        or key == "setup_detail_total_seconds"
        or (
            key.startswith("setup_detail_")
            and key.endswith("_seconds")
        )
    )


def _is_setup_detail_calls_meta(key: str) -> bool:
    return key.startswith("setup_detail_") and key.endswith("_calls")


def run_one(
    wrapper,
    tier: str,
    solver: str,
    out_dir: Path,
    run_index: int = 0,
    repeat: int = 1,
    thermal_warm_start: bool = False,
    loop_kernel_jit: bool = True,
    residual_only_check: bool = True,
    dof_to_quad_cache: bool = True,
    step_predicate_cache: bool = True,
    skip_unused_mechanics_material: bool = True,
    thermal_only_mechanics_surrogate: bool = True,
    quiet_jax_fem_logs: bool = True,
    lazy_output_postprocess: bool = False,
    cell_num_cuts: int | None = None,
    cell_target_batch_size: int | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{tier}_{solver}" if repeat == 1 else f"{tier}_{solver}_r{run_index:02d}"
    profile_json = out_dir / f"{suffix}.json"
    warm_start_flags = (
        ["--xla-thermal-warm-start"] if thermal_warm_start else []
    )
    loop_kernel_jit_flags = (
        [] if loop_kernel_jit else ["--no-xla-jit-loop-kernels"]
    )
    residual_only_check_flags = (
        [] if residual_only_check else ["--no-xla-residual-only-check"]
    )
    dof_to_quad_cache_flags = (
        [] if dof_to_quad_cache else ["--no-xla-dof-to-quad-cache"]
    )
    step_predicate_cache_flags = (
        [] if step_predicate_cache else ["--no-xla-step-predicate-cache"]
    )
    skip_unused_mechanics_material_flags = (
        []
        if skip_unused_mechanics_material
        else ["--no-xla-skip-unused-mechanics-material"]
    )
    thermal_only_mechanics_surrogate_flags = (
        []
        if thermal_only_mechanics_surrogate
        else ["--no-xla-thermal-only-mechanics-surrogate"]
    )
    quiet_jax_fem_log_flags = (
        [] if quiet_jax_fem_logs else ["--no-xla-quiet-jax-fem-logs"]
    )
    lazy_output_postprocess_flags = (
        ["--xla-lazy-output-postprocess"]
        if lazy_output_postprocess
        else []
    )
    cell_num_cuts_flags = (
        ["--xla-cell-num-cuts", str(cell_num_cuts)]
        if cell_num_cuts is not None
        else []
    )
    cell_target_batch_size_flags = (
        ["--xla-cell-target-batch-size", str(cell_target_batch_size)]
        if cell_target_batch_size is not None
        else []
    )
    argv = (
        SOLVER_FLAGS[solver]
        + warm_start_flags
        + loop_kernel_jit_flags
        + residual_only_check_flags
        + dof_to_quad_cache_flags
        + step_predicate_cache_flags
        + skip_unused_mechanics_material_flags
        + thermal_only_mechanics_surrogate_flags
        + quiet_jax_fem_log_flags
        + lazy_output_postprocess_flags
        + cell_num_cuts_flags
        + cell_target_batch_size_flags
        + ["--xla-preallocate", "off",
           "--profile-json", str(profile_json),
           "--profile-label", f"{tier}/{solver}" if repeat == 1 else f"{tier}/{solver}/r{run_index}",
           "--output-dir", str(out_dir / f"{suffix}_run")]
        + TIERS[tier]
    )
    rc = wrapper.main(argv)
    if rc != 0:
        raise RuntimeError(f"{tier}/{solver} exited with {rc}")
    return json.loads(profile_json.read_text())


def summarize_runs(
    tier: str,
    solver: str,
    runs: list[dict],
    discard_first: int = 0,
) -> dict:
    if not runs:
        raise ValueError(f"no runs to summarize for {tier}/{solver}")
    if discard_first < 0:
        raise ValueError("--discard-first must be non-negative")
    samples = runs[discard_first:]
    if not samples:
        raise ValueError(
            f"discarded all runs for {tier}/{solver}: "
            f"repeat={len(runs)} discard_first={discard_first}"
        )

    stage_keys = sorted(
        {key for run in samples for key in run.get("stage_seconds", {})}
    )
    call_keys = sorted(
        {key for run in samples for key in run.get("stage_calls", {})}
    )
    steps = int(round(_mean([float(run.get("steps", 0)) for run in samples])))
    stage_seconds = {
        key: _mean([float(run.get("stage_seconds", {}).get(key, 0.0)) for run in samples])
        for key in stage_keys
    }
    stage_calls = {
        key: int(round(_mean([float(run.get("stage_calls", {}).get(key, 0)) for run in samples])))
        for key in call_keys
    }

    summary = copy.deepcopy(samples[-1])
    summary["label"] = f"{tier}/{solver}/mean"
    summary["wall_seconds"] = _mean([float(run["wall_seconds"]) for run in samples])
    summary["steps"] = steps
    summary["linear_iterations"] = int(
        round(_mean([float(run.get("linear_iterations", 0)) for run in samples]))
    )
    summary["stage_seconds"] = stage_seconds
    summary["stage_calls"] = stage_calls
    summary["per_step_seconds"] = {
        key: (seconds / steps if steps else 0.0)
        for key, seconds in stage_seconds.items()
    }
    summary.setdefault("meta", {})
    for key in COUNTER_META_KEYS:
        if any(key in run.get("meta", {}) for run in runs):
            summary["meta"][key] = int(
                round(
                    _mean([
                        float(run.get("meta", {}).get(key, 0))
                        for run in samples
                    ])
                )
            )
    meta_keys = {key for run in runs for key in run.get("meta", {})}
    for key in sorted(meta_keys):
        if _is_setup_detail_seconds_meta(key):
            summary["meta"][key] = _mean([
                float(run.get("meta", {}).get(key, 0.0))
                for run in samples
            ])
        elif _is_setup_detail_calls_meta(key):
            summary["meta"][key] = int(
                round(
                    _mean([
                        float(run.get("meta", {}).get(key, 0))
                        for run in samples
                    ])
                )
            )
    summary["meta"]["benchmark_repeat"] = len(runs)
    summary["meta"]["benchmark_discard_first"] = discard_first
    summary["meta"]["benchmark_samples"] = len(samples)
    summary["meta"]["benchmark_sample_wall_seconds"] = [
        float(run["wall_seconds"]) for run in samples
    ]
    summary["meta"]["benchmark_raw_runs"] = copy.deepcopy(runs)
    return summary


def run_repeated(
    wrapper,
    tier: str,
    solver: str,
    out_dir: Path,
    repeat: int,
    discard_first: int,
    thermal_warm_start: bool = False,
    loop_kernel_jit: bool = True,
    residual_only_check: bool = True,
    dof_to_quad_cache: bool = True,
    step_predicate_cache: bool = True,
    skip_unused_mechanics_material: bool = True,
    thermal_only_mechanics_surrogate: bool = True,
    quiet_jax_fem_logs: bool = True,
    lazy_output_postprocess: bool = False,
    cell_num_cuts: int | None = None,
    cell_target_batch_size: int | None = None,
) -> dict:
    if repeat <= 0:
        raise ValueError("--repeat must be positive")
    runs = [
        run_one(
            wrapper,
            tier,
            solver,
            out_dir,
            i,
            repeat,
            thermal_warm_start=thermal_warm_start,
            loop_kernel_jit=loop_kernel_jit,
            residual_only_check=residual_only_check,
            dof_to_quad_cache=dof_to_quad_cache,
            step_predicate_cache=step_predicate_cache,
            skip_unused_mechanics_material=skip_unused_mechanics_material,
            thermal_only_mechanics_surrogate=thermal_only_mechanics_surrogate,
            quiet_jax_fem_logs=quiet_jax_fem_logs,
            lazy_output_postprocess=lazy_output_postprocess,
            cell_num_cuts=cell_num_cuts,
            cell_target_batch_size=cell_target_batch_size,
        )
        for i in range(repeat)
    ]
    return summarize_runs(tier, solver, runs, discard_first=discard_first)


def print_table(results: dict) -> None:
    counters = [
        ("dof_hit", "dof_to_quad_cache_hits"),
        ("dof_miss", "dof_to_quad_cache_misses"),
        ("step_hit", "step_predicate_cache_hits"),
        ("step_miss", "step_predicate_cache_misses"),
        ("mech_skip", "loop_kernel_jit_mechanics_disabled_skips"),
        ("mech_surr", "thermal_only_mechanics_surrogate_hits"),
        ("post_skip", "lazy_output_postprocess_skips"),
    ]
    stages = PRINT_STAGE_COLUMNS
    solver_width = max(12, *(len(str(name)) for name in results))
    counter_width = 10
    stage_width = max(16, *(len(stage) + 2 for stage in stages))
    header = f"{'solver':<{solver_width}}{'wall(s)':>10}" + "".join(
        f"{label:>{counter_width}}" for label, _ in counters
    ) + "".join(
        f"{s:>{stage_width}}" for s in stages) + f"{'ms/step':>10}"
    print(header)
    print("-" * len(header))
    for name, rep in results.items():
        ss = rep["stage_seconds"]
        meta = rep.get("meta", {})
        per_step = (rep["wall_seconds"] / rep["steps"] * 1e3
                    if rep["steps"] else float("nan"))
        row = f"{name:<{solver_width}}{rep['wall_seconds']:>10.2f}" + "".join(
            f"{int(meta.get(key, 0) or 0):>{counter_width}d}"
            for _, key in counters
        ) + "".join(
            f"{ss.get(s, 0.0):>{stage_width}.2f}" for s in stages
        ) + f"{per_step:>10.3f}"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tiny", choices=sorted(TIERS))
    ap.add_argument("--solvers", nargs="+", default=["spsolve", "jax"],
                    choices=sorted(SOLVER_FLAGS))
    ap.add_argument("--out", default="benchmarks/out")
    ap.add_argument("--json", default=None,
                    help="write the combined comparison to this file")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each solver this many times in the same process")
    ap.add_argument("--discard-first", type=int, default=0,
                    help="discard this many first runs before averaging")
    ap.add_argument("--thermal-warm-start", action="store_true",
                    help="enable v04 thermal Newton warm-start during runs")
    ap.add_argument("--no-loop-jit", dest="loop_kernel_jit",
                    action="store_false", default=True,
                    help="pass --no-xla-jit-loop-kernels to the wrapper")
    ap.add_argument("--no-residual-only-check", dest="residual_only_check",
                    action="store_false", default=True,
                    help="pass --no-xla-residual-only-check to the wrapper")
    ap.add_argument("--no-dof-to-quad-cache", dest="dof_to_quad_cache",
                    action="store_false", default=True,
                    help="pass --no-xla-dof-to-quad-cache to the wrapper")
    ap.add_argument("--no-step-predicate-cache", dest="step_predicate_cache",
                    action="store_false", default=True,
                    help="pass --no-xla-step-predicate-cache to the wrapper")
    ap.add_argument("--no-skip-unused-mechanics-material",
                    dest="skip_unused_mechanics_material",
                    action="store_false",
                    default=True,
                    help=("pass --no-xla-skip-unused-mechanics-material to "
                          "the wrapper"))
    ap.add_argument("--no-thermal-only-mechanics-surrogate",
                    dest="thermal_only_mechanics_surrogate",
                    action="store_false",
                    default=True,
                    help=("pass --no-xla-thermal-only-mechanics-surrogate "
                          "to the wrapper"))
    ap.add_argument("--no-quiet-jax-fem-logs",
                    dest="quiet_jax_fem_logs",
                    action="store_false",
                    default=True,
                    help="pass --no-xla-quiet-jax-fem-logs to the wrapper")
    ap.add_argument("--lazy-output-postprocess",
                    dest="lazy_output_postprocess",
                    action="store_true",
                    default=False,
                    help="pass --xla-lazy-output-postprocess to the wrapper")
    ap.add_argument("--no-lazy-output-postprocess",
                    dest="lazy_output_postprocess",
                    action="store_false",
                    help="do not pass --xla-lazy-output-postprocess")
    ap.add_argument("--cell-num-cuts", type=_positive_int, default=None,
                    help=("pass --xla-cell-num-cuts to the wrapper for "
                          "cell assembly chunk tuning"))
    ap.add_argument("--cell-target-batch-size", type=_positive_int, default=None,
                    help=("pass --xla-cell-target-batch-size to the wrapper "
                          "for auto cell assembly chunk tuning"))
    args = ap.parse_args()
    if args.repeat <= 0:
        ap.error("--repeat must be positive")
    if args.discard_first < 0:
        ap.error("--discard-first must be non-negative")
    if args.discard_first >= args.repeat:
        ap.error("--discard-first must be smaller than --repeat")

    wrapper = load_wrapper()
    out_dir = Path(args.out)

    results = {}
    for solver in args.solvers:
        try:
            results[solver] = run_repeated(
                wrapper,
                args.tier,
                solver,
                out_dir,
                repeat=args.repeat,
                discard_first=args.discard_first,
                thermal_warm_start=args.thermal_warm_start,
                loop_kernel_jit=args.loop_kernel_jit,
                residual_only_check=args.residual_only_check,
                dof_to_quad_cache=args.dof_to_quad_cache,
                step_predicate_cache=args.step_predicate_cache,
                skip_unused_mechanics_material=(
                    args.skip_unused_mechanics_material
                ),
                thermal_only_mechanics_surrogate=(
                    args.thermal_only_mechanics_surrogate
                ),
                quiet_jax_fem_logs=args.quiet_jax_fem_logs,
                lazy_output_postprocess=args.lazy_output_postprocess,
                cell_num_cuts=args.cell_num_cuts,
                cell_target_batch_size=args.cell_target_batch_size,
            )
        except (
            RuntimeError,
            KeyError,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as exc:
            print(f"!! {solver}: {exc}", file=sys.stderr)

    if not results:
        print("no successful runs", file=sys.stderr)
        return 1

    print_table(results)

    # Acceptance rule: explain, don't force.
    if "spsolve" in results:
        cpu = results["spsolve"]
        for name, rep in results.items():
            if name == "spsolve":
                continue
            gpu_r = wrapper.ProfilingReport(label=name)
            gpu_r.wall_seconds = rep["wall_seconds"]
            gpu_r.stage_seconds.update(rep["stage_seconds"])
            gpu_r.meta.update(rep.get("meta", {}))
            cpu_r = wrapper.ProfilingReport(label="spsolve")
            cpu_r.wall_seconds = cpu["wall_seconds"]
            cpu_r.stage_seconds.update(cpu["stage_seconds"])
            cpu_r.meta.update(cpu.get("meta", {}))
            print(f"\n{name}: {wrapper.explain_gpu_vs_cpu(gpu_r, cpu_r)}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
