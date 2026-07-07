#!/usr/bin/env python3
"""Phase 1 profiling harness for the mech100 XLA wrapper.

Runs the wrapped solver at three problem tiers and across the requested
linear solvers, then prints a comparison table separating:

    wall time / solver time / matrix conversion / transfer / io / python

Usage:
    python benchmarks/bench_mech100_xla.py --tier tiny --solvers spsolve jax
    python benchmarks/bench_mech100_xla.py --tier representative \
        --solvers spsolve jax petsc amgx --json out/bench.json

Acceptance rules enforced here:
  * every comparison is a REAL run of the physics driver (no dry runs);
  * per-stage numbers come from the ProfilingReport injected into the loop;
  * if a GPU solver loses to spsolve, the harness prints the dominant
    GPU-side cost (conversion vs solve vs transfer) instead of hiding it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (REPO_ROOT / "159_local" / "v03"
           / "am_thermal_stress_macro_intersection_mech100_XLA.py")

# Tier definitions. `case_args` are passed through to the base solver;
# adjust the mesh/path knobs to match the real driver's CLI once wired in.
#   tiny            -- seconds; CI smoke + kernel correctness
#   medium          -- minutes; catches per-step overhead regressions
#   representative  -- the real 826k-scan-step workload shape (subsampled
#                      path but same mesh density and output cadence)
TIERS = {
    "tiny": ["--mesh-scale", "0.1", "--path-stride", "1000",
             "--max-steps", "200"],
    "medium": ["--mesh-scale", "0.5", "--path-stride", "50",
               "--max-steps", "20000"],
    "representative": ["--mesh-scale", "1.0", "--path-stride", "1",
                       "--max-steps", "826000"],
}

SOLVER_FLAGS = {
    "spsolve": ["--xla-linear-solver", "spsolve"],
    "jax": ["--xla-linear-solver", "jax"],
    "jax-precond": ["--xla-linear-solver", "jax", "--xla-jax-precond"],
    "petsc": ["--xla-linear-solver", "petsc"],
    "petsc-gpu": ["--xla-linear-solver", "petsc", "--xla-petsc-gpu"],
    "amgx": ["--xla-linear-solver", "amgx"],
}


def load_wrapper():
    spec = importlib.util.spec_from_file_location("macro_mech100_xla", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_one(wrapper, tier: str, solver: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_json = out_dir / f"{tier}_{solver}.json"
    argv = (
        SOLVER_FLAGS[solver]
        + ["--profile-json", str(profile_json),
           "--profile-label", f"{tier}/{solver}"]
        + TIERS[tier]
    )
    rc = wrapper.main(argv)
    if rc != 0:
        raise RuntimeError(f"{tier}/{solver} exited with {rc}")
    return json.loads(profile_json.read_text())


def print_table(results: dict) -> None:
    stages = ["solver", "conversion", "transfer", "assembly", "io",
              "python_overhead"]
    header = f"{'solver':<12}{'wall(s)':>10}" + "".join(
        f"{s:>16}" for s in stages) + f"{'ms/step':>10}"
    print(header)
    print("-" * len(header))
    for name, rep in results.items():
        ss = rep["stage_seconds"]
        per_step = (rep["wall_seconds"] / rep["steps"] * 1e3
                    if rep["steps"] else float("nan"))
        row = f"{name:<12}{rep['wall_seconds']:>10.2f}" + "".join(
            f"{ss.get(s, 0.0):>16.2f}" for s in stages) + f"{per_step:>10.3f}"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tiny", choices=sorted(TIERS))
    ap.add_argument("--solvers", nargs="+", default=["spsolve", "jax"],
                    choices=sorted(SOLVER_FLAGS))
    ap.add_argument("--out", default="benchmarks/out")
    ap.add_argument("--json", default=None,
                    help="write the combined comparison to this file")
    args = ap.parse_args()

    wrapper = load_wrapper()
    out_dir = Path(args.out)

    results = {}
    for solver in args.solvers:
        try:
            results[solver] = run_one(wrapper, args.tier, solver, out_dir)
        except RuntimeError as exc:
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
            cpu_r = wrapper.ProfilingReport(label="spsolve")
            cpu_r.wall_seconds = cpu["wall_seconds"]
            cpu_r.stage_seconds.update(cpu["stage_seconds"])
            print(f"\n{name}: {wrapper.explain_gpu_vs_cpu(gpu_r, cpu_r)}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
