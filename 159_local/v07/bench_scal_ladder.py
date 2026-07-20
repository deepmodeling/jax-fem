# coding: utf-8
"""Scale ladder on the 3D hyperelastic forward problem.

Usage: python bench_scal_ladder.py <Nx> <arm> [result_json]

arm: petsc | phase23 | spsolve. Note: no rlimit guard — RLIMIT_AS would
break CUDA's large virtual reservations. The driver simply does not run
spsolve at Nx=50 (kernel-OOM already measured on 2026-07-20).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as onp

REPO = "/home/user/work/159/jax-fem"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main() -> int:
    nx = int(sys.argv[1])
    arm = sys.argv[2]
    result_json = sys.argv[3] if len(sys.argv) > 3 else None

    import jax

    jax.devices()

    import jax_fem.solver as jfs
    from jax_fem.solver import solver
    from applications.scalability.hyperelastic3d_common import (
        build_hyperelastic3d_problem_classic,
    )

    stats = {"arm": arm, "nx": nx, "status": "ok",
             "solve_calls": 0, "solve_seconds": 0.0}

    if arm == "phase23":
        from pardiso_variants import VariantSolver

        variant = VariantSolver("phase23")

        def inner(A, b, x0, linear_options, timing=None):
            return variant(A, b, x0, linear_options)
    elif arm == "spsolve":
        def inner(A, b, x0, linear_options, timing=None):
            return jfs.scipy_spsolve(A, b)
    elif arm == "petsc":
        inner = None
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    orig = jfs.linear_solver

    def timed(A, b, x0, linear_options, timing=None):
        t0 = time.perf_counter()
        x = (orig(A, b, x0, linear_options, timing=timing) if inner is None
             else inner(A, b, x0, linear_options, timing=timing))
        stats["solve_seconds"] += time.perf_counter() - t0
        stats["solve_calls"] += 1
        return x

    jfs.linear_solver = timed

    problem, mesh = build_hyperelastic3d_problem_classic(nx, nx, nx)
    stats["nodes"] = int(len(mesh.points))
    stats["dofs"] = int(3 * len(mesh.points))

    t0 = time.perf_counter()
    try:
        sol_list = solver(problem, solver_options={"petsc_solver": {}})
        stats["wall_seconds"] = time.perf_counter() - t0
        stats["max_abs_u"] = float(onp.max(onp.abs(onp.asarray(sol_list[0]))))
    except MemoryError:
        stats["status"] = "OOM"
        stats["wall_seconds"] = time.perf_counter() - t0
    except Exception as exc:  # PETSc/scipy raise various wrappers on ENOMEM
        msg = str(exc)[:200]
        stats["status"] = f"FAIL: {msg}"
        stats["wall_seconds"] = time.perf_counter() - t0

    print(f"[bench_scal_ladder] {json.dumps(stats)}")
    if result_json:
        with open(result_json, "w") as f:
            json.dump(stats, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
