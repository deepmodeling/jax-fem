# coding: utf-8
"""Run a jax-fem application under a chosen linear-solver arm, with timing.

Usage:
    python 159_local/v07/bench_apps.py <app_script.py> <arm> [result_json]

arm:
    baseline -- app's own solver_options, untouched (only timed)
    spsolve  -- force scipy spsolve for every linear solve
    pardiso  -- force pypardiso (v04-default behaviour)
    phase23  -- force the v07 winner (symbolic reuse + backsolve shortcut)

The app runs with cwd set to its own directory (apps use relative input/
output paths). All linear solves are intercepted at jax_fem.solver.
linear_solver, so apps do not need to be modified.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import time

import numpy as np


def main() -> int:
    app_path = os.path.abspath(sys.argv[1])
    arm = sys.argv[2]
    result_json = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None

    v07_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize the JAX backend before the app runs: some apps overwrite
    # CUDA_VISIBLE_DEVICES at import time (e.g. scalability sets "2", which
    # hides the single local GPU); once the backend is up, that has no effect.
    import jax

    jax.devices()

    import jax_fem.solver as jfs

    stats = {"arm": arm, "app": app_path, "calls": 0, "solve_seconds": 0.0}
    orig_linear_solver = jfs.linear_solver

    if arm == "baseline":
        inner = orig_linear_solver
    elif arm == "spsolve":
        def inner(A, b, x0, linear_options, timing=None):
            return jfs.scipy_spsolve(A, b)
    elif arm in ("pardiso", "phase23"):
        if arm == "phase23":
            sys.path.insert(0, v07_dir)
            from pardiso_variants import VariantSolver

            variant = VariantSolver("phase23")

            def inner(A, b, x0, linear_options, timing=None):
                return variant(A, b, x0, linear_options)
        else:
            import pypardiso
            import scipy.sparse

            psolver = pypardiso.PyPardisoSolver()

            def inner(A, b, x0, linear_options, timing=None):
                indptr, indices, data = A.getValuesCSR()
                Asp = scipy.sparse.csr_matrix(
                    (data, indices.astype(np.int32, copy=False),
                     indptr.astype(np.int32, copy=False)))
                return pypardiso.spsolve(
                    Asp, np.asarray(b, dtype=np.float64), solver=psolver)
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    def timed_linear_solver(A, b, x0, linear_options, timing=None):
        t0 = time.perf_counter()
        x = inner(A, b, x0, linear_options, timing=timing)
        stats["solve_seconds"] += time.perf_counter() - t0
        stats["calls"] += 1
        return x

    jfs.linear_solver = timed_linear_solver

    os.chdir(os.path.dirname(app_path))
    t0 = time.perf_counter()
    runpy.run_path(app_path, run_name="__main__")
    stats["wall_seconds"] = time.perf_counter() - t0

    print(f"[bench_apps] {json.dumps(stats)}")
    if result_json:
        with open(result_json, "w") as f:
            json.dump(stats, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
