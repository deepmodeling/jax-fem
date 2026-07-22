# coding: utf-8
"""Adjoint (forward+backward) benchmark on 3D hyperelastic inverse problem.

Usage: python bench_adjoint.py <Nx> <arm> [result_json]

arm:
    petsc     -- app default: petsc for forward and adjoint (adjoint solves
                 an explicitly transposed matrix from scratch)
    phase23   -- v07 direct solver for both, but adjoint still factorizes A_T
    phase23T  -- v07 + transpose reuse: adjoint = one iparm(12) backsolve of
                 the forward factorization (no A_T matrix, no refactorization)

Reports per-iteration value_and_grad wall time and gradient fingerprint for
cross-arm consistency.
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

import jax
import jax.numpy as np

import jax_fem.solver as jfs
from jax_fem.solver import ad_wrapper

from applications.scalability.hyperelastic3d_common import (
    build_hyperelastic3d_problem_inverse,
)

WARMUP = 1
REPEATS = 3


def main() -> int:
    nx = int(sys.argv[1])
    arm = sys.argv[2]
    result_json = sys.argv[3] if len(sys.argv) > 3 else None

    jax.devices()  # pin backend before anything overwrites CUDA env

    stats = {"arm": arm, "nx": nx, "solve_calls": 0, "solve_seconds": 0.0,
             "adjoint_transpose_seconds": 0.0}

    variant = None
    if arm in ("phase23", "phase23T"):
        from jax_fem_am.solvers.pardiso import VariantSolver

        variant = VariantSolver("phase23")

        def inner(A, b, x0, linear_options, timing=None):
            return variant(A, b, x0, linear_options)
    elif arm == "petsc":
        inner = jfs.linear_solver
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    orig_linear_solver = jfs.linear_solver

    def timed_linear_solver(A, b, x0, linear_options, timing=None):
        t0 = time.perf_counter()
        x = (inner(A, b, x0, linear_options, timing=timing)
             if arm != "petsc"
             else orig_linear_solver(A, b, x0, linear_options, timing=timing))
        stats["solve_seconds"] += time.perf_counter() - t0
        stats["solve_calls"] += 1
        return x

    jfs.linear_solver = timed_linear_solver

    if arm == "phase23T":
        # Replica of jfs.implicit_vjp with the A_T construction + solve
        # replaced by a transposed backsolve of the forward factorization.
        def implicit_vjp_transpose_reuse(problem, sol_list, params, v_list,
                                         adjoint_solver_options):
            def constraint_fn(dofs, p):
                problem.set_params(p)
                res_fn = problem.compute_residual
                res_fn = jfs.get_flatten_fn(res_fn, problem)
                res_fn = jfs.apply_bc(res_fn, problem)
                return res_fn(dofs)

            def constraint_fn_sol_to_sol(s_list, p):
                dofs = jax.flatten_util.ravel_pytree(s_list)[0]
                con_vec = constraint_fn(dofs, p)
                return problem.unflatten_fn_sol_list(con_vec)

            def get_partial_params_c_fn(s_list):
                def partial_params_c_fn(p):
                    return constraint_fn_sol_to_sol(s_list, p)
                return partial_params_c_fn

            def get_vjp_contraint_fn_params(p, s_list):
                partial_c_fn = get_partial_params_c_fn(s_list)

                def vjp_linear_fn(vl):
                    primals_output, f_vjp = jax.vjp(partial_c_fn, p)
                    val, = f_vjp(vl)
                    return val
                return vjp_linear_fn

            problem.set_params(params)
            problem.newton_update(sol_list)

            A = jfs.get_A(problem)
            v_vec = jax.flatten_util.ravel_pytree(v_list)[0]
            if hasattr(problem, 'P_mat'):
                v_vec = problem.P_mat.T @ v_vec

            t0 = time.perf_counter()
            adjoint_vec = variant.solve_transposed(A, onp.asarray(v_vec))
            stats["adjoint_transpose_seconds"] += time.perf_counter() - t0

            if hasattr(problem, 'P_mat'):
                adjoint_vec = problem.P_mat @ adjoint_vec

            vjp_linear_fn = get_vjp_contraint_fn_params(params, sol_list)
            vjp_result = vjp_linear_fn(
                problem.unflatten_fn_sol_list(adjoint_vec))
            vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)
            return vjp_result

        jfs.implicit_vjp = implicit_vjp_transpose_reuse

    problem, mesh = build_hyperelastic3d_problem_inverse(nx, nx, nx)
    fe = problem.fes[0]
    rho = 0.5 * np.ones((fe.num_cells, fe.num_quads))

    solver_options = {'petsc_solver': {}} if arm == "petsc" else {
        'spsolve_solver': {}}  # key irrelevant for patched arms; petsc real
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    def objective(rho_in):
        sol_list = fwd_pred(rho_in)
        return np.sum(sol_list[0] ** 2)

    value_and_grad = jax.value_and_grad(objective)

    for _ in range(WARMUP):
        j, g = value_and_grad(rho)
        jax.block_until_ready(j)

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        j, g = value_and_grad(rho)
        jax.block_until_ready(j)
        times.append(time.perf_counter() - t0)

    g_arr = onp.asarray(g).ravel()
    stats.update({
        "iteration_seconds": times,
        "iteration_mean": float(onp.mean(times)),
        "objective": float(j),
        "grad_norm": float(onp.linalg.norm(g_arr)),
        "grad_head": [float(v) for v in g_arr[:5]],
        "nodes": int(len(mesh.points)),
    })
    if variant is not None:
        stats["variant_stats"] = dict(variant._stats)
    print(f"[bench_adjoint] {json.dumps(stats)}")
    if result_json:
        with open(result_json, "w") as f:
            json.dump(stats, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
