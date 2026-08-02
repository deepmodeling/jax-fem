#!/usr/bin/env python3
"""Compare host-staged pyamgx with PETSc GPU COO plus native AMGX."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jax
import jax.numpy as jnp

from jax_fem import logger
from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.problem import Problem
from jax_fem.solver import PYAMGX_AVAILABLE, solver


jax.config.update("jax_enable_x64", True)


class HyperElasticity(Problem):
    def get_tensor_map(self):
        def energy(F):
            young_modulus = 100.0
            poisson_ratio = 0.3
            mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
            kappa = young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
            jacobian = jnp.linalg.det(F)
            i1 = jnp.trace(F.T @ F)
            return (
                0.5 * mu * (jacobian ** (-2.0 / 3.0) * i1 - 3.0)
                + 0.5 * kappa * (jacobian - 1.0) ** 2
            )

        stress = jax.grad(energy)

        def first_pk_stress(displacement_gradient):
            return stress(displacement_gradient + jnp.eye(3))

        return first_pk_stress


def build_problem(n, displacement):
    meshio_mesh = box_mesh(n, n, n, 1.0, 1.0, 1.0)
    mesh = Mesh(
        meshio_mesh.points,
        meshio_mesh.cells_dict["hexahedron"],
        ele_type="HEX8",
    )

    def left(point):
        return jnp.isclose(point[0], 0.0, atol=1.0e-8)

    def right(point):
        return jnp.isclose(point[0], 1.0, atol=1.0e-8)

    def zero(point):
        return 0.0

    def prescribed_displacement(point):
        return displacement

    return HyperElasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type="HEX8",
        dirichlet_bc_info=[
            [left, left, left, right],
            [0, 1, 2, 0],
            [zero, zero, zero, prescribed_displacement],
        ],
    )


def solver_options(backend):
    if backend == "amgx":
        linear = {"amgx_solver": {}}
    else:
        linear = {
            "petsc_gpu_solver": {
                "backend": "native_amgx",
                "rtol": 1.0e-10,
                "max_it": int(os.environ.get("JAX_FEM_PETSC_MAX_IT", "10000")),
                "amgx_config_path": os.environ.get("JAX_FEM_AMGX_CONFIG"),
            }
        }
    return {
        "newton": {
            "tol": 1.0e-8,
            "rel_tol": 1.0e-8,
            "linear": linear,
        }
    }


def solve_once(problem, options):
    start = time.perf_counter()
    solution = solver(problem, solver_options=options)[0]
    solution.block_until_ready()
    return time.perf_counter() - start, solution


def gpu_counters(problem):
    tangent = problem._petsc_gpu_tangent_cache
    _, linear = problem._petsc_gpu_native_amgx_cache
    return {
        "coo_seconds": tangent.update_seconds,
        "coo_calls": tangent.update_calls,
        **linear.timings,
    }


def subtract_counters(after, before=None):
    before = before or {name: 0 for name in after}
    return {name: after[name] - before[name] for name in after}


def print_gpu_counters(label, counters):
    print(f"{label} GPU COO:       {counters['coo_seconds']:.6f} s / "
          f"{int(counters['coo_calls'])} calls")
    for name in (
        "matrix_upload",
        "setup",
        "vector_upload",
        "solve",
        "vector_download",
    ):
        print(f"{label} AMGX {name + ':':<16} {counters[name]:.6f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--displacement", type=float, default=0.1)
    args = parser.parse_args()

    backend = os.environ.get("JAX_FEM_GPU_SOLVER", "").strip().lower()
    if backend not in {"amgx", "petsc_amgx"}:
        raise RuntimeError("Set JAX_FEM_GPU_SOLVER to 'amgx' or 'petsc_amgx'.")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected a JAX GPU backend, got {jax.default_backend()}.")
    if backend == "amgx" and not PYAMGX_AVAILABLE:
        raise RuntimeError("JAX_FEM_GPU_SOLVER=amgx requires pyamgx.")

    logger.setLevel(logging.INFO)
    problem = build_problem(args.n, args.displacement)
    options = solver_options(backend)

    first_time, _ = solve_once(problem, options)
    first_counters = gpu_counters(problem) if backend == "petsc_amgx" else None

    repeat_time, solution = solve_once(problem, options)
    repeat_counters = None
    if backend == "petsc_amgx":
        repeat_counters = subtract_counters(gpu_counters(problem), first_counters)

    solution_l2 = float(jnp.linalg.norm(solution))
    solution_min = float(jnp.min(solution))
    solution_max = float(jnp.max(solution))

    print("\n=== 3D hyperelasticity GPU benchmark ===")
    print(f"backend:              {backend}")
    print(f"mesh:                 {args.n} x {args.n} x {args.n} HEX8")
    print(f"cells / dofs:         {problem.num_cells} / {problem.num_total_dofs_all_vars}")
    print(f"first solve time:     {first_time:.6f} s")
    print(f"repeat solve time:    {repeat_time:.6f} s")
    print(
        "solution L2/min/max:  "
        f"{solution_l2:.8e} / {solution_min:.8e} / {solution_max:.8e}"
    )

    if backend == "petsc_amgx":
        tangent = problem._petsc_gpu_tangent_cache
        _, linear = problem._petsc_gpu_native_amgx_cache
        print(f"PETSc matrix / Vec:   {tangent.mat.getType()} / {linear.x.getType()}")
        print("linear solver:        native AMGX BICGSTAB + AMG")
        print(f"AMGX library:         {linear.api.library_path}")
        print(f"last AMGX iterations: {linear.last_iterations}")
        print(f"last relative res:    {linear.last_relative_residual:.8e}")
        print_gpu_counters("first ", first_counters)
        print_gpu_counters("repeat", repeat_counters)

    print(
        f"RESULT backend={backend} n={args.n} "
        f"first_seconds={first_time:.9f} repeat_seconds={repeat_time:.9f} "
        f"solution_l2={solution_l2:.12e}"
    )
    print("\n3D HYPERELASTICITY GPU BENCHMARK: PASS")


if __name__ == "__main__":
    main()
