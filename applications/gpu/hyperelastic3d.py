"""Benchmark AMGX staging against the device-resident PETSc GPU pipeline.

Run this module in the AMGX and PETSc GPU conda environments separately.  Each
invocation solves the same problem twice so the second time excludes most JAX,
PETSc, and CUDA first-use cache costs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def _select_one_gpu_before_jax_import() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    known, _ = parser.parse_known_args(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_select_one_gpu_before_jax_import()

import jax
import jax.numpy as np
import numpy as onp

from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.problem import Problem


jax.config.update("jax_enable_x64", True)

DOMAIN_SIZE = 1.0
YOUNGS_MODULUS = 1.0e3
POISSON_RATIO = 0.3


class HyperElasticity(Problem):
    """Compressible Neo-Hookean solid under prescribed axial extension."""

    def get_tensor_map(self):
        mu = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
        kappa = YOUNGS_MODULUS / (3.0 * (1.0 - 2.0 * POISSON_RATIO))

        def energy_density(F):
            jacobian = np.linalg.det(F)
            isochoric_scale = jacobian ** (-2.0 / 3.0)
            first_invariant = np.trace(F.T @ F)
            return (
                0.5 * mu * (isochoric_scale * first_invariant - 3.0)
                + 0.5 * kappa * (jacobian - 1.0) ** 2
            )

        first_pk = jax.grad(energy_density)

        def stress_map(displacement_gradient):
            deformation_gradient = displacement_gradient + np.eye(self.dim)
            return first_pk(deformation_gradient)

        return stress_map


def build_problem(num_elements: int, stretch: float):
    meshio_mesh = box_mesh(
        num_elements,
        num_elements,
        num_elements,
        DOMAIN_SIZE,
        DOMAIN_SIZE,
        DOMAIN_SIZE,
    )
    cell_type = get_meshio_cell_type("HEX8")
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def bottom(point):
        return np.isclose(point[2], 0.0, atol=1.0e-8)

    def top(point):
        return np.isclose(point[2], DOMAIN_SIZE, atol=1.0e-8)

    def zero(_point):
        return 0.0

    def prescribed_stretch(_point):
        return stretch

    # The bottom face removes rigid modes; the top z displacement creates a
    # finite-strain state while leaving lateral Poisson contraction free.
    dirichlet_bc_info = [
        [bottom, bottom, bottom, top],
        [0, 1, 2, 2],
        [zero, zero, zero, prescribed_stretch],
    ]
    problem = HyperElasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type="HEX8",
        dirichlet_bc_info=dirichlet_bc_info,
    )
    return problem, mesh


def _solve_amgx(problem, args):
    from jax_fem.solver import solver

    options = {
        "newton": {
            "tol": args.tol,
            "rel_tol": args.rel_tol,
            "linear": {"amgx_solver": {}},
        }
    }
    return solver(problem, solver_options=options), None


def _solve_petsc_gpu(problem, args):
    from jax_fem.petsc_gpu_solver import petsc_gpu_solver

    options = {
        "tol": args.tol,
        "rel_tol": args.rel_tol,
        "linear": {
            "ksp_type": args.ksp_type,
            "pc_type": args.pc_type,
            "factor_solver_type": "cusparse",
            "factor_ordering_type": "natural",
            "factor_reuse_ordering": True,
            "factor_levels": args.factor_levels,
            "cg_single_reduction": args.cg_single_reduction,
            "rtol": args.linear_rtol,
            "atol": args.linear_atol,
            "max_it": args.linear_max_it,
        },
    }
    return petsc_gpu_solver(
        problem, solver_options=options, return_info=True
    )


def _solution_summary(solution) -> dict[str, float]:
    return {
        "max_abs": float(np.max(np.abs(solution))),
        "l2_norm": float(np.linalg.norm(solution)),
        "mean_uz": float(np.mean(solution[:, 2])),
    }


def _write_json(path: str, data: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _compare_reference(result: dict, solution, args) -> None:
    if args.reference_json:
        with Path(args.reference_json).open(encoding="utf-8") as stream:
            reference = json.load(stream)
        reference_time = reference["times_s"][1]
        current_time = result["times_s"][1]
        print(
            f"second-run speedup ({reference['backend']} / {result['backend']}): "
            f"{reference_time / current_time:.3f}x "
            f"({reference_time:.6f} s / {current_time:.6f} s)"
        )

    if args.reference_solution:
        reference_solution = onp.load(args.reference_solution)
        current_solution = onp.asarray(solution)
        if reference_solution.shape != current_solution.shape:
            raise ValueError(
                "Reference solution shape mismatch: "
                f"{reference_solution.shape} != {current_solution.shape}."
            )
        difference = current_solution - reference_solution
        absolute_error = onp.linalg.norm(difference)
        relative_error = absolute_error / max(
            onp.linalg.norm(reference_solution), onp.finfo(onp.float64).eps
        )
        print(
            f"solution difference: L2={absolute_error:.6e}, "
            f"relative L2={relative_error:.6e}, "
            f"Linf={onp.max(onp.abs(difference)):.6e}"
        )


def _validate_reference_paths(args) -> None:
    """Fail before the expensive solves when an explicitly requested reference is absent."""
    missing = []
    for option, value in (
        ("--reference-json", args.reference_json),
        ("--reference-solution", args.reference_solution),
    ):
        if value and not Path(value).is_file():
            missing.append(f"{option}: {value}")
    if missing:
        raise FileNotFoundError(
            "Requested AMGX reference file(s) do not exist:\n  "
            + "\n  ".join(missing)
            + "\nRun the AMGX benchmark first on this host, or omit the "
            "corresponding --reference-* option."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("amgx", "petsc_gpu"), required=True)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index.")
    parser.add_argument("--n", type=int, default=40, help="HEX8 elements per axis.")
    parser.add_argument("--stretch", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--rel-tol", type=float, default=1.0e-8)
    parser.add_argument("--ksp-type", default="cg")
    parser.add_argument("--pc-type", default="icc")
    parser.add_argument("--factor-levels", type=int, default=0)
    parser.add_argument(
        "--cg-single-reduction",
        action="store_true",
        help="Use PETSc's single-reduction variant when --ksp-type=cg.",
    )
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-max-it", type=int, default=10_000)
    parser.add_argument("--json", help="Write benchmark metadata to this JSON file.")
    parser.add_argument("--solution", help="Write the final nodal displacement as .npy.")
    parser.add_argument(
        "--reference-json", help="Compare second-run wall time to a JSON result."
    )
    parser.add_argument(
        "--reference-solution", help="Compare against a saved .npy solution."
    )
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to expose first-use cache costs.")
    if args.n < 1:
        parser.error("--n must be positive.")
    return args


def main():
    args = parse_args()
    _validate_reference_paths(args)
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(
            f"Expected exactly one visible GPU after --gpu selection, got {devices}."
        )

    problem, mesh = build_problem(args.n, args.stretch)
    solve_once = _solve_amgx if args.backend == "amgx" else _solve_petsc_gpu
    times = []
    infos = []
    solution = None

    print(
        f"backend={args.backend}, device={devices[0]}, mesh={args.n}^3 HEX8, "
        f"cells={args.n ** 3}, nodes={len(mesh.points)}, "
        f"dofs={problem.num_total_dofs_all_vars}"
    )
    for run in range(args.repeats):
        start = time.perf_counter()
        sol_list, info = solve_once(problem, args)
        solution = sol_list[0]
        solution.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        infos.append(info)
        cache_label = "cold/cache-building" if run == 0 else "cache-warm"
        print(f"run {run + 1}/{args.repeats} ({cache_label}): {elapsed:.6f} s")

    result = {
        "backend": args.backend,
        "gpu": args.gpu,
        "device": str(devices[0]),
        "n": args.n,
        "cells": args.n ** 3,
        "nodes": len(mesh.points),
        "dofs": problem.num_total_dofs_all_vars,
        "stretch": args.stretch,
        "times_s": times,
        "solution": _solution_summary(solution),
    }
    if args.backend == "petsc_gpu":
        result["petsc_runs"] = infos

    print("BENCHMARK_RESULT " + json.dumps(result, sort_keys=True))
    _compare_reference(result, solution, args)

    if args.solution:
        output = Path(args.solution)
        output.parent.mkdir(parents=True, exist_ok=True)
        onp.save(output, onp.asarray(solution))
        print(f"wrote solution: {output}")
    if args.json:
        _write_json(args.json, result)
        print(f"wrote benchmark: {args.json}")


if __name__ == "__main__":
    main()
