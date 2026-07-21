"""Inverse / AD timing on 3D hyperelastic uniaxial tension (``rho`` only)."""

import os
import time

import jax
import jax.numpy as np

from jax_fem.solver import ad_wrapper, solver
from jax_fem.utils import save_sol

from applications.scalability.hyperelastic3d_common import (
    E_BASE,
    Lx,
    Ly,
    Lz,
    STRETCH,
    T_LATERAL,
    build_hyperelastic3d_problem_inverse,
)

Nx, Ny, Nz = 8, 8, 8
RHO = 0.5
TIMING_WARMUP = 1
TIMING_REPEATS = 3
SAVE_VTK = False


def main():
    problem, mesh = build_hyperelastic3d_problem_inverse(Nx, Ny, Nz)
    fe = problem.fes[0]
    rho = RHO * np.ones((fe.num_cells, fe.num_quads))

    solver_options = {'petsc_solver': {}}

    if SAVE_VTK:
        problem.set_params(rho)
        sol_list = solver(problem, solver_options=solver_options)
        vtk_dir = os.path.join(os.path.dirname(__file__), "output", "vtk")
        os.makedirs(vtk_dir, exist_ok=True)
        path = os.path.join(vtk_dir, f"u_inv_{Nx}x{Ny}x{Nz}.vtu")
        save_sol(fe, sol_list[0], path)
        print(f"Wrote {path}")

    fwd_pred = ad_wrapper(
        problem,
        solver_options=solver_options,
        adjoint_solver_options=solver_options,
    )

    def objective(rho_in):
        sol_list = fwd_pred(rho_in)
        return np.sum(sol_list[0] ** 2)

    value_and_grad = jax.value_and_grad(objective)

    for _ in range(TIMING_WARMUP):
        _ = value_and_grad(rho)
    J0, g0 = value_and_grad(rho)
    jax.block_until_ready(J0)
    jax.block_until_ready(g0)

    def timed(fn, x):
        t0 = time.perf_counter()
        y = fn(x)
        jax.block_until_ready(y)
        return y, time.perf_counter() - t0

    print(f"\n=== Inverse timing: cube {Lx}x{Ly}x{Lz}, mesh {Nx}x{Ny}x{Nz}, cells={Nx * Ny * Nz} ===")
    print(
        f"param: rho={RHO} (uniform); E_base={E_BASE}; "
        f"stretch ux={STRETCH}; lateral ty={T_LATERAL}"
    )
    print("BC: x=0 fixed; x=Lx ux=STRETCH, uy=uz=0; y=Ly Neumann t=[0, T_LATERAL, 0]")

    times = []
    for i in range(TIMING_REPEATS):
        (J, g), dt = timed(value_and_grad, rho)
        times.append(dt)
        print(
            f"[{i + 1}/{TIMING_REPEATS}] J={float(J):.6e}, "
            f"|grad|={float(np.linalg.norm(np.asarray(g))):.4e}, "
            f"value_and_grad={dt:.3f}s"
        )

    import numpy as onp

    print(f"\nSummary: value_and_grad avg={onp.mean(times):.3f}s")


if __name__ == "__main__":
    main()
