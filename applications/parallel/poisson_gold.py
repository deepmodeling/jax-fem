#!/usr/bin/env python3
"""
Serial **gold** reference: nonlinear Poisson on the unit square with JAX-FEM ``Problem`` + ``solver``.

PDE (strong form)::

    -Δu + u³ = f   on Ω = (0,1)²,
    u = 0 on ∂Ω.

Manufactured solution::

    u(x,y) = sin(π x) sin(π y),

so ``f = -Δu + u³ = 2π² sin(πx)sin(πy) + (sin(πx)sin(πy))³``.

This module defines :class:`NonlinearManufacturedPoisson` so ``poisson_mpi.py`` can import the
same weak form without duplicating the kernel.

Run from repo root::

    python -m applications.parallel.poisson_gold

**Fair comparison with** ``poisson_mpi.py``: edit ``GRID_*`` and Newton tolerances below;
``poisson_mpi`` imports the same Newton limits and sets its PETSc KSP separately (serial
``solver`` uses default PETSc linear tolerances unless you change ``jax_fem.solver``).

Wall times are printed for mesh+FE init, nonlinear solve, and error check.

Compare with MPI::

    mpiexec -n 4 python -m applications.parallel.poisson_mpi
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as onp

from jax_fem.problem import Problem
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver

# x64 matches jax_fem/solver defaults
jax.config.update("jax_enable_x64", True)

# --- Fair-run knobs (single source of truth; ``poisson_mpi.py`` imports these) --------------
GRID_NX = 500
GRID_NY = 500

# Outer Newton: same rule as ``jax_fem.solver.solver`` —
# ``while (||R||/||R0|| > rel_tol) and (||R|| > tol)``.
NEWTON_TOL = 1e-10
NEWTON_REL_TOL = 1e-10
NEWTON_MAX_IT = 10_000  # mpi only: ``jax_fem.solver.solver`` has no outer Newton cap

# ``poisson_mpi`` imports ``KSP_*`` for a fixed CG+GAMG ``KSP``; align with ``poisson_fenicsx`` spirit.
PETSC_KSP_TYPE = "cg"
PETSC_PC_TYPE_GOLD = "gamg"
KSP_RTOL = 1e-5
KSP_ATOL = 1e-50
KSP_MAX_IT = 10_000
GMRES_RESTART = 30
# -------------------------------------------------------------------------------------------


def fair_solver_options():
    """``solver_options`` for ``jax_fem.solver.solver`` (Newton tol + ``petsc_solver`` kinds)."""
    return {
        "petsc_solver": {"ksp_type": PETSC_KSP_TYPE, "pc_type": PETSC_PC_TYPE_GOLD},
        "tol": NEWTON_TOL,
        "rel_tol": NEWTON_REL_TOL,
    }


def u_exact(x, y):
    """Exact solution u = sin(πx)sin(πy)."""
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def source_f(x, y):
    """Load f = -Δu + u³ for the manufactured u."""
    u = u_exact(x, y)
    lap_u = -2 * jnp.pi**2 * u
    return -lap_u + u**3


def unit_square_triangle_mesh(nx: int, ny: int):
    """Structured TRI3 mesh on [0,1]²; same connectivity as ``applications.parallel.example``."""
    nvx, nvy = nx + 1, ny + 1
    nvert = nvx * nvy
    coords = onp.zeros((nvert, 2), dtype=onp.float64)
    for i in range(nvx):
        for j in range(nvy):
            k = i * nvy + j
            coords[k, 0] = i / nx
            coords[k, 1] = j / ny
    ncell = 2 * nx * ny
    cells = onp.empty((ncell, 3), dtype=onp.int32)
    t = 0
    for i in range(nx):
        for j in range(ny):
            v00 = i * nvy + j
            v10 = (i + 1) * nvy + j
            v11 = (i + 1) * nvy + (j + 1)
            v01 = i * nvy + (j + 1)
            cells[t] = (v00, v10, v11)
            cells[t + 1] = (v00, v11, v01)
            t += 2
    return coords, cells


class NonlinearManufacturedPoisson(Problem):
    """Weak form: ∫ ∇u·∇v + u³ v - f v = 0, with u = 0 on ∂Ω (Dirichlet)."""

    def get_universal_kernel(self):
        def universal_kernel(
            cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars
        ):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            cell_shape_grads = cell_shape_grads[:, : self.fes[0].num_nodes, :]
            cell_JxW = cell_JxW[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, : self.fes[0].num_nodes, :, :]

            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = jnp.sum(u_grads, axis=1)
            val = jnp.sum(u_grads[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))

            N = self.fes[0].shape_vals
            uq = jnp.sum(cell_sol[None, :, :, None] * N[:, :, None, None], axis=1)
            uq = uq[:, 0]
            nl = uq**3
            val_nl = jnp.sum(
                nl[:, None, None] * N[:, :, None] * cell_JxW[:, None, None], axis=0
            )

            xv, yv = x[:, 0], x[:, 1]
            fu = source_f(xv, yv)
            body = fu[:, None]
            val_body = jnp.sum(
                body[:, None, :] * N[:, :, None] * cell_JxW[:, None, None], axis=0
            )

            out = val + val_nl - val_body
            return jax.flatten_util.ravel_pytree(out)[0]

        return universal_kernel


def dirichlet_bc_unit_square():
    eps = 1e-9
    locs = [
        lambda p: jnp.isclose(p[0], 0.0, atol=eps),
        lambda p: jnp.isclose(p[0], 1.0, atol=eps),
        lambda p: jnp.isclose(p[1], 0.0, atol=eps),
        lambda p: jnp.isclose(p[1], 1.0, atol=eps),
    ]
    return [locs, [0] * 4, [lambda p: 0.0 * p[0]] * 4]


def main():
    nx, ny = GRID_NX, GRID_NY

    t_all0 = time.perf_counter()
    points, cells = unit_square_triangle_mesh(nx, ny)
    t_mesh0 = time.perf_counter()
    mesh = Mesh(points, cells, ele_type="TRI3")
    problem = NonlinearManufacturedPoisson(
        mesh,
        vec=1,
        dim=2,
        ele_type="TRI3",
        dirichlet_bc_info=dirichlet_bc_unit_square(),
        location_fns=None,
    )
    t_mesh1 = time.perf_counter()
    ndof = problem.num_total_dofs_all_vars
    ncell = problem.num_cells

    t_solve0 = time.perf_counter()
    sol_list = solver(problem, solver_options=fair_solver_options())
    t_solve1 = time.perf_counter()

    u = onp.array(sol_list[0][:, 0])
    t_err0 = time.perf_counter()
    # Vectorized RMS vs u_exact = sin(πx)sin(πy) at mesh vertices (same as loop, O(N) NumPy).
    x = points[:, 0]
    y = points[:, 1]
    ue = onp.sin(onp.pi * x) * onp.sin(onp.pi * y)
    err = float(onp.sqrt(onp.mean((u - ue) ** 2)))
    t_err1 = time.perf_counter()
    t_all1 = time.perf_counter()

    print(f"[gold] grid nx={nx} ny={ny}  global_cells={ncell}  global_dofs={ndof}")
    print(
        f"[gold] wall time: total {t_all1 - t_all0:.3f}s  "
        f"mesh+FE_init {t_mesh1 - t_mesh0:.3f}s  "
        f"solver {t_solve1 - t_solve0:.3f}s  "
        f"error_check {t_err1 - t_err0:.3f}s"
    )
    print(f"[gold] RMS error vs exact u: {err:.3e}")
    print(f"[gold] max|u|: {onp.max(onp.abs(u)):.6f} (expect ~1)")


if __name__ == "__main__":
    main()
