#!/usr/bin/env python3
"""
FEniCSx (dolfinx) **parallel** driver for the same manufactured nonlinear Poisson problem as
``poisson_gold.py`` / ``poisson_mpi.py``.

Strong form on :math:`\\Omega = (0,1)^2`::

    -\\Delta u + u^3 = f,  \\qquad u|_{\\partial\\Omega} = 0.

Manufactured solution :math:`u(x,y) = \\sin(\\pi x)\\sin(\\pi y)`, hence

    f = 2\\pi^2 \\sin(\\pi x)\\sin(\\pi y) + (\\sin(\\pi x)\\sin(\\pi y))^3.

Weak form (find :math:`u \\in V` with :math:`u=0` on :math:`\\partial\\Omega`)::

    \\int_\\Omega \\nabla u \\cdot \\nabla v + u^3 v - f v \\,\\mathrm{d}x = 0
    \\quad \\forall v \\in V.

Discretization: ``P1`` (Lagrange degree 1) on a structured triangle mesh from
``mesh.create_unit_square`` (two triangles per macro cell), comparable in spirit to the
``TRI3`` grid in the JAX-FEM gold script.

**Solver:** ``dolfinx.fem.petsc.NonlinearProblem`` (PETSc SNES ``newtonls``, no line search).
Inner linear solves: **CG + GAMG**. Serial runs use **direct LU** on the GAMG coarse level;
``mpiexec -n > 1`` uses **redundant** + LU for robustness.

**Outer Newton vs gold:** PETSc SNES declares convergence when **either** ``||F|| < snes_atol``
**or** ``||F||/||F_0|| < snes_rtol`` (plus optional step tests). Shrinking ``snes_atol`` alone
does nothing if the **relative** test already passes: with ``snes_rtol = NEWTON_REL_TOL`` you
can still stop at iteration 4 while ``||F||`` is tiny. Set ``SNES_FNORM_RTOL`` much **smaller**
than ``NEWTON_REL_TOL`` (this file defaults to ``1e-30``) so the relative test does not fire
first; then a stricter ``SNES_FNORM_ATOL`` (or ``NEWTON_TOL``) controls when SNES stops.
``snes_stol = -1`` disables the relative **step** tolerance path.

**Fair comparison with** ``poisson_gold.py``: match ``GRID_NX`` / ``GRID_NY`` and the Newton /
KSP names below (duplicated on purpose so this module does not import JAX).

Run from repo root::

    mpiexec -n 4 python -m applications.parallel.poisson_fenicsx

Serial::

    python -m applications.parallel.poisson_fenicsx

Only rank 0 prints wall times and the global :math:`L^2(\\Omega)` error; all ranks participate
in the solve.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import ufl
from mpi4py import MPI

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem

# --- Same names as ``applications/parallel/poisson_gold.py`` (no JAX import). ------------
GRID_NX = 500
GRID_NY = 500

NEWTON_TOL = 1e-10
NEWTON_REL_TOL = 1e-10
NEWTON_MAX_IT = 10_000

PETSC_KSP_TYPE = "cg"
KSP_RTOL = 1e-5
KSP_MAX_IT = 10_000

# SNES function-norm limits (see module docstring). ``None`` -> inherit ``NEWTON_*``.
# Keep ``SNES_FNORM_RTOL`` << ``NEWTON_REL_TOL`` unless you want PETSc's *relative* ``||F||/||F_0||``
# test to stop the solve before ``snes_atol`` matters (that was why atol=1e-20 still gave 4 iters).
SNES_FNORM_ATOL: float | None = None
SNES_FNORM_RTOL: float | None = 1e-30
# -------------------------------------------------------------------------------------------


def petsc_options(comm: MPI.Comm) -> dict[str, str | None]:
    snes_atol = NEWTON_TOL if SNES_FNORM_ATOL is None else SNES_FNORM_ATOL
    snes_rtol = NEWTON_REL_TOL if SNES_FNORM_RTOL is None else SNES_FNORM_RTOL
    opts: dict[str, str | None] = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_atol": str(snes_atol),
        "snes_rtol": str(snes_rtol),
        "snes_stol": "-1",
        "snes_max_it": "5",                       # 改为 5
        "snes_convergence_test": "skip",          # 跳过收敛测试
        "snes_error_if_not_converged": "true",
        "ksp_type": PETSC_KSP_TYPE,
        "ksp_rtol": str(KSP_RTOL),
        "ksp_max_it": str(KSP_MAX_IT),
        "ksp_error_if_not_converged": "true",
        "pc_type": "gamg",
        "mg_coarse_ksp_type": "preonly",
    }


    if comm.size == 1:
        opts["mg_coarse_pc_type"] = "lu"
    else:
        opts["mg_coarse_pc_type"] = "redundant"
        opts["mg_coarse_redundant_pc_type"] = "lu"
    return opts


def unit_square_boundary(x: np.ndarray) -> np.ndarray:
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)),
    )


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nx, ny = GRID_NX, GRID_NY

    t_all0 = MPI.Wtime()

    t_mesh0 = MPI.Wtime()
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, unit_square_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    x_coord = ufl.SpatialCoordinate(domain)
    u_ex = ufl.sin(ufl.pi * x_coord[0]) * ufl.sin(ufl.pi * x_coord[1])
    f = 2 * ufl.pi**2 * u_ex + u_ex**3

    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u**3 * v * ufl.dx - f * v * ufl.dx

    n_cells_global = domain.topology.index_map(domain.topology.dim).size_global
    n_dofs_global = V.dofmap.index_map.size_global
    t_mesh1 = MPI.Wtime()

    opts = petsc_options(comm)
    problem = NonlinearProblem(
        F,
        u,
        bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix="nl_poisson",
    )

    t_solve0 = MPI.Wtime()
    problem.solve()
    t_solve1 = MPI.Wtime()

    snes = problem.solver
    snes_iters = snes.getIterationNumber()
    snes_reason = snes.getConvergedReason()
    ksp_linear_its = snes.getLinearSolveIterations()

    t_err0 = MPI.Wtime()
    err_l2_sq_local = fem.assemble_scalar(fem.form((u - u_ex) ** 2 * ufl.dx))
    err_l2 = float(np.sqrt(comm.allreduce(err_l2_sq_local, op=MPI.SUM)))
    t_err1 = MPI.Wtime()
    t_all1 = MPI.Wtime()

    u_max_local = np.max(np.abs(u.x.array)) if u.x.array.size else 0.0
    u_max = float(comm.allreduce(u_max_local, op=MPI.MAX))

    if rank == 0:
        print(
            f"[fenicsx] mpi_size={comm.size}  grid nx={nx} ny={ny}  "
            f"global_cells={n_cells_global}  global_dofs={n_dofs_global}"
        )
        print(
            f"[fenicsx] wall time (MPI.Wtime): total {t_all1 - t_all0:.3f}s  "
            f"mesh+FE_init {t_mesh1 - t_mesh0:.3f}s  "
            f"solve {t_solve1 - t_solve0:.3f}s  "
            f"error_check {t_err1 - t_err0:.3f}s"
        )
        print(
            f"[fenicsx] SNES: iterations={snes_iters}  "
            f"linear_KSP_its_total={ksp_linear_its}  "
            f"converged_reason={snes_reason}  (PETSc: positive => success)"
        )
        print(f"[fenicsx] L2 error ||u - u_ex||: {err_l2:.3e}")
        print(f"[fenicsx] max|u| (global): {u_max:.6f} (expect ~1)")

    if snes_reason <= 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
