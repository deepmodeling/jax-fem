#!/usr/bin/env python3
"""
DMPlex MPI driver: same PDE as ``poisson_gold``; DMPlex + per-rank ``Mesh``, Newton + **CG+GAMG**
(``KSP_*`` from ``poisson_gold``). Rank 0 prints a **short** per-phase time inside the Newton loop
(gather, JAX, matrix→PETSc+``setOperators``, KSP, other).

Run: ``mpiexec -n 4 python -m applications.parallel.poisson_mpi``

**petsc4py:** ``b.copy(r)`` is in-place. **Jacobian:** ``dm.createMatrix()``, then
per Newton: ``setValuesRCV`` for rows owned elsewhere + local CSR ``setValuesCSR`` for
owned rows (``Mat.setValues`` in petsc4py is a dense block, not point COO).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.sparse as sp_sparse
import jax
import jax.numpy as jnp
import jax.flatten_util

jax.config.update("jax_enable_x64", True)

from mpi4py import MPI
import petsc4py

petsc4py.init(sys.argv, comm=MPI.COMM_WORLD)
from petsc4py import PETSc

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import apply_bc_vec

from applications.parallel.poisson_gold import (
    GRID_NX,
    GRID_NY,
    KSP_ATOL,
    KSP_MAX_IT,
    KSP_RTOL,
    NEWTON_MAX_IT,
    NEWTON_REL_TOL,
    NEWTON_TOL,
    NonlinearManufacturedPoisson,
    dirichlet_bc_unit_square,
    unit_square_triangle_mesh,
)

PRINT_NEWTON_RESIDUAL = True

# PETSc ``setFromOptions`` for this KSP: ``<prefix>ksp_*``, ``<prefix>pc_type``
_KSP_OPT_PREFIX = "jxf_"


def _ksp_set_cg_gamg(ksp: PETSc.KSP) -> None:
    p = _KSP_OPT_PREFIX
    opts = PETSc.Options()
    for k, v in {
        f"{p}ksp_type": "cg",
        f"{p}ksp_rtol": str(KSP_RTOL),
        f"{p}ksp_atol": str(KSP_ATOL),
        f"{p}ksp_max_it": str(KSP_MAX_IT),
        f"{p}ksp_error_if_not_converged": "true",
        f"{p}pc_type": "gamg",
    }.items():
        opts.setValue(k, v)
    ksp.setFromOptions()
    ksp.setErrorIfNotConverged(True)


def global_dof(gsection, v):
    off = int(gsection.getOffset(v))
    if off < 0:
        return -off - 1
    return off


def _petsc_int(a):
    return np.asarray(a, dtype=np.int32)


def build_local_mesh_from_dm(dm, gsection, coord_sec, coord_arr):
    c_start, c_end = dm.getHeightStratum(0)
    point_to_local: dict[int, int] = {}
    pts: list[list[float]] = []
    cells_loc: list[list[int]] = []
    for cell in range(c_start, c_end):
        closure_pts, _ = dm.getTransitiveClosure(cell, useCone=True)
        verts = [p for p in closure_pts if dm.getPointDepth(p) == 0]
        lid: list[int] = []
        for v in verts:
            if v not in point_to_local:
                point_to_local[v] = len(pts)
                off = coord_sec.getOffset(v)
                dim = dm.getDimension()
                pts.append([float(coord_arr[off + d]) for d in range(dim)])
            lid.append(point_to_local[v])
        cells_loc.append(lid)
    points = np.asarray(pts, dtype=np.float64)
    cells = np.asarray(cells_loc, dtype=np.int32)
    nln = points.shape[0]
    plex_pt = np.empty(nln, dtype=np.int32)
    for p, i in point_to_local.items():
        plex_pt[i] = p
    gdofs = np.array([global_dof(gsection, int(plex_pt[i])) for i in range(nln)], dtype=np.int32)
    return Mesh(points, cells, ele_type="TRI3"), gdofs


def gather_owned_nodal_values(x, gsection, v_start, v_end, rstart, rend, nglobal, comm):
    xloc = x.array_r
    uv = np.zeros(nglobal, dtype=np.float64)
    for v in range(v_start, v_end):
        gd = global_dof(gsection, v)
        if rstart <= gd < rend:
            uv[gd] = float(xloc[gd - rstart])
    out = np.zeros_like(uv)
    comm.Allreduce(uv, out, op=MPI.SUM)
    return out


def scatter_global_coo(
    mat, rows, cols, data, addv=PETSc.InsertMode.ADD_VALUES, timers=None
):
    """Compact MPIAIJ local COO scatter: not-owned -> setValuesRCV, owned -> setValuesCSR."""
    it, rank = PETSc.IntType, MPI.COMM_WORLD.Get_rank()
    tw = MPI.Wtime
    t0 = tw()
    r, c, v = np.asarray(rows, it).ravel(), np.asarray(cols, it).ravel(), np.asarray(data, np.float64).ravel()
    if r.size == 0: return
    ng, _ = mat.getSize()
    rstart, rend = mat.getOwnershipRange()
    nlocal = rend - rstart
    if nlocal == 0:
        if timers and rank == 0: timers["prep"] = timers.get("prep", 0.0) + (tw() - t0)
        return
    t4 = tw()
    own = (r >= rstart) & (r < rend)
    # Off-rows (need RCV, always in COO)
    t_off0 = tw()
    if np.any(~own):
        Sn = sp_sparse.coo_matrix((v[~own], (r[~own], c[~own])), shape=(ng, ng))
        Sn.sum_duplicates()
        Sn.eliminate_zeros()
        if Sn.nnz:
            n = Sn.row.size
            mat.setValuesRCV(
                Sn.row.astype(it).reshape(n, 1),
                Sn.col.astype(it).reshape(n, 1),
                Sn.data.reshape(n, 1), addv=addv
            )
    t_off1 = tw()
    # Owned rows (use efficient CSR scatter if possible)
    t_b0 = tw()
    if np.any(own):
        r0, c0, v0 = r[own], c[own], v[own]
        S = sp_sparse.coo_matrix((v0, ((r0 - rstart).astype(it), c0)), shape=(nlocal, ng))
        S.sum_duplicates()
        S.eliminate_zeros()
        S = S.tocsr()
        t6 = tw()
        if S.nnz:
            mat.setValuesCSR(
                S.indptr.astype(it),
                S.indices.astype(it),
                S.data, addv=addv)
        t7 = tw()
    else:
        t6 = t7 = tw()
    if timers and rank == 0:
        timers["prep"] = timers.get("prep", 0.0) + (t4 - t0)
        timers["setvalues_offrow"] = timers.get("setvalues_offrow", 0.0) + (t_off1 - t_off0)
        timers["coo2csr_owned"] = timers.get("coo2csr_owned", 0.0) + (tw() - t_b0)
        timers["setvalues_csr"] = timers.get("setvalues_csr", 0.0) + (t7 - t6)


def scatter_rhs_vec(vec, gdofs: np.ndarray, res_np: np.ndarray, addv=PETSc.InsertMode.ADD_VALUES) -> None:
    vec.setValues(
        np.asarray(gdofs, np.int32).ravel(), np.asarray(res_np, np.float64).ravel(), addv=addv
    )


def rms_versus_u_exact_on_owned_vertices(comm, dm, gsection, coord_sec, coord_arr, v_start, v_end, rstart, rend, xloc):
    """RMS of ``u - sin(πx)sin(πy)`` on owned depth-0 dofs (``poisson_gold`` manufactured solution)."""
    v = np.arange(v_start, v_end, dtype=np.int32)
    d = int(dm.getDimension())
    g = np.fromiter(
        (global_dof(gsection, int(p)) for p in v), np.int32, count=len(v)
    )
    o = np.fromiter(
        (coord_sec.getOffset(int(p)) for p in v), np.intp, count=len(v)
    )
    x = np.column_stack([coord_arr[o + j] for j in range(d)])
    m = (g >= rstart) & (g < rend)
    e = xloc[g[m] - rstart] - (np.sin(np.pi * x[m, 0]) * np.sin(np.pi * x[m, 1]))
    g_sq = comm.allreduce(float(np.dot(e, e)), op=MPI.SUM)
    g_n = comm.allreduce(int(m.sum()), op=MPI.SUM)
    return (g_sq / g_n) ** 0.5 if g_n else 0.0


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nx, ny = GRID_NX, GRID_NY

    t0 = MPI.Wtime()
    coords, cells = unit_square_triangle_mesh(nx, ny)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, comm=comm)
    dm.distribute(overlap=0)
    dm.setUp()

    v_start, v_end = dm.getDepthStratum(0)

    section = PETSc.Section().create()
    section.setChart(v_start, v_end)
    for v in range(v_start, v_end):
        section.setDof(v, 1)
    section.setUp()
    dm.setLocalSection(section)
    dm.setUp()
    gsection = dm.getGlobalSection()

    coord_sec = dm.getCoordinateSection()
    cvec = dm.getCoordinatesLocal()
    coord_arr = cvec.array_r

    mesh, gdofs = build_local_mesh_from_dm(dm, gsection, coord_sec, coord_arr)
    problem = NonlinearManufacturedPoisson(
        mesh,
        vec=1,
        dim=2,
        ele_type="TRI3",
        dirichlet_bc_info=dirichlet_bc_unit_square(),
        location_fns=None,
    )

    x = dm.createGlobalVec()
    b = dm.createGlobalVec()
    rstart, rend = x.getOwnershipRange()
    nglobal = int(np.asarray(x.getSize()).ravel()[0])

    mat = dm.createMatrix()
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)

    boundary_owned: list[int] = []
    for v in range(v_start, v_end):
        off = coord_sec.getOffset(v)
        xy = np.array([coord_arr[off + d] for d in range(dm.getDimension())], dtype=np.float64)
        if (
            abs(xy[0]) < 1e-9
            or abs(xy[0] - 1.0) < 1e-9
            or abs(xy[1]) < 1e-9
            or abs(xy[1] - 1.0) < 1e-9
        ):
            gd = global_dof(gsection, v)
            if rstart <= gd < rend:
                boundary_owned.append(gd)
    boundary_owned = sorted(set(boundary_owned))

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(mat, mat)
    ksp.setOptionsPrefix(_KSP_OPT_PREFIX)
    _ksp_set_cg_gamg(ksp)

    t_setup = MPI.Wtime()
    x.set(0.0)
    du = dm.createGlobalVec()

    t_newton0 = MPI.Wtime()
    rnorm0 = None
    n_linear_solves = 0
    n_outer_newton = 0
    t_gather = t_jaxfe = t_mat = t_ksp = t_other = 0.0
    t_scat: dict | None = {} if rank == 0 else None
    for it in range(NEWTON_MAX_IT):
        n_outer_newton += 1
        if rank == 0:
            _t0 = MPI.Wtime()
        u_full = gather_owned_nodal_values(x, gsection, v_start, v_end, rstart, rend, nglobal, comm)
        if rank == 0:
            t_gather += MPI.Wtime() - _t0

        if rank == 0:
            _t0 = MPI.Wtime()
        u_loc = u_full[gdofs]
        sol_list = [jnp.asarray(u_loc).reshape(-1, 1)]
        dofs_flat = jax.flatten_util.ravel_pytree(sol_list)[0]

        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs_flat, problem)
        res_np = np.asarray(res_vec, dtype=np.float64)

        if rank == 0:
            t_jaxfe += MPI.Wtime() - _t0

        if rank == 0:
            _t0 = MPI.Wtime()
        b.set(0.0)
        scatter_rhs_vec(b, gdofs, res_np, addv=PETSc.InsertMode.ADD_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()

        g_rows = gdofs[problem.I.astype(np.int64)]
        g_cols = gdofs[problem.J.astype(np.int64)]
        mat.zeroEntries()
        scatter_global_coo(mat, g_rows, g_cols, problem.V, timers=t_scat)
        mat.assemblyBegin()
        mat.assemblyEnd()

        if boundary_owned:
            mat.zeroRows(_petsc_int(boundary_owned), diag=1.0)
            b.setValues(
                _petsc_int(boundary_owned),
                np.zeros(len(boundary_owned), dtype=np.float64),
                addv=PETSc.InsertMode.INSERT_VALUES,
            )
        mat.assemblyBegin()
        mat.assemblyEnd()
        b.assemblyBegin()
        b.assemblyEnd()
        ksp.setOperators(mat, mat)
        if rank == 0:
            t_mat += MPI.Wtime() - _t0

        if rank == 0:
            _t0 = MPI.Wtime()
        norm_r = b.norm()
        if rnorm0 is None:
            rnorm0 = norm_r
        rel_r = norm_r / rnorm0 if rnorm0 else 0.0
        if rank == 0:
            t_other += MPI.Wtime() - _t0
        if rank == 0 and PRINT_NEWTON_RESIDUAL:
            print(
                f"[mpi] Newton {it:2d}  ||R|| (after BC) = {norm_r:.3e}  "
                f"rel||R||/||R0|| = {rel_r:.3e}"
            )
        if not ((rel_r > NEWTON_REL_TOL) and (norm_r > NEWTON_TOL)):
            break

        if rank == 0:
            _t0 = MPI.Wtime()
        r_work = b.duplicate()
        b.copy(r_work)
        r_work.scale(-1.0)
        du.set(0.0)
        ksp.solve(r_work, du)
        if rank == 0:
            t_ksp += MPI.Wtime() - _t0
        if ksp.getConvergedReason() < 0:
            if rank == 0:
                print(f"[mpi] KSP diverged, reason = {ksp.getConvergedReason()}")
            break
        n_linear_solves += 1
        if rank == 0:
            _t0 = MPI.Wtime()
        x.axpy(1.0, du)
        if rank == 0:
            t_other += MPI.Wtime() - _t0

    t_newton1 = MPI.Wtime()
    t_err0 = MPI.Wtime()
    xloc = x.array_r
    rms = rms_versus_u_exact_on_owned_vertices(
        comm, dm, gsection, coord_sec, coord_arr, v_start, v_end, rstart, rend, xloc
    )
    t_err1 = MPI.Wtime()
    t1 = MPI.Wtime()
    if rank == 0:
        ncell_glob = 2 * nx * ny
        print(
            f"[mpi] grid nx={nx} ny={ny}  global_cells={ncell_glob}  global_dofs={nglobal}  "
            f"MPI_ranks={comm.Get_size()}"
        )
        print(
            f"[mpi] wall time (MPI.Wtime, rank 0): total {t1 - t0:.3f}s  "
            f"setup {t_setup - t0:.3f}s  "
            f"newton_loop {t_newton1 - t_newton0:.3f}s  "
            f"error_check {t_err1 - t_err0:.3f}s  "
            f"newton_linear_solves={n_linear_solves}"
        )
        newton_wall = t_newton1 - t_newton0
        accounted = t_gather + t_jaxfe + t_mat + t_ksp + t_other
        unacc = max(0.0, newton_wall - accounted)

        def _pct(t: float) -> float:
            return 100.0 * t / newton_wall if newton_wall > 0 else 0.0

        print(
            f"[mpi] newton_phases  rank0_sum  (outer_iters={n_outer_newton}  % of newton_loop):"
        )
        for name, tsec in (
            ("gather", t_gather),
            ("jaxfe", t_jaxfe),
            ("mat+setKSP", t_mat),
            ("ksp_solve", t_ksp),
            ("other", t_other),
        ):
            print(f"[mpi]   {name:12s}  {tsec:8.3f}s  {_pct(tsec):5.1f}%")
        if unacc > 0.01:
            print(f"[mpi]   {'unacc':12s}  {unacc:8.3f}s  {_pct(unacc):5.1f}%  (newton wall minus sum of phases)")
        print(f"[mpi]   {'newton_wall':12s}  {newton_wall:8.3f}s  100%")
        if t_scat is not None:
            order = (
                ("prep", "ravel+ownership+split"),
                ("setvalues_offrow", "Mat.setValuesRCV (not-owned rows, MPI route)"),
                ("coo2csr_owned", "coo+sum_dup+tocsr (nlocal×ng)"),
                ("setvalues_csr", "Mat.setValuesCSR (owned rows)"),
            )
            ssum = sum(t_scat.get(k, 0.0) for k, _ in order)
            print(
                f"[mpi] scatter_global_coo  rank0  sum over outer iters  "
                f"(sub-phases, s, % of scatter subtotal {ssum:.3f}s):"
            )
            for k, label in order:
                sec = t_scat.get(k, 0.0)
                p = 100.0 * sec / ssum if ssum > 0 else 0.0
                print(f"[mpi]   {label:32s}  {sec:8.3f}  {p:5.1f}%")
        print(f"[mpi] RMS vs exact (owned verts): {rms:.3e}")


if __name__ == "__main__":
    main()
