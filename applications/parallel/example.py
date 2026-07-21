#!/usr/bin/env python3
"""
DMPlex + JAX tutorial: Poisson problem -Δu = f on [0,1]×[0,1] with u = 0 on the boundary.

Mesh: structured triangles built from explicit vertex coordinates and cell connectivity,
then loaded into a ``DMPlex`` via ``createFromCellList``. We avoid
``createBoxMesh(..., simplex=True)`` because that path requires PETSc built with Triangle
(``--download-triangle``), which many conda builds omit.

The mesh is **partitioned** across ``MPI.COMM_WORLD`` via ``DMPlex.distribute``; assembly uses
``DM`` global section dof indices so ``Mat`` / ``Vec`` are the standard MPI parallel objects.

Run from the JAX-FEM repository root::

    python -m applications.parallel.example

Parallel (same module, partitioned mesh + parallel KSP)::

    mpiexec -n 4 python -m applications.parallel.example

Using ``python -m`` prepends the current working directory to ``sys.path`` before resolving
the package name. If you launch the script by path from another directory, the block below
still inserts the repo root so imports like ``jax_fem`` resolve correctly.

Analytical solution (verification)
----------------------------------
We choose a manufactured solution that satisfies homogeneous Dirichlet data on the unit
square and match the load ``f`` to it.

Let::

    u(x, y) = sin(π x) sin(π y).

On ``∂Ω``, either ``x ∈ {0,1}`` or ``y ∈ {0,1}``, so ``u = 0`` on the boundary.

In two dimensions, ``Δu = ∂²u/∂x² + ∂²u/∂y²``::

    ∂²u/∂x² = -π² sin(π x) sin(π y),
    ∂²u/∂y² = -π² sin(π x) sin(π y),

hence ``Δu = -2π² u`` and therefore ``-Δu = 2π² sin(π x) sin(π y)``. The code sets
``f`` to this expression, so ``u`` above is the **exact** solution of the discrete weak
problem only up to quadrature / P1 approximation error on the load.

VTK output (ParaView)
---------------------
Like other examples under ``applications/`` (e.g. ``nodal_stress``), results are written
with ``meshio`` under ``applications/parallel/output/vtk/`` as ``u.vtu``. Point fields:
``u`` (FEM), ``u_exact``, ``u_error``, and ``owner_rank`` (MPI rank that owns that dof row).
Cell field ``partition`` is the MPI rank that **owns the triangle** for assembly. In
ParaView use **Color by** ``partition`` (Cell Data) or ``owner_rank`` (Point Data) to see
how ``dm.distribute`` split the domain across ranks.

Code layout
-----------
Sections in the source are marked explicitly: **core** items participate in the discrete
Poisson solve (mesh, assembly, BCs, KSP); **post-process** covers verification (RMS error)
and **visualization-only** paths (``meshio`` VTU, ``partition`` / ``owner_rank`` fields).

Post-process on rank 0 uses a **small dict keyed by rounded ``(x,y)``** to merge gathered
nodal values and cell-centroid ranks onto the VTU ``coords`` / ``cells`` (rough but short;
PETSc global dof order after ``distribute`` need not match VTK vertex rows).
"""
from pathlib import Path
import os
import sys

_repo_root = Path(__file__).resolve().parents[2]
_root = str(_repo_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import jax.numpy as jnp
import jax
import meshio
from mpi4py import MPI
import petsc4py

# First positional arg is *argv* (must be iterable), not the communicator.
petsc4py.init(sys.argv, comm=MPI.COMM_WORLD)
from petsc4py import PETSc


# =============================================================================
# Core FEM: P1 triangle stiffness / load (used only inside the assembly loop)
# =============================================================================
def element_stiffness(vertices):
    """
    Args:
        vertices: (3, 2) vertex coordinates in physical space.

    Returns:
        (3, 3) element stiffness matrix for the Poisson bilinear form.
    """
    v0, v1, v2 = vertices
    # Triangle area (cross product / 2)
    area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]))
    # Shape function gradients on the reference triangle with nodes (0,0), (1,0), (0,1)
    ref_grads = jnp.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    # Jacobian of the affine map from reference to physical triangle
    J = jnp.array(
        [
            [v1[0] - v0[0], v2[0] - v0[0]],
            [v1[1] - v0[1], v2[1] - v0[1]],
        ]
    )
    invJ = jnp.linalg.inv(J)
    # Physical gradients (rows): grad_x N_i = (grad_ξ N_i) @ J^{-1} with J = dx/dξ.
    # Do not use J^{-T} on the left of grad_ξ rows; that would be the wrong chain rule.
    grads = ref_grads @ invJ
    K = area * (grads @ grads.T)
    return K


def element_load(vertices, f):
    """
    Args:
        vertices: (3, 2) vertex coordinates in physical space.
        f: scalar source f(x, y).

    Returns:
        (3,) element load vector; f is sampled once at the centroid and lumped equally.
    """
    centroid = jnp.mean(vertices, axis=0)
    f_center = f(centroid[0], centroid[1])
    area = 0.5 * abs(
        (vertices[1, 0] - vertices[0, 0]) * (vertices[2, 1] - vertices[0, 1])
        - (vertices[1, 1] - vertices[0, 1]) * (vertices[2, 0] - vertices[0, 0])
    )
    w = area / 3.0 * f_center
    return jnp.array([w, w, w])


def _petsc_scalar_array(a):
    """PETSc Mat/Vec setValues expect NumPy (or list), not JAX arrays."""
    return np.asarray(jnp.asarray(a), dtype=np.float64)


def _petsc_int_array(a):
    """PETSc APIs expect plain NumPy int arrays for index lists."""
    return np.asarray(a, dtype=np.int32)


# -----------------------------------------------------------------------------
# Core: PETSc helpers (assembly / BCs need global dof indices on distributed DM)
# -----------------------------------------------------------------------------
def global_dof(gsection, v):
    """Global dof index for mesh point ``v`` (PETSc encodes ghosts as ``-(g+1)``)."""
    off = int(gsection.getOffset(v))
    if off < 0:
        return -off - 1
    return off


# -----------------------------------------------------------------------------
# Post-process only: round coordinates → dict keys (simple VTK merge on rank 0)
# -----------------------------------------------------------------------------
def xykey(x, y):
    return round(float(x), 10), round(float(y), 10)


# -----------------------------------------------------------------------------
# Core: build serial connectivity (every rank holds the same arrays; PETSc partitions)
# -----------------------------------------------------------------------------
def unit_square_triangle_mesh(nx, ny, lower=(0.0, 0.0), upper=(1.0, 1.0)):
    """
    Build an ``(nx+1)×(ny+1)`` vertex grid on ``[lower, upper]`` and split each macro-quad
    into two triangles (consistent diagonals).

    This avoids ``DMPlex.createBoxMesh(..., simplex=True)``, which needs PETSc linked to
    Triangle (``--download-triangle``), often absent in conda-forge builds.
    """
    lx, ly = float(lower[0]), float(lower[1])
    ux, uy = float(upper[0]), float(upper[1])
    nvx, nvy = nx + 1, ny + 1
    nvert = nvx * nvy
    coords = np.empty((nvert, 2), dtype=np.float64)
    for i in range(nvx):
        for j in range(nvy):
            k = i * nvy + j
            coords[k, 0] = lx + (ux - lx) * (i / nx)
            coords[k, 1] = ly + (uy - ly) * (j / ny)
    ncell = 2 * nx * ny
    cells = np.empty((ncell, 3), dtype=np.int32)
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


# =============================================================================
# Visualization only: meshio → VTU (ParaView). No effect on the discrete solution.
# =============================================================================
def save_poisson_vtu(
    coords, cells, u_fem, vtk_path, cell_partition=None, point_owner_rank=None
):
    """
    Write a VTU file for ParaView (same pattern as ``jax_fem.utils.save_sol`` / other
    ``applications/*/example.py`` scripts that use ``meshio``).
    """
    vtk_dir = os.path.dirname(vtk_path)
    os.makedirs(vtk_dir, exist_ok=True)
    n = coords.shape[0]
    points = np.zeros((n, 3), dtype=np.float64)
    points[:, :2] = np.asarray(coords, dtype=np.float64)
    u_ex = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    point_data = {
        "u": np.asarray(u_fem, dtype=np.float32).reshape(n, 1),
        "u_exact": np.asarray(u_ex, dtype=np.float32).reshape(n, 1),
        "u_error": np.asarray(u_fem - u_ex, dtype=np.float32).reshape(n, 1),
    }
    if point_owner_rank is not None:
        point_data["owner_rank"] = np.asarray(point_owner_rank, dtype=np.float32).reshape(
            n, 1
        )
    cell_data = {}
    if cell_partition is not None:
        cell_data["partition"] = [np.asarray(cell_partition, dtype=np.float32)]
    out = meshio.Mesh(
        points=points,
        cells=[("triangle", np.asarray(cells, dtype=np.int32))],
        point_data=point_data,
        cell_data=cell_data,
    )
    out.write(vtk_path)


# =============================================================================
# Driver: core parallel solve, then optional post-process (error + VTK on rank 0)
# =============================================================================
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nx = ny = 20
    coords, cells = unit_square_triangle_mesh(nx, ny, lower=(0, 0), upper=(1, 1))
    # --- Core: DMPlex from full cell list, then partition across MPI ranks ---
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, comm=comm)
    dm.distribute(overlap=0)
    dm.setUp()

    v_start, v_end = dm.getDepthStratum(0)  # local vertex point ids
    c_start, c_end = dm.getHeightStratum(0)  # local triangle cell ids
    print(
        f"[rank {rank}] depth stratum 0 (vertices): v_start={v_start}  v_end={v_end}  "
        f"(local vertex count = {v_end - v_start})",
        flush=True,
    )

    # --- Core: one scalar dof per mesh vertex; local + global sections for Mat/Vec ---
    section = PETSc.Section().create()
    section.setChart(v_start, v_end)
    for v in range(v_start, v_end):
        section.setDof(v, 1)
    section.setUp()
    dm.setLocalSection(section)
    dm.setUp()
    gsection = dm.getGlobalSection()

    mat = dm.createMatrix()
    x = dm.createGlobalVec()
    b = dm.createGlobalVec()
    rstart, rend = mat.getOwnershipRange()

    if rank == 0:
        nglobal, _ = mat.getSize()
        print(f"Mesh topological dimension: {dm.getDimension()}D")
        print(f"MPI ranks: {comm.Get_size()}  global dofs: {nglobal}")
        print(f"Global cells: {2 * nx * ny}  (each rank owns a subset)")

    # --- Core: read vertex coordinates from the distributed DM (not the numpy coords) ---
    coord_sec = dm.getCoordinateSection()
    cvec = dm.getCoordinatesLocal()
    coord_arr = cvec.array_r

    def vertex_xy(v):
        off = coord_sec.getOffset(v)
        dim = dm.getDimension()
        return np.array([coord_arr[off + d] for d in range(dim)], dtype=np.float64)

    # Body force f = -Δu with u = sin(πx) sin(πy); see module docstring.
    def f_func(x, y):
        return 2 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    # --- Core: parallel Galerkin assembly (each rank touches only its local cells) ---
    for cell in range(c_start, c_end):
        closure_pts, _ = dm.getTransitiveClosure(cell, useCone=True)
        vertices_idx = [p for p in closure_pts if dm.getPointDepth(p) == 0]
        if len(vertices_idx) != 3:
            raise RuntimeError(f"Expected 3 mesh vertices on cell {cell}, got {len(vertices_idx)}")
        vert_coords = jnp.array([vertex_xy(v) for v in vertices_idx])
        dofs = np.array([global_dof(gsection, v) for v in vertices_idx], dtype=np.int32)

        K_e = element_stiffness(vert_coords)
        b_e = element_load(vert_coords, f_func)
        mat.setValues(dofs, dofs, _petsc_scalar_array(K_e), addv=PETSc.InsertMode.ADD_VALUES)
        b.setValues(dofs, _petsc_scalar_array(b_e), addv=PETSc.InsertMode.ADD_VALUES)

    mat.assemblyBegin()
    mat.assemblyEnd()
    b.assemblyBegin()
    b.assemblyEnd()

    # --- Core: Dirichlet u=0 on boundary; only zero *owned* rows (avoid ghost duplicates) ---
    boundary_dofs_set = set()
    for v in range(v_start, v_end):
        coord = vertex_xy(v)
        if (
            abs(coord[0]) < 1e-8
            or abs(coord[0] - 1) < 1e-8
            or abs(coord[1]) < 1e-8
            or abs(coord[1] - 1) < 1e-8
        ):
            gd = global_dof(gsection, v)
            if rstart <= gd < rend:
                boundary_dofs_set.add(gd)
    boundary_dofs = sorted(boundary_dofs_set)
    if boundary_dofs:
        mat.zeroRows(_petsc_int_array(boundary_dofs), diag=1.0)
        for dof in boundary_dofs:
            b.setValue(dof, 0.0)
    mat.assemblyBegin()
    mat.assemblyEnd()
    b.assemblyBegin()
    b.assemblyEnd()

    # --- Core: parallel linear solve ---
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(mat)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    ksp.setFromOptions()
    ksp.solve(b, x)

    # =========================================================================
    # Post-process (I/O and checks only; the PDE solve is complete above.)
    # =========================================================================

    # --- Verification: RMS error vs exact u (manufactured solution; module docstring) ---
    u_exact = lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    xloc = x.array_r
    nglobal, _ = mat.getSize()
    # Gather owned nodal data keyed by rounded (x,y); rank 0 hashes onto VTK ``coords``.
    owned = []
    for v in range(v_start, v_end):
        gd = global_dof(gsection, v)
        if not (rstart <= gd < rend):
            continue
        xy = vertex_xy(v)
        kx, ky = xykey(xy[0], xy[1])
        owned.append((kx, ky, float(xloc[gd - rstart]), rank))
    all_owned = comm.gather(owned, root=0)

    cell_owner = []
    for cell in range(c_start, c_end):
        closure_pts, _ = dm.getTransitiveClosure(cell, useCone=True)
        vertices_idx = [p for p in closure_pts if dm.getPointDepth(p) == 0]
        vc = np.array([vertex_xy(v) for v in vertices_idx], dtype=np.float64)
        cx, cy = float(vc[:, 0].mean()), float(vc[:, 1].mean())
        cell_owner.append((*xykey(cx, cy), rank))
    all_cell_owner = comm.gather(cell_owner, root=0)

    # --- Rank 0 only: build VTK-aligned arrays, print error, write VTU ---
    if rank == 0:
        pos_to_u = {}
        pos_to_owner = {}
        for chunk in all_owned:
            for kx, ky, val, r in chunk:
                pos_to_u[(kx, ky)] = val
                pos_to_owner[(kx, ky)] = r
        nvert = coords.shape[0]
        u_vtk = np.empty(nvert, dtype=np.float64)
        owner_vtk = np.empty(nvert, dtype=np.float32)
        for k in range(nvert):
            key = xykey(coords[k, 0], coords[k, 1])
            u_vtk[k] = pos_to_u[key]
            owner_vtk[k] = pos_to_owner[key]
        error = 0.0
        for (kx, ky), val in pos_to_u.items():
            ue = float(u_exact(kx, ky))
            error += (val - ue) ** 2
        error = np.sqrt(error / len(pos_to_u))
        print(f"Root-mean-square vertex error vs exact u: {error:.2e}")

        centroid_to_rank = {}
        for chunk in all_cell_owner:
            for kx, ky, r in chunk:
                centroid_to_rank[(kx, ky)] = r
        ncell = cells.shape[0]
        part = np.empty(ncell, dtype=np.float32)
        for ci in range(ncell):
            i, j, kk = cells[ci]
            c = (coords[i] + coords[j] + coords[kk]) / 3.0
            part[ci] = centroid_to_rank[xykey(c[0], c[1])]

        data_dir = os.path.join(os.path.dirname(__file__), "output")
        vtk_path = os.path.join(data_dir, "vtk", "u.vtu")
        save_poisson_vtu(
            coords, cells, u_vtk, vtk_path, cell_partition=part, point_owner_rank=owner_vtk
        )
        print(f"Wrote VTK for ParaView: {vtk_path}")
        print("Cell data 'partition' and point data 'owner_rank' show MPI distribution.")
        print("Done.")


if __name__ == "__main__":
    main()
