"""Forward 3D hyperelastic cube (small deformation)."""

import os

from jax_fem.solver import solver
from jax_fem.utils import save_sol

from applications.scalability.hyperelastic3d_common import (
    Lx,
    Ly,
    Lz,
    build_hyperelastic3d_problem_classic,
)

Nx, Ny, Nz = 80, 80, 80


def main():
    problem, mesh = build_hyperelastic3d_problem_classic(Nx, Ny, Nz)

    solver_options = {"petsc_solver": {}}
    sol_list = solver(problem, solver_options=solver_options)

    u = sol_list[0]
    print(
        f"cube {Lx}x{Ly}x{Lz}, mesh {Nx}x{Ny}x{Nz}, cells={Nx * Ny * Nz}, "
        f"nodes={len(mesh.points)}, max|u|={float(abs(u).max()):.6e}, "
        f"max|uy|={float(abs(u[:, 1]).max()):.6e}"
    )

    out_dir = os.path.join(os.path.dirname(__file__), "output", "vtk")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"u_classic_{Nx}x{Ny}x{Nz}.vtu")
    save_sol(problem.fes[0], u, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
