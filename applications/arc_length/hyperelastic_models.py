"""Shared mesh and material for PETSc debug beam examples."""

import jax
import jax.numpy as np

from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

Lx = 50.
Ly = 2.
Nx = 50
Ny = 2
ELE_TYPE = 'QUAD4'


def make_mesh(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny):
    cell_type = get_meshio_cell_type(ELE_TYPE)
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    return Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


def location_left(point):
    """Left edge x = 0 (1-argument fn for Dirichlet / surface location)."""
    return np.isclose(point[0], 0., atol=1e-5)


def location_right(point):
    """Right edge x = Lx (1-argument fn for Dirichlet / surface location)."""
    return np.isclose(point[0], Lx, atol=1e-5)


class BeamHyperelastic(Problem):
    """Neo-Hookean hyperelastic beam (volume energy only)."""

    def get_tensor_map(self):
        def psi(F):
            E = 1e3
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 2.)
            I1 = np.trace(F.T @ F)
            return (mu / 2.) * (Jinv * I1 - 2.) + (kappa / 2.) * (J - 1.)**2.

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            F = u_grad + np.eye(self.dim)
            return P_fn(F)

        return first_PK_stress
