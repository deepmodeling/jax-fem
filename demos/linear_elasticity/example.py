import jax.numpy as np
import os
from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh

from jax_am import logger
import logging
logger.setLevel(logging.INFO)


class LinearElasticity(FEM):

    def get_tensor_map(self):

        def stress(u_grad):
            E = 70e3
            nu = 0.3
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(
                self.dim) + 2 * mu * epsilon
            return sigma

        return stress


ele_type = 'TET10'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 10., 2., 2.
Nx, Ny, Nz = 25, 5, 5
meshio_mesh = box_mesh(Nx=Nx,
                       Ny=Ny,
                       Nz=Nz,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


def left(point):
    return np.isclose(point[0], 0., atol=1e-5)


def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)


def zero_dirichlet_val(point):
    return 0.


dirichlet_bc_info = [[left] * 3, [0, 1, 2], [zero_dirichlet_val] * 3]


def neumann_val(point):
    return np.array([0., 0., -100.])


neumann_bc_info = [[right], [neumann_val]]

problem = LinearElasticity(mesh,
                           vec=3,
                           dim=3,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,
                           neumann_bc_info=neumann_bc_info)
sol = solver(problem, linear=True, use_petsc=True)
vtk_path = os.path.join(data_dir, 'vtk/u.vtu')
save_sol(problem, sol, vtk_path)
