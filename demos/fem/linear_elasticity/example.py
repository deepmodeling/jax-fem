import jax
import jax.numpy as np
import os

from jax_am.fem.models import LinearElasticity
from jax_am.fem.solver import solver
from jax_am.fem.generate_mesh import Mesh, box_mesh
from jax_am.fem.utils import save_sol


def problem():
    """Can be used to test the memory limit of JAX-FEM
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    ele_type = 'HEX8'
 
    meshio_mesh = box_mesh(10, 10, 10, 1., 1., 1., data_dir, ele_type=ele_type)
    # meshio_mesh = box_mesh(100, 100, 100, 1., 1., 1., data_dir)
    # meshio_mesh = box_mesh(300, 100, 100, 1., 1., 1., data_dir)

    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return 0.1

    dirichlet_bc_info = [[left, left, left, right, right, right], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]
 
    problem = LinearElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    sol = solver(problem, linear=True, precond=True)
    vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
    save_sol(problem, sol, vtk_path)


if __name__ == "__main__":
    problem()
