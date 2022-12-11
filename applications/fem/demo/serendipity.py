import jax
import jax.numpy as np
import numpy as onp
import meshio
import os

from jax_am.fem.models import LinearElasticity
from jax_am.fem.solver import solver
from jax_am.fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_am.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def problem():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    ele_type = 'HEX20'
    cell_type = get_meshio_cell_type(ele_type)
 
    mesh_file = os.path.join(data_dir, f"abaqus/cube.inp")
    meshio_mesh = meshio.read(mesh_file)

    meshio_mesh.points[:, 0] = meshio_mesh.points[:, 0] - onp.min(meshio_mesh.points[:, 0])
    meshio_mesh.points[:, 1] = meshio_mesh.points[:, 1] - onp.min(meshio_mesh.points[:, 1])
    meshio_mesh.points[:, 2] = meshio_mesh.points[:, 2] - onp.min(meshio_mesh.points[:, 2])

    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    Lx = onp.max(mesh.points[:, 0])
    Ly = onp.max(mesh.points[:, 1])
    Lz = onp.max(mesh.points[:, 2])

    print(f"Lx = {Lx}, Ly = {Ly}, Lz = {Lz}")

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return 0.1 * Lx

    dirichlet_bc_info = [[left, left, left, right, right, right], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]
 
    problem = LinearElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    sol = solver(problem, linear=True, precond=True)
    vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
    save_sol(problem, sol, vtk_path, cell_type=cell_type)


if __name__ == "__main__":
    problem()
