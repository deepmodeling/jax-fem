import jax
import jax.numpy as np
import numpy as onp
import meshio
import os
import glob

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress


def problem():
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    ele_type = 'HEX20'
    cell_type = get_meshio_cell_type(ele_type)
 
    mesh_file = os.path.join(input_dir, f"abaqus/cube.inp")
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
    sol_list = solver(problem)
    vtk_path = os.path.join(output_dir, f'vtk/u.vtu')
    save_sol(problem.fes[0], sol_list[0], vtk_path)

    prof_dir = os.path.join(output_dir, f'prof')
    os.makedirs(prof_dir, exist_ok=True)

    files = glob.glob(os.path.join(prof_dir, f'*'))
    for f in files:
        os.remove(f)

    jax.profiler.save_device_memory_profile(os.path.join(prof_dir, f'memory.prof'))


if __name__ == "__main__":
    problem()
