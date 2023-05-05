import jax
import jax.numpy as np
import os
import glob

from jax_am.fem.models import LinearElasticity
from jax_am.fem.solver import solver
from jax_am.fem.generate_mesh import Mesh, box_mesh
from jax_am.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

    prof_dir = os.path.join(data_dir, f'prof')
    os.makedirs(prof_dir, exist_ok=True)

    files = glob.glob(os.path.join(prof_dir, f'*'))
    for f in files:
        os.remove(f)

    jax.profiler.save_device_memory_profile(os.path.join(prof_dir, f'memory.prof'))


if __name__ == "__main__":
    problem()
