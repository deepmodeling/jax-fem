import jax
import jax.numpy as np
import os

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.common import rectangle_mesh


class Poisson(FEM):
    def get_tensor_map(self):
        return lambda x: x


ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 1., 1.
meshio_mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)

def dirichlet_val_left(point):
    return 0.

def dirichlet_val_right(point):
    return 0.

location_fns = [left, right]
value_fns = [dirichlet_val_left, dirichlet_val_right]
vecs = [0, 0]
dirichlet_bc_info = [location_fns, vecs, value_fns]

def neumann_val(point):
    return np.array([np.sin(5.*point[0])])

neumann_bc_info = [[bottom, top], [neumann_val, neumann_val]]

def body_force(point):
    return np.array([10*np.exp(-(np.power(point[0] - 0.5, 2) + np.power(point[1] - 0.5, 2)) / 0.02)])

problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
    neumann_bc_info=neumann_bc_info, source_info=body_force)
sol = solver(problem, linear=True, use_petsc=True)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem, sol, vtk_path)
