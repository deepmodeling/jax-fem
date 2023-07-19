import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver, ad_wrapper
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.fem.mma import optimize
from jax_am.common import rectangle_mesh


class Elasticity(FEM):
    def get_tensor_map(self):
        def stress(u_grad, theta):
            Emax = 70.e3
            Emin = 1e-3*Emax
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin)*theta[0]**penal
            epsilon = 0.5*(u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
            sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
            sig12 = E/(1 + nu)*eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
            return sigma
        return stress

    def set_params(self, params):
        """Override base class method.
        """
        thetas = np.repeat(params[:, None, :], self.num_quads, axis=1)
        self.internal_vars['laplace'] = [thetas]


data_path = os.path.join(os.path.dirname(__file__), 'data') 
files = glob.glob(os.path.join(data_path, f'vtk/*'))
for f in files:
    os.remove(f)

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 10., 10.
meshio_mesh = rectangle_mesh(Nx=10, Ny=10, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)
    
def load_location(point):
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))

def dirichlet_val(point):
    return 0.

def neumann_val(point):
    return np.array([0., -100.])

dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
neumann_bc_info = [[load_location], [neumann_val]]
problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)
fwd_pred = ad_wrapper(problem, linear=True, use_petsc=True)

rho = 0.5*np.ones((problem.num_cells, 1))
fwd_pred(rho)


def J_total(rho):
    sol = fwd_pred(rho)
    return np.sum(sol), np.sum(sol)

jacs = jax.jacrev(J_total)(rho)
print(jacs.shape)
