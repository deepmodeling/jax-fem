import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio

from jax_am.fem.generate_mesh import box_mesh, Mesh
from jax_am.fem.solver import solver
from jax_am.fem.core import FEM
from jax_am.fem.utils import save_sol


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')


class PF(FEM):
    G_c = 100.
    l = 0.01

    def get_tensor_map(self):
        def fn(d_grad):
            return G_c*l*d_grad
        return fn
 
    def get_mass_map(self):
        def fn(d, history):
            return G_c/l*d - 2.*(1 - d)*history
        return fn

    def set_params(self, history):
        self.internal_vars['mass_vars'] = history


class Elasticity(FEM):
    E = 70e9
    nu = 0.3

    def get_tensor_map(self):
        _, stress_fn = self.get_maps()
        return stress_fn

    def get_maps(self):
        mu = E/(2.*(1. + nu))
        lmbda = E*nu/((1+nu)*(1-2*nu))
        bulk_mod = lamda + 2.*mu/self.dim

        def psi_plus(epsilon):
            eps_dev = epsilon - 1./self.dim*np.trace(epsilon)*np.eye(self.dim)
            tr_epsilon_plus = np.maximum(np.trace(epsilon), 0.)
            return bulk_mod/2.*tr_epsilon_plus**2 + mu*np.sum(eps_dev*eps_dev)
            
        def psi_minus(epsilon):
            tr_epsilon_minus = np.minimum(np.trace(epsilon), 0.)
            return bulk_mod/2.*tr_epsilon_minus**2

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def stress_fn(u_grad, d):
            epsilon = strain(u_grad)
            sigma = (1 - d[0])**2 * jax.grad(psi_plus)(epsilon) + jax.grad(psi_minus)(epsilon) 
            return sigma

        def psi_plus_fn(u_grad):
            epsilon = strain(u_grad)
            return psi_plus(epsilon)

        return psi_plus_fn, stress_fn

    def compute_history(self, sol, history_old):
        pass

    def set_params(self, d):
        self.internal_vars['laplace_vars'] = d
    

def simulation():

    ele_type = 'HEX8'
    vtk_dir = os.path.join(data_dir, 'vtk')
    problem_name = 'example'
    Nx, Ny, Nz = 100, 100, 1
    Lx, Ly, Lz = 1., 1., 0.01
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])


    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)
        

    Elasticity(mesh, vec=3, dim=3)

if __name__ == "__main__":
    simulation()


