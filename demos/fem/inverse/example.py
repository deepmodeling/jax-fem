import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver, ad_wrapper
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh


class HyperElasticity(FEM):
    def get_tensor_map(self):
        def psi(F, rho):
            E = self.E * rho
            nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, rho):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, rho)
            return P
        return first_PK_stress

    def set_params(self, params):
        E, rho, scale_d, scale_n, scale_s = params
        self.E = E
        self.internal_vars['laplace'] = [rho]
        self.dirichlet_bc_info[-1][-1] = get_dirichlet_bottom(scale_d)
        self.update_Dirichlet_boundary_conditions(self.dirichlet_bc_info)
        self.neumann_value_fns[0] = get_neumann_top(scale_n)
        self.source_info = get_body_force(scale_s)


ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh(Nx=5, Ny=5, Nz=5, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


def get_dirichlet_bottom(scale):
    def dirichlet_bottom(point):
        z_disp = scale*Lz
        return z_disp
    return dirichlet_bottom


def get_neumann_top(scale):
    def neumann_top(point):
        base_traction = 100.
        traction_z = scale*base_traction
        return np.array([0., 0., -traction_z])
    return neumann_top


def get_body_force(scale):
    def body_force(point):
        base_force = 100
        force = scale*base_force
        return np.array([force, 0., 0.])
    return body_force


def zero_dirichlet_val(point):
    return 0.


def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)


def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)


dirichlet_bc_info = [[bottom]*3, [0, 1, 2], [zero_dirichlet_val]*2 + [get_dirichlet_bottom(1.)]]
neumann_bc_info = [[top], [None]]
problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)


rho = 0.5*np.ones((problem.num_cells, problem.num_quads))
E = 1.e6
scale_d, scale_n, scale_s = 1., 1., 1.
params = [E, rho, scale_d, scale_n, scale_s]

fwd_pred = ad_wrapper(problem)
sol = fwd_pred(params)

vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem, sol, vtk_path)

def test_fn(sol):
    return np.sum(sol**2)

def composed_fn(params):
    return test_fn(fwd_pred(params))

val = test_fn(sol)

h = 1e-3

E_plus = (1 + h)*E
params_E = [E_plus, rho, scale_d, scale_n, scale_s]
dE_fd = (composed_fn(params_E) - val)/(h*E)

rho_plus = rho.at[0, 0].set((1 + h)*rho[0, 0])
params_rho = [E, rho_plus, scale_d, scale_n, scale_s]
drho_fd_00 = (composed_fn(params_rho) - val)/(h*rho[0, 0])

scale_d_plus = (1 + h)*scale_d
params_scale_d = [E, rho, scale_d_plus, scale_n, scale_s]
dscale_d_fd = (composed_fn(params_scale_d) - val)/(h*scale_d)

scale_n_plus = (1 + h)*scale_n
params_scale_n = [E, rho, scale_d, scale_n_plus, scale_s]
dscale_n_fd = (composed_fn(params_scale_n) - val)/(h*scale_n)

scale_s_plus = (1 + h)*scale_s
params_scale_s = [E, rho, scale_d, scale_n, scale_s_plus]
dscale_s_fd = (composed_fn(params_scale_s) - val)/(h*scale_s)

dE, drho, dscale_d, dscale_n, dscale_s = jax.grad(composed_fn)(params)

print(f"\nDerivative comparison between automatic differentiation (AD) and finite difference (FD)")
print(f"dE = {dE}, dE_fd = {dE_fd}")
print(f"drho[0, 0] = {drho[0, 0]}, drho_fd_00 = {drho_fd_00}")
print(f"dscale_d = {dscale_d}, dscale_d_fd = {dscale_d_fd}")
print(f"dscale_n = {dscale_n}, dscale_n_fd = {dscale_n_fd}")
print(f"dscale_s = {dscale_s}, dscale_s_fd = {dscale_s_fd}")
