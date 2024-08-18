# Import some useful modules.
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt


# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh


# Define constitutive relationship.
class HyperElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

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

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0., 1e3])

        return [surface_map]

    def set_params(self, params):
        E, rho, scale_d = params
        self.E = E
        self.internal_vars = [rho]
        self.fe.dirichlet_bc_info[-1][-1] = get_dirichlet_bottom(scale_d)
        self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)


# Specify mesh-related information (first-order hexahedron element).
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=5, Ny=5, Nz=5, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# Define Dirichlet boundary values.
def get_dirichlet_bottom(scale):
    def dirichlet_bottom(point):
        z_disp = scale*Lz
        return z_disp
    return dirichlet_bottom

def zero_dirichlet_val(point):
    return 0.


# Define boundary locations.
def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)

dirichlet_bc_info = [[bottom]*3, [0, 1, 2], [zero_dirichlet_val]*2 + [get_dirichlet_bottom(1.)]]
location_fns = [top]


# Create an instance of the problem.
problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)


# Define parameters.
rho = 0.5*np.ones((problem.fe.num_cells, problem.fe.num_quads))
E = 1.e6
scale_d = 1.
params = [E, rho, scale_d]


# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem) 
sol_list = fwd_pred(params)

vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fe, sol_list[0], vtk_path)

def test_fn(sol_list):
    return np.sum(sol_list[0]**2)

def composed_fn(params):
    return test_fn(fwd_pred(params))

val = test_fn(sol_list)

h = 1e-3 # small perturbation


# Forward difference
E_plus = (1 + h)*E
params_E = [E_plus, rho, scale_d]
dE_fd = (composed_fn(params_E) - val)/(h*E)

rho_plus = rho.at[0, 0].set((1 + h)*rho[0, 0])
params_rho = [E, rho_plus, scale_d]
drho_fd_00 = (composed_fn(params_rho) - val)/(h*rho[0, 0])

scale_d_plus = (1 + h)*scale_d
params_scale_d = [E, rho, scale_d_plus]
dscale_d_fd = (composed_fn(params_scale_d) - val)/(h*scale_d)

# Derivative obtained by automatic differentiation
dE, drho, dscale_d = jax.grad(composed_fn)(params)

# Comparison
print(f"\nDerivative comparison between automatic differentiation (AD) and finite difference (FD)")
print(f"\ndrho[0, 0] = {drho[0, 0]}, drho_fd_00 = {drho_fd_00}")
print(f"\ndscale_d = {dscale_d}, dscale_d_fd = {dscale_d_fd}")

print(f"\ndE = {dE}, dE_fd = {dE_fd}, WRONG results! Please avoid gradients w.r.t self.E")
print(f"This is due to the use of glob variable self.E, inside a jax jitted function.")

# TODO: show the following will cause an error
# dE_E, _, _ = jax.grad(composed_fn)(params_E)
