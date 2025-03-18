import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import scipy

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh
from applications.hessian.hess_manager import HessVecProduct, finite_difference_hessp

output_dir = os.path.join(os.path.dirname(__file__), 'output')


class HyperElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):
        def psi(F, rho):
            E = 1e6*rho
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
            return np.array([0., 0, -1e3])

        return [surface_map]

    def set_params(self, params):
        rho1, rho2, rho3 = params
        rho = np.ones((problem.fe.num_cells, problem.fe.num_quads))
        num_cells = problem.fe.num_cells
        rho = rho.at[:num_cells//3, :].set(rho1)
        rho = rho.at[num_cells//3:2*num_cells//3, :].set(rho2)
        rho = rho.at[2*num_cells//3:, :].set(rho3)
        self.internal_vars = [rho]


# Specify mesh-related information (first-order hexahedron element).
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=20, Ny=20, Nz=20, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=output_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


def zero_dirichlet_val(point):
    return 0.

# Define boundary locations.
def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)

dirichlet_bc_info = [[bottom]*3, [0, 1, 2], [zero_dirichlet_val]*3]
location_fns = [top]

# Create an instance of the problem.
problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)


params = [0.1, 0.5, 0.8]

# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem) 
sol_list = fwd_pred(params)

vtk_path = os.path.join(output_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)


def J_fn(u, θ):
    u_vec = jax.flatten_util.ravel_pytree(u)[0]
    θ_vec = jax.flatten_util.ravel_pytree(θ)[0]
    return np.sum(u_vec**3) + np.sum(θ_vec**3) + np.sum(u_vec**2) * np.sum(np.exp(θ_vec))


def J(θ):
    u = fwd_pred(θ)
    return J_fn(u, θ)


θ = params
θ_vec, unflatten_fn_θ = jax.flatten_util.ravel_pytree(θ)
key = jax.random.key(1)
θ_hat_flat = jax.random.normal(key, θ_vec.shape)
θ_hat = unflatten_fn_θ(θ_hat_flat)


hess_vec_prod = HessVecProduct(problem, J_fn, {}, {})
hess_vec_prod.hessp(θ, θ_hat)
finite_difference_hessp(hess_vec_prod, θ, θ_hat)

