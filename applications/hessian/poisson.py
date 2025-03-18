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
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from applications.hessian.hess_manager import HessVecProduct, finite_difference_hessp

output_dir = os.path.join(os.path.dirname(__file__), 'output')


class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x, rho: rho*x

    def get_mass_map(self):
        def mass_map(u, x, rho):
            val = -np.array([10*rho*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            return val
        return mass_map

    def get_surface_maps(self):
        def surface_map(u, x):
            return -np.array([np.sin(5.*x[0])])

        return [surface_map, surface_map]

    def set_params(self, params):
        rho1, rho2, rho3 = params
        rho = np.ones((problem.fes[0].num_cells, problem.fes[0].num_quads))
        num_cells = problem.fes[0].num_cells
        rho = rho.at[:num_cells//3, :].set(rho1)
        rho = rho.at[num_cells//3:2*num_cells//3, :].set(rho2)
        rho = rho.at[2*num_cells//3:, :].set(rho3)
        self.internal_vars = [rho]


ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 1., 1.
meshio_mesh = rectangle_mesh(Nx=2, Ny=1, domain_x=Lx, domain_y=Ly)
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
location_fns = [bottom, top]
problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)


def J_fn(u, θ):
    u_vec = jax.flatten_util.ravel_pytree(u)[0]
    θ_vec = jax.flatten_util.ravel_pytree(θ)[0]
    return np.sum(u_vec**3) + np.sum(θ_vec**3) + np.sum(u_vec**2) * np.sum(np.exp(θ_vec))

fwd_pred = ad_wrapper(problem) 

params = [0.1, 0.5, 0.8]
sol_list = fwd_pred(params)
vtk_path = os.path.join(output_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)


θ = params
θ_vec, unflatten_fn_θ = jax.flatten_util.ravel_pytree(θ)
key = jax.random.key(1)
θ_hat_flat = jax.random.normal(key, θ_vec.shape)
θ_hat = unflatten_fn_θ(θ_hat_flat)

hess_vec_prod = HessVecProduct(problem, J_fn, {}, {})
hess_vec_prod.hessp(θ, θ_hat)
finite_difference_hessp(hess_vec_prod, θ, θ_hat)