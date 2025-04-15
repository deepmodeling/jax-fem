import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import logging
from scipy.optimize import minimize
 
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem import logger
from applications.hessian.hess_manager import HessVecProduct
from applications.hessian.utils import compute_l2_norm_error

logger.setLevel(logging.INFO)

case_name = 'poisson'
data_dir = os.path.join(os.path.dirname(__file__), f'data/{case_name}')
fwd_vtk_dir = os.path.join(data_dir, 'forward/vtk')
fwd_numpy_dir = os.path.join(data_dir, 'forward/numpy')
inv_vtk_dir = os.path.join(data_dir, 'inverse/vtk')


class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x, theta: x

    def get_mass_map(self):
        def mass_map(u, x, theta):
            val = theta
            return np.array([val])
        return mass_map

    def set_params(self, theta):
        self.internal_vars = [theta]


class HessVecProductPoisson(HessVecProduct):
    def callback(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.callback is called for optimization step {self.opt_step} ...")
        self.opt_step += 1
        inv_vtk_dir, = self.args

        # u, F_fn = forward_step(self.problem, self.unflatten(θ), self.solver_options)
        # save_sol(self.problem.fes[0], u[0], os.path.join(self.inv_vtk_dir, f'u_{self.opt_step:05d}.vtu'), 
        #     cell_infos=[('theta', np.mean(self.unflatten(θ), axis=-1))])

        save_sol(self.problem.fes[0], self.cached_vars['u'][0], os.path.join(inv_vtk_dir, f'u_{self.opt_step:05d}.vtu'),
            cell_infos=[('theta', np.mean(self.unflatten(θ_flat), axis=-1))])
        # logger.info(f"current objective J = {self.J(θ)}")


def workflow():
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 1.
    meshio_mesh = rectangle_mesh(Nx=64, Ny=64, domain_x=Lx, domain_y=Ly)
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
    problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    
 
    fwd_pred = ad_wrapper(problem) 
    # (num_cells, num_quads, dim)
    quad_points = problem.fes[0].get_physical_quad_points()

    run_forward_flag = True
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*')) + glob.glob(os.path.join(fwd_numpy_dir, f'*'))
        for f in files:
            os.remove(f)

        theta_true = -10*np.exp(-(np.power(quad_points[:, :, 0] - 0.5, 2) + np.power(quad_points[:, :, 1] - 0.5, 2)) / 0.02)
        sol_list_true = fwd_pred(theta_true)

        save_sol(problem.fes[0], sol_list_true[0], os.path.join(fwd_vtk_dir, f'u.vtu'), cell_infos=[('theta', np.mean(theta_true, axis=-1))])
        os.makedirs(fwd_numpy_dir, exist_ok=True)
        np.save(os.path.join(fwd_numpy_dir, f'u.npy'), sol_list_true[0])

    run_inverse_flag = True
    if run_inverse_flag:
        files = glob.glob(os.path.join(inv_vtk_dir, f'*')) 
        for f in files:
            os.remove(f)

        sol_list_true = [np.load(os.path.join(fwd_numpy_dir, f'u.npy'))]

        def J_fn(u, θ):
            sol_list_pred = u
            l2_error = compute_l2_norm_error(problem, sol_list_pred, sol_list_true)
            θ_vec = jax.flatten_util.ravel_pytree(θ)[0]
            return 1e3*l2_error**2 + 0*np.sum(θ_vec**2)

        # def J_fn(u, θ):
        #     sol_list_pred = u
        #     θ_vec = jax.flatten_util.ravel_pytree(θ)[0]
        #     return np.sum((sol_list_pred[0] - sol_list_true[0])**2) + 0.*np.sum(θ_vec**2)

        theta_ini = 1*np.ones_like(quad_points)[:, :, 0]
        sol_list_ini = fwd_pred(theta_ini)
        save_sol(problem.fes[0], sol_list_ini[0], os.path.join(inv_vtk_dir, f'u_{0:05d}.vtu'), cell_infos=[('theta', np.mean(theta_ini, axis=-1))])

        hess_vec_prod = HessVecProductPoisson(problem, J_fn, theta_ini, {}, {}, inv_vtk_dir)

        # # CG or L-BFGS-B or Newton-CG or SLSQP
        result = minimize(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, method='newton-cg', jac=hess_vec_prod.grad, 
            hessp=hess_vec_prod.hessp, callback=hess_vec_prod.callback, options={'maxiter': 5, 'xtol': 1e-20, 'gtol': 1e-20, 'ftol': 1e-20})

        # Does not converge
        # result = minimize(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, method='newton-cg', jac=hess_vec_prod.grad, 
        #                   callback=hess_vec_prod.callback, options={'xtol': 1e-20, 'gtol': 1e-20, 'ftol': 1e-20})

        # result = minimize(hess_vec_prod.J, hess_vec_prod.θ_ini_flat, method='L-BFGS-B', jac=hess_vec_prod.grad, 
        #     callback=hess_vec_prod.callback, options={'maxiter': 20, 'disp': True, 'gtol': 1e-20, 'xtol': 1e-30}) 

        print(result)


if __name__=="__main__":
    workflow()
