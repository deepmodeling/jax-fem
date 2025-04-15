import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import scipy
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh 
from jax_fem import logger
from applications.hessian.hess_manager import HessVecProduct
from applications.hessian.utils import compute_l2_norm_error


logger.setLevel(logging.INFO)

case_name = 'traction'
data_dir = os.path.join(os.path.dirname(__file__), f'data/{case_name}')
fwd_dir = os.path.join(data_dir, 'forward')
fwd_vtk_dir = os.path.join(data_dir, 'forward/vtk')
fwd_numpy_dir = os.path.join(data_dir, 'forward/numpy')
inv_vtk_dir = os.path.join(data_dir, 'inverse/vtk')
inv_numpy_dir = os.path.join(data_dir, 'inverse/numpy')


class HyperElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):
        def psi(F):
            E = 1e6
            nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x, load_value):
            return np.array([0., -load_value, 0.])
        return [surface_map]

    def set_params(self, params):
        surface_params = params
        # Generally, [[surface1_params1, surface1_params2, ...], [surface2_params1, surface2_params2, ...], ...]
        self.internal_vars_surfaces = [[surface_params]] 


class HessVecProductTraction(HessVecProduct):
    def callback(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.callback is called for optimization step {self.opt_step} ...")
        self.opt_step += 1
        inv_vtk_dir, = self.args
        # save_sol(self.problem.fes[0], self.cached_vars['u'][0], os.path.join(inv_vtk_dir, f'u_{self.opt_step:05d}.vtu'))
        # np.save(os.path.join(inv_numpy_dir, f'traction_{self.opt_step:05d}.npy'), self.unflatten(θ))


def visualize_traction(case, steps=None):
    y_pos, traction = np.load(os.path.join(fwd_numpy_dir, f'traction.npy'))
    fig = plt.figure(figsize=(8, 6)) 
    plt.title(f'Ground truth')
    plt.plot(y_pos, traction, color='blue', marker='o', markersize=4, linestyle='None')
     
    if case == 'inverse':
        for step in steps:
            traction = np.load(os.path.join(inv_numpy_dir, f'traction_{step:05d}.npy'))
            fig = plt.figure(figsize=(8, 6)) 
            plt.title(f'Step {step}')
            plt.plot(y_pos, traction, color='blue', marker='o', markersize=4, linestyle='None')

    plt.show()


def workflow():

    # Specify mesh-related information (first-order hexahedron element).
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly, Lz = 1., 1., 0.05
    meshio_mesh = box_mesh_gmsh(Nx=20, Ny=20, Nz=1, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=fwd_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def zero_dirichlet_val(point):
        return 0.

    # Define boundary locations.
    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    dirichlet_bc_info = [[bottom]*3, [0, 1, 2], [zero_dirichlet_val]*3]
    location_fns = [top]

    # Create an instance of the problem.
    problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    fwd_pred = ad_wrapper(problem) 
    # (num_selected_faces, num_face_quads, dim)
    surface_quad_points = problem.physical_surface_quad_points[0]

    run_forward_flag = True
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*')) + glob.glob(os.path.join(fwd_numpy_dir, f'*'))
        for f in files:
            os.remove(f)

        # traction_true = np.ones((surface_quad_points.shape[0], surface_quad_points.shape[1])) 
        traction_true = 1e5*np.exp(-(np.power(surface_quad_points[:, :, 0] - Lx/2., 2)) / (2.*(Ly/5.)**2))

        sol_list_true = fwd_pred(traction_true)

        save_sol(problem.fes[0], sol_list_true[0], os.path.join(fwd_vtk_dir, f'u.vtu'))
        os.makedirs(fwd_numpy_dir, exist_ok=True)
        np.save(os.path.join(fwd_numpy_dir, f'u.npy'), sol_list_true[0])
        np.save(os.path.join(fwd_numpy_dir, f'traction.npy'), np.stack([surface_quad_points[:, :, 0], traction_true]))

    run_inverse_flag = True
    if run_inverse_flag:
        files = glob.glob(os.path.join(inv_vtk_dir, f'*')) + glob.glob(os.path.join(inv_numpy_dir, f'*'))
        for f in files:
            os.remove(f)

        sol_list_true = [np.load(os.path.join(fwd_numpy_dir, f'u.npy'))]

        def J_fn(u, θ):
            sol_list_pred = u
            l2_error = compute_l2_norm_error(problem, sol_list_pred, sol_list_true)
            θ_vec = jax.flatten_util.ravel_pytree(θ)[0]

            # A good implementation of the optimizer should not depend the scaling factor, but scipy Newton-CG does depend.
            return 1e10*l2_error**2 + 0*np.sum(θ_vec**2)
 

        traction_ini = 1e5*np.ones_like(surface_quad_points)[:, :, 0]
        sol_list_ini = fwd_pred(traction_ini)
        save_sol(problem.fes[0], sol_list_ini[0], os.path.join(inv_vtk_dir, f'u_{0:05d}.vtu'))
        os.makedirs(inv_numpy_dir, exist_ok=True)
        np.save(os.path.join(inv_numpy_dir, f'traction_{0:05d}.npy'), traction_ini)

        hess_vec_prod = HessVecProductTraction(problem, J_fn, traction_ini, {}, {}, inv_vtk_dir)

        # About 60s to achieve J=0.008
        # result = minimize(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, method='newton-cg', jac=hess_vec_prod.grad, 
        #     hessp=hess_vec_prod.hessp, callback=hess_vec_prod.callback, options={'maxiter': 20, 'xtol': 1e-30})

        # Not converging
        # result = minimize(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, method='newton-cg', jac=hess_vec_prod.grad, 
        #      callback=hess_vec_prod.callback, options={'maxiter': 20, 'xtol': 1e-30})


        # About 100s to achieve J=0.008
        result = minimize(hess_vec_prod.J, hess_vec_prod.θ_ini_flat, method='L-BFGS-B', jac=hess_vec_prod.grad, 
            callback=hess_vec_prod.callback, options={'maxiter': 100, 'disp': True, 'gtol': 1e-20, 'xtol': 1e-30}) 

        print(result)
 

if __name__=="__main__":
    workflow()
    # visualize_traction('inverse', [i for i in range(6)])
