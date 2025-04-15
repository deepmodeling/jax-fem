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
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh 
from jax_fem import logger
from applications.hessian.hess_manager import HessVecProduct

logger.setLevel(logging.INFO)

case_name = 'shape'
data_dir = os.path.join(os.path.dirname(__file__), f'data/{case_name}')
fwd_dir = os.path.join(data_dir, 'forward')
fwd_vtk_dir = os.path.join(data_dir, 'forward/vtk')
fwd_numpy_dir = os.path.join(data_dir, 'forward/numpy')
inv_vtk_dir = os.path.join(data_dir, 'inverse/vtk')
inv_numpy_dir = os.path.join(data_dir, 'inverse/numpy')


def scaled_sigmoid(x, lower_lim, upper_lim, p=0.1):
    return lower_lim + (upper_lim - lower_lim)/((1. + np.exp(-x*p)))


def pore_fn(x, pore_center, L0, c1, c2, beta):
    beta = scaled_sigmoid(beta, -np.pi/4., np.pi/4., p=0.1)
    porosity = 0.5
    theta = np.arctan2(x[1] - pore_center[1], x[0] - pore_center[0]) 
    r = np.sqrt(np.sum((x - pore_center)**2))
    x_rel = r*np.cos(theta - beta)
    y_rel = r*np.sin(theta - beta)
    p = 200.
    rho = 1./(1. + np.exp(-(np.abs(x_rel) + np.abs(y_rel) - 0.9*L0/2)*p))
    return rho

pore_fn_vmap = jax.vmap(pore_fn, in_axes=(0, None, None, None, None, None))


class Elasticity(Problem):
    def custom_init(self, Lx, Ly, nx, ny):
        self.fe = self.fes[0]
        # (num_cells, num_quads, dim)
        physical_quad_points = self.fe.get_physical_quad_points()
        L0 = Lx/nx
        self.pore_center_list = []
        self.quad_inds_list = []
        for i in range(nx):
            for j in range(ny):
                pore_center = np.array([i*L0 + L0/2., j*L0 + L0/2.])
                self.pore_center_list.append(pore_center)
                # (num_selected_quad_points, 2)
                quad_inds = np.argwhere((physical_quad_points[:, :, 0] >= i*L0) &
                                        (physical_quad_points[:, :, 0] < (i + 1)*L0) & 
                                        (physical_quad_points[:, :, 1] >= j*L0) &
                                        (physical_quad_points[:, :, 1] < (j + 1)*L0))
                self.quad_inds_list.append(quad_inds)
        self.L0 = L0

    def get_tensor_map(self):
        def psi(F_2d, rho):
            # Plane strain
            F = np.array([[F_2d[0, 0], F_2d[0, 1], 0.], 
                          [F_2d[1, 0], F_2d[1, 1], 0.],
                          [0., 0., 1.]])
            Emax = 1e6  
            Emin = 1e-3*Emax
            E = Emin + (Emax - Emin)*rho
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
            return np.array([0, 1e4])
        return [surface_map]

    def set_params(self, params):
        # Override base class method.
        c1, c2, beta = params
        rhos = np.ones((self.fe.num_cells, self.fe.num_quads))
        for i in range(len(self.pore_center_list)):
            quad_inds = self.quad_inds_list[i]
            # (num_selected_quad_points, dim)
            quad_points = self.physical_quad_points[quad_inds[:, 0], quad_inds[:, 1]]
            pore_center = self.pore_center_list[i]
            rho_vals = pore_fn_vmap(quad_points, pore_center, self.L0, c1[i], c2[i], beta[i])
            rhos = rhos.at[quad_inds[:, 0], quad_inds[:, 1]].set(rho_vals)
        self.internal_vars = [rhos]

    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)    
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val


class HessVecProductPore(HessVecProduct):
    def callback(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.callback is called for optimization step {self.opt_step} ...")
        self.opt_step += 1
        inv_vtk_dir, = self.args
        θ = self.unflatten(θ_flat)
        sol_list = self.cached_vars['u']
        rho = jax.lax.stop_gradient(self.problem.internal_vars[0])
        # save_sol(self.problem.fe, np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
        #          os.path.join(inv_vtk_dir, f'u_{self.opt_step:05d}.vtu'), cell_infos=[('rho', np.mean(rho, axis=-1))])

        logger.info(f"########################## θ = \n{θ}")


def workflow():
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 0.5
    nx, ny = 4, 2 # pore numbers along x-axis and y-axis
    meshio_mesh = rectangle_mesh(Nx=200, Ny=100, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load_location(point):
        return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))

    def dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]

    location_fns = [load_location]

    problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                         location_fns=location_fns, additional_info=(Lx, Ly, nx, ny))
    fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

    run_forward_flag = True
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*'))
        for f in files:
            os.remove(f)

        params = np.array([[0.]*nx*ny, [0.]*nx*ny, [0.]*nx*ny])
        sol_list_true = fwd_pred(params)
        save_sol(problem.fe, np.hstack((sol_list_true[0], np.zeros((len(sol_list_true[0]), 1)))), 
            os.path.join(fwd_vtk_dir, f'u.vtu'), cell_infos=[('rho', np.mean(problem.internal_vars[0], axis=-1))])

    run_inverse_flag = True
    if run_inverse_flag:
        files = glob.glob(os.path.join(inv_vtk_dir, f'*'))
        for f in files:
            os.remove(f)

        def J_fn(u, θ):
            sol_list = u
            compliace = problem.compute_compliance(sol_list[0])
            return 1e5*compliace

        params_ini = np.array([[0.]*nx*ny, [0.]*nx*ny, [0.]*nx*ny])

        sol_list_ini = fwd_pred(params_ini)
 
        save_sol(problem.fe, np.hstack((sol_list_ini[0], np.zeros((len(sol_list_ini[0]), 1)))), 
            os.path.join(inv_vtk_dir, f'u_{0:05d}.vtu'), cell_infos=[('rho', np.mean(problem.internal_vars[0], axis=-1))])

        option_umfpack = {'umfpack_solver': {}}
        hess_vec_prod = HessVecProductPore(problem, J_fn, params_ini, option_umfpack, option_umfpack, inv_vtk_dir)

        result = minimize(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, method='newton-cg', jac=hess_vec_prod.grad, 
            hessp=hess_vec_prod.hessp, callback=hess_vec_prod.callback, options={'maxiter': 6, 'xtol': 1e-30})

        # result = minimize(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, method='newton-cg', jac=hess_vec_prod.grad, 
        #                   callback=hess_vec_prod.callback, options={'maxiter': 6, 'xtol': 1e-30})

        # result = minimize(hess_vec_prod.J, hess_vec_prod.θ_ini_flat, method='L-BFGS-B', jac=hess_vec_prod.grad, 
        #     callback=hess_vec_prod.callback, options={'maxiter': 20, 'disp': True, 'gtol': 1e-20, 'xtol': 1e-30}) 


        print(result)

if __name__=="__main__":
    workflow()