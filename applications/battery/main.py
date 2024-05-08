'''
The main module for solving P2D problems with JAX-FEM

The macro variables (p,c,s,j) are coupled and solved at the same time step.

Last modified: 08/05/2024

'''

# JAX-FEM packages
from jax_fem.solver import solver,ad_wrapper
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem import logger


# Some useful package
import os
import time
import scipy.io as scio
import numpy as onp
import jax
import jax.numpy as np

from jax import config
config.update("jax_enable_x64", True)


# Custom modules for P2D problems
from prep import prep_mesh_macro,assign_init_sol
from para import param_sets_macro, param_sets_micro
from micro import prep_micro_problem, res_node_flux, solve_micro_problem
from matlab_fns import calcKappa_Deriv
from utils import output_sol


# Data dir
input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')

import logging
# logger.setLevel(logging.WARNING)

start_time = time.time()

t_sta = 0.0                    
t_end = 10.0                    
dt = 1.0                                   
steps_time = int((t_end-t_sta)/dt)            
# steps_total = steps_time + 1

# parameters
params_macro = param_sets_macro(dt)
params_micro = param_sets_micro(dt, params_macro.r_an, params_macro.r_ca)

# mesh data from MATLAB
mat_mesh_macro = scio.loadmat(os.path.join(input_dir, f'mesh/macro_mesh.mat')) 
mat_mesh_micro = scio.loadmat(os.path.join(input_dir, f'mesh/micro_mesh.mat')) 
# from utils import plot_macro_micro_mesh
# plot_macro_micro_mesh(mat_mesh_macro, mat_mesh_micro, output_dir)

# mesh data for jax-fem (macro)
mesh_macro = prep_mesh_macro(mat_mesh_macro)
# problem data for jax-fem (micro)
problem_micro = prep_micro_problem(mat_mesh_micro, params_micro, dt, 'matlab')


class macro_P2D(Problem):
    def custom_init(self):
        self.fe_p = self.fes[0]
        self.fe_c = self.fes[1]
        self.fe_s = self.fes[2]
        self.fe_j = self.fes[3]
        
    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, 
                             cell_theta, cell_c_quad_old, cell_sol_micro_old, cell_tag, cell_nodes_tag, cell_nodes_sum):
            
            # ---- Split the input variables ----
            
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            
            # sol of the cell
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            # (num_nodes, vec) -> (4, 1)
            cell_sol_p, cell_sol_c, cell_sol_s, cell_sol_j = cell_sol_list
            
            # shape function gradients of the cell
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            # (num_quads, num_nodes, dim) -> (4, 4, 2)
            cell_shape_grads_p, cell_shape_grads_c, cell_shape_grads_s, cell_shape_grads_j = cell_shape_grads_list
            
            # grad(v)*JxW of the cell
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            # (num_quads, num_nodes, 1, dim) -> (4, 4, 1, 2)
            cell_v_grads_JxW_p, cell_v_grads_JxW_c, cell_v_grads_JxW_s, cell_v_grads_JxW_j = cell_v_grads_JxW_list
            
            # JxW of the cell -> (num_quads,) -> (4,)
            cell_JxW_p, cell_JxW_c, cell_JxW_s, cell_JxW_j = cell_JxW[0], cell_JxW[1], cell_JxW[2], cell_JxW[3]
            
            # ---- Some parameters ----

            lag1 = -1*cell_tag*(cell_tag-2)      # anode - 1
            lag2 = 1/2*cell_tag*(cell_tag-1)     # cathode - 2
            lag3 = 1/2*(cell_tag-1)*(cell_tag-2) # seperator - 0
            
            eps_mat = (lag1*params_macro.eps_mat_an + 
                        lag2*params_macro.eps_mat_ca + 
                        lag3*params_macro.eps_mat_se)
            
            eps_inc = (lag1*params_macro.eps_inc_an+
                        lag2*params_macro.eps_inc_ca)
            
            r = (lag1*params_macro.r_an + lag2*params_macro.r_ca + lag3*1)
            
            sour_ac = (lag1*params_macro.sour_an + lag2*params_macro.sour_ca)
            
            svr = 3 * eps_inc / r
            ratio_mat = (eps_mat)**(params_macro.alpha * cell_theta)
            ratio_inc = (eps_inc)**(params_macro.alpha * cell_theta)
            
            
            # ---- Residual of potential in electrolyte  ----
            
            # R_p = (ka_eff * inner(grad(p), grad(v_p) - kad_eff * inner(grad(ln(c))), grad(v_p) - a*F*j*v_p)*dx
            
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec) -> (num_quads,)
            c = np.sum(cell_sol_c[None,:,:] * self.fe_c.shape_vals[:,:,None], axis=1)[:,0]
            # **** Notes: following params_macro are location dependent (quadrature point location dependent) ****
            # (num_quads,)
            kappa, kaDeriv = jax.vmap(calcKappa_Deriv)(c)
            ka_eff = kappa / params_macro.ka_ref * ratio_mat
            # **** Notes: kad_eff seems not to be consistent with the formulations in the PDF ****
            kad_eff = 2 * ka_eff * params_macro.R * params_macro.T / params_macro.F *(1-params_macro.tp)
            
            
            # Handles the term `ka_eff * inner(grad(p), grad(v_p)*dx`
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            p_grads = cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :]
            p_grads = np.sum(p_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, 1, 1) * (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rp_v1 = np.sum(ka_eff[:,None,None,None] * p_grads[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))

            
            # Handles the term `kad_eff * inner(grad(ln(c))), grad(v_p)*dx`
            # **** Notes: grad(ln(c)) can be written as d(ln(c))/d(c) * grad(c) = 1/c * grad(c) ****
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            c_grads = cell_sol_c[None, :, :, None] * cell_shape_grads_c[:, :, None, :]
            c_grads = np.sum(c_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, 1, 1) * (num_quads, 1, 1, 1) * (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rp_v2 = np.sum(kad_eff[:,None,None,None] * 1./c[:,None,None,None] * c_grads[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))
            
            
            # Handles the term `a*F*j*v_p*dx` (sour_p * a *j * v_p *dx)
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec)
            j = np.sum(cell_sol_j[None,:,:] * self.fe_j.shape_vals[:,:,None], axis=1)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1)  -> (num_nodes, vec)
            Rp_v3 = params_macro.sour_p * svr * np.sum(j[:,None,:] * self.fe_p.shape_vals[:,:,None] * cell_JxW_p[:,None,None], axis=0)
            
            Rp = Rp_v1 - Rp_v2 - Rp_v3
            
            
            # ---- Residual of diffusion in electrolyte  ----
            
            # R_c = (c_crt - c_old)/dt * v_c *dx + df_eff * inner(grad(c),grad(v_c))*dx - (1-t_e)*a*j*v_c*dx
            
            df_eff = params_macro.df * ratio_mat / params_macro.df_ref
            
            # Handles the term ` (c_crt - c_old)/dt * v_c *dx`
            dc = c - cell_c_quad_old
            # (num_quads, 1, 1) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            Rc_v1 = params_macro.dtco_mat * eps_mat * np.sum(dc[:,None,None] * self.fe_c.shape_vals[:,:,None]* cell_JxW_c[:,None,None], axis=0)
            
            
            # Handles the term `df_eff * inner(grad(c),grad(v_c))*dx`
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rc_v2 = df_eff * np.sum(c_grads[:, None, :, :] * cell_v_grads_JxW_c, axis=(0, -1))
            
            
            # Handles the term `(1-t_e)*a*j*v_c*dx`
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1)  -> (num_nodes, vec)
            Rc_v3 = params_macro.sour_c * svr * np.sum(j[:,None,:] * self.fe_c.shape_vals[:,:,None] * cell_JxW_c[:,None,None], axis=0)
            
            Rc = Rc_v1 + Rc_v2 - Rc_v3
            
            
            # ---- Residual of potential in electrode  ----
            
            # Only in anode and cathode!
            
            # R_s = (sig_eff * grad(s) * grad(v_s) + a*F*j*v_s)*dx
            
            sig = params_macro.sigan / params_macro.sigan_ref;
            sig_eff = sig * ratio_inc
            
            # Handles the term `sig_eff * grad(s) * grad(v_s)*dx`
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            s_grads = cell_sol_s[None, :, :, None] * cell_shape_grads_s[:, :, None, :]
            s_grads = np.sum(s_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes, vec)
            Rs_v1 = sig_eff * np.sum(s_grads[:, None, :, :] * cell_v_grads_JxW_s, axis=(0, -1))
            
            # Handles the term `a*F*j*v_s*dx`
            Rs_v2 = sour_ac * svr * np.sum(j[:,None,:] * self.fe_s.shape_vals[:,:,None] * cell_JxW_s[:,None,None], axis=0)
            
            Rs = Rs_v1 + Rs_v2
            
            
            # ---- Residual of pore wall flux ----
            
            # cell_sol_list [(num_nodes,vec),...]
            # cell_sol_micro_old (num_nodes, num_total_nodes_micro)
            # cell_tag (num_nodes,)
        
            Rj = res_node_flux(cell_sol_list, cell_sol_micro_old, cell_nodes_tag, problem_micro, params_macro, params_micro)
            
            # Rj = ((lag1+lag2)*Rj + lag3*np.zeros((len(cell_sol_j),1)))/cell_nodes_sum
            
            Rj = Rj/cell_nodes_sum
            
            weak_form = [Rp, Rc, Rs, Rj] # [(num_nodes, vec), ...]
            
            # jax.debug.breakpoint()
            
            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel
    
    
    def get_universal_kernels_surface(self):
        '''
        Neumann boundary conditions for phi_s
        '''
        def current(u):
            return np.array([params_macro.I_bc*params_macro.i_ca])
        
        def current_neumann(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_face_quads, dim)
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # face_nanson_scale: (num_vars, num_face_quads)
            
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_s = cell_sol_list[2]
            face_shape_vals = face_shape_vals[:, -self.fes[2].num_nodes:]
            face_nanson_scale = face_nanson_scale[0]
            
            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol_s[None, :, :] * face_shape_vals[:, :, None], axis=1)
            
            u_physics = jax.vmap(current)(u)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val_s = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)
            
            # only for the electrode potential
            val = [val_s*0.,val_s*0.,val_s,val_s*0.]
            
            return jax.flatten_util.ravel_pytree(val)[0]
        
        return [current_neumann]


    def set_params(self, params):
        """
        Input variables to the kernel
        """
        cells_theta, sol_macro_old, sol_c_old, sol_micro_old, cells_tag, cells_nodes_tag, cells_nodes_sum = params
        
        self.initial_guess = sol_macro_old
        
        # (num_cells, num_quads)
        sol_c_old = self.fe_c.convert_from_dof_to_quad(sol_c_old.reshape((-1,1)))[:,:,0]
        # (num_cells, num_nodes, num_micro_nodes)
        sol_micro_old = sol_micro_old[self.cells_list[0]]
        
        self.internal_vars = [cells_theta, sol_c_old, sol_micro_old, cells_tag, cells_nodes_tag, cells_nodes_sum]


def battery():
    
    '''
    Solve the macro P2D problem (phi_e, c_e, phi_s ,j)
    '''
    
    # mesh for JAX-FEM
    ele_type = 'QUAD4'
    jax_mesh = Mesh(mesh_macro.points, mesh_macro.cells, ele_type=ele_type)
    
    cells_tag, cells_nodes_tag, cells_nodes_sum = mesh_macro.cells_vars
    nodes_tag = mesh_macro.nodes_vars[0]
    
    cells_nodes_sum = cells_nodes_sum.astype(onp.float64)
    
    
    dofs_p, dofs_c, dofs_s, dofs_j = mesh_macro.dofs
    
    # boundary conditions
    # only for the electrode potential (s) 
    x_max = (jax_mesh.points).max(0)[0] # 225.
    x_min = (jax_mesh.points).min(0)[0] # 0.
    
    x_anright = (jax_mesh.points[mesh_macro.nodes_bound_anright,:]).max(0)[0] # 100.
    x_caleft = (jax_mesh.points[mesh_macro.nodes_bound_caleft,:]).max(0)[0]   # 125.
    
    def left(point):
        return np.isclose(point[0], x_min, atol=1e-5)
    
    def right(point):
        return np.isclose(point[0], x_max, atol=1e-5)
    
    def separator(point):
        return (point[0] > x_anright) & (point[0] < x_caleft)

    def zero_dirichlet(point):
        return 0.

    dirichlet_bc_info_s = [[left, separator], [0,0], [zero_dirichlet]*2]
    dirichlet_bc_info_j = [[separator], [0], [zero_dirichlet]]
    
    location_fns_s = [right]
    
    
    # macro problem for (p,c,s,j)
    problem_macro = macro_P2D([jax_mesh]*4, vec = [1]*4, dim=2, 
                        ele_type = [ele_type]*4, gauss_order=[2]*4,
                        dirichlet_bc_info = [None,None,dirichlet_bc_info_s,dirichlet_bc_info_j],
                        location_fns = location_fns_s)
    
    
    def fwd_pred_seq(cells_theta):
        
        # sol_macro_time: (4*nnode_macro, timesteps)
        # sol_micro_time: (nnode_macro, nnode_micro, timesteps)
        
        logger.debug(f"\nGet the solution for the first time step...")
        copy_time = -1
        sol_macro_time, sol_micro_time = assign_init_sol(mesh_macro, problem_micro, params_macro, 
                                                         steps_total, input_dir, copy_time)
        
        options = {'ksp_type': 'bcgsl', 'pc_type': 'jacobi'}
        fwd_pred = ad_wrapper(problem_macro, use_petsc=True, petsc_options=options,
                                             use_petsc_adjoint=True, petsc_options_adjoint=options)
            
        for itime in range(1, steps_total): # time steps [1,...,10]
        
            logger.debug(f"\nStep {itime} in {steps_total-1}")
            
            sol_macro_old = sol_macro_time[:,itime-1]
            sol_micro_old = sol_micro_time[:,:,itime-1]   
            sol_c_old = sol_macro_old[dofs_c] # c_crt - c_old = 0
            
            sol_p, sol_c, sol_s, sol_j = fwd_pred([cells_theta, sol_macro_old, sol_c_old, sol_micro_old, 
                                                   cells_tag, cells_nodes_tag, cells_nodes_sum])
            
            sol_macro_time = sol_macro_time.at[dofs_p,itime].set(sol_p.reshape(-1))
            sol_macro_time = sol_macro_time.at[dofs_c,itime].set(sol_c.reshape(-1))
            sol_macro_time = sol_macro_time.at[dofs_s,itime].set(sol_s.reshape(-1))
            sol_macro_time = sol_macro_time.at[dofs_j,itime].set(sol_j.reshape(-1))

            sol = np.vstack((sol_p, sol_c, sol_s, sol_j))
            
            logger.debug(f"\nUpdate micro solution - need to solve the micro problem again...")
            for ind, node_sets in enumerate([mesh_macro.nodes_anode,mesh_macro.nodes_cathode]):
                for inode in node_sets:
                    node_flux = sol_macro_time[dofs_j[inode],itime]
                    sol_micro_old = sol_micro_time[inode,:,itime-1].reshape(-1,1)
                    sol_micro = solve_micro_problem(problem_micro, params_micro, nodes_tag[inode], sol_micro_old, node_flux).reshape(-1)
                    sol_micro_time = sol_micro_time.at[inode,:,itime].set(sol_micro)
        
        if forward_flag:
            output_sol(mesh_macro, sol_macro_time, sol_micro_time, input_dir, output_dir)
        
        return sol
    
    
    def get_macro_sol(cells_theta):
        sol = fwd_pred_seq(cells_theta)
        return sol
    
    def J_total(cells_theta):
        sol = get_macro_sol(cells_theta)
        J = sol[371,0]
        return J
    
    def test_fun_fwd(theta):
        cells_theta = theta*np.ones((len(jax_mesh.cells),1))
        sol = get_macro_sol(cells_theta)
        return sol
    
    def test_fun_grad(theta):
        cells_theta = theta*np.ones((len(jax_mesh.cells),1))
        J = J_total(cells_theta)
        return J
    
    
    forward_flag = True
    if forward_flag:
        logger.debug(f"\nCheck the forward problem...")
        steps_total = 11
        theta = 1.
        sol_jax = test_fun_fwd(theta)
        
        
    gradient_flag = False
    if gradient_flag:
        logger.debug(f"\nCheck the gradient...")
        steps_total = 11
        h = 1e-3
        theta = 1.
        J_minus = test_fun_grad(theta - h)
        J_plus = test_fun_grad(theta + h)
        fd_gradient = (J_plus - J_minus)/(2*h)
        ad_gradient = jax.grad(test_fun_grad)(theta)
        print(f"Step {steps_total-1}")
        print(f"fd_gradient = {fd_gradient:.10f}\nad_gradient = {ad_gradient:.10f}")
        print(f"error:{np.abs(fd_gradient-ad_gradient)/np.abs(fd_gradient)*100:.6f}%")
        
    end_time = time.time()
    total_time = end_time - start_time
    logger.debug(f"\nFinish! Total time cost:{total_time:.2f}s")


if __name__ == "__main__":
    battery()