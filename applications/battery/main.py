'''
The main module for solving P2D problems with JAX-FEM

The macro variables (p,c,s,j) are coupled and solved at the same time step.

Last modified: 21/03/2024

'''

# JAX-FEM packages
from jax_fem.solver import solver
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


# Custom modules for P2D problems
from para import param_sets_macro, param_sets_micro
from micro import prep_micro_problem, res_node_flux, solve_micro_problem
from matlab_fns import assign_init_sol,calcKappa_Deriv
from utils import plot_macro_micro_mesh, plot_micro_verify_data, postprocess


# Data dir
input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')


start_time = time.time()

# ==================== Initialization ====================

# time
t_start = 0.0                              # start time
t_end = 10.0                               # end time
dt = 1.0                                   # time step size
num_t = int((t_end-t_start)/dt)            # total time = num_t * dt
timesteps = num_t + 1                      # initial + time evolution

# parameters
params_macro = param_sets_macro(dt)
params_micro = param_sets_micro(dt, params_macro.r_an, params_macro.r_ca)

# mesh
macro_mesh = scio.loadmat(os.path.join(input_dir, f'mesh/macro_mesh.mat')) 
micro_mesh = scio.loadmat(os.path.join(input_dir, f'mesh/micro_mesh.mat')) 
# plot_macro_micro_mesh(macro_mesh, micro_mesh, output_dir)

# micro problem-related variables sets
micro_problem = prep_micro_problem(micro_mesh, params_micro, dt)

# nodes
nnode_macro = int(macro_mesh['nnode'])
nodes_anode = (onp.unique(macro_mesh['connect_anode'])-1).astype(onp.int32)
nodes_cathode = (onp.unique(macro_mesh['connect_cathode'])-1).astype(onp.int32)

nodes_separator = (onp.unique(macro_mesh['connect_separator'])-1).astype(onp.int32)
bound_anright = (onp.unique(macro_mesh['bound_anright'])-1).astype(onp.int32)
ca_left = (onp.unique(macro_mesh['bound_caleft'])-1).astype(onp.int32)
nodes_separator = onp.setdiff1d(nodes_separator, onp.union1d(bound_anright, ca_left))

# dofs
dofs_p = onp.linspace(0, nnode_macro-1, nnode_macro, dtype=onp.int32)
dofs_p_an = nodes_anode
dofs_p_ca = nodes_cathode

dofs_c = dofs_p + nnode_macro
dofs_c_an = nnode_macro + nodes_anode
dofs_c_ca = nnode_macro + nodes_cathode

# following dofs (s, j) are defined in all domain but only meaningful in electrode

dofs_s = dofs_c + nnode_macro 
dofs_s_an = 2*nnode_macro + nodes_anode
dofs_s_ca = 2*nnode_macro + nodes_cathode

dofs_pcs = onp.hstack((dofs_p,dofs_c,dofs_s))

dofs_j = dofs_s + nnode_macro
dofs_j_an = 3*nnode_macro + nodes_anode
dofs_j_ca = 3*nnode_macro + nodes_cathode



# ==================== The weak form ====================

class macro_P2D(Problem):
    def custom_init(self):
        self.fe_p = self.fes[0]
        self.fe_c = self.fes[1]
        self.fe_s = self.fes[2]
        self.fe_j = self.fes[3]
        
    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, 
                             cell_c_quad_old, cell_sol_micro_old, cell_tag, cell_nodes_tag, cell_nodes_sum):
            
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
            ratio_mat = (eps_mat)**(params_macro.alpha)
            ratio_inc = (eps_inc)**(params_macro.alpha)
            
            
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
        
            Rj = res_node_flux(cell_sol_list, cell_sol_micro_old, cell_nodes_tag, micro_problem, params_macro, params_micro)
            
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
        sol_c_old, sol_micro_old, cells_tag, cells_nodes_tag, cells_nodes_sum = params
        
        # (num_cells, num_quads)
        sol_c_old = self.fe_c.convert_from_dof_to_quad(sol_c_old.reshape((-1,1)))[:,:,0]
        # (num_cells, num_nodes, num_micro_nodes)
        sol_micro_old = sol_micro_old[self.cells_list[0]]
        
        self.internal_vars = [sol_c_old, sol_micro_old, cells_tag, cells_nodes_tag, cells_nodes_sum]




# ==================== Some helpful functions ====================

def get_macro_mesh(macro_mesh, ele_type):
    
    # mesh
    points = onp.array( macro_mesh['coords'], dtype=onp.float64)
    
    cells_anode = onp.array(macro_mesh['connect_anode']-1, dtype=onp.int32)
    cells_separator = onp.array(macro_mesh['connect_separator']-1, dtype=onp.int32)
    cells_cathode = onp.array(macro_mesh['connect_cathode']-1, dtype=onp.int32)
    cells = onp.vstack((cells_anode,cells_separator,cells_cathode))
    
    mesh = Mesh(points, cells, ele_type=ele_type)
    
    # cells tag
    cells_tag = onp.zeros((len(cells),1))
    cells_tag[0:len(cells_anode)] = 1
    cells_tag[-len(cells_cathode):] = 2
    
    # nodes tag
    nodes_tag = onp.zeros((len(points)))
    nodes_tag[nodes_anode] = 1
    nodes_tag[nodes_cathode] = 2
    
    # cell nodes tag
    cells_nodes_tag = onp.take(nodes_tag, cells)
    
    # repeat num of each node
    nodes,nodes_sum = onp.unique(cells,return_counts=True)
    cells_nodes_sum = onp.take(nodes_sum,cells)[:,:,None]
    
    return mesh, cells_tag, cells_nodes_tag, cells_nodes_sum, nodes_tag


def modify_init_sol(sol_macro_time, sol_micro_time, itime):
    '''
    This function copy the solution in MATLAB to replace the initial solution.
    
    '''
    if itime>=0:
        # itime = 1 # the end time step to be copied
        
        # macro
        macro_ref = scio.loadmat(os.path.join(input_dir, f'data/sol_macro.mat'))
        sol_macro_time[dofs_p,0:itime+1] = macro_ref['sol_p'][:,0:itime+1]
        sol_macro_time[dofs_c,0:itime+1] = macro_ref['sol_c'][:,0:itime+1] 
        sol_macro_time[dofs_s_an,0:itime+1] = macro_ref['sol_s_an'][:,0:itime+1] 
        sol_macro_time[dofs_s_ca,0:itime+1] = macro_ref['sol_s_ca'][:,0:itime+1]
        sol_macro_time[dofs_j_an,0:itime+1] = macro_ref['sol_j_an'][:,0:itime+1] 
        sol_macro_time[dofs_j_ca,0:itime+1] = macro_ref['sol_j_ca'][:,0:itime+1] 
        
        # micro
        micro_ref = scio.loadmat(os.path.join(input_dir, f'data/sol_micro.mat')) 
        sol_micro_time[nodes_anode,:,0:itime+1] = ((micro_ref['sol_micro_an']).transpose(2,0,1))[:,:,0:itime+1] 
        sol_micro_time[nodes_cathode,:,0:itime+1] = ((micro_ref['sol_micro_ca']).transpose(2,0,1))[:,:,0:itime+1] 
    
    return sol_macro_time, sol_micro_time


def verify_micro_problem(micro_problem, nodes_tag, sol_macro_time, sol_micro_time):
    '''
    To verify the micro problems
    '''
    itime = 2  # time step to verify
    
    from micro import Bulter_Volmer
    from micro import solve_micro_problem
    
    css = onp.ones((nnode_macro,1))
    j = onp.ones((nnode_macro,1))
    
    for ind, node_sets in enumerate([nodes_anode,nodes_separator,nodes_cathode]):
        for inode in node_sets:
            node_flux = sol_macro_time[dofs_j[inode],itime-1]
            sol_micro_old = sol_micro_time[inode,:,itime-1].reshape(-1,1)
            sol_micro = solve_micro_problem(micro_problem, params_micro, nodes_tag[inode], sol_micro_old, node_flux)
            css[inode] = sol_micro[micro_problem.bound_right]
            sol_p = sol_macro_time[dofs_p[inode],itime-1]
            sol_c = sol_macro_time[dofs_c[inode],itime-1]
            sol_s = sol_macro_time[dofs_s[inode],itime-1]
            j[inode] = Bulter_Volmer(sol_p, sol_c, sol_s, css[inode], nodes_tag[inode], params_macro)
    
    # data obtained from MATLAB
    node_flux_ref = scio.loadmat(os.path.join(input_dir, f'data/sol_j_t2.mat'))
    
    j_mat = onp.zeros((nnode_macro,1))
    j_mat[nodes_anode] = node_flux_ref['j0_an'] * node_flux_ref['BV_an']
    j_mat[nodes_cathode] = node_flux_ref['j0_ca'] * node_flux_ref['BV_ca']
    
    css_mat = onp.zeros((nnode_macro,1))
    css_mat[nodes_anode] = node_flux_ref['css_an']
    css_mat[nodes_cathode] = node_flux_ref['css_ca']
    
    plot_micro_verify_data(j,j_mat,css,css_mat,output_dir)
    
    return None


def switch_A_sp_jax(micro_problem, jax_flag):
    
    if jax_flag:
        micro_problem.A_an_inv = micro_problem.A_an_inv_jax
        micro_problem.A_ca_inv = micro_problem.A_ca_inv_jax
    else:
        micro_problem.A_an_inv = micro_problem.A_an_inv_sp
        micro_problem.A_ca_inv = micro_problem.A_ca_inv_sp
    
    return micro_problem

# ==================== The main function to solve P2D problems ====================


def macro_problem():
    
    '''
    Solve the macro P2D problem (phi_e, c_e, phi_s ,j)
    '''
    
    # ---- mesh object for JAX-FEM ----
    
    # macro
    ele_type = 'QUAD4'
    mesh, cells_tag, cells_nodes_tag, cells_nodes_sum, nodes_tag = get_macro_mesh(macro_mesh, ele_type)
    
    
    # ---- boundary conditions ----
    
    # only for the electrode potential (s) 
    
    min_x = np.min(mesh.points[:,0]) # x = 0.
    max_x = np.max(mesh.points[:,0]) # x = 225.
    
    # x = 100.
    # anright_x = onp.unique(mesh.points[(macro_mesh['bound_anright']-1).astype(onp.int32),0]) 
    # x = 125.
    # caleft_x = onp.unique(mesh.points[(macro_mesh['bound_caleft']-1).astype(onp.int32),0])   
    anright_x = 100.
    caleft_x = 125.
    
    def left(point):
        return np.isclose(point[0], min_x, atol=1e-5)
    
    def right(point):
        return np.isclose(point[0], max_x, atol=1e-5)
    
    def separator(point):
        return (point[0] > anright_x) & (point[0] < caleft_x)
    
    def dirichlet_s_location(point):
        return np.logical_or(left(point), separator(point))

    def zero_dirichlet(point):
        return 0.

    dirichlet_bc_info_s = [[dirichlet_s_location], [0], [zero_dirichlet]]
    dirichlet_bc_info_j = [[separator], [0], [zero_dirichlet]]
    
    location_fns_s = [right]
    
    
    # ---- macro problem ----
    
    # macro problem for (p,c,s,j)
    problem_macro = macro_P2D([mesh]*4, vec = [1]*4, dim=2, 
                        ele_type = [ele_type]*4, gauss_order=[2]*4,
                        dirichlet_bc_info = [None,None,dirichlet_bc_info_s,dirichlet_bc_info_j],
                        location_fns = location_fns_s)
    
    
    # ---- initial sol ----
    
    # macro: (4*nnode_macro, timesteps)
    # micro: (nnode_macro, nnode_micro, timesteps)
    
    sol_macro_time, sol_micro_time = assign_init_sol(macro_mesh, micro_mesh, timesteps, params_macro)
    
    mod_itime = 0
    sol_macro_time, sol_micro_time = modify_init_sol(sol_macro_time, sol_micro_time, mod_itime)
    
    
    # verify_micro_problem(micro_problem, nodes_tag, sol_macro_time, sol_micro_time)
    
    
    # ---- time evolution ----
    
    for itime in range(1, timesteps): # time steps [1,...,10]
    
        logger.debug(f'time step {itime}')
        
        switch_A_sp_jax(micro_problem, True)
        
        # ---- preparations ----
        
        # sol at the previous step
        if itime>0:
            sol_macro_old = sol_macro_time[:,itime-1]
            sol_micro_old = sol_micro_time[:,:,itime-1]
        else:
            sol_macro_old = sol_macro_time[:,itime]
            sol_micro_old = sol_micro_time[:,:,itime]
        # sol_crt = sol_old   
        initial_guess = sol_macro_old
        # c_crt - c_old = 0
        sol_c_old = sol_macro_old[dofs_c]
        # set parameters
        problem_macro.set_params([sol_c_old, sol_micro_old, cells_tag, cells_nodes_tag, cells_nodes_sum])
        
        
        # ---- solve ----
        
        sol_p, sol_c, sol_s, sol_j = solver(problem_macro, linear=False, precond=False, 
                                     initial_guess=initial_guess, use_petsc=True)
        
        # ---- update macro variables ----
        
        sol_macro_time[dofs_p,itime] = sol_p.reshape(-1)
        sol_macro_time[dofs_c,itime] = sol_c.reshape(-1)
        sol_macro_time[dofs_s,itime] = sol_s.reshape(-1)
        sol_macro_time[dofs_j,itime] = sol_j.reshape(-1)
        
        
        # ---- update micro variables ----
        switch_A_sp_jax(micro_problem, False)
        for ind, node_sets in enumerate([nodes_anode,nodes_cathode]):
            for inode in node_sets:
                node_flux = sol_macro_time[dofs_j[inode],itime]
                sol_micro_old = sol_micro_time[inode,:,itime-1].reshape(-1,1)
                sol_micro_time[inode,:,itime] = solve_micro_problem(micro_problem, params_micro, nodes_tag[inode], sol_micro_old, node_flux).reshape(-1)
        
        
    # ---- Post-processing ----
    
    sol_macro_list = [sol_macro_time[dofs_p,:],sol_macro_time[dofs_c,:],
                      sol_macro_time[dofs_s[nodes_anode],:],sol_macro_time[dofs_s[nodes_cathode],:],
                      sol_macro_time[dofs_j[nodes_anode],:],sol_macro_time[dofs_j[nodes_cathode],:]]
    
    sol_micro_list = [sol_micro_time[nodes_anode,:,:],sol_micro_time[nodes_cathode,:,:]]
    
    postprocess(mesh, sol_macro_list, sol_micro_list, input_dir, output_dir)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    
    logger.debug(f'Finish! Total time cost:{total_time:.2f}s')


if __name__ == "__main__":
    # main
    macro_problem()
