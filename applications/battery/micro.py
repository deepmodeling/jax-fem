'''
This module is for solving microscopic diffusion problem of particle in P2D problem with 1D FEM.

The JAX-FEM package is not used here.

Last modified: 08/05/2024

'''
from dataclasses import dataclass

import os

import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO

from jax import config
config.update("jax_enable_x64", True)


import basix
import scipy
import numpy as onp
from matlab_fns import calcJ0, calcUoc, calcBV


@dataclass
class micro_core:
    '''
    to store micro problem-relateed variables
    '''
    dt:float # time step size


def prep_micro_problem(micro_mesh, params_micro, dt, stiff_flag='matlab'):
    '''
    This function initializes the mciro problem
    '''
    
    def basis_interval():
        '''
        For 1D element
        '''
        # Lagrange
        element_family = basix.ElementFamily.P
        basix_ele = basix.CellType.interval
        degree = 1
        gauss_order = 2
        
        # Reference domain
        # Following codes mainly come from jax_fem.basis
        quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
        element = basix.create_element(element_family, basix_ele, degree)
        vals_and_grads = element.tabulate(1, quad_points)[:, :, :, :]
        shape_values = vals_and_grads[0, :, :, 0]
        shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
        
        return shape_values, shape_grads_ref, weights, quad_points
    
    def get_shape_grads(problem):
        '''
        Get FEM-related variables in physical domain
        
        Follwing codes are mainly copied from jax_fem.fe.get_shape_grads
        '''
        physical_coos = onp.take(problem.points, problem.cells,
                                 axis=0)  # (num_cells, num_nodes, dim)
        # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
        jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] *
                                   problem.shape_grads_ref[None, :, :, None, :],
                                   axis=2,
                                   keepdims=True)
        jacobian_det = onp.linalg.det(
                            jacobian_dx_deta)[:, :, 0]  # (num_cells, num_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)
        # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
        # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
        shape_grads_physical = (problem.shape_grads_ref[None, :, :, None, :]
                                @ jacobian_deta_dx)[:, :, :, 0, :]
        JxW = jacobian_det * problem.quad_weights[None, :]
        
        # (num_cells, num_quads, num_nodes, 1, dim)
        v_grads_JxW = shape_grads_physical[:, :, :, None, :] * JxW[:, :, None, None, None]
        
        return shape_grads_physical, JxW, v_grads_JxW
    
    def micro_stiffness(problem, params_micro):
        ''''
        Return the global stiffness entities for micro problems
        
        '''
        
        def explicit_stiffness(dtco_inc, df):
            
            # (num_quads, num_nodes)
            shape_vals = problem.shape_vals
            # (num_cells, num_quads, num_nodes, dim)
            shape_grads = problem.shape_grads
            # (num_cells, num_quads)
            JxW = problem.JxW
            # (num_cells, num_quads, num_nodes, 1, dim)
            v_grads_JxW = problem.v_grads_JxW
            
            x = problem.quad_coords_physical
            
            # (num_cells, num_quads, 1, 1) * 
            # (1, num_quads, num_nodes, 1) @ (1, num_quads, 1, num_nodes) *
            # (num_cells, num_quads, 1, 1)
            # +
            # (num_cells, num_quads, 1, 1) * 
            # (num_cells, num_quads, num_nodes, 1) @ (num_cells, num_quads, 1, num_nodes) *
            # (num_cells, num_quads, 1, 1)
            
            V = (dtco_inc * 4 * onp.pi * x[:,:,None,None]**2 *
                    shape_vals[None,:,:,None] @ shape_vals[None,:,None,:] *
                    JxW[:,:,None,None]
                    +
                    df * 4 * onp.pi * x[:,:,None,None]**2 *
                    shape_grads[:,:,:,None,0] @ shape_grads[:,:,None,:,0] * JxW[:,:,None,None]
                    ) 
            V = onp.sum(V,axis=1)
            return V
        
        # anode
        dtco_inc = params_micro.dtco_inc_an
        df = params_micro.ds_an / params_micro.ds_ref_an
        V_an = explicit_stiffness(dtco_inc, df).reshape(-1)

        # cathode
        dtco_inc = params_micro.dtco_inc_ca
        df = params_micro.ds_ca / params_micro.ds_ref_ca
        V_ca = explicit_stiffness(dtco_inc, df).reshape(-1)
        
        return V_an, V_ca
    
    # variable sets
    problem = micro_core(dt)
    
    # Mesh
    problem.points = onp.array(micro_mesh['coords'], dtype=onp.float64)
    problem.cells = onp.array(micro_mesh['connect']-1, dtype=onp.int32)
    problem.num_nodes = len(problem.points)
    problem.ndofs = len(problem.points)
    problem.right = onp.max(problem.points) # x = 1.
    problem.bound_right = onp.array(micro_mesh['bound_right']-1, dtype=onp.int32).reshape(-1)
    
    # 1D FE
    problem.shape_vals, problem.shape_grads_ref, problem.quad_weights, problem.quad_points = basis_interval()
    problem.shape_grads, problem.JxW , problem.v_grads_JxW = get_shape_grads(problem)
    
    if stiff_flag == 'jax':
        # Notes: the quadrature interval in basix is [0,1]
        # (num_quads, )
        quad_coords_ref = problem.quad_points.reshape(-1)
        cells_coords = onp.take(problem.points, problem.cells)
        quad_coords_physical = ((cells_coords[:,1] - cells_coords[:,0])[:,None]* 
                                quad_coords_ref[None,:] + cells_coords[:,0][:,None])
        # (num_cells, num_quads)
        problem.quad_coords_physical = quad_coords_physical
        
    elif stiff_flag == 'matlab':
        # To be consistent with MATLAB. However, this seems wrong!
        quad_coords_ref = onp.array([-1/np.sqrt(3),1/np.sqrt(3)])
        # (num_cells, num_quads) --> (1, num_quads)
        problem.quad_coords_physical = quad_coords_ref[None,:]
    else:
        raise ValueError("Wrong input for 'stiff_flag'")
    
    # Assembly
    problem.I = onp.repeat(problem.cells,repeats=2,axis=1).reshape(-1)
    problem.J = onp.repeat(problem.cells,repeats=2,axis=0).reshape(-1)
    problem.V_an, problem.V_ca = micro_stiffness(problem, params_micro)
    
    problem.A_an = scipy.sparse.csr_matrix((problem.V_an, (problem.I, problem.J)), 
                                   shape=(problem.ndofs, problem.ndofs))
    problem.A_ca = scipy.sparse.csr_matrix((problem.V_ca, (problem.I, problem.J)), 
                                   shape=(problem.ndofs, problem.ndofs))
    

    problem.A_an_inv_sp = scipy.sparse.linalg.inv(problem.A_an)
    problem.A_ca_inv_sp = scipy.sparse.linalg.inv(problem.A_ca)

    problem.A_an_inv_jax = BCOO.from_scipy_sparse(scipy.sparse.linalg.inv(problem.A_an))
    problem.A_ca_inv_jax = BCOO.from_scipy_sparse(scipy.sparse.linalg.inv(problem.A_ca))
    
    
    problem.A_an_inv = np.array(problem.A_an_inv_sp.toarray(),dtype=np.float64)
    problem.A_ca_inv = np.array(problem.A_ca_inv_sp.toarray(),dtype=np.float64)
    
    
    # problem.A_an_inv = problem.A_an_inv_jax
    # problem.A_ca_inv = problem.A_ca_inv_jax
    
    return problem



def compute_micro_residual(problem, params_micro, nodes_tag, sol_micro, sol_micro_old):
    '''
    Compute the residual for micro problems
    '''
    
    # (num_cells, num_nodes, vec)
    sol_micro = np.take(sol_micro, problem.cells)[:,:,None]
    # (num_cells, num_nodes, vec)
    sol_micro_old = np.take(sol_micro_old, problem.cells)[:,:,None]
    
    # (num_quads, num_nodes)
    shape_vals = problem.shape_vals
    # (num_cells, num_quads, num_nodes, dim)
    shape_grads = problem.shape_grads
    # (num_cells, num_quads)
    JxW = problem.JxW
    # (num_cells, num_quads, num_nodes, 1, dim)
    v_grads_JxW = problem.v_grads_JxW
    # (num_cells, num_quads)
    x = problem.quad_coords_physical
    
    # parameters
    lag1 = -1*nodes_tag*(nodes_tag-2)       # anode - 1
    lag2 = 1/2*nodes_tag*(nodes_tag-1)      # cathode - 2
    lag3 = 1/2*(nodes_tag-1)*(nodes_tag-2)  # seperator - 0
    
    # (1,)
    dtco_inc = lag1 * params_micro.dtco_inc_an + lag2 * params_micro.dtco_inc_ca
    df = lag1 * (params_micro.ds_an / params_micro.ds_ref_an) + lag2 * (params_micro.ds_ca / params_micro.ds_ref_ca)
    
    # residual
    # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
    sol_micro_crt = np.sum(sol_micro[:,None,:,:] * shape_vals[None,:,:,None], axis=2)
    sol_micro_old = np.sum(sol_micro_old[:,None,:,:] * shape_vals[None,:,:,None], axis=2)
    d_cs = sol_micro_crt - sol_micro_old
    
    # (num_cells, num_quads, 1, 1) * (num_cells, num_quads, 1, vec) *
    # (1, num_quads, num_nodes, 1) * (num_cells, num_quads, 1, 1) --> (num_cells,num_nodes,vec)
    res = 4 * np.pi * dtco_inc * np.sum(x[:,:,None,None]**2 * d_cs[:,:,None,:] 
                                        * shape_vals[None,:,:,None] * JxW[:,:,None,None],axis=1)
    # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, vec, dim)
    cs_grads = np.sum(sol_micro[:, None, :, :, None] * shape_grads[:, :, :, None, :],axis=2)
    # (num_cells, num_quads, 1, 1, 1) * (num_cells, num_quads, 1, vec, dim) * (num_cells, num_quads, num_nodes, 1, dim) -> 
    res = res + 4 * np.pi * df * np.sum(x[:,:,None,None,None]**2 * cs_grads[:,:,None,:,:] * v_grads_JxW, axis=(1,-1))
    
    weak_form = np.zeros((len(problem.points)))
    weak_form = weak_form.at[problem.cells.reshape(-1)].add(res.reshape(-1))
    
    return weak_form.reshape(-1,1)


def solve_micro_problem(problem, params_micro, node_tag, sol_micro_old, node_flux):
    '''
    Solve the micro problem
    '''
    # parameters
    lag1 = -1*node_tag*(node_tag-2)       # anode - 1
    lag2 = 1/2*node_tag*(node_tag-1)      # cathode - 2
    lag3 = 1/2*(node_tag-1)*(node_tag-2)  # seperator - 0
    
    # micro stiffness
    A_inv = lag1 * problem.A_an_inv + lag2 * problem.A_ca_inv
    
    # neumann
    q_bc = lag1 * params_micro.q_bc_an + lag2 * params_micro.q_bc_ca
    
    def newton_update(sol_micro):
        res = compute_micro_residual(problem, params_micro, node_tag, sol_micro, sol_micro_old)
        # apply neumann BCs
        res = res.at[problem.bound_right].add(q_bc * node_flux)
        res_val = np.linalg.norm(res)
        return res, res_val
    
    # previous time step solutions as initial solutions
    sol_micro = sol_micro_old
    res, res_val = newton_update(sol_micro)
    # jax.debug.print('Solve micro problem tag {}', node_tag)
    # jax.debug.print('Before  residual values: {}',res_val)
    tol = 1e-6
    num = 0
    # while res_val > tol:
    num = num + 1
    sol_micro_inc = A_inv @ (-res)
    sol_micro = sol_micro + sol_micro_inc
    res, res_val = newton_update(sol_micro)
    # jax.debug.print('Iter {}  Micro-res: {}',num, res_val)
    return sol_micro



def Bulter_Volmer(sol_p, sol_c, sol_s, css, node_tag, params_macro):
    '''
    Bulter-Volmer equation
    '''
    # j0
    j0 = calcJ0(sol_c, css, node_tag, params_macro)
    # Uoc
    Uoc = calcUoc(css, node_tag, params_macro)
    # eta
    eta = sol_s - sol_p - Uoc
    # BV
    BV = calcBV(eta)
    # j
    sol_j = j0 * BV
    return sol_j



def res_node_flux(cell_sol_list, sol_micro_old, cell_tag, micro_problem, params_macro, params_micro):
    '''
    To calculate the residual for the nodal variable pore wall flux j
    
    which requires the micro solution and the BV equation.
    
    '''
    
    # These variables are all defined on the element nodes!
    cell_sol_p, cell_sol_c, cell_sol_s, cell_sol_j = cell_sol_list

    # micro Li+ on the surface
    j_BV = np.zeros((len(cell_sol_j),1))
    
    for i in range(len(cell_sol_j)):
        sol_micro = solve_micro_problem(micro_problem, params_micro, 
                                                cell_tag[i], sol_micro_old[i,:].reshape(-1,1), cell_sol_j[i])
        css = sol_micro[micro_problem.bound_right].reshape(-1)
        
        # BV
        j = Bulter_Volmer(cell_sol_p[i], cell_sol_c[i], cell_sol_s[i], css, cell_tag[i], params_macro)
        j_BV = j_BV.at[i].set(j)
        
    # Res
    Rj = cell_sol_j - j_BV
    
    return Rj








