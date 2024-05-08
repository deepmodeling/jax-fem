'''
The main module for solving P2D problems with JAX-FEM

Initialize the data for P2D problems

Last modified: 20/04/2024

'''

import os
import numpy as onp
import scipy.io as scio
import jax.numpy as np

from para import data_sets


def prep_mesh_macro(mat_mesh_macro):
    
    # data class
    mesh = data_sets('mat_mesh_macro') 
    
    # nodes
    nnode_macro = int(mat_mesh_macro['nnode'])
    nodes_anode = (onp.unique(mat_mesh_macro['connect_anode'])-1).astype(onp.int32)
    nodes_cathode = (onp.unique(mat_mesh_macro['connect_cathode'])-1).astype(onp.int32)

    nodes_separator = (onp.unique(mat_mesh_macro['connect_separator'])-1).astype(onp.int32)
    bound_anright = (onp.unique(mat_mesh_macro['bound_anright'])-1).astype(onp.int32)
    ca_left = (onp.unique(mat_mesh_macro['bound_caleft'])-1).astype(onp.int32)
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
    
    # points
    points = onp.array( mat_mesh_macro['coords'], dtype=onp.float64)
    
    cells_anode = onp.array(mat_mesh_macro['connect_anode']-1, dtype=onp.int32)
    cells_separator = onp.array(mat_mesh_macro['connect_separator']-1, dtype=onp.int32)
    cells_cathode = onp.array(mat_mesh_macro['connect_cathode']-1, dtype=onp.int32)
    cells = onp.vstack((cells_anode,cells_separator,cells_cathode))
    
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
    
    # Assembly
    
    # nodes
    mesh.num_nodes = nnode_macro
    mesh.nodes_anode = nodes_anode
    mesh.nodes_cathode = nodes_cathode
    mesh.nodes_separator = nodes_separator
    
    # dofs
    mesh.dofs = [dofs_p, dofs_c, dofs_s, dofs_j]
    mesh.dofs_an = [dofs_p_an, dofs_c_an, dofs_s_an, dofs_j_an]
    mesh.dofs_ca = [dofs_p_ca, dofs_c_ca, dofs_s_ca, dofs_j_ca]
    
    # points
    mesh.points = points
    
    # cells
    mesh.cells = cells
    
    # others
    mesh.nodes_bound_anright = (mat_mesh_macro['bound_anright']-1).astype(onp.int32).reshape(-1)
    mesh.nodes_bound_caleft = (mat_mesh_macro['bound_caleft']-1).astype(onp.int32).reshape(-1)
    
    mesh.cells_vars = [cells_tag, cells_nodes_tag, cells_nodes_sum]
    mesh.nodes_vars = [nodes_tag]
    
    return mesh


def assign_init_sol(mesh_macro, problem_micro, params_macro, timesteps, input_dir, copy_time):
    
    # Get the initial solutions with the matched dimension for JAX-FEM
    
    # For p, c, s, j
    ndof_macro = 4 # QUAD4 elements
    nnode_macro = mesh_macro.num_nodes
    
    ndof_micro = 1 # Interval elements
    nnode_micro = problem_micro.num_nodes
    
    nodes_anode = mesh_macro.nodes_anode
    nodes_separator = mesh_macro.nodes_separator
    nodes_cathode = mesh_macro.nodes_cathode
    
    dofs_p, dofs_c, dofs_s, dofs_j = mesh_macro.dofs
    dofs_s_an = mesh_macro.dofs_an[2]
    dofs_s_ca = mesh_macro.dofs_ca[2]
    
    # (num_macro_nodes * num_dofs, timesteps)
    sol_init_macro = onp.zeros((ndof_macro*nnode_macro,timesteps)) 
    
    # The separator has no pore wall flux j, however, it is still considered here.
    # (num_macro_nodes * num_micro_nodes * num_dofs, timesteps)
    sol_init_micro = onp.zeros((nnode_macro, nnode_micro*ndof_micro, timesteps)) 
    
    # macro solution
    sol_init_macro[dofs_p,0] = params_macro.phi0_el
    sol_init_macro[dofs_c,0] = params_macro.c0_el
    sol_init_macro[dofs_s_an,0] = params_macro.phi0_an
    sol_init_macro[dofs_s_ca,0] = params_macro.phi0_ca
    
    # sol_init_macro[dofs_p,0] = -1*params_macro.phi0_an
    # sol_init_macro[dofs_c,0] = params_macro.c0_el
    # sol_init_macro[dofs_s_an,0] = 0
    # sol_init_macro[dofs_s_ca,0] = params_macro.phi0_ca - params_macro.phi0_an
    
    # micro solution
    sol_init_micro[nodes_anode,:,0] = params_macro.c0_an
    sol_init_micro[nodes_cathode,:,0] = params_macro.c0_ca
    
    # copy solution from MATLAB
    if copy_time>=0:
        sol_init_macro, sol_init_micro = modify_init_sol(mesh_macro, sol_init_macro, sol_init_micro, copy_time, input_dir)
    
    sol_init_macro = np.array(sol_init_macro)
    sol_init_micro = np.array(sol_init_micro)
        
    return sol_init_macro, sol_init_micro

def modify_init_sol(mesh_macro, sol_macro_time, sol_micro_time, copy_time, input_dir):
    '''
    This function copy the solution in MATLAB to replace the initial solution.
    
    '''
    dofs_p, dofs_c, dofs_s, dofs_j = mesh_macro.dofs
    dofs_s_an, dofs_j_an = mesh_macro.dofs_an[2:]
    dofs_s_ca, dofs_j_ca = mesh_macro.dofs_ca[2:]
    
    if copy_time>=0:
        
        # copy_time = 1 # the end time step to be copied
        
        # macro
        macro_ref = scio.loadmat(os.path.join(input_dir, f'data/sol_macro.mat'))
        sol_macro_time[dofs_p,0:copy_time+1] = macro_ref['sol_p'][:,0:copy_time+1]
        sol_macro_time[dofs_c,0:copy_time+1] = macro_ref['sol_c'][:,0:copy_time+1] 
        sol_macro_time[dofs_s_an,0:copy_time+1] = macro_ref['sol_s_an'][:,0:copy_time+1] 
        sol_macro_time[dofs_s_ca,0:copy_time+1] = macro_ref['sol_s_ca'][:,0:copy_time+1]
        sol_macro_time[dofs_j_an,0:copy_time+1] = macro_ref['sol_j_an'][:,0:copy_time+1] 
        sol_macro_time[dofs_j_ca,0:copy_time+1] = macro_ref['sol_j_ca'][:,0:copy_time+1] 
        
        # micro
        micro_ref = scio.loadmat(os.path.join(input_dir, f'data/sol_micro.mat')) 
        sol_micro_time[mesh_macro.nodes_anode,:,0:copy_time+1] = ((micro_ref['sol_micro_an']).transpose(2,0,1))[:,:,0:copy_time+1] 
        sol_micro_time[mesh_macro.nodes_cathode,:,0:copy_time+1] = ((micro_ref['sol_micro_ca']).transpose(2,0,1))[:,:,0:copy_time+1] 
    
    return sol_macro_time, sol_micro_time