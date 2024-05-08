'''
Some useful functions for solving P2D problems with JAX-FEM.

Last modified: 29/02/2024

'''
import os
import numpy as onp
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def verify_micro_problem(mesh_macro, problem_micro, params_macro, params_micro, nodes_tag, 
                         sol_macro_time, sol_micro_time, input_dir, output_dir):
    '''
    To verify the micro problems
    
    Move the follwoing codes into 'main.py':
    
    # # copy_time = 1
    # from utils import verify_micro_problem
    # switch_A_sp_jax(problem_micro, False)
    # verify_micro_problem(mesh_macro, problem_micro, params_macro, params_micro, nodes_tag, 
    #                      sol_macro_time, sol_micro_time, input_dir, output_dir)
    
    '''
    itime = 2  # time step to verify
    
    from micro import Bulter_Volmer
    from micro import solve_micro_problem
    
    dofs_p, dofs_c, dofs_s, dofs_j = mesh_macro.dofs
    nnode_macro = mesh_macro.num_nodes
    nodes_anode = mesh_macro.nodes_anode
    nodes_separator = mesh_macro.nodes_separator
    nodes_cathode = mesh_macro.nodes_cathode
    
    css = onp.ones((nnode_macro,1))
    j = onp.ones((nnode_macro,1))
    
    for ind, node_sets in enumerate([nodes_anode,nodes_separator,nodes_cathode]):
        for inode in node_sets:
            node_flux = sol_macro_time[dofs_j[inode],itime-1]
            sol_micro_old = sol_micro_time[inode,:,itime-1].reshape(-1,1)
            sol_micro = solve_micro_problem(problem_micro, params_micro, nodes_tag[inode], sol_micro_old, node_flux)
            css[inode] = sol_micro[problem_micro.bound_right]
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
    
    from utils import plot_micro_verify_data
    plot_micro_verify_data(j,j_mat,css,css_mat,output_dir)
    
    return None


def show_macro_mesh(cells_an, cells_sp, cells_ca, points, fc='w', ec='k'):
    # Show the macro
    fig, ax = plt.subplots()
    
    # Separator
    for i in range(len(cells_sp)):
        cell_points = points[cells_sp[i,:], :]
        poly = pat.Polygon(cell_points, facecolor='w', edgecolor='k')
        plt.gca().add_patch(poly)

    
    # Anode
    for i in range(len(cells_an)):
        cell_points = points[cells_an[i,:], :]
        poly = pat.Polygon(cell_points, facecolor='w', edgecolor='r')
        plt.gca().add_patch(poly)

    # Cathode
    for i in range(len(cells_ca)):
        cell_points = points[cells_ca[i,:], :]
        poly = pat.Polygon(cell_points, facecolor='w', edgecolor='b')
        plt.gca().add_patch(poly)
    
    plt.text(25, 15, 'anode')
    plt.text(98, 15, 'seperator')
    plt.text(175, 15, 'cathode')
    
    
    ax.set_xlim(-5,230)
    ax.set_xticks([0,50,100,150,200,225])
    ax.set_ylim(-20,30)
    ax.set_yticks([0,10])
    ax.set_aspect(1)
    
    plt.show()
    
    return fig


def show_micro_mesh(cells, points, lc='b'):
    # Show the micro
    fig, ax = plt.subplots()
    for i in range(len(cells)):
        cell_points = points[cells[i,:], :]
        cell_points = onp.hstack((cell_points, cell_points*0))
        plt.plot(cell_points[:,0],cell_points[:,1], color=lc)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylim(-0.05,0.05)
    ax.set_yticks([])
    plt.show()
    
    return fig


def plot_macro_micro_mesh(macro_mesh, micro_mesh, output_dir):
    
    # macro mesh
    fig = show_macro_mesh(macro_mesh['connect_anode']-1, 
                          macro_mesh['connect_separator']-1,
                          macro_mesh['connect_cathode']-1,
                          macro_mesh['coords'])
    fig.savefig(os.path.join(output_dir, f'mesh/macro_mesh.svg'),
                dpi=300, format="svg",bbox_inches='tight')
    
    # mciro mesh
    fig = show_micro_mesh(micro_mesh['connect']-1, 
                          micro_mesh['coords'])
    fig.savefig(os.path.join(output_dir, f'mesh/micro_mesh.svg'),
                dpi=300, format="svg",bbox_inches='tight')

    return None



def plot_micro_verify_data(j,j_mat,css,css_mat,output_dir):
    
    # j
    fig, ax = plt.subplots()
    plt.plot(j,color='r',linestyle='-',label='JAX-FEM')
    plt.plot(j_mat,color='b',linestyle='--',label='MATLAB')
    plt.text(10,-2.7e-5,
            f'Max abs error:{onp.max(onp.abs(j-j_mat)):.4e}')
    plt.title("pore wall flux $j$")
    plt.xlabel("nodes index")
    plt.ylabel("$j$")
    plt.xlim(0, 140)
    plt.ylim(-3e-5, 3e-5)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(output_dir, f'data/sol_j_t2.svg'),
                dpi=300, format="svg",bbox_inches='tight')
    
    # # scatter
    # fig, ax = plt.subplots()
    # plt.scatter(mesh.points[:,0], sol_j, color='none', 
    #             marker='o', edgecolors='r', s=8)
    # plt.title("pore wall flux $j$")
    # plt.xlabel("coordinates ($x$)")
    # plt.ylabel("$j$")
    # plt.xlim(0, 225)
    # plt.ylim(-3e-5, 3e-5)
    # fig.savefig(os.path.join(output_dir, f'data/sol_j_x_t2.svg'),
    #             dpi=300, format="svg",bbox_inches='tight')
    
    # css
    fig, ax = plt.subplots()
    plt.plot(css,color='r',linestyle='-',label='JAX-FEM')
    plt.plot(css_mat,color='b',linestyle='--',label='MATLAB')
    plt.text(10,22000,
            f'Max abs error:{onp.max(onp.abs(css-css_mat)):.4e}')
    plt.title("micro surface concentration of $Li^+$")
    plt.xlabel("nodes index")
    plt.ylabel("$c_{s,s}$")
    plt.xlim(0, 140)
    plt.ylim(-1000, 26000)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(output_dir, f'data/sol_css_t2.svg'),
                dpi=300, format="svg",bbox_inches='tight')
    
    # scatter
    # fig, ax = plt.subplots()
    # plt.scatter(mesh.points[:,0], css, color='none', 
    #             marker='o', edgecolors='r', s=8)
    # plt.title("micro surface concentration of $Li^+$")
    # plt.xlabel("coordinates ($x$)")
    # plt.ylabel("$c_{s,s}$")
    # plt.xlim(0, 225)
    # plt.ylim(-1000, 26000)
    # fig.savefig(os.path.join(output_dir, f'data/sol_css_x_t2.svg'),
    #             dpi=300, format="svg",bbox_inches='tight')
        
    
    return None



def output_sol(mesh_macro, sol_macro_time, sol_micro_time, input_dir, output_dir):
    
    dofs_p, dofs_c, dofs_s, dofs_j = mesh_macro.dofs
    nodes_anode = mesh_macro.nodes_anode
    nodes_cathode = mesh_macro.nodes_cathode
    nodes_separator = mesh_macro.nodes_separator
    
    # solution obtained from JAX-FEM
    sol_p = sol_macro_time[dofs_p,:]
    sol_c = sol_macro_time[dofs_c,:]
    sol_s_an = sol_macro_time[dofs_s[nodes_anode],:]
    sol_s_ca = sol_macro_time[dofs_s[nodes_cathode],:]
    sol_j_an = sol_macro_time[dofs_j[nodes_anode],:]
    sol_j_ca = sol_macro_time[dofs_j[nodes_cathode],:]
    
    sol_micro_an, sol_micro_ca = [sol_micro_time[nodes_anode,:,:],sol_micro_time[nodes_cathode,:,:]]
    
    # solution obtained from MATLAB
    macro_ref = scio.loadmat(os.path.join(input_dir, f'data/sol_macro.mat'))
    sol_p_mat =  macro_ref['sol_p']
    sol_c_mat =  macro_ref['sol_c']
    sol_s_an_mat =  macro_ref['sol_s_an']
    sol_s_ca_mat =  macro_ref['sol_s_ca']
    sol_j_an_mat =  macro_ref['sol_j_an']
    sol_j_ca_mat =  macro_ref['sol_j_ca']
    
    micro_ref = scio.loadmat(os.path.join(input_dir, f'data/sol_micro.mat')) 
    sol_micro_an_mat =  micro_ref['sol_micro_an'].transpose(2,0,1)
    sol_micro_ca_mat =  micro_ref['sol_micro_ca'].transpose(2,0,1)
    
    # p
    fig, ax = plt.subplots()
    plt.plot(sol_p[:,-1],color='r',linestyle='-',label='JAX-FEM')
    plt.plot(sol_p_mat[:,-1],color='b',linestyle='--',label='MATLAB')
    # plt.xlabel("node index")
    plt.ylabel("electrolyte phase potential $\phi_e$")
    plt.title("$\phi_e$ (t = 10s)")
    plt.text(20,-0.185,
            f'Max relative error:{onp.max(onp.abs(sol_p[:,-1]-sol_p_mat[:,-1])/onp.abs(sol_p_mat[:,-1])):.8%}')
    plt.ylim(-0.21, -0.18)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(output_dir, f'data/sol_p_t10.svg'),
                dpi=300, format="svg",bbox_inches='tight')
    
    
    # c
    fig, ax = plt.subplots()
    plt.plot(sol_c[:,-1],color='r',linestyle='-',label='JAX-FEM')
    plt.plot(sol_c_mat[:,-1],color='b',linestyle='--',label='MATLAB')
    # plt.xlabel("node index")
    plt.ylabel("electrolyte phase concentration $c_e$")
    plt.title("$c_e$ (t = 10s)")
    plt.text(20,1060,
            f'Max relative error:{onp.max(onp.abs(sol_c[:,-1]-sol_c_mat[:,-1])/onp.abs(sol_c_mat[:,-1])):.8%}')
    plt.ylim(920, 1080)
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig(os.path.join(output_dir, f'data/sol_c_t10.svg'),
                dpi=300, format="svg",bbox_inches='tight')
    
    
    # s_an
    nonzero_index = onp.setdiff1d(onp.linspace(0,62,63), onp.array([0,3,42])).astype(onp.int32)
    fig, ax = plt.subplots()
    plt.plot(sol_s_an[:,-1],color='r',linestyle='-',label='JAX-FEM')
    plt.plot(sol_s_an_mat[:,-1],color='b',linestyle='--',label='MATLAB')
    # plt.xlabel("node index")
    plt.ylabel("solid phase potential $\phi_s$")
    plt.title("$\phi_s$ for anode (t = 10s)")
    plt.text(10,0.5e-5,
            f'Max relative error:{onp.max(onp.abs(sol_s_an[nonzero_index,-1]-sol_s_an_mat[nonzero_index,-1])/onp.abs(sol_s_an_mat[nonzero_index,-1])):.8%}')
    plt.ylim(-4e-5, 1e-5)
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig(os.path.join(output_dir, f'data/sol_s_an_t10.svg'),
                dpi=300, format="svg",bbox_inches='tight')
    
    
    # s_ca
    fig, ax = plt.subplots()
    plt.plot(sol_s_ca[:,-1],color='r',linestyle='-',label='JAX-FEM')
    plt.plot(sol_s_ca_mat[:,-1],color='b',linestyle='--',label='MATLAB')
    # plt.xlabel("node index")
    plt.ylabel("electrode potential $\phi_s$")
    plt.title("$\phi_s$ for cathode (t = 10s)")
    plt.text(10,4.133+0.00055,
            f'Max relative error:{onp.max(onp.abs(sol_s_ca[:,-1]-sol_s_ca_mat[:,-1])/onp.abs(sol_s_ca_mat[:,-1])):.8%}')
    plt.ylim(4.1330, 4.1336)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(output_dir, f'data/sol_s_ca_t10.svg'),
                dpi=300, format="svg",bbox_inches='tight')

    
    # j_an
    fig, ax = plt.subplots()
    plt.plot(sol_j_an[:,-1],color='r',linestyle='-',label='JAX-FEM')
    plt.plot(sol_j_an_mat[:,-1],color='b',linestyle='--',label='MATLAB')
    plt.xlabel("node index")
    plt.ylabel("pore wall flux $j$")
    plt.title("$j$ for anode (t = 10s)")
    plt.text(10,2.65e-5,
            f'Max relative error:{onp.max(onp.abs(sol_j_an[:,-1]-sol_j_an_mat[:,-1])/onp.abs(sol_j_an_mat[:,-1])):.8%}')
    plt.ylim(1.3e-5, 2.8e-5)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(output_dir, f'data/sol_j_an_t10.svg'),
                dpi=300, format="svg",bbox_inches='tight')
    
    # j_ca
    fig, ax = plt.subplots()
    plt.plot(sol_j_ca[:,-1],color='r',linestyle='-',label='JAX-FEM')
    plt.plot(sol_j_ca_mat[:,-1],color='b',linestyle='--',label='MATLAB')
    plt.xlabel("node index")
    plt.ylabel("pore wall flux $j$")
    plt.title("$j$ for cathode (t = 10s)")
    plt.text(10,-1.8e-5,
            f'Max relative error:{onp.max(onp.abs(sol_j_an[:,-1]-sol_j_an_mat[:,-1])/onp.abs(sol_j_ca_mat[:,-1])):.8%}')
    plt.ylim(-2.5e-5, -1.7e-5)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(output_dir, f'data/sol_j_ca_t10.svg'),
                dpi=300, format="svg",bbox_inches='tight')
    
    
    
    # # c_e in PDF
    # fig, ax = plt.subplots()
    # plt.scatter(mesh.points[:,0], sol_c_mat[:,-1], color='none', 
    #             marker='o', edgecolors='r', s=8)
    
    # plt.title("$Li^+$ concentration in electrolyte ($t = 11$s)")
    # plt.xlabel("coordinates ($x$)")
    # plt.ylabel("$c_e$")
    # plt.xlim(0, 225)
    # # plt.ylim(900, 1100)
    # plt.ylim(600, 1400)
    # plt.show()
    # fig.savefig(os.path.join(output_dir, f'data/sol_c.svg'),
    #             dpi=300, format="svg",bbox_inches='tight')
    
    return None



# ------------- Some test codes for main.py -------------


# Objevtive curve

# theta = np.linspace(1.-2e-4, 1.+2e-4, 5)
# J_curve = np.zeros_like(theta)
# for i in range(len(J_curve)):
#     J_curve = J_curve.at[i].set(test_fun_grad(theta[i]))
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# plt.plot(theta, J_curve, color='r',linestyle='-',marker='o')
# plt.text(0.15, 0.9, f'slope:{(J_curve[0]-J_curve[-1])/(theta[0]-theta[-1]):.8e}',transform=ax.transAxes)
# plt.text(0.05, 0.55, f'slope:{(J_curve[1]-J_curve[-2])/(theta[1]-theta[-2]):.8e}',transform=ax.transAxes)
# ax.set_xticks(theta)
# plt.xlabel("$\\theta$")
# plt.ylabel("objective")
# plt.title("t=10s")
# plt.show()


# Get the initial solution (MATLAB)

# def get_init_sol_mat(sol_macro_time, sol_micro_time, cells_theta):
#     '''
#     get the initial solution for the first time step (MATLAB)
#     '''
#     init_val_p = lambda point: params_macro.phi0_el
#     init_val_s_an = lambda point: params_macro.phi0_an
#     init_val_s_ca = lambda point: params_macro.phi0_ca
    
#     dirichlet_p_init = [[left], [0], [init_val_p]]
#     dirichlet_s_init = [[left, right, separator], [0]*3, [init_val_s_an, init_val_s_ca, zero_dirichlet]]
    
#     problem_macro_init = macro_P2D([jax_mesh]*4, vec = [1]*4, dim=2, 
#                                     ele_type = [ele_type]*4, gauss_order=[2]*4,
#                                     dirichlet_bc_info = [dirichlet_p_init,None,dirichlet_s_init,dirichlet_bc_info_j],
#                                     location_fns = [lambda point:False])
    
#     sol_macro_old = sol_macro_time[:,0]
#     sol_micro_old = sol_micro_time[:,:,0]   
#     sol_c_old = sol_macro_old[dofs_c] # c_crt - c_old = 0
    
#     problem_macro_init.set_params([cells_theta, sol_macro_old, sol_c_old, sol_micro_old, cells_tag, cells_nodes_tag, cells_nodes_sum])
    
#     sol_p, sol_c, sol_s, sol_j = solver(problem_macro_init, linear=False, precond=True, 
#                                  initial_guess=sol_macro_old, use_petsc=True)
    
#     return sol_macro_time, sol_micro_time

    