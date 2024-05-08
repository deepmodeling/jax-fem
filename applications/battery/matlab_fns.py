'''
The following functions are refered to P2D codes in MATLAB (Zhou, 2019)

Last modified: 29/02/2024

'''

import numpy as onp
import jax.numpy as np

def calcUoc(c_ss,node_tag,params_macro):
    
    # F = 96485.33289; % Faraday's constant (C/mol)
    # R = 8.31447; % Gas constant
    # T = 373.15; % Absolute temperature
    
    # maximum conc in cathode active material (mol/m^3)
    
    c_max = params_macro.c_max;
    
    # parameters
    lag1 = -1*node_tag*(node_tag-2)       # anode - 1
    lag2 = 1/2*node_tag*(node_tag-1)      # cathode - 2
    lag3 = 1/2*(node_tag-1)*(node_tag-2)  # seperator - 0
    
    # alias
    exp = np.exp
    tanh = np.tanh
    
    # anode
    y = c_ss / c_max[0]
    Uoc_an = (0.194 + 1.5 * exp(-120.0 * y)
            +0.0351 * tanh((y - 0.286) / 0.083) 
            -0.0045 * tanh((y - 0.849) / 0.119) 
            -0.035 * tanh((y - 0.9233) / 0.05) 
            -0.0147 * tanh((y - 0.5) / 0.034) 
            -0.102 * tanh((y - 0.194) / 0.142) 
            -0.022 * tanh((y - 0.9) / 0.0164) 
            -0.011 * tanh((y - 0.124) / 0.0226) 
            +0.0155 * tanh((y - 0.105) / 0.029))
    
    # cathode
    y = c_ss / c_max[1]
    Uoc_ca = (2.16216 + 0.07645 * tanh(30.834 - 54.4806 * y) 
           +2.1581 * tanh(52.294 - 50.294 * y) 
           -0.14169 * tanh(11.0923 - 19.8543 * y) 
           +0.2051 * tanh(1.4684 - 5.4888 * y) 
           +0.2531 * tanh((-y + 0.56478) / 0.1316) 
           -0.02167 * tanh((y - 0.525) / 0.006))
    
    Uoc = lag1 * Uoc_an + lag2 * Uoc_ca
    
    return Uoc
    

def calcKappa(c):
    
    # coeffieients from Zhang, Du
    a0 = 0.0911;
    a1 = 1.9101e-3;
    a2 = -1.052e-6;
    a3 = 0.1554e-9;
    
    kappa = a0 + a1 * c + a2 * c**2 + a3 * c**3;
    
    return kappa


def calcKappa_Deriv(c):
    
    # coeffieients from Zhang, Du
    
    a0 = 0.0911;
    a1 = 1.9101e-3;
    a2 = -1.052e-6;
    a3 = 0.1554e-9;
    
    kappa      = a0 + a1 * c + a2 * c**2 + a3 * c**3;
    kappaDeriv = a1 + 2 * a2 * c + 3 * a3 * c**2;
    
    return kappa, kappaDeriv


def calcJ0(c_e, c_ss, node_tag, params_macro):
    
    # input c_s, c_e should have unit of A/m^2
    # i = 1 or 2,
    # 1-anode_spe interface,
    # 2-cathode_spe interface
    
    # F = 96485.33289; % Faraday's constant (C/mol)
    
    # parameters
    lag1 = -1*node_tag*(node_tag-2)       # anode - 1
    lag2 = 1/2*node_tag*(node_tag-1)      # cathode - 2
    lag3 = 1/2*(node_tag-1)*(node_tag-2)  # seperator - 0
    
    # nominal Reaction rates (A/m^2)*(mol^3/mol)^(1+alpha)
    k_s = lag1 * params_macro.k_s[0] + lag2 * params_macro.k_s[1]
    
    # maximum conc in cathode active material (mol/m^3)
    c_max = lag1 * params_macro.c_max[0] + lag2 * params_macro.c_max[1] 
    
    alpha_a = 0.5
    alpha_c = 0.5
    
    j0 = k_s * (c_max - c_ss)**alpha_a * (c_e)**alpha_a * (c_ss)**alpha_c
    
    return j0



def calcBV(eta):
    
    F = 96485.33289; # Faraday's constant (C/mol)
    R = 8.31447;     # Gas constant
    T = 298.15;      # Absolute temperature
    
    BV = np.exp(0.5 * F / R / T * eta ) - np.exp(-0.5 * F / R / T * eta );
    
    return BV





# -----------------------------------------------------------------------------
#
#
# Follwing functions are aborted ... (2024.02)
#
#
#
# -----------------------------------------------------------------------------


# def assign_dofs_init_sol(macro_mesh, timesteps, params_macro):
    
#     # This function combines 'assignDofs.m' and initial sol assignment operation
    
#     # The structure of the return value does not fit the JAX-FEM

#     nnode = int(macro_mesh['nnode'])
#     ndof = 4 # QUAD4 elements
    
#     nodes_anode = onp.unique(macro_mesh['connect_anode'])
#     nodes_separator = onp.unique(macro_mesh['connect_separator'])
#     nodes_cathode = onp.unique(macro_mesh['connect_cathode'])
    
#     dofArray = onp.zeros((nnode, ndof))
    
#     # for phi_e at all nodes
#     dofArray[:, 0] = onp.linspace(1,nnode,nnode)-1
    
#     # for phi_c at all nodes
#     dofArray[:, 1] = onp.linspace(1,nnode,nnode)-1 + nnode
    
#     ndofs = 2 * nnode
    
#     # for phi_s at nodes in anode and cathode,
#     # including interface nodes shared with electrolyte
#     dofArray[nodes_anode-1, 2] = ndofs +  onp.linspace(1,len(nodes_anode),len(nodes_anode))-1
#     ndofs = ndofs + len(nodes_anode)
    
#     dofArray[nodes_cathode-1, 2] = ndofs +  onp.linspace(1,len(nodes_cathode),len(nodes_cathode))-1
#     ndofs = ndofs + len(nodes_cathode)
    
    
#     # for j at nodes in anode and cathode,
#     # including interface nodes shared with electrolyte
#     dofArray[nodes_anode-1, 3] = ndofs +  onp.linspace(1,len(nodes_anode),len(nodes_anode))-1
#     ndofs = ndofs + len(nodes_anode)
    
#     dofArray[nodes_cathode-1, 3] = ndofs +  onp.linspace(1,len(nodes_cathode),len(nodes_cathode))-1
#     ndofs = ndofs + len(nodes_cathode)
    
#     # Transform ndarray type
#     dofArray = dofArray.astype(int)
    
#     # potential dofs electrolyte
#     dofsPhi_e = dofArray[:,0]
    
#     # concentration dofs electrolyte
#     dofsC_e = dofArray[:,1]
    
#     # potential dofs solid particle
#     dofsPhi_an = dofArray[nodes_anode-1, 2]
#     dofsPhi_ca = dofArray[nodes_cathode-1, 2]
    
#     # pore wall flux j
#     ij =  onp.concatenate((nodes_anode,nodes_cathode))
#     dofsFlux = dofArray[ij-1,3]
    
    
#     # Assign the initial solutions for all dofs
    
#     # The return values should be transfered to jax_fem.solver()
#     # i.e., jax_fem.solver(..., initial_guess = sol_init, ...)
    
#     sol_init = onp.zeros((ndofs,timesteps)) # initial solutions
#     sol_init[dofsC_e,0] = params_macro.c0_el
#     sol_init[dofsPhi_e,0] = params_macro.phi0_el
#     sol_init[dofsPhi_an,0] = params_macro.phi0_an
#     sol_init[dofsPhi_ca,0] = params_macro.phi0_ca
    
#     return ndofs, dofsPhi_e, dofsC_e, dofsPhi_an, dofsPhi_ca, dofsFlux, sol_init
