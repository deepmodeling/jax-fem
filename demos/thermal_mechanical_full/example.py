"""This example refers to
https://comet-fenics.readthedocs.io/en/latest/demo/thermoelasticity/thermoelasticity_transient.html

Also, see jax-fem/demos/thermal_mechanical_full/fenics.py
"""

# Import some useful modules.
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio


# Import JAX-FEM specific modules.
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file
from jax_fem.problem import Problem


# Define the coupling problems.
class thermal_mechanical_full(Problem):
    
    def custom_init(self):
        self.fe_u = self.fes[0]
        self.fe_T = self.fes[1]
    
    def set_params(self, params):
        sol_u_old, sol_dT_old, dt = params
        self.internal_vars = [self.fe_u.sol_to_grad(sol_u_old), 
                              self.fe_T.convert_from_dof_to_quad(sol_dT_old),
                              dt * np.ones(self.num_cells)]
    
    def get_universal_kernel(self):
        
        def strain(u_grad):
            return 0.5 * (u_grad + u_grad.T)
        
        def stress(u_grad, dT):
                   
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon \
                - kappa * dT * np.eye(self.dim)
            
            return sigma
        
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, u_grads_old, T_old, dt):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            
            ## Split
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol_u, cell_sol_T = cell_sol_list
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_u, cell_shape_grads_T = cell_shape_grads_list
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_u, cell_v_grads_JxW_T = cell_v_grads_JxW_list
            cell_JxW_u, cell_JxW_T = cell_JxW[0], cell_JxW[1]
            
            ## Handles the term 'C * (T_crt-T_old)/dt * Q * dx'
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec)
            T = np.sum(cell_sol_T[None,:,:] * self.fe_T.shape_vals[:,:,None],axis=1)
            # (num_quads, 1, num_vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            val1 = C /dt * np.sum((T - T_old)[:,None,:] * self.fe_T.shape_vals[:,:,None] * cell_JxW_T[:,None,None],axis = 0)
            
            ## Handles the term 'kappa * T0 * tr((epsilon_crt-epsilon_old)/dt) * Q * dx'
            # (num_quads, vec, dim)
            epsilon_old = jax.vmap(strain)(u_grads_old)
            u_grads = np.sum(cell_sol_u[None,:,:,None] * cell_shape_grads_u[:,:,None,:], axis=1)
            epsilon = jax.vmap(strain)(u_grads)
            # (num_quads,)
            dtr_dt = np.trace((epsilon-epsilon_old)/dt,axis1=1,axis2=2)
            # (num_quads, 1) * (num_quads, num_nodes) * (num_quads, 1) -> (num_nodes,) -> (num_nodes, vec)
            val2 = kappa * T0 * np.sum(dtr_dt[:,None] * self.fe_T.shape_vals *cell_JxW_T[:,None],axis=0)[:,None]
            
            ## Handles the term 'k * inner(grad(T_crt),grad(Q)) * dx'
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim) 
            # -> (num_quads, vec, dim)
            T_grads = np.sum(cell_sol_T[None,:,:,None] * cell_shape_grads_T[:,:,None,:], axis=1)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) ->  (num_nodes, vec) 
            val3 = np.sum(k * T_grads[:,None,:,:] * cell_v_grads_JxW_T,axis=(0,-1))
            
            ## Handles the term 'inner(sigma,grad(v)) * dx'
            u_physics = jax.vmap(stress)(u_grads,T)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) ->  (num_nodes, vec) 
            val4 = np.sum(u_physics[:,None,:,:] * cell_v_grads_JxW_u,axis=(0,-1))
        
            weak_form = [val4, val1 + val2 + val3]
            
            return jax.flatten_util.ravel_pytree(weak_form)[0]
        
        return universal_kernel
    

# A little program to find orientation of 3 points
# Coplied from https://www.geeksforgeeks.org/orientation-3-ordered-points/
class Point:
    # to store the x and y coordinates of a point
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
def orientation(p1, p2, p3):
    # To find the orientation of  an ordered triplet (p1,p2,p3) function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
    val = (float(p2.y - p1.y) * (p3.x - p2.x)) - (float(p2.x - p1.x) * (p3.y - p2.y))
    if (val > 0):
        # Clockwise orientation
        return 1
    elif (val < 0):
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0
 
def transform_cells(cells, points, ele_type):
    """FEniCS triangular mesh is not always counter-clockwise. We need to fix it.
    """
    new_cells = []
    for cell in cells:
        pts = points[cell[:3]]
        p1 = Point(pts[0, 0], pts[0, 1])
        p2 = Point(pts[1, 0], pts[1, 1])
        p3 = Point(pts[2, 0], pts[2, 1])
         
        o = orientation(p1, p2, p3)
         
        if (o == 0):
            print(f"Linear")
            print(f"Can't be linear, somethign wrong!")
            exit()
        elif (o == 1):
            # print(f"Clockwise")
            if ele_type == 'TRI3':
                new_celll = cell[[0, 2, 1]]
            elif ele_type == 'TRI6':
                new_celll = cell[[0, 2, 1, 5, 4, 3]]
            else:
                print(f"Wrong element type, can't be transformed")
                exit()
            new_cells.append(new_celll)
        else:
            # print(f"CounterClockwise")
            new_cells.append(cell)

    return onp.stack(new_cells)


# Define parameters
T0 = 293. # ambient temperature
theta_hole = 10. # temperature change at hole boundary
E = 70e3
nu = 0.3
mu = E/(2.*(1. + nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
rho = 2700. # density
alpha = 2.31e-5 # thermal expansion coefficient
kappa = alpha*(2*mu + 3*lmbda)
C = 910e-6 * rho # specific heat per unit volume at constant strain
k = 237e-6 # thermal conductivity

def main_fns():
    
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    # First run `python -m demos.thermal_mechanical_full.fenics` to generate these numpy files
    ele_type = 'TRI3'
    points = onp.load(os.path.join(input_dir, f'numpy/points.npy'))
    cells = onp.load(os.path.join(input_dir, f'numpy/cells.npy'))
    cells = transform_cells(cells, points, ele_type)
    mesh = Mesh(points, cells)
    
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
    
    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)
    
    def zero_dirichlet(point):
        return 0.
    
    def hole(point):
        R = 0.1
        return np.isclose(point[0]**2+point[1]**2, R**2, atol=1e-3)
    
    def theta_dirichlet(point):
        return theta_hole
        
    dirichlet_bc_info_u = [[left, bottom],[0,1],[zero_dirichlet]*2]
    dirichlet_bc_info_T = [[hole],[0],[theta_dirichlet]]
    
    problem = thermal_mechanical_full([mesh, mesh], vec=[2, 1], dim=2, ele_type=[ele_type, ele_type], gauss_order=[2, 2],
                                      dirichlet_bc_info=[dirichlet_bc_info_u, dirichlet_bc_info_T])
    
    sol_u = 0*np.ones((len(mesh.points), 2))
    sol_dT = 0*np.ones((len(mesh.points), 1))
    
    # Start the major loop of time iteration.
    Nincr = 200
    t = onp.logspace(1,4,Nincr+1)
    for (i, dt_i) in enumerate(onp.diff(t)):
        print(f'Increment {i+1}; dt {i+1}:{dt_i}')
        problem.set_params([sol_u, sol_dT, dt_i])
        sol_list = solver(problem, solver_options={'petsc_solver': {}})
        sol_u, sol_dT = sol_list
        # Store the solution to local file.
        vtk_path_u = os.path.join(output_dir, f'vtk/u_{i}.vtk')
        vtk_path_dT = os.path.join(output_dir, f'vtk/theta_{i}.vtu')
        save_sol(problem.fes[0], sol_list[0], vtk_path_u)
        save_sol(problem.fes[1], sol_list[1], vtk_path_dT)
    
    print(f"Max u = {onp.max(sol_u)}, Min u = {onp.min(sol_u)}")
    print(f"Max theta = {onp.max(sol_dT)}, Min theta = {onp.min(sol_dT)}")

if __name__ == "__main__":
    main_fns()
    