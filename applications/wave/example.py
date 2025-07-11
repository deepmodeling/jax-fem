"""
This example deals with scalar wave equations.
Also, see jax-fem/demos/wave/fenics.py
"""

# Import some useful modules.
import os
import jax
import jax.numpy as np
import jax.flatten_util
import meshio
import numpy as onp
import matplotlib.pyplot as plt


# Import JAX-FEM specific modules.
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.utils import save_sol, modify_vtu_file
from jax_fem.problem import Problem


class wave(Problem):
    
    def set_params(self, params):
        sol_2dt, sol_dt = params
        self.internal_vars = [self.fes[0].convert_from_dof_to_quad(sol_2dt), 
                              self.fes[0].convert_from_dof_to_quad(sol_dt)]
    
    def get_universal_kernel(self):
           
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, cell_sol_2dt, cell_sol_dt):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            
            ## Split
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol = cell_sol_list[0]
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads = cell_shape_grads_list[0]
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW = cell_v_grads_JxW_list[0]
            cell_JxW = cell_JxW[0]
            
            ## Handles the term 'c^2 * dt^2 * inner(grad(u_crt),grad(v)) * dx'
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim) 
            # -> (num_quads, vec, dim)
            u_grad = np.sum(cell_sol[None,:,:,None] * cell_shape_grads[:,:,None,:],axis=1)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) ->  (num_nodes, vec) 
            val1 = np.sum(c**2 * dt**2 * u_grad[:,None,:,:] * cell_v_grads_JxW,axis=(0,-1))
            
            ## Handles the term 'inner(u_crt,v) * dx'
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec) 
            u = np.sum(cell_sol[None,:,:] * self.fes[0].shape_vals[:,:,None],axis=1)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec) 
            val2 = np.sum(u[:,None,:] * self.fes[0].shape_vals[:,:,None] * cell_JxW[:,None,None],axis=0)
            
            ## Handles the term '((2*u_old_dt - u_old_2dt) * v) * dx'
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec) 
            val3 = np.sum((2 * cell_sol_dt - cell_sol_2dt)[:,None,:] * self.fes[0].shape_vals[:,:,None] * cell_JxW[:,None,None],axis=0) 

            weak_form = val1 + val2 - val3
            
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
dt = 1/250000 # temporal sampling interval
c = 5000 # speed of sound
steps = 200

def main_fns():
    
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    # First run `python -m demos.wave.fenics` to generate these numpy files
    ele_type = 'TRI3'
    Lx, Ly = 1.0, 1.0
    points = onp.load(os.path.join(input_dir, f'numpy/points.npy'))
    cells = onp.load(os.path.join(input_dir, f'numpy/cells.npy'))
    cells = transform_cells(cells, points, ele_type)
    mesh = Mesh(points, cells)
    
    def ones_dirichlet(point):
        return 1.
    
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
    
    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)
    
    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)
    
    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)
    
    dirichlet_bc_info = [[left, right, bottom, top],[0]*4,[ones_dirichlet]*4]
    
    problem = wave(mesh, vec=1, dim=2, ele_type = ele_type, gauss_order=2, dirichlet_bc_info = dirichlet_bc_info)
    sol_2dt = np.zeros((len(points),1))
    sol_dt = np.zeros((len(points),1))
    
    # Start the major loop of time iteration.
    for i in range(steps):
        print(f'Time increment {i+1}\n')
        problem.set_params([sol_2dt,sol_dt])
        sol = solver(problem)[0]
        sol_2dt = sol_dt
        sol_dt = sol
        # Store the solution to local file.
        vtk_path_u = os.path.join(output_dir, f'vtk/u_{i}.vtk')
        save_sol(problem.fes[0], sol, vtk_path_u)
    
    print(f"Max u = {onp.max(sol)}, Min u = {onp.min(sol)}")

if __name__ == "__main__":
    main_fns()
