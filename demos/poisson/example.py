# Import some generally useful packages.
import jax
import jax.numpy as np
import os


# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh


# Define constitutive relationship. 
class Poisson(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div.f(u_grad) = b. Here, we define f to be the indentity function. 
    # We will see how f is deined as more complicated to solve non-linear problems 
    # in later examples.
    def get_tensor_map(self):
        return lambda x: x
    
    # Define the source term b
    def get_mass_map(self):
        def mass_map(u, x):
            val = -np.array([10*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            return val
        return mass_map

    def get_surface_maps(self):
        def surface_map(u, x):
            return -np.array([np.sin(5.*x[0])])

        return [surface_map, surface_map]


# Specify mesh-related information. 
# We make use of the external package 'meshio' and create a mesh named 'meshio_mesh', 
# then converting it into a JAX-FEM compatible one.
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 1., 1.
meshio_mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)


# Define Dirichlet boundary values. 
# This means on the 'left' side, we apply the function 'dirichlet_val_left' 
# to the 0 component of the solution variable; on the 'right' side, we apply 
# 'dirichlet_val_right' to the 0 component.
def dirichlet_val_left(point):
    return 0.

def dirichlet_val_right(point):
    return 0.

location_fns = [left, right]
value_fns = [dirichlet_val_left, dirichlet_val_right]
vecs = [0, 0]
dirichlet_bc_info = [location_fns, vecs, value_fns]


# Define Neumann boundary locations.
# This means on the 'bottom' and 'top' side, we will perform the surface integral 
# with the function 'get_surface_maps' defined in the class 'Poisson'.
location_fns = [bottom, top]


# Create an instance of the Class 'Poisson'. 
# Here, vec is the number of components for the solution.
problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)


# Solve the problem. 
# solver_options can be changed for other linear solver options
sol = solver(problem)
# sol = solver(problem, solver_options={'umfpack_solver': {}})
# sol = solver(problem, solver_options={'petsc_solver': {'ksp_type': 'bcgsl', 'pc_type': 'ilu'}})

# Save the solution to a local folder that can be visualized with ParaWiew.
data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol[0], vtk_path)
