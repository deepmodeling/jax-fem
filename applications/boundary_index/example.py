"""
This example shows how to specify boundary locations with point and/or index

It addresses the Github issue https://github.com/deepmodeling/jax-fem/issues/20
"""
import jax
import jax.numpy as np
import os

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh


class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x: x

    def get_surface_maps(self):
        def surface_map(u, x):
            return -np.array([np.sin(5.*x[0])])

        return [surface_map, surface_map]

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 1., 1.
meshio_mesh = rectangle_mesh(Nx=2, Ny=2, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# We consider a problem having global DOF numbering and global cell numbering like the following:

# 2-----5-----8
# |     |     |
# | (1) | (3) |
# |     |     |
# 1-----4-----7
# |     |     |
# | (0) | (2) |
# |     |     |
# 0-----3-----6

# The local face indexing is as the following:

#  3---[3]---2
#  |         |
# [1]       [2]
#  |         |
#  0---[0]---1

# You may define the right boundary point collection and the top boundary point collection as
ind_set_right = np.array([6, 7, 8])
ind_set_top = np.array([2, 5, 8])

# Define boundary locations.
def left(point):
    """
    If one argument is passed, it is treated as "point".
    """
    return np.isclose(point[0], 0., atol=1e-5)

def right(point, ind):
    """
    If two arguments are passed, the first will be "point" and the second will be "point index".
    """
    return np.isin(ind, ind_set_right) 

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5) 

def top(point, ind):
    return np.isin(ind, ind_set_top) 

def dirichlet_val_left(point):
    return 0.

def dirichlet_val_right(point):
    return 0.

location_fns1 = [left, right]
value_fns = [dirichlet_val_left, dirichlet_val_right]
vecs = [0, 0]
dirichlet_bc_info = [location_fns1, vecs, value_fns]

location_fns2 = [bottom, top]

problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns2)

print(f"\n\nlocation_fns1 is processed to generate Dirichlet node indices: \n{problem.fes[0].node_inds_list}")
print(f"\nwhere node_inds_list[l][j] returns the jth selected node index in Dirichlet set l")

print(f"\n\nlocation_fns2 is processed to generate boundary indices list: \n{problem.boundary_inds_list}")
print(f"\nwhere boundary_inds_list[k][i, 0] returns the global cell index of the ith selected face of boundary subset k")
print(f"      boundary_inds_list[k][i, 1] returns the local face index of the ith selected face of boundary subset k")
