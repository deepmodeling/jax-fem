"""This example exactly reproduces 
https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/stokes-taylor-hood/python/documentation.html

Also, see jax-fem/applications/stokes/fenics.py
"""
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file
from jax_fem.problem import Problem


class StokesFlow(Problem):
    def custom_init(self):
        self.fe_u = self.fes[0]
        self.fe_p = self.fes[1]

    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            # cell_sol_u: (num_nodes_u, vec), cell_sol_p: (num_nodes, vec)
            cell_sol_u, cell_sol_p = cell_sol_list
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_u, cell_shape_grads_p = cell_shape_grads_list
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_u, cell_v_grads_JxW_p = cell_v_grads_JxW_list
            cell_JxW_u, cell_JxW_p = cell_JxW[0], cell_JxW[1]

            # Handles the term `inner(grad(u), grad(v)*dx`
            # (1, num_nodes_u, vec_u, 1) * (num_quads, num_nodes_u, 1, dim) -> (num_quads, num_nodes_u, vec_u, dim)
            u_grads = cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec_u, dim)
            # (num_quads, num_nodes_u, vec_u, dim) -> (num_nodes_u, vec_u)
            val1 = np.sum(u_grads[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))

            # Handles the term `div(v)*p*dx`
            # (1, num_nodes_p, vec_p) * (num_quads, num_nodes_p, 1) -> (num_quads, num_nodes_p, vec_p) -> (num_quads, vec_p)
            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0]
            # Be careful about this step to find divergence!
            # (num_quads, num_nodes_u, 1, dim) -> (num_quads, num_nodes_u, vec_u)
            div_v = cell_v_grads_JxW_u[:, :, 0, :]
            # (num_quads, 1, 1) * (num_quads, num_nodes_u, vec_u) -> (num_nodes_u, vec_u) 
            val2 = np.sum(p[:, None, None] * div_v, axis=0)

            # Handles the term `q*div(u))*dx`
            # (num_quads, vec_u, dim) -> (num_quads, )
            div_u = u_grads[:, 0, 0] + u_grads[:, 1, 1]  
            # (num_quads, 1) * (num_quads, num_nodes_p) * (num_quads, 1) -> (num_nodes_p,) -> (num_nodes_p, vec_p)
            val3 = np.sum(div_u[:, None] * self.fe_p.shape_vals * cell_JxW_p[:, None], axis=0)[:, None]

            weak_form = [val1 + val2, val3] # [(num_nodes, vec), ...]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel

    def configure_Dirichlet_BC_for_dolphin(self):
        """FEniCS dolfin example has interior boundaries that can't be directly imported
        Here, we manually find the boundaries containing the 'dolphin' contour
        """
        cells_u = self.fe_u.cells
        points_u = self.fe_u.points
        v, c = onp.unique(cells_u, return_counts=True) 
        boundary_mid_nodes = v[c==1]

        def ind_map(ind):
            assert ind == 3 or ind == 4 or ind == 5, f"Wrong face ind!"
            if ind == 3:
                return 0, 1
            if ind == 4:
                return 1, 2
            if ind == 5:
                return 2, 0

        boundary_inds = []
        for cell in cells_u:
            for c in cell:
                if c in boundary_mid_nodes:
                    pos_ind = onp.argwhere(cell==c)[0, 0]
                    node1, node2 = ind_map(pos_ind)
                    boundary_inds += [c, cell[node1], cell[node2]]
 
        boundary_inds = onp.array(list(set(boundary_inds)))
        valid_inds = onp.argwhere((points_u[boundary_inds][:, 0] < 1 - 1e-5) &  
                                  (points_u[boundary_inds][:, 0] > 1e-5)  ).reshape(-1)
        boundary_inds = boundary_inds[valid_inds]

        vec_inds_1 = onp.zeros_like(boundary_inds, dtype=onp.int32)
        vec_inds_2 = onp.ones_like(boundary_inds, dtype=onp.int32)
        values = onp.zeros_like(boundary_inds, dtype=onp.float32)

        self.fes[0].node_inds_list += [boundary_inds]*2 
        self.fes[0].vec_inds_list += [vec_inds_1, vec_inds_2] 
        self.fes[0].vals_list += [values]*2 


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


def problem():
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    # First run `python -m applications.stokes.fenics` to generate these numpy files
    ele_type_u = 'TRI6'
    points_u = onp.load(os.path.join(input_dir, f'numpy/points_u.npy'))
    cells_u = onp.load(os.path.join(input_dir, f'numpy/cells_u.npy'))
    cells_u = transform_cells(cells_u, points_u, ele_type_u)
    mesh_u = Mesh(points_u, cells_u)

    ele_type_p = 'TRI3'
    points_p = onp.load(os.path.join(input_dir, f'numpy/points_p.npy'))
    cells_p = onp.load(os.path.join(input_dir, f'numpy/cells_p.npy'))
    cells_p = transform_cells(cells_p, points_p, ele_type_p)
    mesh_p = Mesh(points_p, cells_p)

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return -np.sin(point[1]*np.pi)

    dirichlet_bc_info1 = [[right, right], 
                          [0,  1], 
                          [dirichlet_val, zero_dirichlet_val]]
 

    dirichlet_bc_info2 = [[left], 
                          [0], 
                          [zero_dirichlet_val]]

    problem = StokesFlow([mesh_u, mesh_p], vec=[2, 1], dim=2, ele_type=[ele_type_u, ele_type_p], gauss_order=[2, 2],
                                dirichlet_bc_info=[dirichlet_bc_info1, dirichlet_bc_info2])
    problem.configure_Dirichlet_BC_for_dolphin()

    # Preconditioning is very important for a problem like this. See discussions:
    # https://fenicsproject.discourse.group/t/steady-stokes-equation-3d-dolfinx/9709/4
    # https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/stokes-iterative/demo_stokes-iterative.py.html
    # Here, we choose 'ksp_type' to be 'tfqmr' and 'pc_type' to be 'ilu'
    # But see a variety of other choices in PETSc:
    # https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/index.html
    sol_list = solver(problem, solver_options={'petsc_solver': {'ksp_type': 'tfqmr', 'pc_type': 'ilu'}})

    # Alternatively, you may use the UMFPACK solver
    # sol_list = solver(problem, solver_options={'umfpack_solver': {}})

    u, p = sol_list
    print(f"Max u = {onp.max(u)}, Min u = {onp.min(u)}")
    print(f"Max p = {onp.max(p)}, Min p = {onp.min(p)}")
 
    vtk_path_u = os.path.join(output_dir, f'vtk/jax-fem_velocity.vtu')
    vtk_path_p = os.path.join(output_dir, f'vtk/jax-fem_pressure.vtu')
    sol_to_save = np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1))))
    save_sol(problem.fes[0], sol_to_save, vtk_path_u)
    save_sol(problem.fes[1], sol_list[1], vtk_path_p)


if __name__ == "__main__":
    problem()
