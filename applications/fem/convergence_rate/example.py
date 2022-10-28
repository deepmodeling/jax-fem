import jax
import jax.numpy as np
import numpy as onp
import meshio
import gmsh
import os

from jax_am.fem.jax_fem import Mesh, LinearPoisson
from jax_am.fem.solver import solver
from jax_am.fem.generate_mesh import box_mesh, get_meshio_cell_type
from jax_am.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class LinearPoissonConvergence(LinearPoisson):
    def __init__(self, mesh, ele_type, lag_order, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):  
        super().__init__("some_name", mesh, ele_type, lag_order, dirichlet_bc_info, neumann_bc_info, source_info) 

    def compute_l2_norm_error(self, sol, true_u_fn):
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
        physical_quad_points = self.get_physical_quad_points() # (num_cells, num_quads, dim) 
        true_u = jax.vmap(jax.vmap(true_u_fn))(physical_quad_points) # (num_cells, num_quads, vec)
        # (num_cells, num_quads, vec) * (num_cells, num_quads, 1)
        l2_error = np.sqrt(np.sum((u - true_u)**2 * self.JxW[:, :, None]))
        return l2_error


    # def compute_h1_norm_error(self, sol, true_u_fn):
    #     cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
    #     # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
    #     u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
    #     physical_quad_points = self.get_physical_quad_points() # (num_cells, num_quads, dim) 
    #     true_u = jax.vmap(jax.vmap(true_u_fn))(physical_quad_points) # (num_cells, num_quads, vec)
    #     # (num_cells, num_quads, vec) * (num_cells, num_quads, 1)
    #     l2_error = np.sqrt(np.sum((u - true_u)**2 * self.JxW[:, :, None]))
    #     return l2_error


def problem():
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data') # applications/fem/convergence_rate/data

    ele_type = 'tetrahedron'
    lag_order = 2
    cell_type = get_meshio_cell_type(ele_type, lag_order)
    meshio_mesh = box_mesh(16, 16, 16, 1., 1., 1., data_dir, ele_type, lag_order)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def true_u_fn(point):
        x, y, z = point
        return np.array([1e2*((x - 0.5)**3 + 2.*(y - 0.5)**3 + 1e1*np.exp(-(z - 0.5)**2))])

    def body_force(point):
        val = -np.trace(np.squeeze(jax.hessian(true_u_fn)(point)))
        return np.array([val])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def front(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def back(point):
        return np.isclose(point[1], 1., atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], 1., atol=1e-5)

    def dirichlet_val(point):
        return true_u_fn(point)[0]

    dirichlet_bc_info = [[left, right, front, back, bottom, top], 
                         [0]*6, 
                         [dirichlet_val]*6]
 
    problem = LinearPoissonConvergence(mesh, ele_type, lag_order, 
                                       dirichlet_bc_info=dirichlet_bc_info, source_info=body_force)

    sol = solver(problem, linear=True, precond=True)

    print(f"l2_error = {problem.compute_l2_norm_error(sol, true_u_fn)}")

    vtk_dir = os.path.join(data_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_file = os.path.join(vtk_dir, f"u.vtu")
    save_sol(problem, sol, vtk_file, cell_type=cell_type)


if __name__ == "__main__":
    problem()
