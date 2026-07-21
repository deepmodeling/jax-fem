import jax
import jax.numpy as np
import numpy as onp
import meshio
import gmsh
import os

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol


class LinearPoisson(Problem):
    def get_tensor_map(self):
        return lambda x: x

    def get_mass_map(self):
        def mass_map(u, x):
            val = np.trace(np.squeeze(jax.hessian(true_u_fn)(x)))
            return np.array([val])
        return mass_map

    def compute_l2_norm_error(self, sol, true_u_fn):
        cells_sol = sol[self.fes[0].cells] # (num_cells, num_nodes, vec)
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.fes[0].shape_vals[None, :, :, None], axis=2)
        physical_quad_points = self.fes[0].get_physical_quad_points() # (num_cells, num_quads, dim)
        true_u = jax.vmap(jax.vmap(true_u_fn))(physical_quad_points) # (num_cells, num_quads, vec)
        # (num_cells, num_quads, vec) * (num_cells, num_quads, 1)
        l2_error = np.sqrt(np.sum((u - true_u)**2 * self.fes[0].JxW[:, :, None]))
        return l2_error

    def compute_h1_norm_error(self, sol, true_u_fn):
        cells_sol = sol[self.fes[0].cells] # (num_cells, num_nodes, vec)
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.fes[0].shape_vals[None, :, :, None], axis=2)
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = cells_sol[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        physical_quad_points = self.fes[0].get_physical_quad_points() # (num_cells, num_quads, dim)
        true_u = jax.vmap(jax.vmap(true_u_fn))(physical_quad_points) # (num_cells, num_quads, vec)
        true_u_grads = jax.vmap(jax.vmap(jax.jacrev(true_u_fn)))(physical_quad_points) # (num_cells, num_quads, vec, dim)
        # (num_cells, num_quads, vec) * (num_cells, num_quads, 1)
        val_l2_error = np.sqrt(np.sum((u - true_u)**2 * self.fes[0].JxW[:, :, None]))
        # (num_cells, num_quads, vec, dim) * (num_cells, num_quads, 1, 1)
        grad_l2_error = np.sqrt(np.sum((u_grads - true_u_grads)**2 * self.fes[0].JxW[:, :, None, None]))
        h1_error = val_l2_error + grad_l2_error
        return h1_error


def true_u_fn(point):
    """Some arbitrarily created analytical solution
    """
    x, y, z = point
    return np.array([1e2*((x - 0.5)**3 + 2.*(y - 0.5)**3 + 1e1*np.exp(-(z - 0.5)**2))])


def problem(ele_type, N, data_dir):
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(N, N, N, 1., 1., 1., data_dir, ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

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
 
    problem = LinearPoisson(mesh, vec=1, dim=3, ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info)

    sol_list = solver(problem)
    sol = sol_list[0]
    vtk_dir = os.path.join(data_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_file = os.path.join(vtk_dir, f"u.vtu")
    save_sol(problem.fes[0], sol, vtk_file)

    l2_error = problem.compute_l2_norm_error(sol, true_u_fn)
    h1_error = problem.compute_h1_norm_error(sol, true_u_fn)
    return l2_error, h1_error


def convergence_test():
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data') # applications/fem/convergence_rate/data
    ele_types = ['TET4', 'TET10']
    degrees = [1, 2]
    Ns = onp.array([8, 16])
    l2_errors_orders = []
    h1_errors_orders = []
    for ele_type in ele_types:
        l2_errors = []
        h1_errors = []        
        for N in Ns:
            l2_error, h1_error = problem(ele_type, N, data_dir)
            l2_errors.append(l2_error)
            h1_errors.append(h1_error)
        l2_errors_orders.append(l2_errors)
        h1_errors_orders.append(h1_errors)

    l2_errors_orders = onp.array(l2_errors_orders)
    h1_errors_orders = onp.array(h1_errors_orders)

    print(f"l2_errors_orders = \n{l2_errors_orders}")
    print(f"h1_errors_orders = \n{h1_errors_orders}")
    print(f"Expect 2, got {onp.log2(l2_errors_orders[0][-2]/l2_errors_orders[0][-1])}")
    print(f"Expect 1, got {onp.log2(h1_errors_orders[0][-2]/h1_errors_orders[0][-1])}")
    print(f"Expect 3, got {onp.log2(l2_errors_orders[1][-2]/l2_errors_orders[1][-1])}")
    print(f"Expect 2, got {onp.log2(h1_errors_orders[1][-2]/h1_errors_orders[1][-1])}")


if __name__ == "__main__":
    convergence_test()
