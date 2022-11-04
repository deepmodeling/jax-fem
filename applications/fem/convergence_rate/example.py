import jax
import jax.numpy as np
import numpy as onp
import meshio
import gmsh
import os

from jax_am.fem.models import LinearPoisson
from jax_am.fem.solver import solver
from jax_am.fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_am.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def problem(ele_type, lag_order, N, data_dir):
    cell_type = get_meshio_cell_type(ele_type, lag_order)
    meshio_mesh = box_mesh(N, N, N, 1., 1., 1., data_dir, ele_type, lag_order)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def true_u_fn(point):
        """Some arbitrarily created analytical solution
        """
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
 
    problem = LinearPoisson(mesh, vec=1, dim=3, ele_type=ele_type, lag_order=lag_order, 
                            dirichlet_bc_info=dirichlet_bc_info, source_info=body_force)

    sol = solver(problem, linear=True, precond=True)
    vtk_dir = os.path.join(data_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_file = os.path.join(vtk_dir, f"u.vtu")
    save_sol(problem, sol, vtk_file, cell_type=cell_type)

    l2_error = problem.compute_l2_norm_error(sol, true_u_fn)
    h1_error = problem.compute_h1_norm_error(sol, true_u_fn)
    return l2_error, h1_error


def convergence_test():
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data') # applications/fem/convergence_rate/data
    ele_type = 'tetrahedron'
    lag_orders = [1, 2]
    Ns = onp.array([8, 16])
    l2_errors_orders = []
    h1_errors_orders = []
    for lag_order in lag_orders:
        l2_errors = []
        h1_errors = []        
        for N in Ns:
            l2_error, h1_error = problem(ele_type, lag_order, N, data_dir)
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
