import jax
import jax.numpy as np
import numpy as onp
import os
import meshio
import gmsh
import scipy

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol, modify_vtu_file
from jax_fem.basis import get_elements


def gmsh_mesh(data_dir, degree):
    """
    Generate a mesh
    Reference:
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_8_3/tutorial/python/t1.py
    """
    msh_dir = os.path.join(data_dir, f'msh')
    os.makedirs(msh_dir, exist_ok=True)
    file_path = os.path.join(msh_dir, f't1.msh')

    gmsh.initialize()
    gmsh.model.add("t1")
    lc = 1e-2
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1., 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1., 1., 0, lc, 3)
    p4 = gmsh.model.geo.addPoint(0, 1., 0, lc)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(3, 2, 2)
    gmsh.model.geo.addLine(3, p4, 3)
    gmsh.model.geo.addLine(4, 1, p4)
    gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
    ps = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, ps, "My surface")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(file_path)

    return file_path


def periodic_boundary_conditions(periodic_bc_info, mesh, vec):
    """
    Construct the 'P' matrix
    Reference: https://fenics2021.com/slides/dokken.pdf
    """
    p_node_inds_list_A = []
    p_node_inds_list_B = []
    p_vec_inds_list = []

    location_fns_A, location_fns_B, mappings, vecs = periodic_bc_info
    for i in range(len(location_fns_A)):
        node_inds_A = onp.argwhere(jax.vmap(location_fns_A[i])(mesh.points)).reshape(-1)
        node_inds_B = onp.argwhere(jax.vmap(location_fns_B[i])(mesh.points)).reshape(-1)
        points_set_A = mesh.points[node_inds_A]
        points_set_B = mesh.points[node_inds_B]

        EPS = 1e-5
        node_inds_B_ordered = []
        for node_ind in node_inds_A:
            point_A = mesh.points[node_ind]
            dist = onp.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
            node_ind_B_ordered = node_inds_B[onp.argwhere(dist < EPS)].reshape(-1)
            node_inds_B_ordered.append(node_ind_B_ordered)

        node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(-1)
        vec_inds = onp.ones_like(node_inds_A, dtype=onp.int32) * vecs[i]

        p_node_inds_list_A.append(node_inds_A)
        p_node_inds_list_B.append(node_inds_B_ordered)
        p_vec_inds_list.append(vec_inds)
        assert len(node_inds_A) == len(node_inds_B_ordered)

    # For mutiple variables (e.g, stokes flow, u-p coupling), offset will be nonzero.
    offset = 0
    inds_A_list = []
    inds_B_list = []
    for i in range(len(p_node_inds_list_A)):
        inds_A_list.append(onp.array(p_node_inds_list_A[i] * vec + p_vec_inds_list[i] + offset, dtype=onp.int32))
        inds_B_list.append(onp.array(p_node_inds_list_B[i] * vec + p_vec_inds_list[i] + offset, dtype=onp.int32))

    inds_A = onp.hstack(inds_A_list)
    inds_B = onp.hstack(inds_B_list)

    num_total_nodes = len(mesh.points)
    num_total_dofs = num_total_nodes * vec
    N = num_total_dofs
    M = num_total_dofs - len(inds_B)

    # The use of 'reduced_inds_map' seems to be a smart way to construct P_mat
    reduced_inds_map = onp.ones(num_total_dofs, dtype=onp.int32)
    reduced_inds_map[inds_B] = -(inds_A + 1)
    reduced_inds_map[reduced_inds_map == 1] = onp.arange(M)

    I = []
    J = []
    V = []
    for i in range(num_total_dofs):
        I.append(i)
        V.append(1.)
        if reduced_inds_map[i] < 0:
            J.append(reduced_inds_map[-reduced_inds_map[i] - 1])
        else:
            J.append(reduced_inds_map[i])
 
    P_mat = scipy.sparse.csr_array((onp.array(V), (onp.array(I), onp.array(J))), shape=(N, M))

    return P_mat


class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x, theta: x * theta

    def get_mass_map(self):
        def mass_map(u, x, theta):
            dx = x[0] - 0.5
            dy = x[1] - 0.5
            val = x[0]*np.sin(5.0*np.pi*x[1]) + 1.0*np.exp(-(dx*dx + dy*dy)/0.02)
            return np.array([-val])
        return mass_map

    def set_params(self, theta):
        thetas = theta * np.ones((self.fes[0].num_cells, self.fes[0].num_quads))
        self.internal_vars = [thetas]


def problem():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ele_type = 'TRI6'
    _, _, _, _, degree, _ = get_elements(ele_type)
    msh_file_path = gmsh_mesh(data_dir, degree)
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(msh_file_path)
    mesh = Mesh(meshio_mesh.points[:, :2], meshio_mesh.cells_dict[cell_type])
    vec = 1
    dim = 2

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], 1., atol=1e-5)

    def dirichlet_val(point):
        return 0.

    def mapping_x(point_A):
        point_B = point_A + np.array([1., 0])
        return point_B

    location_fns_A = [left] 
    location_fns_B = [right] 
    mappings = [mapping_x]
    vecs = [0]
    periodic_bc_info = [location_fns_A, location_fns_B, mappings, vecs]

    dirichlet_bc_info = [[bottom, top], 
                         [0]*2, 
                         [dirichlet_val, dirichlet_val]]

    P_mat = periodic_boundary_conditions(periodic_bc_info, mesh, vec)
    problem = Poisson(mesh, vec=vec, dim=dim, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    problem.P_mat = P_mat

    # Other solvers can be 'jax_solver', 'umfpack_solver'
    fwd_pred = ad_wrapper(problem, solver_options={'petsc_solver': {}}, adjoint_solver_options={'petsc_solver': {}}) 
    theta = 1.
    sol_list = fwd_pred(theta)
    vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
    save_sol(problem.fes[0], sol_list[0], vtk_path)

    # Test AD
    def J(theta):
        sol_list = fwd_pred(theta)
        return np.sum(np.sum(sol_list[0]**2))

    h = 1e-3
    fd_result = (J(theta + h) - J(theta - h))/(2*h)
    ad_result = jax.grad(J)(theta)
    print(f"Finite difference: {fd_result}") 
    print(f"Automatic differentiation: {ad_result}")


if __name__ == "__main__":
    problem()
