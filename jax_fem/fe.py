import numpy as onp
import jax
import jax.numpy as np
import sys
import time
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Union
from jax_fem.common import timeit
from jax_fem.generate_mesh import Mesh
from jax_fem.basis import get_face_shape_vals_and_grads, get_shape_vals_and_grads
from jax_fem import logger


onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=5)


@dataclass
class FiniteElement:
    """
    Defines finite element related to one variable (can be vector valued)

    Attributes
    ----------
    mesh : Mesh object
        The mesh object stores points (coordinates) and cells (connectivity).
    vec : int
        The number of vector variable components of the solution.
        E.g., a 3D displacement field has u_x, u_y and u_z components, so vec=3
    dim : int
        The dimension of the problem.
    ele_type : str
        Element type
    dirichlet_bc_info : [location_fns, vecs, value_fns]
        location_fns : List[Callable]
            Callable : a function that inputs a point and returns if the point satisfies the location condition
        vecs: List[int]
            integer value must be in the range of 0 to vec - 1,
            specifying which component of the (vector) variable to apply Dirichlet condition to
        value_fns : List[Callable]
            Callable : a function that inputs a point and returns the Dirichlet value
    periodic_bc_info : [location_fns_A, location_fns_B, mappings, vecs]
        location_fns_A : List[Callable]
            Callable : location function for boundary A
        location_fns_B : List[Callable]
            Callable : location function for boundary B
        mappings : List[Callable]
            Callable: function mapping a point from boundary A to boundary B
        vecs: List[int]
            which component of the (vector) variable to apply periodic condition to
    """
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str
    gauss_order: int
    dirichlet_bc_info: Optional[List[Union[List[Callable], List[int], List[Callable]]]]
    periodic_bc_info: Optional[List[Union[List[Callable], List[Callable], List[Callable], List[int]]]] = None

    def __post_init__(self):
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)
        self.num_total_dofs = self.num_total_nodes * self.vec

        start = time.time()
        logger.debug(f"Computing shape function values, gradients, etc.")

        self.shape_vals, self.shape_grads_ref, self.quad_weights = get_shape_vals_and_grads(self.ele_type, self.gauss_order)
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, self.face_normals, self.face_inds \
        = get_face_shape_vals_and_grads(self.ele_type, self.gauss_order)
        self.num_quads = self.shape_vals.shape[0]
        self.num_nodes = self.shape_vals.shape[1]
        self.num_faces = self.face_shape_vals.shape[0]
        self.shape_grads, self.JxW = self.get_shape_grads()
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(self.dirichlet_bc_info)
        self.p_node_inds_list_A, self.p_node_inds_list_B, self.p_vec_inds_list = self.periodic_boundary_conditions()
        # (num_cells, num_quads, num_nodes, 1, dim)
        self.v_grads_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]
        self.num_face_quads = self.face_quad_weights.shape[1]

        end = time.time()
        compute_time = end - start

        logger.debug(f"Done pre-computations, took {compute_time} [s]")
        logger.info(f"Solving a problem with {len(self.cells)} cells, {self.num_total_nodes}x{self.vec} = {self.num_total_dofs} dofs.")

    def get_shape_grads(self):
        """Compute shape function gradient value
        The gradient is w.r.t physical coordinates.
        See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
        Page 147, Eq. (3.9.3)

        Returns
        -------
        shape_grads_physical : onp.ndarray
            (num_cells, num_quads, num_nodes, dim)
        JxW : onp.ndarray
            (num_cells, num_quads)
        """
        assert self.shape_grads_ref.shape == (self.num_quads, self.num_nodes, self.dim)
        physical_coos = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
        jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] *
                                   self.shape_grads_ref[None, :, :, None, :], axis=2, keepdims=True)
        jacobian_det = onp.linalg.det(jacobian_dx_deta)[:, :, 0]  # (num_cells, num_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)
        # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
        # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
        shape_grads_physical = (self.shape_grads_ref[None, :, :, None, :]
                                @ jacobian_deta_dx)[:, :, :, 0, :]
        JxW = jacobian_det * self.quad_weights[None, :]
        return shape_grads_physical, JxW

    def get_face_shape_grads(self, boundary_inds):
        """Face shape function gradients and JxW (for surface integral)
        Nanson's formula is used to map physical surface ingetral to reference domain
        Reference: https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            (num_selected_faces, 2)

        Returns
        -------
        face_shape_grads_physical : onp.ndarray
            (num_selected_faces, num_face_quads, num_nodes, dim)
        nanson_scale : onp.ndarray
            (num_selected_faces, num_face_quads)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes, dim)
        selected_f_normals = self.face_normals[boundary_inds[:, 1]]  # (num_selected_faces, dim)

        # (num_selected_faces, 1, num_nodes, dim, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        # (num_selected_faces, num_face_quads, num_nodes, dim, dim) -> (num_selected_faces, num_face_quads, dim, dim)
        jacobian_dx_deta = onp.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
        jacobian_det = onp.linalg.det(jacobian_dx_deta)  # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)  # (num_selected_faces, num_face_quads, dim, dim)

        # (1, num_face_quads, num_nodes, 1, dim) @ (num_selected_faces, num_face_quads, 1, dim, dim)
        # (num_selected_faces, num_face_quads, num_nodes, 1, dim) -> (num_selected_faces, num_face_quads, num_nodes, dim)
        face_shape_grads_physical = (selected_f_shape_grads_ref[:, :, :, None, :] @ jacobian_deta_dx[:, :, None, :, :])[:, :, :, 0, :]

        # (num_selected_faces, 1, 1, dim) @ (num_selected_faces, num_face_quads, dim, dim)
        # (num_selected_faces, num_face_quads, 1, dim) -> (num_selected_faces, num_face_quads)
        nanson_scale = onp.linalg.norm((selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :], axis=-1)
        selected_weights = self.face_quad_weights[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads)
        nanson_scale = nanson_scale * jacobian_det * selected_weights
        return face_shape_grads_physical, nanson_scale

    def get_physical_quad_points(self):
        """Compute physical quadrature points

        Returns
        -------
        physical_quad_points : onp.ndarray
            (num_cells, num_quads, dim)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        # (1, num_quads, num_nodes, 1) * (num_cells, 1, num_nodes, dim) -> (num_cells, num_quads, dim)
        physical_quad_points = onp.sum(self.shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)
        return physical_quad_points

    def get_physical_surface_quad_points(self, boundary_inds):
        """Compute physical quadrature points on the surface

        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            ndarray shape: (num_selected_faces, 2)

        Returns
        -------
        physical_surface_quad_points : ndarray
            (num_selected_faces, num_face_quads, dim)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim)
        physical_surface_quad_points = onp.sum(selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2)
        return physical_surface_quad_points

    def Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Indices and values for Dirichlet B.C.

        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]

        Returns
        -------
        node_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to num_total_nodes - 1
        vec_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to to vec - 1
        vals_List : List[ndarray]
            Dirichlet values to be assigned
        """
        node_inds_list = []
        vec_inds_list = []
        vals_list = []
        if dirichlet_bc_info is not None:
            location_fns, vecs, value_fns = dirichlet_bc_info
            assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs)
            for i in range(len(location_fns)):
                num_args = location_fns[i].__code__.co_argcount
                if num_args == 1:
                    location_fn = lambda point, ind: location_fns[i](point)
                elif num_args == 2:
                    location_fn = location_fns[i]
                else:
                    raise ValueError(f"Wrong number of arguments for location_fn: must be 1 or 2, get {num_args}")

                node_inds = onp.argwhere(jax.vmap(location_fn)(self.mesh.points, np.arange(self.num_total_nodes))).reshape(-1)
                vec_inds = onp.ones_like(node_inds, dtype=onp.int32) * vecs[i]
                values = jax.vmap(value_fns[i])(self.mesh.points[node_inds].reshape(-1, self.dim)).reshape(-1)
                node_inds_list.append(node_inds)
                vec_inds_list.append(vec_inds)
                vals_list.append(values)
        return node_inds_list, vec_inds_list, vals_list

    def update_Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Reset Dirichlet boundary conditions.
        Useful when a time-dependent problem is solved, and at each iteration the boundary condition needs to be updated.

        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]
        """
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(dirichlet_bc_info)

    def periodic_boundary_conditions(self):
        """Not working
        """
        p_node_inds_list_A = []
        p_node_inds_list_B = []
        p_vec_inds_list = []
        if self.periodic_bc_info is not None:
            location_fns_A, location_fns_B, mappings, vecs = self.periodic_bc_info
            for i in range(len(location_fns_A)):
                node_inds_A = onp.argwhere(jax.vmap(location_fns_A[i])(self.mesh.points)).reshape(-1)
                node_inds_B = onp.argwhere(jax.vmap(location_fns_B[i])(self.mesh.points)).reshape(-1)
                points_set_A = self.mesh.points[node_inds_A]
                points_set_B = self.mesh.points[node_inds_B]

                EPS = 1e-5
                node_inds_B_ordered = []
                for node_ind in node_inds_A:
                    point_A = self.mesh.points[node_ind]
                    dist = onp.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
                    node_ind_B_ordered = node_inds_B[onp.argwhere(dist < EPS)].reshape(-1)
                    node_inds_B_ordered.append(node_ind_B_ordered)

                node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(-1)
                vec_inds = onp.ones_like(node_inds_A, dtype=onp.int32) * vecs[i]

                p_node_inds_list_A.append(node_inds_A)
                p_node_inds_list_B.append(node_inds_B_ordered)
                p_vec_inds_list.append(vec_inds)
                assert len(node_inds_A) == len(node_inds_B_ordered)

        return p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list

    def get_boundary_conditions_inds(self, location_fns):
        """Given location functions, compute which faces satisfy the condition.

        Parameters
        ----------
        location_fns : List[Callable]
            Callable: a location function that inputs a point (ndarray) and returns if the point satisfies the location condition
                      e.g., lambda x: np.isclose(x[0], 0.)
                      If this location function takes 2 arguments, then the first is point and the second is index.
                      e.g., lambda x, ind: np.isclose(x[0], 0.) & np.isin(ind, np.array([1, 3, 10]))

        Returns
        -------
        boundary_inds_list : List[onp.ndarray]
            (num_selected_faces, 2)
            boundary_inds_list[k][i, 0] returns the global cell index of the ith selected face of boundary subset k
            boundary_inds_list[k][i, 1] returns the local face index of the ith selected face of boundary subset k
        """
        # TODO: assume this works for all variables, and return the same result
        cell_points = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1)  # (num_cells, num_faces, num_face_vertices, dim)
        cell_face_inds = onp.take(self.cells, self.face_inds, axis=1) # (num_cells, num_faces, num_face_vertices)
        boundary_inds_list = []
        if location_fns is not None:
            for i in range(len(location_fns)):
                num_args = location_fns[i].__code__.co_argcount
                if num_args == 1:
                    location_fn = lambda point, ind: location_fns[i](point)
                elif num_args == 2:
                    location_fn = location_fns[i]
                else:
                    raise ValueError(f"Wrong number of arguments for location_fn: must be 1 or 2, get {num_args}")

                vmap_location_fn = jax.vmap(location_fn)
                def on_boundary(cell_points, cell_inds):
                    boundary_flag = vmap_location_fn(cell_points, cell_inds)
                    return onp.all(boundary_flag)

                vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
                boundary_flags = vvmap_on_boundary(cell_face_points, cell_face_inds)
                boundary_inds = onp.argwhere(boundary_flags)  # (num_selected_faces, 2)
                boundary_inds_list.append(boundary_inds)

        return boundary_inds_list

    def convert_from_dof_to_quad(self, sol):
        """Obtain quad values from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)

        Returns
        -------
        u : np.DeviceArray
            (num_cells, num_quads, vec)
        """
        # (num_total_nodes, vec) -> (num_cells, num_nodes, vec)
        cells_sol = sol[self.cells]
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
        return u

    def convert_from_dof_to_face_quad(self, sol, boundary_inds):
        """Obtain surface solution from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)
        boundary_inds : int

        Returns
        -------
        u : np.DeviceArray
            (num_selected_faces, num_face_quads, vec)
        """
        cells_old_sol = sol[self.cells]  # (num_cells, num_nodes, vec)
        selected_cell_sols = cells_old_sol[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, vec))
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) 
        # -> (num_selected_faces, num_face_quads, vec)
        u = np.sum(selected_cell_sols[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
        return u

    def sol_to_grad(self, sol):
        """Obtain solution gradient from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)

        Returns
        -------
        u_grads : np.DeviceArray
            (num_cells, num_quads, vec, dim)
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_quads, vec, dim)
        return u_grads

    def print_BC_info(self):
        """Print boundary condition information for debugging purposes.

        TODO: Not working
        """
        if hasattr(self, 'neumann_boundary_inds_list'):
            print(f"\n\n### Neumann B.C. is specified")
            for i in range(len(self.neumann_boundary_inds_list)):
                print(f"\nNeumann Boundary part {i + 1} information:")
                print(self.neumann_boundary_inds_list[i])
                print(
                    f"Array.shape = (num_selected_faces, 2) = {self.neumann_boundary_inds_list[i].shape}"
                )
                print(f"Interpretation:")
                print(
                    f"    Array[i, 0] returns the global cell index of the ith selected face"
                )
                print(
                    f"    Array[i, 1] returns the local face index of the ith selected face"
                )
        else:
            print(f"\n\n### No Neumann B.C. found.")

        if len(self.node_inds_list) != 0:
            print(f"\n\n### Dirichlet B.C. is specified")
            for i in range(len(self.node_inds_list)):
                print(f"\nDirichlet Boundary part {i + 1} information:")
                bc_array = onp.stack([
                    self.node_inds_list[i], self.vec_inds_list[i],
                    self.vals_list[i]
                ]).T
                print(bc_array)
                print(
                    f"Array.shape = (num_selected_dofs, 3) = {bc_array.shape}")
                print(f"Interpretation:")
                print(
                    f"    Array[i, 0] returns the node index of the ith selected dof"
                )
                print(
                    f"    Array[i, 1] returns the vec index of the ith selected dof"
                )
                print(
                    f"    Array[i, 2] returns the value assigned to ith selected dof"
                )
        else:
            print(f"\n\n### No Dirichlet B.C. found.")
