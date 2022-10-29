import numpy as onp
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
import scipy
import os
import sys
import time
import meshio
import matplotlib.pyplot as plt
from functools import partial

from jax_am.fem.basis import get_face_shape_vals_and_grads, get_shape_vals_and_grads

from jax.config import config
config.update("jax_enable_x64", True)

onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=5)


class FEM:
    def __init__(self, mesh, ele_type='hexahedron', lag_order=1, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        """
        Attributes
        ----------
        self.mesh: Mesh object
            The mesh object stores points (coordinates) and cells (connectivity).
        self.points: ndarray
            shape: (num_total_nodes, dim) 
            The physical mesh nodal coordinates.
        self.dim: int
            The dimension of the problem.
        self.num_quads: int
            Number of quadrature points for each hex element.
        self.num_faces: int
            Number of faces for each hex element.
        self.num_cells: int
            Number of hex elements.
        self.num_total_nodes: int
            Number of total nodes.
        """
        self.mesh = mesh
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.dim = len(self.points[0])
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)
        self.num_total_dofs = self.num_total_nodes*self.vec

        start = time.time()

        # Some re-used quantities can be pre-computed and stored for better performance.
        self.shape_vals, self.shape_grads_ref, self.quad_weights = get_shape_vals_and_grads(ele_type, lag_order)
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, self.face_normals, self.face_inds \
        = get_face_shape_vals_and_grads(ele_type, lag_order)

        self.num_quads = self.shape_vals.shape[0]
        self.num_nodes = self.shape_vals.shape[1]
        self.num_faces = self.face_shape_vals.shape[0]

        self.shape_grads, self.JxW = self.get_shape_grads()

        # Note: Assume Dirichlet B.C. must be provided. This is probably true for all the problems we will encounter.
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(dirichlet_bc_info)

        end = time.time()
        compute_time = end - start
        print(f"\nDone pre-computations, took {compute_time} [s]")
        print(f"Solving a problem with {len(self.cells)} cells, {self.num_total_nodes}x{self.vec} = {self.num_total_dofs} dofs.")

    def get_shape_grads(self):
        """Pre-compute shape function gradient value
        The gradient is w.r.t physical coordinates.
        See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
        Page 147, Eq. (3.9.3)

        Returns
        -------
        shape_grads_physical: ndarray
            (num_cells, num_quads, num_nodes, dim)  
        JxW: ndarray
            (num_cells, num_quads)
        """
        assert self.shape_grads_ref.shape == (self.num_quads, self.num_nodes, self.dim)
        physical_coos = onp.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)
        # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
        jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] * self.shape_grads_ref[None, :, :, None, :], axis=2, keepdims=True)
        jacobian_det = onp.linalg.det(jacobian_dx_deta)[:, :, 0] # (num_cells, num_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)
        # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim) 
        # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
        shape_grads_physical = (self.shape_grads_ref[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
        JxW = jacobian_det * self.quad_weights[None, :]
        return shape_grads_physical, JxW

    def get_face_shape_grads(self, boundary_inds):
        """Face shape function gradients and JxW (for surface integral)
        Nanson's formula is used to map physical surface ingetral to reference domain
        Reference: https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

        Parameters
        ----------
        boundary_inds: list[ndarray]
            ndarray shape: (num_selected_faces, 2)

        Returns
        ------- 
        face_shape_grads_physical: ndarray
            (num_selected_faces, num_face_quads, num_nodes, dim)
        nanson_scale: ndarray
            (num_selected_faces, num_face_quads)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes, dim)
        selected_f_normals = self.face_normals[boundary_inds[:, 1]] # (num_selected_faces, dim)

        # (num_selected_faces, 1, num_nodes, dim, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        # (num_selected_faces, num_face_quads, num_nodes, dim, dim) -> (num_selected_faces, num_face_quads, dim, dim)
        jacobian_dx_deta = onp.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
        jacobian_det = onp.linalg.det(jacobian_dx_deta) # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta) # (num_selected_faces, num_face_quads, dim, dim)

        # (1, num_face_quads, num_nodes, 1, dim) @ (num_selected_faces, num_face_quads, 1, dim, dim)
        # (num_selected_faces, num_face_quads, num_nodes, 1, dim) -> (num_selected_faces, num_face_quads, num_nodes, dim)
        face_shape_grads_physical = (selected_f_shape_grads_ref[:, :, :, None, :] @ jacobian_deta_dx[:, :, None, :, :])[:, :, :, 0, :]

        # (num_selected_faces, 1, 1, dim) @ (num_selected_faces, num_face_quads, dim, dim)
        # (num_selected_faces, num_face_quads, 1, dim) -> (num_selected_faces, num_face_quads)
        nanson_scale = onp.linalg.norm((selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :], axis=-1)
        selected_weights = self.face_quad_weights[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads)
        nanson_scale = nanson_scale * jacobian_det * selected_weights
        return face_shape_grads_physical, nanson_scale

    def get_physical_quad_points(self):
        """Compute physical quadrature points
 
        Returns
        ------- 
        physical_quad_points: ndarray
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
        boundary_inds: list[ndarray]
            ndarray shape: (num_selected_faces, 2)

        Returns
        ------- 
        physical_surface_quad_points: ndarray
            (num_selected_faces, num_face_quads, dim) 
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, dim)
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)  
        # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim) 
        physical_surface_quad_points = onp.sum(selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2)
        return physical_surface_quad_points

    def Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Indices and values for Dirichlet B.C. 

        Parameters
        ----------
        dirichlet_bc_info: [location_fns, vecs, value_fns]
            location_fns: list[callable]
                callable: a function that inputs a point (ndarray) and returns if the point satisfies the location condition
            vecs: list[int]
                integer value must be in the range of 0 to vec - 1, 
                specifying which component of the (vector) variable to apply Dirichlet condition to
            value_fns: list[callable]
                callable: a function that inputs a point (ndarray) and returns the Dirichlet value

        Returns
        ------- 
        node_inds_list: list[ndarray]
            The ndarray ranges from 0 to num_total_nodes - 1
        vec_inds_list: list[ndarray]
            The ndarray ranges from 0 to to vec - 1
        vals_list: list[ndarray]
            Dirichlet values to be assigned
        """
        location_fns, vecs, value_fns = dirichlet_bc_info
        # TODO: add assertion for vecs, vecs must only contain 0 or 1 or 2, and must be integer
        assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs)
        node_inds_list = []
        vec_inds_list = []
        vals_list = []
        for i in range(len(location_fns)):
            node_inds = onp.argwhere(jax.vmap(location_fns[i])(self.mesh.points)).reshape(-1)
            vec_inds = onp.ones_like(node_inds, dtype=onp.int32)*vecs[i]
            values = jax.vmap(value_fns[i])(self.mesh.points[node_inds].reshape(-1, self.dim)).reshape(-1)
            node_inds_list.append(node_inds)
            vec_inds_list.append(vec_inds)
            vals_list.append(values)
        return node_inds_list, vec_inds_list, vals_list

    def update_Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Reset Dirichlet boundary conditions.
        Useful when a time-dependent problem is solved, and at each iteration the boundary condition needs to be updated.
        """
        # TODO: use getter and setter!
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(dirichlet_bc_info)

    def Neuman_boundary_conditions_inds(self, location_fns):
        """Given location functions, compute which faces satisfy the condition. 

        Parameters
        ----------
        location_fns: list[callable]
            callable: a function that inputs a point (ndarray) and returns if the point satisfies the location condition
                      e.g., lambda x: np.isclose(x[0], 0.)

        Returns
        ------- 
        boundary_inds_list: list[ndarray]
            ndarray shape: (num_selected_faces, 2)
            boundary_inds_list[k][i, j] returns the index of face j of cell i of surface k
        """
        # face_inds = get_face_inds()
        cell_points = onp.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1) # (num_cells, num_faces, num_face_nodes, dim)
        boundary_inds_list = []
        for i in range(len(location_fns)):
            vmap_location_fn = jax.vmap(location_fns[i])
            def on_boundary(cell_points):
                boundary_flag = vmap_location_fn(cell_points)
                return onp.all(boundary_flag)
            vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
            boundary_flags = vvmap_on_boundary(cell_face_points)
            boundary_inds = onp.argwhere(boundary_flags) # (num_selected_faces, 2)
            boundary_inds_list.append(boundary_inds)
        return boundary_inds_list

    def Neuman_boundary_conditions_vals(self, value_fns, boundary_inds_list):
        """Compute traction values on the face quadrature points.

        Parameters
        ----------
        value_fns: list[callable]
            callable: a function that inputs a point (ndarray) and returns the value
                      e.g., lambda x: x[0]**2
        boundary_inds_list: list[ndarray]
            ndarray shape: (num_selected_faces, 2)    

        Returns
        ------- 
            traction_list: list[ndarray]
            ndarray shape: (num_selected_faces, num_face_quads, vec)
        """
        traction_list = []
        for i in range(len(value_fns)):
            boundary_inds = boundary_inds_list[i]
            # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
            subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)
            traction = jax.vmap(jax.vmap(value_fns[i]))(subset_quad_points) # (num_selected_faces, num_face_quads, vec)
            assert len(traction.shape) == 3
            traction_list.append(traction)
        return traction_list


class Laplace(FEM):
    """Solving problems whose weak form is (f(u_grad), v_grad) * dx - (traction, v) * ds - (body_force, v) * dx = 0,
    where u and v are trial and test functions, respectively, and f is a general function.
    This covers
        - Poisson's problem
        - Linear elasticity
        - Hyper-elasticity
        - Plasticity
        ...
    """
    # TODO: Better way to write this __ini__ thing?
    def __init__(self, mesh, ele_type='hexahedron', lag_order=1, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        super().__init__(mesh, ele_type, lag_order, dirichlet_bc_info, neumann_bc_info, source_info) 
        # Some pre-computations   
        self.body_force = self.compute_body_force(source_info)
        self.neumann = self.compute_Neumann_integral(neumann_bc_info)
        # (num_cells, num_quads, num_nodes, 1, dim)
        self.v_grads_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]
        self.mass_kernel_flag = False
        self.laplace_kernel_flag = True

    def get_tensor_map(self):
        raise NotImplementedError(f"Child class must override this function.")

    def get_mass_map(self):
        raise NotImplementedError(f"Child class must override this function.")

    def get_laplace_kernel(self, tensor_map):
        def laplace_kernel(cell_sol, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars):
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :] 
            u_grads = np.sum(u_grads, axis=1) # (num_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim) # (num_quads, vec, dim) 
            # (num_quads, vec, dim) 
            u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape) 
            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            return val
        return laplace_kernel

    def get_mass_kernel(self, mass_map):
        def mass_kernel(cell_sol, cell_JxW, *cell_internal_vars):
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
            u = np.sum(cell_sol[None, :, :] * self.shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(mass_map)(u, *cell_internal_vars) # (num_quads, vec) 
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, None, None) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * self.shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)
            return val
        return mass_kernel    

    def unpack_kernels_vars(self, **internal_vars):
        if 'mass' in internal_vars.keys():
            mass_internal_vars = internal_vars['mass']
        else:
            mass_internal_vars = ()

        if 'laplace' in internal_vars.keys():
            laplace_internal_vars = internal_vars['laplace']
        else:
            laplace_internal_vars = ()

        return [mass_internal_vars, laplace_internal_vars]        

    def get_kernel_fn(self):
        def kernel(cell_sol, cell_shape_grads, cell_JxW, cell_v_grads_JxW, cell_mass_internal_vars, cell_laplace_internal_vars):
            if self.mass_kernel_flag:
                mass_kernel = self.get_mass_kernel(self.get_mass_map())
                mass_val = mass_kernel(cell_sol, cell_JxW, *cell_mass_internal_vars)
            else:
                mass_val = 0.

            if self.laplace_kernel_flag:
                laplace_kernel = self.get_laplace_kernel(self.get_tensor_map())
                laplace_val = laplace_kernel(cell_sol, cell_shape_grads, cell_v_grads_JxW, *cell_laplace_internal_vars)
            else:
                laplace_val = 0.
            
            return laplace_val + mass_val

        def D_fn(cell_sol, *args):
            return jax.jacfwd(kernel)(cell_sol, *args)

        return kernel, D_fn

    def split_and_compute(self, cells_sol, fn, np_version, **internal_vars):
        vmap_fn = jax.jit(jax.vmap(fn))
        kernal_vars = self.unpack_kernels_vars(**internal_vars)
        num_cuts = 20
        if num_cuts > len(self.cells):
            num_cuts = len(self.cells)
        batch_size = len(self.cells) // num_cuts
        input_collection = [cells_sol, self.shape_grads, self.JxW, self.v_grads_JxW, *kernal_vars]
        values = []
        for i in range(num_cuts):
            if i < num_cuts - 1:
                input_col = jax.tree_map(lambda x: x[i*batch_size:(i + 1)*batch_size], input_collection)
            else:
                input_col = jax.tree_map(lambda x: x[i*batch_size:], input_collection)

            val = vmap_fn(*input_col)
            values.append(val)

        # np_version set to jax.numpy allows for auto diff, but uses GPU memory
        # np_version set to ordinary numpy saves GPU memory, but can't use auto diff 
        values = np_version.vstack(values)
        return values

    def compute_residual_vars(self, sol, **internal_vars):
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        kernel, _ = self.get_kernel_fn()
        weak_form = self.split_and_compute(cells_sol, kernel, np, **internal_vars) # (num_cells, num_nodes, vec)
        weak_form = weak_form.reshape(-1, self.vec) # (num_cells*num_nodes, vec)
        res = np.zeros_like(sol)
        res = res.at[self.cells.reshape(-1)].add(weak_form) - self.body_force - self.neumann
        return res 

    def compute_residual(self, sol):
        """Child class should override if internal variables exist
        """
        return self.compute_residual_vars(sol)

    def newton_vars(self, sol, **internal_vars):
        print(f"Update solution, internal variable...")
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        _, D_fn = self.get_kernel_fn()
        print(f"Compute D...")
        D = self.split_and_compute(cells_sol, D_fn, onp, **internal_vars)
        V = D.reshape(-1)
        inds = (self.vec * self.cells[:, :, None] + onp.arange(self.vec)[None, None, :]).reshape(self.num_cells, -1)
        I = onp.repeat(inds[:, :, None], self.num_nodes*self.vec, axis=2).reshape(-1)
        J = onp.repeat(inds[:, None, :], self.num_nodes*self.vec, axis=1).reshape(-1)
        # print(f"type(V) = {type(V)}, type(I) = {type(I)}, type(J) = {type(J)}")
        print(f"Creating sparse matrix with scipy...")
        self.A_sp_scipy = scipy.sparse.csc_array((V, (I, J)), shape=(self.num_total_dofs, self.num_total_dofs))
        print(f"Creating sparse matrix from scipy using JAX BCOO...")
        self.A_sp = BCOO.from_scipy_sparse(self.A_sp_scipy).sort_indices()
        print(f"self.A_sp.data.shape = {self.A_sp.data.shape}")
        print(f"Global sparse matrix takes about {self.A_sp.data.shape[0]*8*3/2**30} G memory to store.")

    def newton_update(self, sol):
        """Child class should override if internal variables exist
        """
        return self.newton_vars(sol)

    def compute_linearized_residual(self, dofs):
        return self.A_sp @ dofs

    def compute_body_force(self, source_info):
        """In the weak form, we have (body_force, v) * dx, and this function computes this.

        Parameters
        ----------
        source_info: callable
            A function that inputs a point (ndarray) and returns the body force at this point.

        Returns
        -------
        body_force: ndarray
            (num_total_nodes, vec)
        """
        rhs = np.zeros((self.num_total_nodes, self.vec))
        if source_info is not None:
            body_force_fn = source_info
            physical_quad_points = self.get_physical_quad_points() # (num_cells, num_quads, dim) 
            body_force = jax.vmap(jax.vmap(body_force_fn))(physical_quad_points) # (num_cells, num_quads, vec) 
            assert len(body_force.shape) == 3
            v_vals = np.repeat(self.shape_vals[None, :, :, None], self.num_cells, axis=0) # (num_cells, num_quads, num_nodes, 1)
            v_vals = np.repeat(v_vals, self.vec, axis=-1) # (num_cells, num_quads, num_nodes, vec)
            # (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
            rhs_vals = np.sum(v_vals * body_force[:, :, None, :] * self.JxW[:, :, None, None], axis=1).reshape(-1, self.vec) 
            rhs = rhs.at[self.cells.reshape(-1)].add(rhs_vals) 
        return rhs

    def compute_Neumann_integral(self, neumann_bc_info):
        """In the weak form, we have the Neumann integral: (traction, v) * ds, and this function computes this.

        Parameters
        ----------
        neumann_bc_info: [location_fns, value_fns]
            location_fns: list[callable]
            value_fns: list[callable]

        Returns
        -------
        integral: ndarray
            (num_total_nodes, vec)
        """
        integral = np.zeros((self.num_total_nodes, self.vec))
        if neumann_bc_info is not None:
            location_fns, value_fns = neumann_bc_info
            integral = np.zeros((self.num_total_nodes, self.vec))
            boundary_inds_list = self.Neuman_boundary_conditions_inds(location_fns)
            traction_list = self.Neuman_boundary_conditions_vals(value_fns, boundary_inds_list)
            for i, boundary_inds in enumerate(boundary_inds_list):
                traction = traction_list[i]
                _, nanson_scale = self.get_face_shape_grads(boundary_inds) # (num_selected_faces, num_face_quads)
                # (num_faces, num_face_quads, num_nodes) ->  (num_selected_faces, num_face_quads, num_nodes)
                v_vals = np.take(self.face_shape_vals, boundary_inds[:, 1], axis=0)
                v_vals = np.repeat(v_vals[:, :, :, None], self.vec, axis=-1) # (num_selected_faces, num_face_quads, num_nodes, vec)
                subset_cells = np.take(self.cells, boundary_inds[:, 0], axis=0) # (num_selected_faces, num_nodes)
                # (num_selected_faces, num_nodes, vec) -> (num_selected_faces*num_nodes, vec)
                int_vals = np.sum(v_vals * traction[:, :, None, :] * nanson_scale[:, :, None, None], axis=1).reshape(-1, self.vec) 
                integral = integral.at[subset_cells.reshape(-1)].add(int_vals)   
        return integral

    def surface_integral(self, location_fn, surface_fn, sol):
        """Compute surface integral specified by surface_fn: f(u_grad) * ds
        For post-processing only.
        Example usage: compute the total force on a certain surface.

        Parameters
        ----------
        location_fn: callable
            A function that inputs a point (ndarray) and returns if the point satisfies the location condition.
        surface_fn: callable
            A function that inputs a point (ndarray) and returns the value.
        sol: ndarray
            (num_total_nodes, vec)

        Returns
        -------
        int_val: ndarray
            (vec,)
        """
        boundary_inds = self.Neuman_boundary_conditions_inds([location_fn])[0]
        face_shape_grads_physical, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        u_grads_face = sol[self.cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
        u_grads_face = np.sum(u_grads_face, axis=2) # (num_selected_faces, num_face_quads, vec, dim)
        traction = surface_fn(u_grads_face) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, vec) * (num_selected_faces, num_face_quads, 1)
        int_val = np.sum(traction * nanson_scale[:, :, None], axis=(0, 1))
        return int_val


class LinearPoisson(Laplace):
    def __init__(self, name, mesh, ele_type='hexahedron', lag_order=1, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 1
        super().__init__(mesh, ele_type, lag_order, dirichlet_bc_info, neumann_bc_info, source_info) 

    def get_tensor_map(self):
        return lambda x: x
 

class LinearElasticity(Laplace):
    def __init__(self, name, mesh, ele_type='hexahedron', lag_order=1, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, ele_type, lag_order, dirichlet_bc_info, neumann_bc_info, source_info) 
    
    def get_tensor_map(self):
        def stress(u_grad):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    def compute_surface_area(self, location_fn, sol):
        """For post-processing only
        """
        def unity_fn(u_grads):
            unity = np.ones_like(u_grads)[:, :, :, 0]
            return unity
        unity_integral_val = self.surface_integral(location_fn, unity_fn, sol)
        return unity_integral_val

    def compute_traction(self, location_fn, sol):
        """For post-processing only
        TODO: duplicated code
        """
        stress = self.get_tensor_map()
        vmap_stress = jax.vmap(stress)
        def traction_fn(u_grads):
            """
            Returns
            ------- 
            traction: ndarray
                (num_selected_faces, num_face_quads, vec)
            """
            # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
            sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
            # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
            # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
            normals = np.array([0., 0., 1.]).reshape((self.dim, 1))
            traction = (sigmas @ normals[None, None, :, :])[:, :, :, 0]
            return traction

        traction_integral_val = self.surface_integral(location_fn, traction_fn, sol)
        return traction_integral_val


class HyperElasticity(Laplace):
    def __init__(self, name, mesh, ele_type='hexahedron', lag_order=1, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, ele_type, lag_order, dirichlet_bc_info, neumann_bc_info, source_info)

    def get_tensor_map(self):
        def psi(F):
            E = 1e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        return first_PK_stress

    def compute_traction(self, location_fn, sol):
        """For post-processing only
        """
        first_PK_stress = self.get_tensor_map()
        vmap_stress = jax.vmap(first_PK_stress)
        def traction_fn(u_grads):
            """
            Returns
            ------- 
            traction: ndarray
                (num_selected_faces, num_face_quads, vec)
            """
            # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
            sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
            # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
            # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
            normals = np.array([0., 0., 1.]).reshape((self.dim, 1))
            traction = (sigmas @ normals[None, None, :, :])[:, :, :, 0]
            return traction

        traction_integral_val = self.surface_integral(location_fn, traction_fn, sol)
        return traction_integral_val


class Plasticity(Laplace):
    def __init__(self, name, mesh, ele_type='hexahedron', lag_order=1, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, ele_type, lag_order, dirichlet_bc_info, neumann_bc_info, source_info)
        self.epsilons_old = onp.zeros((len(self.cells), self.num_quads, self.vec, self.dim))
        self.sigmas_old = onp.zeros_like(self.epsilons_old)

    def get_tensor_map(self):
        _, stress_return_map = self.get_maps()
        return stress_return_map

    def newton_update(self, sol):
        return self.newton_vars(sol, laplace=[self.sigmas_old, self.epsilons_old])

    def compute_residual(self, sol):
        return self.compute_residual_vars(sol, laplace=[self.sigmas_old, self.epsilons_old])

    def get_maps(self):
        EPS = 1e-10
        # TODO
        def safe_sqrt(x):  
            safe_x = np.where(x > 0., x, EPS)
            return np.sqrt(safe_x)

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def stress(epsilon):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            sig0 = 250.
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc) + sigma_old

            s_dev = sigma_trial - 1./self.dim*np.trace(sigma_trial)*np.eye(self.dim)

            # s_norm = np.sqrt(3./2.*np.sum(s_dev*s_dev))
            s_norm = safe_sqrt(3./2.*np.sum(s_dev*s_dev))

            f_yield = s_norm - sig0
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            # TODO
            sigma = sigma_trial - f_yield_plus*s_dev/(s_norm + EPS)
            return sigma

        return strain, stress_return_map

    def stress_strain_fns(self):
        strain, stress_return_map = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        self.sigmas_old = vmap_stress_rm(u_grads, self.sigmas_old, self.epsilons_old)
        self.epsilons_old = vmap_strain(u_grads)

    def compute_avg_stress(self):
        """For post-processing only
        """
        # num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1)
        sigma = np.sum(self.sigmas_old.reshape(-1, self.vec, self.dim) * self.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.JxW)
        avg_sigma = sigma/vol
        return avg_sigma


class Mesh():
    """A custom mesh manager might be better than just use third-party packages like meshio?
    """
    def __init__(self, points, cells):
        # TODO: Assert that cells must have correct orders 
        self.points = points
        self.cells = cells
