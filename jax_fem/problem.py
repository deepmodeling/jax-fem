import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Union
import functools

from jax_fem.utils import timeit 
from jax_fem.generate_mesh import Mesh
from jax_fem.fe import FiniteElement
from jax_fem import logger


@dataclass
class Problem:
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str = 'HEX8'
    gauss_order: int = None
    dirichlet_bc_info: Optional[List[Union[List[Callable], List[int], List[Callable]]]] = None
    location_fns: Optional[List[Callable]] = None
    additional_info: Any = ()

    def __post_init__(self):

        if type(self.mesh) != type([]):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.gauss_order = [self.gauss_order]
            self.dirichlet_bc_info = [self.dirichlet_bc_info]

        self.num_vars = len(self.mesh)

        self.fes = [FiniteElement(mesh=self.mesh[i], 
                                  vec=self.vec[i], 
                                  dim=self.dim, 
                                  ele_type=self.ele_type[i], 
                                  gauss_order=self.gauss_order[i] if type(self.gauss_order) == type([]) else self.gauss_order,
                                  dirichlet_bc_info=self.dirichlet_bc_info[i] if type(self.dirichlet_bc_info) == type([]) else self.dirichlet_bc_info) \
                    for i in range(self.num_vars)] 

        self.cells_list = [fe.cells for fe in self.fes]
        # Assume all fes have the same number of cells, same dimension
        self.num_cells = self.fes[0].num_cells
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        self.offset = [0] 
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

        def find_ind(*x):
            inds = []
            for i in range(len(x)):
                x[i].reshape(-1)
                crt_ind = self.fes[i].vec * x[i][:, None] + np.arange(self.fes[i].vec)[None, :] + self.offset[i]
                inds.append(crt_ind.reshape(-1))

            return np.hstack(inds)

        # (num_cells, num_nodes*vec + ...)
        inds = onp.array(jax.vmap(find_ind)(*self.cells_list))
        self.I = onp.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
        self.J = onp.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list] # [(num_selected_faces, num_nodes), ...]
            inds_face = onp.array(jax.vmap(find_ind)(*cells_list_face)) # (num_selected_faces, num_nodes*vec + ...)
            I_face = onp.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
            J_face = onp.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
            self.I = onp.hstack((self.I, I_face))
            self.J = onp.hstack((self.J, J_face))
            self.cells_list_face_list.append(cells_list_face)
     
        self.cells_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*self.cells_list) # (num_cells, num_nodes + ...)

        dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
        # TODO: dumb_array_dof is useless?
        dumb_array_node = [np.zeros(fe.num_nodes) for fe in self.fes]
        # _, unflatten_fn_node = jax.flatten_util.ravel_pytree(dumb_array_node)
        _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)
        
        dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(dumb_sol_list)
        self.num_total_dofs_all_vars = len(dumb_dofs)

        self.num_nodes_cumsum = onp.cumsum([0] + [fe.num_nodes for fe in self.fes])
        # (num_cells, num_vars, num_quads)
        self.JxW = onp.transpose(onp.stack([fe.JxW for fe in self.fes]), axes=(1, 0, 2)) 
        # (num_cells, num_quads, num_nodes +..., dim)
        self.shape_grads = onp.concatenate([fe.shape_grads for fe in self.fes], axis=2)
        # (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_JxW = onp.concatenate([fe.v_grads_JxW for fe in self.fes], axis=2)

        # TODO: assert all vars quad points be the same
        # (num_cells, num_quads, dim)
        self.physical_quad_points = self.fes[0].get_physical_quad_points()  

        self.selected_face_shape_grads = []
        self.nanson_scale = []
        self.selected_face_shape_vals = []
        self.physical_surface_quad_points = []
        for boundary_inds in self.boundary_inds_list:
            s_shape_grads = []
            n_scale = []
            s_shape_vals = []
            for fe in self.fes:
                # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
                face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(boundary_inds)  
                selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
                s_shape_grads.append(face_shape_grads_physical)
                n_scale.append(nanson_scale)
                s_shape_vals.append(selected_face_shape_vals)

            # (num_selected_faces, num_face_quads, num_nodes + ..., dim)
            s_shape_grads = onp.concatenate(s_shape_grads, axis=2)
            # (num_selected_faces, num_vars, num_face_quads)
            n_scale = onp.transpose(onp.stack(n_scale), axes=(1, 0, 2))  
            # (num_selected_faces, num_face_quads, num_nodes + ...)
            s_shape_vals = onp.concatenate(s_shape_vals, axis=2)
            # (num_selected_faces, num_face_quads, dim)
            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(boundary_inds) 

            self.selected_face_shape_grads.append(s_shape_grads)
            self.nanson_scale.append(n_scale)
            self.selected_face_shape_vals.append(s_shape_vals)
            # TODO: assert all vars face quad points be the same
            self.physical_surface_quad_points.append(physical_surface_quad_points)

        self.internal_vars = ()
        self.internal_vars_surfaces = [() for _ in range(len(self.boundary_inds_list))]
        self.custom_init(*self.additional_info)
        self.pre_jit_fns()

    def custom_init(self):
        """Child class should override if more things need to be done in initialization
        """
        pass

    def get_laplace_kernel(self, tensor_map):

        def laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, :self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, :self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec

            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, vec, self.dim)  # (num_quads, vec, dim)
            # (num_quads, vec, dim)
            u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape)
            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
            return val

        return laplace_kernel

    def get_mass_kernel(self, mass_map):

        def mass_kernel(cell_sol_flat, x, cell_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_JxW: (num_vars, num_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            cell_JxW = cell_JxW[0]
            vec = self.fes[0].vec
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
            u = np.sum(cell_sol[None, :, :] * self.fes[0].shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(mass_map)(u, x, *cell_internal_vars)  # (num_quads, vec)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * self.fes[0].shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)
            val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
            return val

        return mass_kernel

    def get_surface_kernel(self, surface_map):

        def surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # x: (num_face_quads, dim)
            # face_nanson_scale: (num_vars, num_face_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            face_shape_vals = face_shape_vals[:, :self.fes[0].num_nodes]
            face_nanson_scale = face_nanson_scale[0]

            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(surface_map)(u, x, *cell_internal_vars_surface)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)

            return jax.flatten_util.ravel_pytree(val)[0]

        return surface_kernel

    def pre_jit_fns(self):
        def value_and_jacfwd(f, x):
            pushfwd = functools.partial(jax.jvp, f, (x, ))
            basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
            return y, jac

        def get_kernel_fn_cell():
            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
                """
                universal_kernel should be able to cover all situations (including mass_kernel and laplace_kernel).
                mass_kernel and laplace_kernel are from legacy JAX-FEM. They can still be used, but not mandatory.
                """

                # TODO: If there is no kernel map, returning 0. is not a good choice. 
                # Return a zero array with proper shape will be better.
                if hasattr(self, 'get_mass_map'):
                    mass_kernel = self.get_mass_kernel(self.get_mass_map())
                    mass_val = mass_kernel(cell_sol_flat, physical_quad_points, cell_JxW, *cell_internal_vars)
                else:
                    mass_val = 0.

                if hasattr(self, 'get_tensor_map'):
                    laplace_kernel = self.get_laplace_kernel(self.get_tensor_map())
                    laplace_val = laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars)
                else:
                    laplace_val = 0.

                if hasattr(self, 'get_universal_kernel'):
                    universal_kernel = self.get_universal_kernel()
                    universal_val = universal_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, 
                        cell_v_grads_JxW, *cell_internal_vars)
                else:
                    universal_val = 0.

                return laplace_val + mass_val + universal_val


            def kernel_jac(cell_sol_flat, *args):
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(kernel_partial, cell_sol_flat)  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)

            return kernel, kernel_jac

        def get_kernel_fn_face(ind):
            def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
                """
                universal_kernel should be able to cover all situations (including surface_kernel).
                surface_kernel is from legacy JAX-FEM. It can still be used, but not mandatory.
                """
                if hasattr(self, 'get_surface_maps'):
                    surface_kernel = self.get_surface_kernel(self.get_surface_maps()[ind])
                    surface_val = surface_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    surface_val = 0.

                if hasattr(self, 'get_universal_kernels_surface'):
                    universal_kernel = self.get_universal_kernels_surface()[ind]
                    universal_val = universal_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    universal_val = 0.

                return surface_val + universal_val

            def kernel_jac(cell_sol_flat, *args):
                # return jax.jacfwd(kernel)(cell_sol_flat, *args)
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(kernel_partial, cell_sol_flat)  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)

            return kernel, kernel_jac

        kernel, kernel_jac = get_kernel_fn_cell()
        kernel = jax.jit(jax.vmap(kernel))
        kernel_jac = jax.jit(jax.vmap(kernel_jac))
        self.kernel = kernel
        self.kernel_jac = kernel_jac

        num_surfaces = len(self.boundary_inds_list)
        if hasattr(self, 'get_surface_maps'):
            assert num_surfaces == len(self.get_surface_maps())
        elif hasattr(self, 'get_universal_kernels_surface'):
            assert num_surfaces == len(self.get_universal_kernels_surface()) 
        else:
            assert num_surfaces == 0, "Missing definitions for surface integral"
            

        self.kernel_face = []
        self.kernel_jac_face = []
        for i in range(len(self.boundary_inds_list)):
            kernel_face, kernel_jac_face = get_kernel_fn_face(i)
            kernel_face = jax.jit(jax.vmap(kernel_face))
            kernel_jac_face = jax.jit(jax.vmap(kernel_jac_face))
            self.kernel_face.append(kernel_face)
            self.kernel_jac_face.append(kernel_jac_face)

    @timeit
    def split_and_compute_cell(self, cells_sol_flat, np_version, jac_flag, internal_vars):
        """Volume integral in weak form
        """
        vmap_fn = self.kernel_jac if jac_flag else self.kernel
        num_cuts = 20
        if num_cuts > self.num_cells:
            num_cuts = self.num_cells
        batch_size = self.num_cells // num_cuts
        input_collection = [cells_sol_flat, self.physical_quad_points, self.shape_grads, self.JxW, self.v_grads_JxW, *internal_vars]

        if jac_flag:
            values = []
            jacs = []
            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree_map(lambda x: x[i * batch_size:(i + 1) * batch_size], input_collection)
                else:
                    input_col = jax.tree_map(lambda x: x[i * batch_size:], input_collection)

                val, jac = vmap_fn(*input_col)
                values.append(val)
                jacs.append(jac)
            values = np_version.vstack(values)
            jacs = np_version.vstack(jacs)

            return values, jacs
        else:
            values = []
            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree_map(lambda x: x[i * batch_size:(i + 1) * batch_size], input_collection)
                else:
                    input_col = jax.tree_map(lambda x: x[i * batch_size:], input_collection)

                val = vmap_fn(*input_col)
                values.append(val)
            values = np_version.vstack(values)
            return values

    def compute_face(self, cells_sol_flat, np_version, jac_flag, internal_vars_surfaces):
        """Surface integral in weak form
        """
        if jac_flag:
            values = []
            jacs = []
            for i, boundary_inds in enumerate(self.boundary_inds_list):
                vmap_fn = self.kernel_jac_face[i]
                selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
                input_collection = [selected_cell_sols_flat, self.physical_surface_quad_points[i], self.selected_face_shape_vals[i], 
                                    self.selected_face_shape_grads[i], self.nanson_scale[i], *internal_vars_surfaces[i]]

                val, jac = vmap_fn(*input_collection)
                values.append(val)
                jacs.append(jac)
            return values, jacs
        else:
            values = []
            for i, boundary_inds in enumerate(self.boundary_inds_list):
                vmap_fn = self.kernel_face[i]
                selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
                # TODO: duplicated code
                input_collection = [selected_cell_sols_flat, self.physical_surface_quad_points[i], self.selected_face_shape_vals[i], 
                                    self.selected_face_shape_grads[i], self.nanson_scale[i], *internal_vars_surfaces[i]]
                val = vmap_fn(*input_collection)
                values.append(val)
            return values

    def compute_residual_vars_helper(self, weak_form_flat, weak_form_face_flat):
        res_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        weak_form_list = jax.vmap(lambda x: self.unflatten_fn_dof(x))(weak_form_flat) # [(num_cells, num_nodes, vec), ...]
        res_list = [res_list[i].at[self.cells_list[i].reshape(-1)].add(weak_form_list[i].reshape(-1, 
            self.fes[i].vec)) for i in range(self.num_vars)]

        for ind, cells_list_face in enumerate(self.cells_list_face_list):
            weak_form_face_list = jax.vmap(lambda x: self.unflatten_fn_dof(x))(weak_form_face_flat[ind]) # [(num_selected_faces, num_nodes, vec), ...]
            res_list = [res_list[i].at[cells_list_face[i].reshape(-1)].add(weak_form_face_list[i].reshape(-1, 
                self.fes[i].vec)) for i in range(self.num_vars)]   

        return res_list

    def compute_residual_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        logger.debug(f"Computing cell residual...")
        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)] # [(num_cells, num_nodes, vec), ...]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list) # (num_cells, num_nodes*vec + ...)
        weak_form_flat = self.split_and_compute_cell(cells_sol_flat, np, False, internal_vars)  # (num_cells, num_nodes*vec + ...)
        weak_form_face_flat = self.compute_face(cells_sol_flat, np, False, internal_vars_surfaces)  # [(num_selected_faces, num_nodes*vec + ...), ...]
        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_newton_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        logger.debug(f"Computing cell Jacobian and cell residual...")
        cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)] # [(num_cells, num_nodes, vec), ...]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list) # (num_cells, num_nodes*vec + ...)
        # (num_cells, num_nodes*vec + ...),  (num_cells, num_nodes*vec + ..., num_nodes*vec + ...)
        weak_form_flat, cells_jac_flat = self.split_and_compute_cell(cells_sol_flat, onp, True, internal_vars)
        self.V = onp.array(cells_jac_flat.reshape(-1))

        # [(num_selected_faces, num_nodes*vec + ...,), ...], [(num_selected_faces, num_nodes*vec + ..., num_nodes*vec + ...,), ...]
        weak_form_face_flat, cells_jac_face_flat = self.compute_face(cells_sol_flat, onp, True, internal_vars_surfaces)
        for cells_jac_f_flat in cells_jac_face_flat:
            self.V = onp.hstack((self.V, onp.array(cells_jac_f_flat.reshape(-1))))

        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_residual(self, sol_list):
        return self.compute_residual_vars(sol_list, self.internal_vars, self.internal_vars_surfaces)

    def newton_update(self, sol_list):
        return self.compute_newton_vars(sol_list, self.internal_vars, self.internal_vars_surfaces)

    def set_params(self, params):
        """Used for solving inverse problems.
        """
        raise NotImplementedError("Child class must implement this function!")