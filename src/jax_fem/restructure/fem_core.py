"""
Here, we aim to keep the essential attributes and methods related to setting up
the FEM problem. This includes initializing the mesh, element types, and
boundary conditions.
"""

import numpy as onp
import jax
import jax.numpy as np
import sys
import time
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Union

from jax_fem.basis import (get_face_shape_vals_and_grads,
                           get_shape_vals_and_grads)

# Define immutable containers for the boundary conditions
from typing import NamedTuple, List, Callable

import equinox as eqx

class FEM(eqx.Module):

    """
    Solving second-order elliptic PDE problems whose FEM weak form is
    (f(u_grad), v_grad) * dx - (traction, v) * ds - (body_force, v) * dx = 0,
    where u and v are trial and test functions, respectively, and f is a general function.
    This covers
        - Poisson's problem
        - Heat equation
        - Linear elasticity
        - Hyper-elasticity
        - Plasticity

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

    dirichlet_bcs : NamedTuple
    periodic_bcs : NamedTuple
    neumann_bcs : NamedTuple
    cauchy_bcs: NamedTuple
    source_info: Callable
        A function that inputs a point and returns the body force at this point
    """


    def __init__(self,
                 mesh: Mesh,
                 ele_type: str = "HEX8",
                 vec: int,
                 dim: int,
                 dirichlet_bc_info: Optional[NamedTuple] = None,
                 periodic_bc_info: Optional[NamedTuple] = None,
                 neumann_bc_info: Optional[NamedTuple] = None,
                 cauchy_bc_info: Optional[NamedTuple] = None,
                 source_info: Callable = None,):

        self.mesh = mesh
        self.ele_type = ele_type
        self.vec = vec
        self.dim = dim
        self.dirichlet_bc_info = dirichlet_bc_info
        self.periodic_bc_info = periodic_bc_info
        self.neumann_bc_info = neumann_bc_info
        self.cauchy_bc_info = cauchy_bc_info
        self.source_info = source_info

        self.initalize_params(ele_type)

    def initialize_mesh_params_and_grads(self):

        self.num_cells = len(self.mesh.cells)
        self.num_total_nodes = len(self.mesh.points)
        self.num_total_dofs = self.num_total_nodes * self.vec

        # physical coords shape is (num_cells, num_nodes, dim)
        self.physical_coos = onp.take(self.mesh.points, self.mesh.cells, axis=0)

        # Get shape values and gradients for prescribed element type
        (self.shape_vals,
         self.shape_grads_ref,
         self.quad_weights) = get_shape_vals_and_grads(self.ele_type)

        # Get face shape values, gradients, weights, face normals, and indices
        (self.face_shape_vals,
         self.face_shape_grads_ref,
         self.face_quad_weights,
         self.face_normals,
         self.face_inds) = get_face_shape_vals_and_grads(self.ele_type)

        self.num_quads, self.num_nodes = self.shape_vals.shape[:2]
        self.num_faces = self.face_shape_vals.shape[0]

        self.shape_grads, self.JxW = self.get_shape_fn_grads()

        self.v_grads_JxW = np.einsum('abcd,e->abcde',
                                     self.shape_grads,
                                     self.JxW)

    def get_shape_fn_grads(self):
        """
        Compute shape function gradients w.r.t physical coordinates.
        Returns
        -------
        shape_fn_grads : ndarray
            An array representing the shape function gradients w.r.t physical
            coordinates. The array dimensions are:
            (num_cells, num_quads, num_nodes, dim)

        JxW : ndarray
            An array representing the product of the determinant of the
            Jacobian matrix and quadrature weights. The array dimensions are:
            (num_cells, num_quads)
        """


        # Jacobian matrix (derivatives of physical coordinates w.r.t.
        # reference coordinates)
        jacobian_dx_deta = np.einsum('ijk,lmk->ijlm',
                                      self.physical_coos,
                                      self.shape_grads_ref)

        # Compute the inverse of the Jacobian matrix
        jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)

        # Compute the determinant of the Jacobian matrix
        jacobian_det = np.linalg.det(jacobian_dx_deta)  # (num_cells, num_quads)


        # Compute physical gradients of the shape functions
        shape_grads_physical = np.einsum('ijk,ijklm->ijlm',
                                         self.shape_grads_ref,
                                         jacobian_deta_dx)

        # Product of the determinant of the Jacobian and quadrature weights
        JxW = jacobian_det * self.quad_weights[:, np.newaxis]

        return shape_grads_physical, JxW

    def get_face_shape_grads(self, boundary_inds):
        """
        Face shape function gradients and JxW (for surface integral) Nanson's
        formula is used to map physical surface ingetral to reference domain
        Reference:
        https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

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

        # (num_selected_faces, num_nodes, dim)
        selected_coos = self.physical_coos[boundary_inds[:, 0]]

        # (num_selected_faces, num_face_quads, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[boundary_inds[:, 1]]

        # (num_selected_faces, dim)
        selected_f_normals = self.face_normals[boundary_inds[:, 1]]

        # Calculating the Jacobian
        jacobian_dx_deta = np.einsum("ijke,ijkl->ijel",
                                     selected_coos,
                                     selected_f_shape_grads_ref)

        jacobian_det = np.linalg.det(jacobian_dx_deta)
        jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)

        # Calculating physical gradients of face shape functions
        face_shape_grads_physical = np.einsum("ijkl,ijel->ijek",
                                              selected_f_shape_grads_ref,
                                              jacobian_deta_dx)

        # Calculating Nanson's formula scale
        nanson_scale = np.linalg.norm(np.einsum("ij,ijel->iel",
                                                selected_f_normals,
                                                jacobian_deta_dx), axis=-1)

        selected_weights = self.face_quad_weights[boundary_inds[:, 1]]
        nanson_scale *= jacobian_det * selected_weights

        return face_shape_grads_physical, nanson_scale

    def get_physical_quad_points(self):
        # TODO: REWRITE AND OPTIMIZE

    def get_physical_surface_quad_points(self, boundary_inds):
        # TODO: REWRITE AND OPTIMIZE


    def set_boundary_conditions(self):

        # Discretizing boundary conditions -
        _ = self.Dirichlet_boundary_conditions(self.dirichlet_bcs)
        _  = self.periodic_boundary_conditions()

        # Term which occurs in all integrations (all constants)
        self.v_grads_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]

        self.internal_vars = {}
        self.compute_Neumann_boundary_inds()


    # TODO: Get rid of the repetitive calculation of physical_coos
    # if possible

    # The methods below are all one-time calls, even in repeated analysis
    def get_shape_grads(self):
        # ...

    def get_face_shape_grads(self, boundary_inds):
        # ...

    def get_physical_quad_points(self):
        # ...

    def get_physical_surface_quad_points(self, boundary_inds):
        # ...

    # This is effectively a discretized version of the BC - change the name
    # accordingly?

    def Dirichlet_boundary_conditions(self, dirichlet_bcs):
        # ...
        # Again, all we are doing here is calculating the indices and concrete
        # values for the boundary conditions. Can we do this in a more general
        # way?

    def periodic_boundary_conditions(self):
        # ...

    # This is a very general function - makes sense to reduce the unpacking
    # behaviour to the minimum

    # Note: onp.where is quite slow - do we really need to use it?
    # If we do, let's minimize the number of times we use it.

    def get_boundary_conditions_inds(self, location_fns):
        # ...

    # This is still pre-processing:
    def compute_Neumann_boundary_inds(self):
        # ....

    ## IMPORTANT: This is where we are actually processing
    # How this part will be set will largely depend on the handling of 'kernels'

    # External force (?)
    def compute_Neumann_integral_vars(self, **internal_vars):
        # ...

    # Processing - source term in the residual - two methods:
    def compute_body_force_by_fn(self):
        # ...

    def compute_body_force_by_sol(self, sol, mass_map):
        # ...

    # For each kernel, we have a function which returns a function that can
    # be applied cellwise to calculate the weak form terms.

    # The logic for kernels must be separated - kernels can be defined outside
    # of the FEM class (see kernels.py), and here, we can just call them on the
    #'tensor maps' etc. This would make it very similar to MOOSE

    def get_laplace_kernel(self, tensor_map):
        def laplace_kernel(
            cell_sol, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars
        ):
            # ...
            return val

        return laplace_kernel

    def get_mass_kernel(self, mass_map):
        def mass_kernel(cell_sol, cell_JxW, *cell_internal_vars):
            # ....
            return val

        return mass_kernel

    def get_cauchy_kernel(self, cauchy_map):
        def cauchy_kernel(cell_sol, face_shape_vals, face_nanson_scale):
            # ....
            return val

        return cauchy_kernel

    # This is used for topopt - e.g. theta would be an internal variable

    # Internal variables have terms corresponding to each one of the kernels
    # why is that? We could make it more general by having an enum for kernels
    # like 1: LAPLACE, 2: MASS, 3: CAUCHY, etc. The internal variables could
    # use those enums as keys and initialize with zeros/ None values.

    def unpack_kernels_vars(self, **internal_vars):
        # ...

    ## Important
    def split_and_compute_cell(self, cells_sol, np_version, jac_flag, **internal_vars):

        # There is a more general way to write the two functions below, like
        # we used in TO-JAX (you can ask Igor about it.)

        # These are utils
        def value_and_jacrev(f, x):
            # ...

        def value_and_jacfwd(f, x):
            # ...

        # This needs to be redesigned as well to make more modular

        # Assembles all the kernels for a single cell
        def get_kernel_fn_cell():

            # Note that the kernel is always a function of the same things
            # This means we can separate the kernel from the FEM class
            # definition
            def kernel(
                cell_sol,
                cell_shape_grads,
                cell_JxW,
                cell_v_grads_JxW,
                cell_mass_internal_vars,
                cell_laplace_internal_vars,
            ):
                if hasattr(self, "get_mass_map"):
                    mass_kernel = self.get_mass_kernel(self.get_mass_map())
                    mass_val = mass_kernel(cell_sol, cell_JxW, *cell_mass_internal_vars)
                else:
                    mass_val = 0.0

                if hasattr(self, "get_tensor_map"):
                    laplace_kernel = self.get_laplace_kernel(self.get_tensor_map())
                    laplace_val = laplace_kernel(
                        cell_sol,
                        cell_shape_grads,
                        cell_v_grads_JxW,
                        *cell_laplace_internal_vars,
                    )
                else:
                    laplace_val = 0.0

                return laplace_val + mass_val

            # Effectively, an elementwise stiffness matrix - should have
            # caching!!! dr/du is constant for each element
            def kernel_jac(cell_sol, *args):
                kernel_partial = lambda cell_sol: kernel(cell_sol, *args)
                return value_and_jacfwd(
                    kernel_partial, cell_sol
                )  # kernel(cell_sol, *args), jax.jacfwd(kernel)(cell_sol, *args)

            return kernel, kernel_jac

        kernel, kernel_jac = get_kernel_fn_cell()
        fn = kernel_jac if jac_flag else kernel
        vmap_fn = jax.jit(jax.vmap(fn))

        ## The naming and logic behind this code is questionable
        # What is the best way to process a batch in JAX?

        # Also, it would be nice if we had a chunked_vmap function that deals
        # with this and is defined elsewhere, similar to:
        # the solution described here:
        #  https://github.com/google/jax/issues/11319

        # Find terms corresponding to each kernel
        kernal_vars = self.unpack_kernels_vars(**internal_vars)
        num_cuts = 20
        if num_cuts > len(self.mesh.cells):
            num_cuts = len(self.mesh.cells)
        batch_size = len(self.mesh.cells) // num_cuts

        input_collection = [
            cells_sol,
            self.shape_grads,
            self.JxW,
            self.v_grads_JxW,
            *kernal_vars,
        ]


        if jac_flag:
            values = []
            jacs = []
            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree_map(
                        lambda x: x[i * batch_size : (i + 1) * batch_size],
                        input_collection,
                    )
                else:
                    input_col = jax.tree_map(
                        lambda x: x[i * batch_size :], input_collection
                    )

                val, jac = vmap_fn(*input_col)

                values.append(val)
                jacs.append(jac)

            # This we should probably get rid of fully - the overhead from
            # transferring is considerable.


            # np_version set to jax.numpy allows for auto diff, but uses GPU memory
            if np_version.__name__ == "jax.numpy":
                values = np_version.vstack(values)
                jacs = np_version.vstack(jacs)
            else:
                # np_version set to ordinary numpy saves GPU memory
                # values = jax_array_list_to_numpy_diff(values)
                # jacs = jax_array_list_to_numpy_diff(jacs)

                # Putting the large matrix on CPU - This allows
                # differentiation as well
                cpu = jax.devices("cpu")[0]
                values = np.vstack(jax.device_put(values, cpu))
                jacs = np.vstack(jax.device_put(jacs, cpu))

            return values, jacs
        else:
            values = []
            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree_map(
                        lambda x: x[i * batch_size : (i + 1) * batch_size],
                        input_collection,
                    )
                else:
                    input_col = jax.tree_map(
                        lambda x: x[i * batch_size :], input_collection
                    )

                val = vmap_fn(*input_col)
                values.append(val)
            values = np_version.vstack(values)
            return values

    # Not even Cauchy, these are Robin BCs
    # I find it weird that this logic is separated from other kernels, I think
    # it should not be the case, even though we integrate over the boundary
    # (not full domain)

    def compute_face(self, cells_sol, np_version, jac_flag):
        def get_kernel_fn_face(cauchy_map):
            def kernel(cell_sol, face_shape_vals, face_nanson_scale):
                cauchy_kernel = self.get_cauchy_kernel(cauchy_map)
                val = cauchy_kernel(cell_sol, face_shape_vals, face_nanson_scale)
                return val

            def kernel_jac(cell_sol, *args):
                return jax.jacfwd(kernel)(cell_sol, *args)

            return kernel, kernel_jac

        # TODO: Better to move the following to __init__ function?
        location_fns, value_fns = self.cauchy_bc_info
        boundary_inds_list = self.get_boundary_conditions_inds(location_fns)
        values = []
        selected_cells = []
        for i, boundary_inds in enumerate(boundary_inds_list):
            selected_cell_sols = cells_sol[
                boundary_inds[:, 0]
            ]  # (num_selected_faces, num_nodes, vec))
            selected_face_shape_vals = self.face_shape_vals[
                boundary_inds[:, 1]
            ]  # (num_selected_faces, num_face_quads, num_nodes)
            _, nanson_scale = self.get_face_shape_grads(
                boundary_inds
            )  # (num_selected_faces, num_face_quads)
            kernel, kernel_jac = get_kernel_fn_face(value_fns[i])
            fn = kernel_jac if jac_flag else kernel
            vmap_fn = jax.jit(jax.vmap(fn))
            val = vmap_fn(selected_cell_sols, selected_face_shape_vals, nanson_scale)
            values.append(val)
            selected_cells.append(self.mesh.cells[boundary_inds[:, 0]])

        values = np_version.vstack(values)
        selected_cells = onp.vstack(selected_cells)

        assert len(values) == len(selected_cells)

        return values, selected_cells

    # These seem to be helper utilities or even postprocessing - could they be
    # used to utils to
    # declutter the FEM module?

    def convert_from_dof_to_quad(self, sol):
        # ...

    def convert_neumann_from_dof(self, sol, index):
        # ...

    # Should this be here? Looks like post-processing in a way
    def sol_to_grad(self, sol):
        # ...

    # IT would be amazing to move all this to nonlinear solver file:

    def compute_residual_vars_helper(self, sol, weak_form, **internal_vars):
        # ...


    def compute_residual_vars(self, sol, **internal_vars):
        # ...


    def compute_newton_vars(self, sol, **internal_vars):
        # ...


    def compute_residual(self, sol):
        # ...

    ## THIs should be separated out into 'Nonlinear solvers'
    def newton_update(self, sol):
        # ...

    def set_params(self, params):
        # ...


    def print_BC_info(self):
        # ...
