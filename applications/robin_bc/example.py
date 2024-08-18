import jax
import jax.numpy as np
import numpy as onp
import os
import meshio
import gmsh

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol, modify_vtu_file
from jax_fem.basis import get_elements

from applications.stokes.example import transform_cells


class Poisson(Problem):

    def get_universal_kernel(self):

        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_vars, num_quads, num_nodes, dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_vars, num_quads, num_nodes, 1, dim)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, :self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_JxW = cell_JxW[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, :self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec

            dx = x[:, 0] - 0.5
            dy = x[:, 1] - 0.5
            body_val = x[:, 0]*np.sin(5.0*np.pi*x[:, 1]) + 1.0*np.exp(-(dx*dx + dy*dy)/0.02)
            body_val = body_val[:, None] # (num_quads, vec)

            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
            val = np.sum(u_grads[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))

            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            body_val = np.sum(body_val[:, None, :] * self.fes[0].shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)

            val = val - body_val

            return jax.flatten_util.ravel_pytree(val)[0]

        return universal_kernel

    # Alternative way:
    # Either define get_surface_maps or get_universal_kernels_surface

    # def get_surface_maps(self):
    #     def surface_map(u, x):
    #         return 5*u**2
    #     return [surface_map, surface_map]

    def get_universal_kernels_surface(self):
        def robin_map(u):
            return 5*u**2

        def robin_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_face_quads, dim)
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # face_nanson_scale: (num_vars, num_face_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            face_shape_vals = face_shape_vals[:, :self.fes[0].num_nodes]
            face_nanson_scale = face_nanson_scale[0]

            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(robin_map)(u)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)

            return jax.flatten_util.ravel_pytree(val)[0]

        return [robin_kernel, robin_kernel]


def problem():
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    vec = 1
    dim = 2
    ele_type = 'TRI3'
    points = onp.load(os.path.join(input_dir, f'numpy/points_u.npy'))
    cells = onp.load(os.path.join(input_dir, f'numpy/cells_u.npy'))
    cells = transform_cells(cells, points, ele_type)
    mesh = Mesh(points, cells)

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], 1., atol=1e-5)

    def dirichlet_val(point):
        return 1.

    dirichlet_bc_info = [[bottom, top], 
                         [0]*2, 
                         [dirichlet_val, dirichlet_val]]

    location_fns = [left, right]

    # gauss_order=2 produces the same result with FEniCS
    problem = Poisson(mesh, vec, dim, ele_type, gauss_order=2, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

    sol_list = solver(problem)

    vtk_file = os.path.join(output_dir, f"vtk/u_jax-fem.vtu")
    save_sol(problem.fes[0], sol_list[0], vtk_file)


if __name__ == "__main__":
    problem()
