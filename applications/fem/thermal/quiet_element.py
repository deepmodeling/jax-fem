import jax
import jax.numpy as np
import numpy as onp
import os

from jax_am.fem.jax_fem import Mesh, LinearElasticity
from jax_am.fem.solver import solver
from jax_am.fem.generate_mesh import box_mesh
from jax_am.fem.utils import save_sol
from jax_am.fem.basis import get_face_shape_vals_and_grads



# def get_external_face(mesh):
#     ele_type = 'hexahedron'
#     lag_order = 1
#     _, _, _, _, face_inds = get_face_shape_vals_and_grads(ele_type, lag_order)

#     print(f"Start")
#     cells_face = mesh.cells[:, face_inds]
#     cells_face_sorted = onp.sort(cells_face)
#     hash_map = {}
#     inner_faces = []
#     all_faces = []

#     for cell_id in range(len(cells_face)):
#         for face_id in range(len(cells_face[cell_id])):
#             key = tuple(cells_face[cell_id, face_id].tolist())
#             if key in hash_map.keys():
#                 inner_faces.append(hash_map[key])
#                 inner_faces.append((cell_id, face_id))
#             else:
#                 hash_map[key] = (cell_id, face_id)
#             all_faces.append((cell_id, face_id))

#     outer_faces = onp.array(list((set(all_faces) - set(inner_faces))))

#     print(f"outer_faces.shape = {outer_faces.shape}")

#     outer_points = mesh.points[cells_face[outer_faces[:, 0], outer_faces[:, 1]]]




def get_active_cells(mesh, active_cell_inds):
    active_cell_inds = onp.sort(active_cell_inds)
    active_cells = mesh.cells[active_cell_inds]
    active_points_truth_tab = onp.zeros(len(mesh.points), dtype=bool)
    active_points_truth_tab[active_cells.reshape(-1)] = True
    points_map_active_to_full = onp.argwhere(active_points_truth_tab).reshape(-1)
    points_map_full_to_active = onp.zeros(len(mesh.points), dtype=onp.int32)
    points_map_full_to_active[points_map_active_to_full] = onp.arange(len(points_map_active_to_full))
    active_cells = points_map_full_to_active[active_cells]
    active_points = mesh.points[active_points_truth_tab]
    return active_cells, active_points



def problem():
    """Can be used to test the memory limit of JAX-FEM
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    problem_name = f'linear_elasticity'
    meshio_mesh = box_mesh(4, 4, 4, 1., 1., 1., data_dir)
    # meshio_mesh = box_mesh(50, 50, 50, 1., 1., 1., data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])


    get_active_cells(mesh, onp.array([1, 2]))
    get_external_face(mesh)


if __name__ == "__main__":
    problem()