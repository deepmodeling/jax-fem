import numpy as onp
import jax
import jax.numpy as np
import os

from jax_am.fem.generate_mesh import Mesh
from jax_am.fem.core import FEM
from jax_am.fem.basis import get_face_shape_vals_and_grads


class Thermal(FEM):
    def custom_init(self, old_sol, rho, Cp, dt, external_faces):
        self.old_sol = old_sol
        self.rho = rho
        self.Cp = Cp
        self.dt = dt
        self.external_faces = external_faces

    def get_tensor_map(self):
        def fn(u_grad):
            k = 15.
            return k*u_grad
        return fn
 
    def get_mass_map(self):
        def T_map(T):
            # fl = np.where(T < Ts, 0., np.where(T > Tl, 1., (T - Ts)/(Tl - Ts))) 
            # h = Cp*(T - T0) + L*fl
            return self.rho*self.Cp*T/self.dt
        return T_map

    def compute_residual(self, sol):
        self.body_force = self.compute_body_force_by_sol(self.old_sol, self.get_mass_map())
        self.neumann = self.compute_Neumann_integral_custom()
        return self.compute_residual_vars(sol)

    def compute_Neumann_integral_custom(self):
        self.neumann_boundary_inds_list = self.update_Neumann_boundary_inds()
        surface_old_T = self.get_surface_old_T(self.old_sol)
        return self.compute_Neumann_integral_vars(surface_old_T)

    def get_surface_old_T(self, sol):
        """TODO: Should we move this function to jax-am library?
        """
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        surface_old_T = []
        crt_t = []
        for i in range(len(self.neumann_value_fns)):
            boundary_inds = self.neumann_boundary_inds_list[i]
            selected_cell_sols = cells_sol[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, vec))
            selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)
            # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) -> (num_selected_faces, num_face_quads, vec) 
            u = np.sum(selected_cell_sols[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
            surface_old_T.append(u)
        return surface_old_T

    def update_Neumann_boundary_inds(self):
        cell_points = onp.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1) # (num_cells, num_faces, num_face_vertices, dim)
        external_cell_face_points = cell_face_points[self.external_faces[:, 0], self.external_faces[:, 1]] # (num_external_faces, num_face_vertices, dim)
      
        def top(face_points):
            face_points_z = face_points[:, 2]
            face_points_z = face_points_z - face_points_z[0]
            no_bottom = face_points[0, 2] > 0.
            return np.logical_and(np.all(np.isclose(face_points_z, 0., atol=1e-5)), no_bottom)

        def walls(face_points):
            return True

        boundary_inds_list = []
        for i in range(len(self.neumann_value_fns)):
            if i == 0:
                vmap_filter_fn = jax.vmap(walls)
            else:
                vmap_filter_fn = jax.vmap(top)
            boundary_flags = vmap_filter_fn(external_cell_face_points)
            inds_flags = onp.argwhere(boundary_flags).reshape(-1)
            boundary_inds = self.external_faces[inds_flags] # (num_selected_faces, 2)
            boundary_inds_list.append(boundary_inds)

        return boundary_inds_list


def hash_map_for_faces(active_cell_truth_tab, cells_face, hash_map, inner_faces, all_faces, cell_inds):
    """Use a hash table to store inner faces
    """
    for i, cell_id in enumerate(cell_inds):
        if active_cell_truth_tab[cell_id]:
            for face_id in range(len(cells_face[cell_id])):
                key = tuple(cells_face[cell_id, face_id].tolist())
                if key in hash_map.keys():
                    inner_faces.append(hash_map[key])
                    inner_faces.append((cell_id, face_id))
                else:
                    hash_map[key] = (cell_id, face_id)
                all_faces.append((cell_id, face_id))
    external_faces = onp.array(list((set(all_faces) - set(inner_faces))))
    return external_faces


def initialize_hash_map(full_mesh, active_cell_truth_tab, cells_map_full, ele_type):
    print(f"Initializing hash map for external faces...")
     # (num_faces, num_face_vertices)
    _, _, _, _, face_inds = get_face_shape_vals_and_grads(ele_type)
    cells_face = full_mesh.cells[:, face_inds] # (num_cells, num_faces, num_face_vertices)
    cells_face = onp.sort(cells_face)
    hash_map = {}
    inner_faces = []
    all_faces = []
    cell_inds = onp.arange(len(cells_face))
    external_faces = hash_map_for_faces(active_cell_truth_tab, cells_face, hash_map, inner_faces, all_faces, cell_inds)
    external_faces[:, 0] = cells_map_full[external_faces[:, 0]]
    return external_faces, cells_face, hash_map, inner_faces, all_faces


def update_hash_map(active_cell_truth_tab_old, active_cell_truth_tab_new, cells_map_full, cells_face, hash_map, inner_faces, all_faces):
    print(f"Updating hash map for external faces...")
    new_born_cell_inds = onp.argwhere(active_cell_truth_tab_old != active_cell_truth_tab_new).reshape(-1)
    external_faces = hash_map_for_faces(active_cell_truth_tab_new, cells_face, hash_map, inner_faces, all_faces, new_born_cell_inds)
    external_faces[:, 0] = cells_map_full[external_faces[:, 0]]
    return external_faces, hash_map, inner_faces, all_faces


def get_active_mesh(mesh, active_cell_truth_tab):
    active_cell_inds = onp.argwhere(active_cell_truth_tab).reshape(-1)
    cell_map_active = onp.sort(active_cell_inds)
    active_cells = mesh.cells[cell_map_active]
    cells_map_full = onp.zeros(len(mesh.cells), dtype=onp.int32)
    cells_map_full[cell_map_active] = onp.arange(len(cell_map_active))
    active_points_truth_tab = onp.zeros(len(mesh.points), dtype=bool)
    active_points_truth_tab[active_cells.reshape(-1)] = True
    points_map_active = onp.argwhere(active_points_truth_tab).reshape(-1)
    points_map_full = onp.zeros(len(mesh.points), dtype=onp.int32)
    points_map_full[points_map_active] = onp.arange(len(points_map_active))
    active_cells = points_map_full[active_cells]
    active_points = mesh.points[active_points_truth_tab]
    active_mesh = Mesh(active_points, active_cells)
    return active_mesh, points_map_active, cells_map_full
