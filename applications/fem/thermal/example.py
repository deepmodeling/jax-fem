"""Test the latent heat nonlinearity for Shuheng.
Newton solver fails.
"""
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import os
import meshio
import time

from jax_am.fem.generate_mesh import box_mesh, Mesh
from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dt = 1e-4
T0 = 300.
Cp = 500.
L = 290e3
rho = 8440.
Ts = 1563
Tl = 1623
h = 50.

class Thermal(FEM):
    def get_tensor_map(self):
        def fn(u_grad):
            k = 10.
            return k*u_grad
        return fn
 
    def get_mass_map(self):
        def T_map(T):
            # fl = np.where(T < Ts, 0., np.where(T > Tl, 1., (T - Ts)/(Tl - Ts))) 
            # h = Cp*(T - T0) + L*fl
            return rho*Cp*T/dt
        return T_map

    def get_body_force_old_T(self, sol):
        mass_kernel = self.get_mass_kernel(self.get_mass_map())
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        val = jax.vmap(mass_kernel)(cells_sol, self.JxW) # (num_cells, num_nodes, vec)
        val = val.reshape(-1, self.vec) # (num_cells*num_nodes, vec)
        body_force = np.zeros_like(sol)
        body_force = body_force.at[self.cells.reshape(-1)].add(val) 
        return body_force 

    def compute_residual(self, sol):
        self.body_force = self.get_body_force_old_T(self.old_sol)
        self.neumann = self.compute_Neumann_integral()
        return self.compute_residual_vars(sol)

    def compute_Neumann_integral(self):
        """Child class should override if internal variables exist
        """
        if not hasattr(self, 'old_sol'):
            self.old_sol = T0*np.ones((self.num_total_nodes, self.vec))

        surface_old_T, crt_t = self.get_surface_old_T(self.old_sol)
        return self.compute_Neumann_integral_vars(surface_old_T, crt_t)

    def get_surface_old_T(self, sol):
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
            if not hasattr(self, 't'):
                self.t = 0.
            crt_t.append(self.t*np.ones((u.shape[0], u.shape[1]))) 
        return surface_old_T, crt_t

    def additional_neumann_filter(self, boundary_inds_list):
        cell_points = onp.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1) # (num_cells, num_faces, num_face_vertices, dim)
        filtered_boundary_inds_list = []

        def filter_fn(face_points):
            face_points_z = face_points[:, 2]
            face_points_z = face_points_z - face_points_z[0]
            return np.all(np.isclose(face_points_z, 0., atol=1e-5))

        for i in range(len(boundary_inds_list)):
            if i == 0:
                boundary_inds = boundary_inds_list[i]
                cell_face_points_crt = cell_face_points[boundary_inds[:, 0], boundary_inds[:, 1]]
                vmap_filter_fn = jax.vmap(filter_fn)
                boundary_flags = vmap_filter_fn(cell_face_points_crt)
                inds_flags = onp.argwhere(boundary_flags).reshape(-1)
                filtered_boundary_inds = boundary_inds[inds_flags] # (num_selected_faces, 2)
            else:
                filtered_boundary_inds = boundary_inds
            filtered_boundary_inds_list.append(filtered_boundary_inds)

        return filtered_boundary_inds_list


def problem():
    t_total = 1e-3
    vel = 0.5
    # ts = np.arange(0., t_total, dt)
    ts = np.arange(0., 10*dt, dt)
    data_dir = os.path.join(os.path.dirname(__file__), 'data') 
    vtk_dir = os.path.join(data_dir, 'vtk')

    problem_name = f'thermal'
    Nx, Ny, Nz = 300, 50, 30
    Lx, Ly, Lz = 6e-3, 1e-3, 6e-4
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def walls(point):
        return point[2] < Lz

    def neumann_top(point, old_T, t):
        # q is the heat flux into the domain
        laser_center = np.array([Lx*0.2 + vel*t, Ly/2., Lz])
        eta = 0.4
        P = 100.
        rb = 1e-3
        d2 = np.sum((point - laser_center)**2)
        q_laser = 2*eta*P/(np.pi*rb**2) * np.exp(-2*d2/rb**2)
        q_conv = h*(T0 - old_T[0])
        q = q_laser + q_conv
        return np.array([q])

    def neumann_walls(point, old_T, t):
        # q is the heat flux into the domain
        q_conv = h*(T0 - old_T[0])
        q = q_conv
        return np.array([q])

    neumann_bc_info = [[top, walls], [neumann_top, neumann_walls]]
    problem = Thermal(mesh, vec=1, dim=3, dirichlet_bc_info=[[],[],[]], neumann_bc_info=neumann_bc_info)

    files = glob.glob(os.path.join(vtk_dir, f'{problem_name}/*'))
    for f in files:
        os.remove(f)

    vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{0:05d}.vtu")
    save_sol(problem, problem.old_sol, vtk_path)

    for i in range(len(ts)):
        print(f"\nStep {i}, total step = {len(ts)}")
        problem.old_sol = solver(problem)
        problem.t = ts[i]
        vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{i:05d}.vtu")
        save_sol(problem, problem.old_sol, vtk_path)


if __name__ == "__main__":
    problem()

