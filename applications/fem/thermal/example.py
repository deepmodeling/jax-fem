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

data_dir = os.path.join(os.path.dirname(__file__), 'data') 


class Thermal(FEM):
    def custom_init(self, old_sol, rho, Cp, dt):
        self.old_sol = old_sol
        self.rho = rho
        self.Cp = Cp
        self.dt = dt

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
        return self.compute_residual_vars(sol)

    def compute_Neumann_integral(self):
        """Overriding base class method
        """
        surface_old_T = self.get_surface_old_T(self.old_sol)
        return self.compute_Neumann_integral_vars(surface_old_T)

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
        return surface_old_T

    def additional_neumann_filter(self, boundary_inds_list):
        cell_points = onp.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1) # (num_cells, num_faces, num_face_vertices, dim)
        filtered_boundary_inds_list = []

        def filter_fn(face_points):
            face_points_z = face_points[:, 2]
            face_points_z = face_points_z - face_points_z[0]
            return np.all(np.isclose(face_points_z, 0., atol=1e-5))

        for i in range(len(boundary_inds_list)):
            boundary_inds = boundary_inds_list[i]
            cell_face_points_crt = cell_face_points[boundary_inds[:, 0], boundary_inds[:, 1]]
            vmap_filter_fn = jax.vmap(filter_fn)
            boundary_flags = vmap_filter_fn(cell_face_points_crt)
            inds_flags = onp.argwhere(boundary_flags).reshape(-1)
            if i == 0:
                filtered_boundary_inds = boundary_inds[inds_flags] # (num_selected_faces, 2)
            else:
                filtered_boundary_inds = boundary_inds[~inds_flags]
            filtered_boundary_inds_list.append(filtered_boundary_inds)

        return filtered_boundary_inds_list


def bare_plate_single_track():
    t_total = 5.
    vel = 0.01
    dt = 1e-2
    T0 = 300.
    Cp = 500.
    L = 290e3
    rho = 8440.
    Ts = 1563
    Tl = 1623
    h = 50.
    rb = 1e-3
    eta = 0.4
    P = 500.

    ts = np.arange(0., 10e5, dt)
    # ts = np.arange(0., 10*dt, dt)
    data_dir = os.path.join(os.path.dirname(__file__), 'data') 
    vtk_dir = os.path.join(data_dir, 'vtk')

    problem_name = f'bare_plate'
    Nx, Ny, Nz = 150, 30, 10
    Lx, Ly, Lz = 30e-3, 6e-3, 2e-3
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def top(point):
        return point[2] > 0.

    def walls(point):
        return True

    def neumann_top(point, old_T):
        # q is the heat flux into the domain
        d2 = np.sum((point - laser_center)**2)
        q_laser = 2*eta*P/(np.pi*rb**2) * np.exp(-2*d2/rb**2)
        q_conv = h*(T0 - old_T[0])
        q = q_laser + q_conv
        return np.array([q])

    def neumann_walls(point, old_T):
        # q is the heat flux into the domain
        q_conv = h*(T0 - old_T[0])
        q = q_conv
        return np.array([q])

    neumann_bc_info = [[top, walls], [neumann_top, neumann_walls]]

    vec = 1
    dim = 3
    old_sol = T0*np.ones((len(mesh.points), vec))

    problem = Thermal(mesh, vec=vec, dim=dim, dirichlet_bc_info=[[],[],[]], neumann_bc_info=neumann_bc_info, 
                      additional_info=(old_sol, rho, Cp, dt))

    files = glob.glob(os.path.join(vtk_dir, f'{problem_name}/*'))
    for f in files:
        os.remove(f)

    vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{0:05d}.vtu")
    save_sol(problem, problem.old_sol, vtk_path)

    for i in range(len(ts[1:])):
        print(f"\nStep {i + 1}, total step = {len(ts)}, laser_x = {Lx*0.2 + vel*ts[i + 1]}")
        laser_center = np.array([Lx*0.2 + vel*ts[i + 1], Ly/2., Lz])
        problem.old_sol = solver(problem)
        if (i + 1) % 10 == 0:
            vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{i + 1:05d}.vtu")
            save_sol(problem, problem.old_sol, vtk_path)

        if Lx*0.2 + vel*ts[i + 1] > Lx*0.4:
            break


def get_active_cells(mesh, active_cell_inds):
    active_cell_inds = onp.sort(active_cell_inds)
    active_cells = mesh.cells[active_cell_inds]
    active_points_truth_tab = onp.zeros(len(mesh.points), dtype=bool)
    active_points_truth_tab[active_cells.reshape(-1)] = True
    map_active_to_full = onp.argwhere(active_points_truth_tab).reshape(-1)
    map_full_to_active = onp.zeros(len(mesh.points), dtype=onp.int32)
    map_full_to_active[map_active_to_full] = onp.arange(len(map_active_to_full))
    active_cells = map_full_to_active[active_cells]
    active_points = mesh.points[active_points_truth_tab]
    return active_cells, active_points, map_active_to_full


def direct_energy_deposition():
    T0 = 300.
    Cp = 500.
    L = 290e3
    rho = 8440.
    h = 50.
    rb = 1e-3
    eta = 0.4
    P = 500.
    base_plate_height = 20.*1e-3 # TODO: How do we get this information? 
    path_resolution = 0.25*1e-3 # element x size = 0.5*1e-3
    vec = 1
    dim = 3

    problem_name = f'thinwall'
    vtk_dir = os.path.join(data_dir, f'vtk/{problem_name}')
    os.makedirs(vtk_dir, exist_ok=True)
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    abaqus_root = os.path.join(data_dir, f'abaqus')
    abaqus_file = os.path.join(abaqus_root, f'thinwall.inp')
    meshio_mesh = meshio.read(abaqus_file)
    full_mesh = Mesh(meshio_mesh.points/1e3, meshio_mesh.cells_dict['hexahedron'])
    active_cell_truth_tab = onp.zeros(len(full_mesh.cells), dtype=bool)
    centroids = onp.mean(full_mesh.points[full_mesh.cells], axis=1)
    active_cell_truth_tab[centroids[:, 2] <= base_plate_height] = True
    active_cell_inds = onp.argwhere(active_cell_truth_tab).reshape(-1)
    active_cells, active_points, _ = get_active_cells(full_mesh, active_cell_inds)
    base_plate_mesh = meshio.Mesh(points=active_points, cells={'hexahedron': active_cells})
    base_plate_mesh.write(os.path.join(vtk_dir, f"base_plate_mesh.vtu"))
    thinwall_mesh = meshio.Mesh(points=full_mesh.points, cells={'hexahedron': full_mesh.cells})
    thinwall_mesh.write(os.path.join(vtk_dir, f"thinwall_mesh.vtu"))

    toolpath = onp.loadtxt(os.path.join(data_dir, f'toolpath/thinwall_toolpath.crs'))
    toolpath[:, 1:4] = toolpath[:, 1:4]/1e3

    def top(point):
        return point[2] > 0.

    def walls(point):
        return True

    def neumann_top(point, old_T):
        # q is the heat flux into the domain
        d2 = (point[0] - laser_center[0])**2 + (point[1] - laser_center[1])**2
        q_laser = 2*eta*P/(np.pi*rb**2) * np.exp(-2*d2/rb**2)
        q_conv = h*(T0 - old_T[0])
        q = q_laser + q_conv
        return np.array([q])

    def neumann_walls(point, old_T):
        # q is the heat flux into the domain
        q_conv = h*(T0 - old_T[0])
        q = q_conv
        return np.array([q])

    for i in range(1,toolpath.shape[0]):
        if toolpath[i,4] == 0:
            continue
        direction = toolpath[i,1:4]-toolpath[i-1,1:4]
        d = np.linalg.norm(direction)
        dir_norm = direction/d
        num = round(d/path_resolution)
        print(f"partion num = {num}")
        t = onp.linspace(toolpath[i-1,0],toolpath[i,0],num+1)
        X = onp.interp(t,[toolpath[i-1,0],toolpath[i,0]],[toolpath[i-1,1],toolpath[i,1]])
        Y = onp.interp(t,[toolpath[i-1,0],toolpath[i,0]],[toolpath[i-1,2],toolpath[i,2]])

        full_sol = T0*np.ones((len(full_mesh.points), vec))

        for j in range (0, num):
            laser_center = np.array([X[j], Y[j], toolpath[i,3] + base_plate_height])
            print(f"laser center = {laser_center}, dt = {t[j + 1] - t[j]}")
            flag_1 = centroids[:, 2] < laser_center[2]
            flag_2 = (centroids[:, 0] - laser_center[0])**2 + (centroids[:, 1] - laser_center[1])**2 <= rb**2
            active_cell_truth_tab = onp.logical_or(active_cell_truth_tab, onp.logical_and(flag_1, flag_2))
            active_cell_inds = onp.argwhere(active_cell_truth_tab).reshape(-1)
            print(len(active_cell_inds))
            active_cells, active_points, map_active_to_full = get_active_cells(full_mesh, active_cell_inds)
            active_mesh = Mesh(active_points, active_cells)
            old_sol = full_sol[map_active_to_full]
            neumann_bc_info = [[top, walls], [neumann_top, neumann_walls]]
            dt = t[j + 1] - t[j]
            problem = Thermal(active_mesh, vec=vec, dim=dim, dirichlet_bc_info=[[],[],[]], neumann_bc_info=neumann_bc_info, 
                              additional_info=(old_sol, rho, Cp, dt))
            sol = solver(problem)
            full_sol = full_sol.at[map_active_to_full].set(sol)
            vtk_path = os.path.join(vtk_dir, f"u_active_{j:05d}.vtu")
            save_sol(problem, sol, vtk_path)

            if j > 10:
                exit()


if __name__ == "__main__":
    # bare_plate_single_track()
    direct_energy_deposition()
