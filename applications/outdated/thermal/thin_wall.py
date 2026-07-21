import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio
import time

from jax_fem.generate_mesh import Mesh
from jax_fem.core import FEM
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from applications.fem.thermal.models import Thermal, initialize_hash_map, update_hash_map, get_active_mesh

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_dir = os.path.join(os.path.dirname(__file__), 'data') 


def ded_thin_wall():
    T0 = 300.
    Cp = 500.
    L = 290e3
    rho = 8440.
    h = 50.
    rb = 1e-3
    eta = 0.4
    P = 500.
    base_plate_height = 20.*1e-3 # TODO: How do we get this information? 

    # path_resolution = 0.25*1e-3 # element x size = 0.5*1e-3
    path_resolution = 0.125*1e-3 # element x size = 0.5*1e-3

    vec = 1
    dim = 3
    ele_type = 'HEX8'
    problem_name = 'thin_wall'

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
    active_mesh, points_map_active, cells_map_full = get_active_mesh(full_mesh, active_cell_truth_tab)
    base_plate_mesh = meshio.Mesh(points=active_mesh.points, cells={'hexahedron': active_mesh.cells})
    base_plate_mesh.write(os.path.join(vtk_dir, f"base_plate_mesh.vtu"))
    thinwall_mesh = meshio.Mesh(points=full_mesh.points, cells={'hexahedron': full_mesh.cells})
    thinwall_mesh.write(os.path.join(vtk_dir, f"thinwall_mesh.vtu"))
    active_cell_truth_tab_old = active_cell_truth_tab

    external_faces, cells_face, hash_map, inner_faces, all_faces = initialize_hash_map(full_mesh, 
        active_cell_truth_tab, cells_map_full, ele_type)

    toolpath = onp.loadtxt(os.path.join(data_dir, f'toolpath/thinwall_toolpath.crs'))
    toolpath[:, 1:4] = toolpath[:, 1:4]/1e3

    def neumann_top(point, old_T):
        # q is the heat flux into the domain
        d2 = (point[0] - laser_center[0])**2 + (point[1] - laser_center[1])**2
        q_laser = 2*eta*P/(np.pi*rb**2) * np.exp(-2*d2/rb**2)
        q = q_laser
        return np.array([q])

    def neumann_walls(point, old_T):
        # q is the heat flux into the domain
        q_conv = h*(T0 - old_T[0])
        q = q_conv
        return np.array([q])

    neumann_bc_info_laser_on = [None, [neumann_walls, neumann_top]]
    neumann_bc_info_laser_off = [None, [neumann_walls]]

    full_sol = T0*np.ones((len(full_mesh.points), vec))  
    for i in range(1, toolpath.shape[0]):
        if toolpath[i, 4] == 0:
            if i == 1:
                num_laser_off = 2
            else:
                num_laser_off = 10
            t = onp.linspace(toolpath[i - 1, 0], toolpath[i, 0], num_laser_off + 1)
            dt = t[1] - t[0]
            sol = full_sol[points_map_active] 
            problem = Thermal(active_mesh, vec=vec, dim=dim, dirichlet_bc_info=[[],[],[]], neumann_bc_info=neumann_bc_info_laser_off, 
                              additional_info=(sol, rho, Cp, dt, external_faces))
            for j in range(num_laser_off):
                print(f"\n############################################################")
                print(f"Laser off: i = {i} in {toolpath.shape[0]} , j = {j} in {num_laser_off}")
                sol = solver(problem, linear=True)
                problem.update_int_vars(sol)
                full_sol = full_sol.at[points_map_active].set(sol)
                vtk_path = os.path.join(vtk_dir, f"u_active_{i:05d}_{j:05d}.vtu")
                save_sol(problem, sol, vtk_path)
        else:
            direction = toolpath[i, 1:4] - toolpath[i - 1 , 1:4]
            d = np.linalg.norm(direction)
            # dir_norm = direction/d
            num_laser_on = round(d/path_resolution)
            print(f"num_laser_on = {num_laser_on}")
            t = onp.linspace(toolpath[i - 1, 0], toolpath[i, 0], num_laser_on + 1)
            X = onp.interp(t, [toolpath[i - 1, 0], toolpath[i, 0]], [toolpath[i - 1, 1], toolpath[i, 1]])
            Y = onp.interp(t, [toolpath[i - 1, 0], toolpath[i, 0]], [toolpath[i - 1, 2], toolpath[i, 2]])

            for j in range(num_laser_on):
                print(f"\n############################################################")
                print(f"Laser on: i = {i} in {toolpath.shape[0]} , j = {j} in {num_laser_on}")
                laser_center = np.array([X[j], Y[j], toolpath[i,3] + base_plate_height])
                print(f"laser center = {laser_center}, dt = {t[j + 1] - t[j]}")
                flag_1 = centroids[:, 2] < laser_center[2]
                flag_2 = (centroids[:, 0] - laser_center[0])**2 + (centroids[:, 1] - laser_center[1])**2 <= rb**2
                active_cell_truth_tab = onp.logical_or(active_cell_truth_tab, onp.logical_and(flag_1, flag_2))
                active_mesh, points_map_active, cells_map_full = get_active_mesh(full_mesh, active_cell_truth_tab)
                sol = full_sol[points_map_active]
                dt = t[j + 1] - t[j]
                external_faces, hash_map, inner_faces, all_faces = update_hash_map(active_cell_truth_tab_old, 
                    active_cell_truth_tab, cells_map_full, cells_face, hash_map, inner_faces, all_faces)

                if onp.all(active_cell_truth_tab == active_cell_truth_tab_old):
                    print(f"No element born")
                    # problem.old_sol = old_sol
                else:
                    print(f"New elements born")
                    problem = Thermal(active_mesh, vec=vec, dim=dim, dirichlet_bc_info=[[],[],[]], neumann_bc_info=neumann_bc_info_laser_on, 
                                      additional_info=(sol, rho, Cp, dt, external_faces))
                sol = solver(problem, linear=True)
                problem.update_int_vars(sol)
                full_sol = full_sol.at[points_map_active].set(sol)
                if j % 10 == 0:
                    vtk_path = os.path.join(vtk_dir, f"u_active_{i:05d}_{j:05d}.vtu")
                    save_sol(problem, sol, vtk_path)

                active_cell_truth_tab_old = active_cell_truth_tab

                # if j > 10:
                #     exit()


if __name__ == "__main__":
    ded_thin_wall()
