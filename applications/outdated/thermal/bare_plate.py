import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio

from jax_fem.generate_mesh import box_mesh_gmsh, Mesh
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from applications.fem.thermal.models import Thermal, initialize_hash_map, update_hash_map, get_active_mesh

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
data_dir = os.path.join(os.path.dirname(__file__), 'data') 


def bare_plate_single_track():
    t_total = 5.
    vel = 0.01
    dt = 1e-2
    T0 = 300.
    Cp = 500.
    L = 290e3
    rho = 8440.
    h = 50.
    rb = 1e-3
    eta = 0.4
    P = 500.
    vec = 1
    dim = 3
    ele_type = 'HEX8'

    ts = np.arange(0., 10e5, dt)
    # ts = np.arange(0., 10*dt, dt)

    vtk_dir = os.path.join(data_dir, 'vtk')

    problem_name = f'bare_plate'
    Nx, Ny, Nz = 150, 30, 10
    Lx, Ly, Lz = 30e-3, 6e-3, 2e-3
    meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
    full_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def top(point):
        return point[2] > 0.

    def walls(point):
        return True

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

    neumann_bc_info = [None, [neumann_top, neumann_walls]]

    active_cell_truth_tab = onp.ones(len(full_mesh.cells), dtype=bool)
    active_mesh, points_map_active, cells_map_full = get_active_mesh(full_mesh, active_cell_truth_tab)
    external_faces, cells_face, hash_map, inner_faces, all_faces = initialize_hash_map(full_mesh, 
        active_cell_truth_tab, cells_map_full, ele_type)
    sol = T0*np.ones((len(active_mesh.points), vec))

    problem = Thermal(active_mesh, vec=vec, dim=dim, neumann_bc_info=neumann_bc_info, 
                      additional_info=(sol, rho, Cp, dt, external_faces))

    files = glob.glob(os.path.join(vtk_dir, f'{problem_name}/*'))
    for f in files:
        os.remove(f)

    vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{0:05d}.vtu")
    save_sol(problem, sol, vtk_path)

    for i in range(len(ts[1:])):
        print(f"\nStep {i + 1}, total step = {len(ts)}, laser_x = {Lx*0.2 + vel*ts[i + 1]}")
        laser_center = np.array([Lx*0.2 + vel*ts[i + 1], Ly/2., Lz])
        sol = solver(problem)
        problem.update_int_vars(sol)
        if (i + 1) % 10 == 0:
            vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{i + 1:05d}.vtu")
            save_sol(problem, sol, vtk_path)

        if Lx*0.2 + vel*ts[i + 1] > Lx*0.4:
            break


if __name__ == "__main__":
    bare_plate_single_track()
