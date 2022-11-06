import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio

from jax_am.fem.generate_mesh import box_mesh, Mesh
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol

from applications.fem.thermal.models import Thermal

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


if __name__ == "__main__":
    bare_plate_single_track()
