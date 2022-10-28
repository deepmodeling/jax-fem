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

from jax_am.fem.generate_mesh import box_mesh
from jax_am.fem.jax_fem import Mesh, Laplace
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# dt = 2e-5
# T0 = 300.

dt = 1e-5
T0 = 1400.
Cp = 500.
L = 290e3
rho = 8440.
Ts = 1563
Tl = 1623

class Thermal(Laplace):
    def __init__(self, name, mesh, dt, dirichlet_bc_info=[[],[],[]], neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 1
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info) 
        self.mass_kernel_flag = True
        self.old_sol = None

    def get_tensor_map(self):
        def fn(u_grad):
            k = 10.
            return k*u_grad
        return fn 
 
    def get_mass_map(self):
        def T_map(T):
            """        
            Parameters
            ----------
            u: ndarray
                (vec,) 
            Returns
            -------
            val: ndarray
                (vec,) 
            """
            fl = np.where(T < Ts, 0., np.where(T > Tl, 1., (T - Ts)/(Tl - Ts))) 
            h = Cp*(T - T0) + L*fl
            return rho*h/dt
        return T_map

    def body_force_old_T(self, sol):
        mass_kernel = self.get_mass_kernel(self.get_mass_map())
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        val = jax.vmap(mass_kernel)(cells_sol, self.JxW) # (num_cells, num_nodes, vec)
        val = val.reshape(-1, self.vec) # (num_cells*num_nodes, vec)
        body_force = np.zeros_like(sol)
        body_force = body_force.at[self.cells.reshape(-1)].add(val) 
        return body_force 

    def compute_residual(self, sol):
        self.body_force = self.body_force_old_T(self.old_sol)
        return self.compute_residual_vars(sol)


def problem():
    ts = np.arange(0., 1e-3, dt)
    data_dir = os.path.join(os.path.dirname(__file__), 'data') 
    vtk_dir = os.path.join(data_dir, 'vtk')

    problem_name = f'thermal'
    Nx, Ny, Nz = 300, 50, 30
    Lx, Ly, Lz = 6e-3, 1e-3, 6e-4
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def neumann_val(point):
        alpha = 1e1
        return np.array([alpha * 1e2/(Lx*Ly)])

    neumann_bc_info = [[top], [neumann_val]]
    problem = Thermal(problem_name, mesh, dt, neumann_bc_info=neumann_bc_info)

    files = glob.glob(os.path.join(vtk_dir, f'{problem_name}/*'))
    for f in files:
        os.remove(f)

    problem.old_sol = T0*np.ones((problem.num_total_nodes, problem.vec))
    vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{0:05d}.vtu")
    save_sol(problem, problem.old_sol, vtk_path)

    for i in range(len(ts)):
        print(f"\nStep {i}, total step = {len(ts)}")
        problem.old_sol = solver(problem)
        vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{i:05d}.vtu")
        save_sol(problem, problem.old_sol, vtk_path)


if __name__ == "__main__":
    problem()

