"""Reference
Simo, Juan C., and Thomas JR Hughes. Computational inelasticity. Vol. 7. Springer Science & Business Media, 2006.
Chapter 9: Phenomenological Plasticity Models
"""
import jax
import jax.numpy as np
import jax.flatten_util
import os
import glob
import matplotlib.pyplot as plt

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh

from applications.forming.model import Plasticity

def simulation():

    class SingleCell(Plasticity):
        def set_params(self, params):
            int_vars, scale = params
            self.internal_vars = int_vars
            self.fe.dirichlet_bc_info[-1][-1] = get_dirichlet_top(scale)
            self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)

    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    vtk_dir = os.path.join(data_dir, 'vtk')

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    Lx, Ly, Lz = 1., 1., 1.
    meshio_mesh = box_mesh_gmsh(Nx=1, Ny=1, Nz=1, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def get_dirichlet_top(scale):
        def val_fn(point):
            z_disp = scale*Lz
            return z_disp
        return val_fn

    def dirichlet_val_bottom(point):
        return 0.

    scales = 0.01*np.hstack((np.linspace(0., 1., 11), np.linspace(1, 0., 11)))

    location_fns = [bottom, top]
    vecs = [2, 2]
    value_fns = [dirichlet_val_bottom, get_dirichlet_top(0.)]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    problem = SingleCell(mesh, ele_type=ele_type, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)

    sol_list = [np.zeros(((problem.fe.num_total_nodes, problem.fe.vec)))]
 
    int_vars = problem.internal_vars

    for i, scale in enumerate(scales):
        print(f"\nStep {i} in {len(scales)}, scale = {scale}")
        problem.set_params([int_vars, scale])
        sol_list = solver(problem, solver_options={'initial_guess': sol_list})   
        int_vars_copy = int_vars
        int_vars = problem.update_int_vars_gp(sol_list[0], int_vars)
        sigmas = problem.compute_stress(sol_list[0], int_vars_copy).mean(axis=1)
        print(f"max alpha = \n{np.max(int_vars[-1])}")
        print(sigmas[0])
        vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
        save_sol(problem.fe, sol_list[0], vtk_path, cell_infos=[('s_norm', np.linalg.norm(sigmas, axis=(1, 2)))])


if __name__=="__main__":
    simulation()
