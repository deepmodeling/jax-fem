import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import meshio
import matplotlib.pyplot as plt

from jax_am.fem.solver import solver
from jax_am.fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_am.fem.utils import save_sol
from jax_am.phase_field.neper import pre_processing

from applications.fem.crystal_plasticity.models import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

case_name = 'polycrystal_neper'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, 'csv')
msh_dir = os.path.join(data_dir, 'msh')
neper_folder = os.path.join(data_dir, 'neper')


def problem():
    pf_args = {}
    pf_args['data_dir'] = data_dir
    pf_args['num_grains'] = 100
    pf_args['id'] = 0
    pf_args['domain_x'] = 0.1
    pf_args['domain_y'] = 0.1
    pf_args['domain_z'] = 0.1
    pf_args['num_oris'] = 10
    pre_processing(pf_args)

    ele_type = 'hexahedron'
    lag_order = 1
    cell_type = get_meshio_cell_type(ele_type, lag_order)
    meshio_mesh = meshio.read(os.path.join(neper_folder, f"domain.msh"))
 
    cell_grain_inds = meshio_mesh.cell_data['gmsh:physical'][0] - 1
    grain_oris_inds = onp.random.randint(pf_args['num_oris'], size=pf_args['num_grains'])
    cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)


    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    quat_file = os.path.join(csv_dir, f"quat.txt")
    quat = onp.loadtxt(quat_file)[:pf_args['num_oris'], 1:]

    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    Lz = np.max(mesh.points[:, 2])

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    disps = np.linspace(0., 0.01*Lx, 51)
    ts = np.linspace(0., 1., 51)

    # disps = np.linspace(0., 0.002*Lx, 11)
    # ts = np.linspace(0., 0.2, 11)

    def corner(point):
        flag_x = np.isclose(point[0], 0., atol=1e-3)
        flag_y = np.isclose(point[1], 0., atol=1e-3)
        flag_z = np.isclose(point[2], Lz, atol=1e-3)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

    def corner2(point):
        flag_x = np.isclose(point[0], 0., atol=1e-3)
        flag_y = np.isclose(point[1], 0., atol=1e-3)
        flag_z = np.isclose(point[2], 0., atol=1e-3)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
            def val_fn(point):
                return disp
            return val_fn

    def left(point):
        return np.isclose(point[0], 0., atol=1e-3)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-3)

    def front(point):
        return np.isclose(point[1], 0., atol=1e-3)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-3)


    dirichlet_bc_info = [[corner, corner, corner2, left, right], 
                         [1, 2, 1, 0, 0], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]


    # dirichlet_bc_info = [[left, front, bottom, right], 
    #                      [0, 1, 2, 0], 
    #                      [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]


    # dirichlet_bc_info = [[left, left, left, right, right, right], 
    #                      [2, 1, 0, 2, 1, 0], 
    #                      [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]

    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, lag_order=lag_order, 
                                dirichlet_bc_info=dirichlet_bc_info, additional_info=(quat, cell_ori_inds))

    results_to_save = []

    sol = np.zeros((problem.num_total_nodes, problem.vec))

    for i in range(len(ts) - 1):
        problem.dt = ts[i + 1] - ts[i]
        print(f"\nStep {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}, dt = {problem.dt}")

        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
        problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)

        # sol = solver(problem)
        sol = solver(problem, initial_guess=sol)

        print(f"Computing stress...")
        sigma_cell_data = problem.compute_avg_stress(sol)[:, 0, 0]
        print(f"Updating int vars...")
        F_p_zz, slip_resistance_0, slip_inc_dt_index_0 = problem.update_int_vars_gp(sol)
        print(f"stress = {sigma_cell_data[0]}, max stress = {np.max(sigma_cell_data)}")
        
        vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
        save_sol(problem, sol, vtk_path, cell_infos=[('cell_ori_inds', cell_ori_inds), ('sigma', sigma_cell_data)], cell_type=cell_type)


if __name__ == "__main__":
    problem()
