import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import meshio
import matplotlib.pyplot as plt

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol


from applications.crystal_plasticity.models import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

case_name = 'polycrystal_neper'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')
neper_folder = os.path.join(data_dir, f'neper/{case_name}')



def pre_processing(pf_args, neper_path='neper'):
    """We use Neper to generate polycrystal structure.
    Neper has two major functions: generate a polycrystal structure, and mesh it.
    See https://neper.info/ for more information.
    """
    neper_path = os.path.join(pf_args['data_dir'], neper_path)
    os.makedirs(neper_path, exist_ok=True)

    if not os.path.exists(os.path.join(neper_path, 'domain.msh')):
        print(f"You don't have neper mesh file ready, try generating them...")
        os.system(f'''neper -T -n {pf_args['num_grains']} -id {pf_args['id']} -regularization 0 -domain "cube({pf_args['domain_x']},\
                   {pf_args['domain_y']},{pf_args['domain_z']})" \
                    -o {neper_path}/domain -format tess,obj,ori''')
        os.system(f"neper -T -loadtess {neper_path}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area")
        os.system(f"neper -M -rcl 1 -elttype hex -faset faces {neper_path}/domain.tess")
    else:
        print(f"You already have neper mesh file.")



def problem():
    pf_args = {}
    pf_args['data_dir'] = data_dir
    pf_args['num_grains'] = 100
    pf_args['id'] = 0
    pf_args['domain_x'] = 0.1
    pf_args['domain_y'] = 0.1
    pf_args['domain_z'] = 0.1
    pf_args['num_oris'] = 10
    pre_processing(pf_args, neper_path=f'neper/{case_name}')

    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
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

    # disps = np.linspace(0., 0.01*Lx, 51)
    # ts = np.linspace(0., 1., 51)

    disps = np.linspace(0., 0.002*Lx, 11)
    ts = np.linspace(0., 0.2, 11)

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

    def back(point):
        return np.isclose(point[1], Ly, atol=1e-3)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-3)

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-3)


    dirichlet_bc_info = [[corner, corner, corner2, left, right], 
                         [1, 2, 1, 0, 0], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]


    # dirichlet_bc_info = [[left, front, back, bottom, top, right], 
    #                      [0, 1, 1, 2, 2, 0], 
    #                      [zero_dirichlet_val]*5 + [get_dirichlet_top(disps[0])]]


    # dirichlet_bc_info = [[left, left, left, right, right, right], 
    #                      [2, 1, 0, 2, 1, 0], 
    #                      [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]

    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, 
                                dirichlet_bc_info=dirichlet_bc_info, additional_info=(quat, cell_ori_inds))

    results_to_save = []
    sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
    params = problem.internal_vars

    for i in range(len(ts) - 1):
        problem.dt = ts[i + 1] - ts[i]
        print(f"\nStep {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}, dt = {problem.dt}")

        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
        problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        problem.set_params(params)

        sol_list = solver(problem, solver_options={'petsc_solver':{}, 'initial_guess': sol_list})   

        print(f"Computing stress...")
        sigma_cell_data = problem.compute_avg_stress(sol_list[0], params)[:, 0, 0]

        print(f"Updating int vars...")
        params = problem.update_int_vars_gp(sol_list[0], params)

        F_p_zz, slip_resistance_0, slip_0 = problem.inspect_interval_vars(params)
        print(f"stress = {sigma_cell_data[0]}, max stress = {np.max(sigma_cell_data)}")
        vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
        save_sol(problem.fes[0], sol_list[0], vtk_path, cell_infos=[('cell_ori_inds', cell_ori_inds), ('sigma', sigma_cell_data)])


if __name__ == "__main__":
    problem()
