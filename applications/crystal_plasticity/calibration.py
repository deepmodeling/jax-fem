import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt

from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol

from applications.crystal_plasticity.models import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

case_name = 'calibration'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')


def problem():
    ele_type = 'HEX8'
    Nx, Ny, Nz = 1, 1, 1
    Lx, Ly, Lz = 1., 1., 1.

    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir, ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    disps = np.linspace(0., 0.005, 11)
    ts = np.linspace(0., 0.5, 11)

    def corner(point):
        flag_x = np.isclose(point[0], 0., atol=1e-5)
        flag_y = np.isclose(point[1], 0., atol=1e-5)
        flag_z = np.isclose(point[2], 0., atol=1e-5)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
            def val_fn(point):
                return disp
            return val_fn

    dirichlet_bc_info = [[corner, corner, bottom, top], 
                         [0, 1, 2, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]

    quat = onp.array([[1, 0., 0., 0.]])
    cell_ori_inds = onp.zeros(len(mesh.cells), dtype=onp.int32)
    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                                additional_info=(quat, cell_ori_inds))
    

    # sol = np.zeros((problem.num_total_nodes, problem.vec))

    fwd_pred = ad_wrapper(problem)

    def simulation(alpha):

        params = problem.internal_vars
        params[1] = alpha*params[1]

        results_to_save = []
        for i in range(10):
            problem.dt = ts[i + 1] - ts[i]
            print(f"\nStep {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}, dt = {problem.dt}")
            dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
            problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)

            sol_list = fwd_pred(params)
            sol = sol_list[0]

            stress_zz = problem.compute_avg_stress(sol, params)[0, 2, 2]

            params = problem.update_int_vars_gp(sol, params)

            # F_p_zz, slip_resistance_0, slip_0 = problem.inspect_interval_vars(params)
            # print(f"stress_zz = {stress_zz}")
            # vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
            # save_sol(problem, sol, vtk_path)
            # results_to_save.append([ts[i + 1], disps[i + 1]/Lz, F_p_zz, slip_resistance_0, slip_0, stress_zz])

        # results_to_save = onp.array(results_to_save)
        # os.makedirs(numpy_dir, exist_ok=True)
        # onp.save(os.path.join(numpy_dir, 'jax_fem_out.npy'), results_to_save)

        return stress_zz

    # 1.01 - 169.65845175597966
    # 0.99 - 166.3895137772737
    # FDM grad - 163.44689893529818
    # AD grad - 163.44798021546643

    grads = jax.grad(simulation)(1.)
    print(grads)

    # print(simulation(1.))


def plot_stress_strain():
    plt.rcParams.update({
        "text.latex.preamble": r"\usepackage{amsmath}",
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})


    # slip_inc_dt_index_0 = (slip_new_gp[0, 0, 0] - self.slip_old_gp[0, 0, 0])/self.dt
    # print(f"slip inc dt index 0 = {slip_inc_dt_index_0}, max slip = {np.max(np.absolute(slip_new_gp))}")

    # time, e_zz, fp_zz, gss, pk2, slip_increment_dt, stress_zz
    moose_out = onp.loadtxt(os.path.join(csv_dir, 'update_method_test_out.csv'), delimiter=',')

    # time, strain, fp_zz, gss, slip, stress_zz
    jax_fem_out = onp.load(os.path.join(numpy_dir, 'jax_fem_out.npy'))

    fig = plt.figure(figsize=(8, 6))
    plt.plot(jax_fem_out[:, 1], moose_out[:, -1], label='MOOSE', color='blue', linestyle="-", linewidth=2)
    plt.plot(jax_fem_out[:, 1], jax_fem_out[:, -1], label='JAX-FEM', color='red', marker='o', markersize=8, linestyle='None') 
    plt.xlabel(r'Strain', fontsize=20)
    plt.ylabel(r'Stress [MPa]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=20, frameon=False)

    print((jax_fem_out[1:, -2] - jax_fem_out[:-1, -2])/jax_fem_out[0, 0])


if __name__ == "__main__":
    problem()
    # plot_stress_strain()
    # plt.show()
