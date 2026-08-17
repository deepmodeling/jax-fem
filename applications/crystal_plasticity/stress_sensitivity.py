import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt

from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol

from applications.crystal_plasticity.models import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

case_name = 'stress_sensitivity'

base_dir = os.path.dirname(__file__)
input_dir = os.path.join(base_dir, 'input', case_name)
output_dir = os.path.join(base_dir, 'output', case_name)
numpy_dir = os.path.join(output_dir, 'numpy')
vtk_dir = os.path.join(output_dir, 'vtk')
figure_dir = os.path.join(output_dir, 'figures')


def problem():
    ele_type = 'HEX8'
    Nx, Ny, Nz = 1, 1, 1
    Lx, Ly, Lz = 1., 1., 1.

    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, output_dir, ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    files = glob.glob(os.path.join(vtk_dir, '*.vtu'))
    for f in files:
        os.remove(f)

    num_steps = 10
    disps = np.linspace(0., 0.005, num_steps + 1)
    ts = np.linspace(0., 0.5, num_steps + 1)

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
    initial_params = problem.internal_vars
    fwd_pred = ad_wrapper(problem)

    def simulation(alpha, save_results=False):
        params = list(initial_params)
        params[1] = alpha*initial_params[1]

        results_to_save = []
        for i in range(num_steps):
            problem.dt = ts[i + 1] - ts[i]
            print(f"\nStep {i + 1} in {num_steps}, disp = {disps[i + 1]}, dt = {problem.dt}")
            dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
            problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)

            sol_list = fwd_pred(params)
            sol = sol_list[0]

            stress_zz = problem.compute_avg_stress(sol, params)[0, 2, 2]

            params = problem.update_int_vars_gp(sol, params)

            if save_results:
                Fp_inv_gp, slip_resistance_gp, slip_gp, _ = params
                F_p_zz = np.linalg.inv(Fp_inv_gp[0, 0])[2, 2]
                slip_resistance_0 = slip_resistance_gp[0, 0, 0]
                slip_0 = slip_gp[0, 0, 0]
                engineering_strain = disps[i + 1]/Lz
                green_strain = 0.5*((1. + engineering_strain)**2 - 1.)

                vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
                save_sol(problem.fes[0], sol, vtk_path)
                results_to_save.append([
                    ts[i + 1], green_strain, F_p_zz,
                    slip_resistance_0, slip_0, stress_zz
                ])

        if save_results:
            os.makedirs(numpy_dir, exist_ok=True)
            results_to_save = onp.asarray(results_to_save)
            onp.save(os.path.join(numpy_dir, 'jax_fem_out.npy'), results_to_save)

        return stress_zz

    final_stress = simulation(1., save_results=True)
    print(f"Final stress_zz = {final_stress}")
    plot_stress_strain()

    # 1.01 - 169.65845175597966
    # 0.99 - 166.3895137772737
    # FDM grad - 163.44689893529818
    # AD grad - 163.44798021546643

    grads = jax.grad(simulation)(1.)
    print(f"d stress_zz / d alpha = {grads}")


def plot_stress_strain():
    moose_out = onp.loadtxt(os.path.join(input_dir, 'moose_reference.csv'), delimiter=',')
    jax_fem_out = onp.load(os.path.join(numpy_dir, 'jax_fem_out.npy'))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(moose_out[:, 1], moose_out[:, -1], label='MOOSE',
            color='blue', linewidth=2)
    ax.plot(jax_fem_out[:, 1], jax_fem_out[:, -1], label='JAX-FEM',
            color='red', marker='o', markersize=7, linestyle='None')
    ax.set_xlabel('Green-Lagrange strain', fontsize=16)
    ax.set_ylabel('Cauchy stress [MPa]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()

    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, 'stress_strain_comparison.png')
    fig.savefig(figure_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    max_error = onp.max(onp.abs(jax_fem_out[:, -1] - moose_out[:, -1]))
    print(f"Saved stress-strain comparison to {figure_path}")
    print(f"Maximum absolute stress difference from MOOSE = {max_error} MPa")


if __name__ == "__main__":
    problem()
