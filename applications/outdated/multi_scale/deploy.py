import numpy as onp
import jax
import jax.numpy as np
import os 
import matplotlib.pyplot as plt
import time

from jax_fem.generate_mesh import Mesh, box_mesh_gmsh
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from applications.fem.multi_scale.arguments import args
from applications.fem.multi_scale.utils import tensor_to_flat, tensor_to_flat
from applications.fem.multi_scale.trainer import H_to_C, get_nn_batch_forward
from applications.fem.multi_scale.fem_model import HyperElasticity

args.device = 1

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)


def debug():
    H_bar = np.array([[-0.009, 0., 0.],
                      [0., -0.009, 0.],
                      [0., 0., 0.025]])

    H_flat = tensor_to_flat(H_bar)
    C_flat, _ = H_to_C(H_flat)
    hyperparam = 'MLP2'
    nn_batch_forward = get_nn_batch_forward(hyperparam)
    energy = nn_batch_forward(C_flat[None, :])
    print(energy)


def walltime(func):
    def wrapper(*list_args, **keyword_args):
        start_time = time.time()
        return_values = func(*list_args, **keyword_args)
        end_time = time.time()
        time_elapsed = end_time - start_time
        platform = jax.lib.xla_bridge.get_backend().platform
        print(f"Time elapsed {time_elapsed} of function {func.__name__} on platform {platform}")
        return return_values
    return wrapper
    

@walltime
def homogenization_problem(case, dns_info=None):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    problem_name = case if dns_info is None else case + "_" + dns_info
    args.num_units_x = 10
    args.num_units_y = 2
    args.num_units_z = 10
    L = args.L
    if case == 'nn':
        num_hex = 5
    else:
        num_hex = args.num_hex

    meshio_mesh = box_mesh_gmsh(num_hex*args.num_units_x, num_hex*args.num_units_y, num_hex*args.num_units_z,
                           L*args.num_units_x, L*args.num_units_y, L*args.num_units_z, data_dir)

    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def corner(point):
        return np.isclose(np.linalg.norm(point), 0., atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], args.num_units_z*L, atol=1e-5)

    dirichlet_zero = lambda _: 0. 
    dirichlet_top_z = lambda x: lambda _: 0.1*args.num_units_z*L
  
    def get_dirichlet_z(rel_disp):
        def val_fn(point):
            return rel_disp*args.num_units_z*L
        return val_fn

    dirichlet_bc_info = [[bottom, bottom, bottom, top, top, top], 
                         [0, 1, 2, 0, 1, 2], 
                         [dirichlet_zero, dirichlet_zero, dirichlet_zero, 
                          dirichlet_zero, dirichlet_zero, get_dirichlet_z(0.)]]

 
    problem = HyperElasticity(jax_mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, additional_info=(case, dns_info))
   
    rel_disps = np.linspace(0., 0.1, 11)
    energies = []
    forces = []

    sol = np.zeros((problem.num_total_nodes, problem.vec))
    energy = problem.compute_energy(sol)
    traction = problem.compute_traction(top, sol)
    energies.append(energy)
    forces.append(traction[-1])

    for i, rel_disp in enumerate(rel_disps[1:]):
        print(f"\nStep {i} in {len(rel_disps) - 1}, rel_disp = {rel_disp}, problem_name = {problem_name}")
        dirichlet_bc_info[-1][-1] = get_dirichlet_z(rel_disp)
        problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        sol = solver(problem)
        energy = problem.compute_energy(sol)
        traction = problem.compute_traction(top, sol)
        energies.append(energy)
        forces.append(traction[-1])
        vtu_path = os.path.join(data_dir, f"vtk/deploy/{problem_name}/u_{i + 1:03d}.vtu")
        save_sol(problem, sol, vtu_path)

    numpy_dir = os.path.join(data_dir, f'numpy/deploy/{problem_name}')
    os.makedirs(numpy_dir, exist_ok=True)
    print(f"energies = {energies}")
    print(f"forces = {forces}")

    data_to_save = np.array([rel_disps, energies, forces])

    onp.save(os.path.join(numpy_dir, 'tensile.npy'), data_to_save)


def run_tensile():
    homogenization_problem('nn')
    homogenization_problem('dns')
    # homogenization_problem('dns', 'in')
    # homogenization_problem('dns', 'out')


def plot_results():
    data_dir = os.path.join(crt_file_path, 'data')
    problem_names = ['dns_in', 'nn', 'dns', 'dns_out']
    colors = ['orange', 'red', 'blue', 'green']
    markers = ['^', 'o', 's', 'v']
    labels = ['Hard', 'NN', 'DNS', 'Soft']

    fig = plt.figure(figsize=(8, 6)) 
    for i in range(len(problem_names)):
        rel_disps, energies, _ = onp.load(os.path.join(data_dir, f'numpy/deploy/{problem_names[i]}/tensile.npy'))
        plt.plot(rel_disps, energies, color=colors[i], marker=markers[i], markersize=10, linestyle='-', linewidth=2, label=labels[i])  
        plt.xlabel("Relative displacement", fontsize=20)
        plt.ylabel(r"Energy [$\mu$J]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False)   
    plt.savefig(os.path.join(data_dir, f'pdf/energy.pdf'), bbox_inches='tight')

    fig = plt.figure(figsize=(8, 6)) 
    for i in range(len(problem_names)):
        rel_disps, _, forces = onp.load(os.path.join(data_dir, f'numpy/deploy/{problem_names[i]}/tensile.npy'))
        plt.plot(rel_disps, forces, color=colors[i], marker=markers[i], markersize=10, linestyle='-', linewidth=2, label=labels[i])  
        plt.xlabel("Relative displacement", fontsize=20)
        plt.ylabel(r"Force [mN]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False) 
    plt.savefig(os.path.join(data_dir, f'pdf/force.pdf'), bbox_inches='tight')


if __name__=="__main__":
    run_tensile()
    # debug()
    # plot_results()
    # plt.show()

