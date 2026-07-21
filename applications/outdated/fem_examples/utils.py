import matplotlib.pyplot as plt
import numpy as onp
import os

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def plot_plastic_stress_strain():
    problem_names = ["linear_elasticity", "hyperelasticity", "plasticity"]
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    y_lables = [r'Force on top surface [N]', r'Force on top surface [N]', r'Volume averaged stress (z-z) [MPa]']
    ratios = [1e-3, 1e-3, 1.]

    for i in range(len(problem_names)):
        disps_path = os.path.join(data_dir, 'numpy', problem_names[i], 'fenicsx/disps.npy')
        fenicsx_forces_path = os.path.join(data_dir, 'numpy', problem_names[i], 'fenicsx/forces.npy')
        jax_fem_forces_path = os.path.join(data_dir, 'numpy', problem_names[i], 'jax_fem/forces.npy')
        fenicsx_forces = onp.load(fenicsx_forces_path)
        jax_fem_forces = onp.load(jax_fem_forces_path)
        disps = onp.load(disps_path)
        fig = plt.figure(figsize=(8, 6)) 
        plt.plot(disps, fenicsx_forces*ratios[i], label='FEniCSx', color='blue', linestyle="-", linewidth=2)
        plt.plot(disps, jax_fem_forces*ratios[i], label='JAX-FEM', color='red', marker='o', markersize=8, linestyle='None') 
        plt.xlabel(r'Displacement of top surface [mm]', fontsize=20)
        plt.ylabel(y_lables[i], fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(os.path.join(data_dir, f'pdf/{problem_names[i]}_stress_strain.pdf'), bbox_inches='tight')


def plot_performance():
    data_dir = f"applications/fem/fem_examples/data/"
    abaqus_cpu_time = onp.loadtxt(os.path.join(data_dir, f"txt/abaqus_fem_time_cpu.txt"))
    abaqus_time_np_12 = onp.loadtxt(os.path.join(data_dir, f"txt/abaqus_fem_time_mpi_np_12.txt"))
    abaqus_time_np_24 = onp.loadtxt(os.path.join(data_dir, f"txt/abaqus_fem_time_mpi_np_24.txt"))
    fenicsx_time_np_1 = onp.loadtxt(os.path.join(data_dir, f"txt/fenicsx_fem_time_mpi_np_1.txt"))
    fenicsx_time_np_2 = onp.loadtxt(os.path.join(data_dir, f"txt/fenicsx_fem_time_mpi_np_2.txt"))
    fenicsx_time_np_4 = onp.loadtxt(os.path.join(data_dir, f"txt/fenicsx_fem_time_mpi_np_4.txt"))
    jax_time_cpu = onp.loadtxt(os.path.join(data_dir, f"txt/jax_fem_cpu_time.txt"))  
    jax_time_gpu = onp.loadtxt(os.path.join(data_dir, f"txt/jax_fem_gpu_time.txt"))  
    cpu_dofs = onp.loadtxt(os.path.join(data_dir, f"txt/jax_fem_cpu_dof.txt"))   
    gpu_dofs = onp.loadtxt(os.path.join(data_dir, f"txt/jax_fem_gpu_dof.txt"))   

    plt.figure(figsize=(12, 9))
    plt.plot(gpu_dofs[1:], abaqus_cpu_time[1:], linestyle='-', marker='o', markersize=12, linewidth=2, color='blue', label='Abaqus CPU')
    plt.plot(gpu_dofs[1:], abaqus_time_np_12[1:], linestyle='-', marker='s', markersize=12, linewidth=2, color='blue', label='Abaqus CPU MPI 12')
    plt.plot(gpu_dofs[1:], abaqus_time_np_24[1:], linestyle='-', marker='^', markersize=12, linewidth=2, color='blue', label='Abaqus CPU MPI 24')
    plt.plot(cpu_dofs[1:], fenicsx_time_np_1[1:], linestyle='-', marker='o', markersize=12, linewidth=2, color='green', label='FEniCSx CPU')
    plt.plot(cpu_dofs[1:], fenicsx_time_np_2[1:], linestyle='-', marker='s', markersize=12, linewidth=2, color='green', label='FEniCSx CPU MPI 2')
    plt.plot(cpu_dofs[1:], fenicsx_time_np_4[1:], linestyle='-', marker='^', markersize=12, linewidth=2, color='green', label='FEniCSx CPU MPI 4')
    plt.plot(cpu_dofs[1:], jax_time_cpu[1:], linestyle='-', marker='s', markersize=12, linewidth=2, color='red', label='JAX-FEM CPU')
    plt.plot(gpu_dofs[1:], jax_time_gpu[1:], linestyle='-', marker='o', markersize=12, linewidth=2, color='red', label='JAX-FEM GPU')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of DOFs", fontsize=20)
    plt.ylabel("Wall time [s]", fontsize=20)
    plt.tick_params(labelsize=20)
    ax = plt.gca()
    # ax.get_xaxis().set_tick_params(which='minor', size=0)
    # plt.xticks(plt_tmp, tick_labels)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
 
    plt.savefig(os.path.join(data_dir, f'pdf/performance.pdf'), bbox_inches='tight')

if __name__ == '__main__':
    # plot_plastic_stress_strain()
    plot_performance()
    plt.show()
