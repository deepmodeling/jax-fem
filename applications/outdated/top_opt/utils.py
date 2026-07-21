import matplotlib.pyplot as plt
import numpy as onp
import os

from jax_fem.utils import make_video

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

data_dir = f"applications/fem/top_opt/data/"


def plot_topopt():
    plate_obj = onp.load(os.path.join(data_dir, f"numpy/plate_outputs.npy")).reshape(-1)
    freecad_obj = onp.load(os.path.join(data_dir, f"numpy/computer_design_outputs.npy")).reshape(-1)
    plate_obj = plate_obj
    data = [plate_obj, freecad_obj]

    cases = ['plate', 'freecad']
    for i in range(len(cases)):
        print(data[i][-1])
        plt.figure(figsize=(8, 6))
        # plt.plot(steps, data[i], linestyle='-', marker='o', markersize=10, linewidth=2, color='black')
        plt.plot(onp.arange(len(data[i])) + 1, data[i], linestyle='-', linewidth=2, color='black')
        if cases[i] == 'freecad':
            plt.plot(onp.arange(len(data[i])) + 1, 3.07*onp.ones_like(data[i]), linestyle='-', linewidth=2, color='red')
        plt.xlabel(r"Optimization step", fontsize=20)
        plt.ylabel(r"Compliance [$\mu$J]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.savefig(os.path.join(data_dir, f'pdf/{cases[i]}_obj.pdf'), bbox_inches='tight')


def plot_plasticity():
    plasticity_obj = onp.load(os.path.join(data_dir, f"numpy/plasticity_outputs.npy")).reshape(-1)
    single_obj = onp.load(os.path.join(data_dir, f"numpy/plasticity_tmp_outputs.npy")).reshape(-1)
   
    plt.figure(figsize=(8, 6))
    # plt.plot(steps, data[i], linestyle='-', marker='o', markersize=10, linewidth=2, color='black')
    plt.plot(5*onp.arange(len(plasticity_obj)) + 1, plasticity_obj, linestyle='-', linewidth=2, color='red', label='top + mat')
    plt.plot(5*onp.arange(len(single_obj)) + 1, single_obj, linestyle='-', linewidth=2, color='blue', label='top')
    plt.legend(fontsize=20)

    plt.xlabel(r"Optimization step", fontsize=20)
    plt.ylabel(r"Compliance [$\mu$J]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)


def plot_eigen():
    def helper(problem_name):
        obj = onp.load(os.path.join(data_dir, f"numpy/{problem_name}_outputs.npy"))
        eigen_vals = onp.load(os.path.join(data_dir, f"numpy/{problem_name}_eigen_vals.npy"))
        plt.figure(figsize=(8, 6))
        plt.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
        plt.xlabel(r"Optimization step", fontsize=20)
        plt.ylabel(r"Compliance [J]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.ylim((0., 1800))
        plt.savefig(os.path.join(data_dir, f'pdf/{problem_name}_obj.pdf'), bbox_inches='tight')

        plt.figure(figsize=(8, 6))
        if problem_name == 'eigen_w_cstr':
            plt.plot(onp.arange(len(eigen_vals)) + 1, 1.*onp.ones(len(eigen_vals)), linestyle='-', linewidth=2, color='black', label='Lower bound')
        colors = ['red', 'blue', 'green']
        labels = ['1st eigenvalue', '2nd eigenvalue', '3rd eigenvalue']
        for i in range(eigen_vals.shape[1]):
            plt.plot(onp.arange(len(eigen_vals)) + 1, eigen_vals[:, i]/1e6, linestyle='-', linewidth=2, color=colors[i], label=labels[i])
        plt.xlabel(r"Optimization step", fontsize=20)
        plt.ylabel(r"Eigenvalue $\omega^2$ [kHz$^2$]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.ylim((0., 2.8))
        plt.legend(fontsize=18, frameon=False)
        plt.savefig(os.path.join(data_dir, f'pdf/{problem_name}_eigen_vals.pdf'), bbox_inches='tight')

    helper('eigen_w_cstr')
    helper('eigen_no_cstr')


def plot_L_shape():
    problem_name_w_cstr = 'L_shape_w_cstr'
    obj_w = onp.load(os.path.join(data_dir, f"numpy/{problem_name_w_cstr}_outputs.npy"))
    max_vm_stress_w = onp.load(os.path.join(data_dir, f"numpy/{problem_name_w_cstr}_max_vm_stresses.npy"))

    problem_name_no_cstr = 'L_shape_no_cstr'
    obj_no = onp.load(os.path.join(data_dir, f"numpy/{problem_name_no_cstr}_outputs.npy"))
    max_vm_stress_no = onp.load(os.path.join(data_dir, f"numpy/{problem_name_no_cstr}_max_vm_stresses.npy"))

    plt.figure(figsize=(8, 6))
    plt.plot(onp.arange(len(obj_w)) + 1, obj_w, linestyle='-', linewidth=2, color='red', label='With stress constraint')
    plt.plot(onp.arange(len(obj_no)) + 1, obj_no, linestyle='-', linewidth=2, color='blue', label='Without stress constraint')
    plt.xlabel(r"Optimization step", fontsize=20)
    plt.ylabel(r"Compliance [J]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)
    # plt.ylim((0., 1800))
    plt.savefig(os.path.join(data_dir, f'pdf/L_shape_obj.pdf'), bbox_inches='tight')

    plt.figure(figsize=(8, 6))

    plt.plot(onp.arange(len(max_vm_stress_w)) + 1, 3.5*onp.ones(len(max_vm_stress_w)), linestyle='-', linewidth=2, color='black', label='Stress upper bound')

    plt.plot(onp.arange(len(max_vm_stress_w)) + 1, max_vm_stress_w/1e6, linestyle='-', linewidth=2, color='red', label='With stress constraint')
    plt.plot(onp.arange(len(max_vm_stress_no)) + 1, max_vm_stress_no/1e6, linestyle='-', linewidth=2, color='blue', label='Without stress constraint')
    plt.xlabel(r"Optimization step", fontsize=20)
    plt.ylabel(r"Max von Mises stress [MPa]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    # plt.ylim((0., 2.8))
    plt.legend(fontsize=20, frameon=False)
    plt.savefig(os.path.join(data_dir, f'pdf/L_shape_max_vm_stress.pdf'), bbox_inches='tight')

  
def plot_box():
    obj = onp.load(os.path.join(data_dir, f"numpy/box_outputs.npy"))
    print(obj)
    print(onp.diff(obj))
    plt.figure(figsize=(8, 6))
    plt.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
    plt.xlabel(r"Optimization step", fontsize=20)
    plt.ylabel(r"Compliance [J]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.savefig(os.path.join(data_dir, f'pdf/box_obj.pdf'), bbox_inches='tight')


if __name__=="__main__":
    # plot_topopt()
    # plot_plasticity()
    # plot_eigen()
    # plot_L_shape()
    plot_box()
    # plt.show()
    # make_video(data_dir)
