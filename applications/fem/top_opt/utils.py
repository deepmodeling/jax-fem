import matplotlib.pyplot as plt
import numpy as onp
import os

from jax_am.common import make_video

# Latex style plot
# plt.rcParams.update({
#     "text.latex.preamble": r"\usepackage{amsmath}",
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

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


if __name__=="__main__":
    # plot_topopt()
    # plot_plasticity()
    # plt.show()
    make_video(data_dir)
