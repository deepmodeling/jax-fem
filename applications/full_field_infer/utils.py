import matplotlib.pyplot as plt
import numpy as onp
import os

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def plot_results():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    obj_val, rel_error_sol, rel_error_force = onp.load(os.path.join(data_dir, f"numpy/outputs.npy"))
    truncate = 21
    obj_val, rel_error_sol, rel_error_force = obj_val[:truncate], rel_error_sol[:truncate], rel_error_force[:truncate]
    steps = onp.arange(len(obj_val))
    print(rel_error_sol[-1])

    plt.figure(figsize=(8, 6))
    plt.plot(steps, obj_val, linestyle='-', marker='o', markersize=10, linewidth=2, color='black')
    plt.xlabel("Optimization step", fontsize=20)
    plt.ylabel("Objective value", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.savefig(os.path.join(data_dir, f'pdf/loss.pdf'), bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.plot(steps, rel_error_sol, linestyle='-', marker='o', markersize=10, linewidth=2, color='black')
    plt.xlabel("Optimization step", fontsize=20)
    plt.ylabel("Inference error", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.savefig(os.path.join(data_dir, f'pdf/error.pdf'), bbox_inches='tight')


    hs, res_zero, res_first = onp.load(os.path.join(data_dir, f"numpy/res.npy"))

    ref_zero = [1/5.*res_zero[-1]/hs[-1] * h for h in hs]
    ref_first = [1/5.*res_first[-1]/hs[-1]**2 * h**2 for h in hs]

    plt.figure(figsize=(10, 8))
    plt.plot(hs, res_zero, linestyle='-', marker='o', markersize=10, linewidth=2, color='blue', label=r"$r_{\textrm{zeroth}}$")
    plt.plot(hs, ref_zero, linestyle='--', linewidth=2, color='blue', label='First order reference')
    plt.plot(hs, res_first, linestyle='-', marker='o', markersize=10, linewidth=2, color='red', label=r"$r_{\textrm{first}}$")
    plt.plot(hs, ref_first, linestyle='--', linewidth=2, color='red', label='Second order reference')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Step size $h$", fontsize=20)
    plt.ylabel("Residual", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   

    plt.savefig(os.path.join(data_dir, f'pdf/res.pdf'), bbox_inches='tight')



if __name__=="__main__":
    plot_results()
    plt.show()
