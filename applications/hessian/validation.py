import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import time
 
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem import logger
from applications.hessian.hess_manager import HessVecProduct
from applications.hessian.utils import compute_l2_norm_error


onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=10)

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


logger.setLevel(logging.INFO)

case_name = 'validation'
data_dir = os.path.join(os.path.dirname(__file__), f'data/{case_name}')
fwd_vtk_dir = os.path.join(data_dir, 'forward/vtk')
fwd_numpy_dir = os.path.join(data_dir, 'forward/numpy')
val_numpy_dir = os.path.join(data_dir, 'inverse/validation/numpy')
val_pdf_dir = os.path.join(data_dir, 'inverse/validation/pdf')
prof_numpy_dir = os.path.join(data_dir, 'inverse/profiling/numpy')
prof_pdf_dir = os.path.join(data_dir, 'inverse/profiling/pdf')


class NonlinearPoisson(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, theta):
            # Handles the term `exp(theta*u) * inner(grad(u), grad(v)*dx`

            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            # theta: (num_quads,)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            # cell_sol_u: (num_nodes_u, vec)
            cell_sol_u, = cell_sol_list
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            # (num_quads, num_nodes_u, dim)
            cell_shape_grads_u, = cell_shape_grads_list
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            # (num_quads, num_nodes_u, 1, dim)
            cell_v_grads_JxW_u, = cell_v_grads_JxW_list

            # (1, num_nodes_u, vec_u) * (num_quads, num_nodes_u, 1) -> (num_quads, num_nodes_u, vec_u) -> (num_quads, vec_u)
            u = np.sum(cell_sol_u[None, :, :] * self.fe.shape_vals[:, :, None], axis=1)

            # (1, num_nodes_u, vec_u, 1) * (num_quads, num_nodes_u, 1, dim) -> (num_quads, num_nodes_u, vec_u, dim)
            u_grads = cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec_u, dim)

            # (num_quads, num_nodes_u, vec_u, dim) -> (num_nodes_u, vec_u)
            val = np.sum(np.exp(theta[:, None, None, None] *  u[:, None, :, None]) * u_grads[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))
            weak_form = [val] # [(num_nodes, vec), ...]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel

    def get_mass_map(self):
        def mass_map(u, x, theta):
            val = -np.array([10*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            return val
        return mass_map

    def get_surface_maps(self):
        def surface_map(u, x):
            return -np.array([np.sin(5.*x[0])])

        return [surface_map, surface_map]

    def set_params(self, theta):
        self.internal_vars = [theta]


def profile_hessp(hess_vec_prod):
    θ_flat = jax.random.normal(jax.random.key(1), hess_vec_prod.θ_ini_flat.shape)
    θ_hat_flat = jax.random.normal(jax.random.key(2), hess_vec_prod.θ_ini_flat.shape)

    # filename = time.perf_counter_ns()

    hessp_options = ['fwd_rev', 'rev_fwd', 'rev_rev']
    num_loops = 11
    J_times = []
    F_times = []
    for index, hessp_option in enumerate(hessp_options):
        hess_vec_prod.hessp_option = hessp_option
        J_times.append([])
        F_times.append([])
        for i in range(num_loops):
            hess_vec_prod.hessp(θ_flat, θ_hat_flat)
            J_time, F_time = hess_vec_prod.profile_info
            J_times[-1].append(J_time)
            F_times[-1].append(F_time)
            print(f"option = {hessp_option}, J_time = {J_time}, F_time = {F_time}")

    profile_results = np.array([J_times, F_times])
    os.makedirs(prof_numpy_dir, exist_ok=True)
    np.save(os.path.join(prof_numpy_dir, f'profile_results_{time.perf_counter_ns()}.npy'), profile_results)


def finite_difference_hessp(hess_vec_prod, θ_flat, θ_hat_flat, h):
    θ_minus_flat = θ_flat - h*θ_hat_flat
    θ_plus_flat  = θ_flat + h*θ_hat_flat
    value_plus = hess_vec_prod.grad(θ_plus_flat)
    value_minus = hess_vec_prod.grad(θ_minus_flat)
    dθ_dθ_J_θ_hat = jax.tree_util.tree_map(lambda x, y: (x - y)/(2*h), value_plus, value_minus)
    logger.debug(f"FD = {dθ_dθ_J_θ_hat}")
    return dθ_dθ_J_θ_hat


def hessian_validation(hess_vec_prod, h, seed_1=1, seed_2=2, seed_3=3):
    θ_flat = jax.random.normal(jax.random.key(seed_1), hess_vec_prod.θ_ini_flat.shape)
    θ_hat_flat = jax.random.normal(jax.random.key(seed_2), hess_vec_prod.θ_ini_flat.shape)
    θ_tilde_flat = jax.random.normal(jax.random.key(seed_3), hess_vec_prod.θ_ini_flat.shape)
    hess_v_ad = hess_vec_prod.hessp(θ_flat, θ_hat_flat)
    hess_v_fd = finite_difference_hessp(hess_vec_prod, θ_flat, θ_hat_flat, h)
    hess_v_ad_flat = jax.flatten_util.ravel_pytree(hess_v_ad)[0]
    hess_v_fd_flat = jax.flatten_util.ravel_pytree(hess_v_fd)[0]
    v_hess_v_ad = np.dot(θ_tilde_flat, hess_v_ad_flat)
    v_hess_v_fd = np.dot(θ_tilde_flat, hess_v_fd_flat)
    rel_err = np.linalg.norm(hess_v_ad_flat - hess_v_fd_flat)/np.linalg.norm(hess_v_ad_flat)
    logger.info(f"\n")
    logger.info(f"v_hess_v_ad = {v_hess_v_ad}, v_hess_v_fd = {v_hess_v_fd}")
    logger.info(f"\n")
    return v_hess_v_ad, v_hess_v_fd, rel_err


def taylor_remainder_test(hess_vec_prod, h):
    θ_flat = jax.random.normal(jax.random.key(1), hess_vec_prod.θ_ini_flat.shape)
    θ_hat_flat = jax.random.normal(jax.random.key(2), hess_vec_prod.θ_ini_flat.shape)

    f_val = hess_vec_prod.J(θ_flat)
    θ_plus_flat = θ_flat + h*θ_hat_flat
    f_plus_val = hess_vec_prod.J(θ_plus_flat)

    f_grad = hess_vec_prod.grad(θ_flat)
    f_grad_flat = jax.flatten_util.ravel_pytree(f_grad)[0]
    v_f_grad = h*np.dot(θ_hat_flat, f_grad_flat)

    hess_v = hess_vec_prod.hessp(θ_flat, θ_hat_flat)
    hess_v_flat = jax.flatten_util.ravel_pytree(hess_v)[0]
    v_hess_v = h**2./2.*np.dot(θ_hat_flat, hess_v_flat)

    return f_plus_val, f_val, v_f_grad, v_hess_v


def workflow():
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 1.
    meshio_mesh = rectangle_mesh(Nx=64, Ny=64, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)


    def dirichlet_val_left(point):
        return 0.

    def dirichlet_val_right(point):
        return 0.

    location_fns_dirichlet = [left, right]
    value_fns = [dirichlet_val_left, dirichlet_val_right]
    vecs = [0, 0]
    dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

    location_fns = [bottom, top]
    problem = NonlinearPoisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    fwd_pred = ad_wrapper(problem) 

    # (num_cells, num_quads, dim)
    quad_points = problem.fes[0].get_physical_quad_points()

    run_forward_flag = False
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*')) + glob.glob(os.path.join(fwd_numpy_dir, f'*'))
        for f in files:
            os.remove(f)

        theta_true =  np.ones_like(quad_points)[:, :, 0]
        sol_list_true = fwd_pred(theta_true)

        save_sol(problem.fes[0], sol_list_true[0], os.path.join(fwd_vtk_dir, f'u.vtu'), cell_infos=[('theta', np.mean(theta_true, axis=-1))])
        os.makedirs(fwd_numpy_dir, exist_ok=True)
        np.save(os.path.join(fwd_numpy_dir, f'u.npy'), sol_list_true[0])

    run_inverse_flag = True
    if run_inverse_flag:
        sol_list_true = [np.load(os.path.join(fwd_numpy_dir, f'u.npy'))]
        def J_fn(u, θ):
            sol_list_pred = u
            l2_u = compute_l2_norm_error(problem, sol_list_pred, sol_list_true)
            return l2_u**2

        theta_ini = np.zeros_like(quad_points)[:, :, 0]
        option_umfpack = {'umfpack_solver': {}}
        hess_vec_prod = HessVecProduct(problem, J_fn, theta_ini, option_umfpack, option_umfpack, None)

        run_profiling_flag = True
        if run_profiling_flag:
            profile_hessp(hess_vec_prod)

        run_validation_flag = False
        if run_validation_flag:
            files = glob.glob(os.path.join(val_numpy_dir, f'*')) 
            for f in files:
                os.remove(f)

            hessian_validation(hess_vec_prod, h=1e-3)
     
            hs = [1e-1, 1e-2, 1e-3, 1e-4]

            taylor_results = []
            for h in hs:
                f_plus_val, f_val, v_f_grad, v_hess_v = taylor_remainder_test(hess_vec_prod, h)
                taylor_results.append([h, f_plus_val, f_val, v_f_grad, v_hess_v])

            taylor_results = np.array(taylor_results)
            os.makedirs(val_numpy_dir, exist_ok=True)
            np.save(os.path.join(val_numpy_dir, f'taylor_results.npy'), taylor_results)

            num_seeds = 100
            vHv_results = []
            for h in hs:
                vHv_results.append([])
                for i in range(num_seeds):
                    print(f"\n\n######################## Random testing h = {h}, index = {i + 1} ")
                    seed_1 = i + 1
                    seed_2 = i + 1 + num_seeds
                    seed_3 = i + 1 + 2*num_seeds
                    v_hess_v_ad, v_hess_v_fd, rel_err = hessian_validation(hess_vec_prod, h, seed_1, seed_2, seed_3)
                    vHv_results[-1].append([v_hess_v_ad, v_hess_v_fd, rel_err])

            vHv_results = np.array(vHv_results)
            np.save(os.path.join(val_numpy_dir, f'vHv_results.npy'), vHv_results)


def generate_figures():
    # # Figure set 1: Taylor remainder test
    # taylor_results = np.load(os.path.join(val_numpy_dir, f"taylor_results.npy"))

    # hs, f_plus_val, f_val, v_f_grad, v_hess_v = taylor_results.T
    # res_zero = np.abs(f_plus_val - f_val)
    # res_first = np.abs(f_plus_val - f_val - v_f_grad)
    # res_second = np.abs(f_plus_val - f_val - v_f_grad - v_hess_v)

    # ref_zero = [1/5.*res_zero[0]/hs[0] * h for h in hs]
    # ref_first = [1/5.*res_first[0]/hs[0]**2 * h**2 for h in hs]
    # ref_second = [1/5.*res_second[0]/hs[0]**3 * h**3 for h in hs]

    # plt.figure(figsize=(10, 10))
    # plt.plot(hs, res_zero, linestyle='-', marker='o', markersize=10, linewidth=2, color='blue', label=r"$r_{\textrm{zeroth}}$")
    # plt.plot(hs, ref_zero, linestyle='--', linewidth=2, color='blue', label='First order reference')
    # plt.plot(hs, res_first, linestyle='-', marker='o', markersize=10, linewidth=2, color='red', label=r"$r_{\textrm{first}}$")
    # plt.plot(hs, ref_first, linestyle='--', linewidth=2, color='red', label='Second order reference')
    # plt.plot(hs, res_second, linestyle='-', marker='o', markersize=10, linewidth=2, color='green', label=r"$r_{\textrm{second}}$")
    # plt.plot(hs, ref_second, linestyle='--', linewidth=2, color='green', label='Third order reference')

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r"Step size $h$", fontsize=20)
    # plt.ylabel("Residual", fontsize=20)
    # plt.tick_params(labelsize=20)
    # plt.tick_params(labelsize=20)
    # plt.legend(fontsize=20, frameon=False)   

    # os.makedirs(val_pdf_dir, exist_ok=True)
    # plt.savefig(os.path.join(val_pdf_dir, f'taylor_results.pdf'), bbox_inches='tight')

    # # Figure set 2: Random sampling Hv or vHv
    # vHv_results = np.load(os.path.join(val_numpy_dir, f"vHv_results.npy"))

    # num_bins = 30
    # colors = ['red', 'blue', 'green', 'orange']
    # labels = [f'h={hs[i]}' for i in range(len(hs))]

    # # Set flag to 'Hv' or 'vHv'
    # flag = 'Hv'
    # for i, h in enumerate(hs):
    #     plt.figure(figsize=(10, 6))
    #     relative_errors_Hv = vHv_results[i][:, 2]
    #     relative_errors_vHv = np.abs((vHv_results[i][:, 0] - vHv_results[i][:, 1])/vHv_results[i][:, 0])

    #     data = relative_errors_Hv if flag == 'Hv' else relative_errors_vHv

    #     plt.hist(data,
    #              bins=num_bins,
    #              color=colors[i],
    #              alpha=0.5,  # Transparency for overlapping regions
    #              edgecolor='black',
    #              label=labels[i])

    #     # plt.title(f'Histogram of relative errors')
    #     plt.tick_params(labelsize=15)
    #     plt.tick_params(labelsize=15)
    #     plt.xlabel('Relative difference', fontsize=20)
    #     plt.ylabel('Count', fontsize=20)
    #     plt.legend(fontsize=20, frameon=False)
    #     # plt.grid(axis='y', alpha=0.75)
    #     plt.savefig(os.path.join(val_pdf_dir, f'{flag}_{i:03d}.pdf'), bbox_inches='tight')

    # Figure set 3: Profiling
    profile_results = np.load(os.path.join(prof_numpy_dir, f"profile_results_30091068549.npy"))
    labels = ['fwd-rev', 'rev-fwd', 'rev-rev']

    J_times, F_times = profile_results
    F_means = np.mean(F_times[:, 1:], axis=1)
    F_stds =  np.std(F_times[:, 1:], axis=1)
    J_means = np.mean(J_times[:, 1:], axis=1)
    J_stds =  np.std(J_times[:, 1:], axis=1)

    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(labels))
    bar_width = 0.8

    bars = plt.bar(x_pos, F_means, 
                   yerr=F_stds, 
                   width=bar_width,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black',
                   error_kw=dict(elinewidth=2, ecolor='black', capsize=5))

    plt.ylabel('Execution time [s]', fontsize=20)
    plt.xticks(x_pos, labels, fontsize=20)
    plt.yticks(fontsize=20)
    # plt.grid(axis='y', alpha=0.75)
    os.makedirs(prof_pdf_dir, exist_ok=True)
    plt.savefig(os.path.join(prof_pdf_dir, f'F.pdf'), bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(labels))
    bar_width = 0.8

    # Create bars with error bars
    bars = plt.bar(x_pos, J_means, 
                   yerr=J_stds, 
                   width=bar_width,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black',
                   error_kw=dict(elinewidth=2, ecolor='black', capsize=5))

    # Customize plot
    plt.ylabel('Execution time [s]', fontsize=20)
    plt.xticks(x_pos, labels, fontsize=20)
    plt.yticks(fontsize=20)
    # plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(prof_pdf_dir, f'J.pdf'), bbox_inches='tight')

    plt.show()

if __name__=="__main__":
    # workflow()
    generate_figures()
