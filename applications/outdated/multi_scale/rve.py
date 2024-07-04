import numpy as onp
import jax
import jax.numpy as np
import time
import os
import glob
from functools import partial
from scipy.stats import qmc

from jax_fem.generate_mesh import Mesh, box_mesh_gmsh
from jax_fem.solver import solver, assign_bc, get_A_fn_linear_fn
from jax_fem.utils import save_sol

from applications.fem.multi_scale.arguments import args
from applications.fem.multi_scale.utils import flat_to_tensor
from applications.fem.multi_scale.fem_model import HyperElasticity

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)


def rve_problem(data_dir):
    args.num_units_x = 1
    args.num_units_y = 1
    args.num_units_z = 1

    L = args.L
    meshio_mesh = box_mesh_gmsh(args.num_hex*args.num_units_x, args.num_hex*args.num_units_y, args.num_hex*args.num_units_z,
                           L*args.num_units_x, L*args.num_units_y, L*args.num_units_z, data_dir)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def corner(point):
        return np.isclose(np.linalg.norm(point), 0., atol=1e-5)

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], L, atol=1e-5)

    def front(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def back(point):
        return np.isclose(point[1], L, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], L, atol=1e-5)

    def dirichlet(point):
        return 0.

    def mapping_x(point_A):
        point_B = point_A + np.array([L, 0., 0.])
        return point_B

    def mapping_y(point_A):
        point_B = point_A + np.array([0., L, 0.])
        return point_B

    def mapping_z(point_A):
        point_B = point_A + np.array([0., 0., L])
        return point_B

    location_fns = [corner]*3
    value_fns = [dirichlet]*3
    vecs = [0, 1, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    location_fns_A = [left]*3 + [front]*3 + [bottom]*3
    location_fns_B = [right]*3 + [back]*3 + [top]*3
    mappings = [mapping_x]*3 + [mapping_y]*3 + [mapping_z]*3
    vecs = [0, 1, 2]*3

    periodic_bc_info = [location_fns_A, location_fns_B, mappings, vecs]
    problem = HyperElasticity(jax_mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, 
        periodic_bc_info=periodic_bc_info, additional_info=('rve', None))
    problem.p_num_eps = 1e2 # For numerical stability of imposing periodic B.C.
    return problem


def exp():
    """Do not delete. We use this to generate RVE demo.
    """
    problem_name = 'rve_debug'
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    problem = rve_problem(data_dir)
    H_bar = np.array([[-0.009, 0., 0.],
                      [0., -0.009, 0.],
                      [0., 0., 0.025]])

    problem.H_bar = H_bar

    # material = np.where(problem.E > 2*1e2, 0., 1.)

    sol_fluc_ini = np.zeros((problem.num_total_nodes, problem.vec))
    sol_fluc_ini = assign_bc(sol_fluc_ini, problem)
    energy = problem.compute_energy(sol_fluc_ini)
    print(f"Initial energy = {energy}")

    sol_disp_ini = problem.fluc_to_disp(sol_fluc_ini)
    jax_vtu_path = os.path.join(data_dir, f'vtk/{problem_name}/sol_disp_ini.vtu')
    save_sol(problem, sol_disp_ini, jax_vtu_path, [("E", problem.E.reshape((problem.num_cells, problem.num_quads))[:, 0])])

    sol_fluc = aug_solve(problem)

    # ratios = [1.5, 1.8, 2.]
    # for ratio in ratios:
    #     problem.H_bar = ratio * H_bar
    #     sol_fluc = aug_solve(problem, initial_guess=sol_fluc)

    energy = problem.compute_energy(sol_fluc)
    print(f"Final energy = {energy}")

    sol_disp = problem.fluc_to_disp(sol_fluc)
    jax_vtu_path = os.path.join(data_dir, f'vtk/{problem_name}/sol_disp.vtu')
    save_sol(problem, sol_disp, jax_vtu_path)

    jax_vtu_path = os.path.join(data_dir, f'vtk/{problem_name}/sol_fluc.vtu')
    save_sol(problem, sol_fluc, jax_vtu_path)

    a = sol_fluc[problem.p_node_inds_list_A[0], problem.p_vec_inds_list[0]]
    b = sol_fluc[problem.p_node_inds_list_B[0], problem.p_vec_inds_list[0]]
    ap = problem.mesh.points[problem.p_node_inds_list_A[0]]
    bp = problem.mesh.points[problem.p_node_inds_list_B[0]]
    print(np.hstack((ap, bp, a[:, None], b[:, None]))[:10])


def check_one_rve():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, f"numpy/training/09052022/00684.npy")
    print(onp.load(file_path))


def solve_rve_problem(problem, sample_H_bar):
    base_H_bar = flat_to_tensor(sample_H_bar)
    problem.H_bar = base_H_bar
    sol_fluc = solver(problem)
    energy = problem.compute_energy(sol_fluc)
    ratios = [0.25, 0.5, 0.75, 0.9, 1.]
    if np.any(np.isnan(energy)):
        print(f"Solve with quasi-static steps...")
        sol_fluc = np.zeros((problem.num_total_nodes, problem.vec))
        for ratio in ratios:
            problem.H_bar = ratio * base_H_bar
            sol_fluc = solver(problem)
        energy = problem.compute_energy(sol_fluc)

    return sol_fluc, np.hstack((sample_H_bar, energy))


def generate_samples():
    dim_H = 6
    sampler = qmc.Sobol(d=dim_H, scramble=False, seed=0)
    sample = sampler.random_base2(m=10)
    lower_val = -0.2
    upper_val = 0.2
    l_bounds = [lower_val]*dim_H
    u_bounds = [upper_val]*dim_H
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
    return scaled_sample
 

def collect_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    problem = rve_problem(data_dir)
    date = f"11012022"
    root_numpy = os.path.join(data_dir, 'numpy/training', date)
 
    if not os.path.exists(root_numpy):
        os.makedirs(root_numpy)

    root_vtk = os.path.join(data_dir, 'vtk/training', date)
    if not os.path.exists(root_vtk):
        os.makedirs(root_vtk)

    samples = generate_samples()
    complete = [i for i in range(len(samples))]

    onp.random.seed(args.device)
    while True:
        files = glob.glob(root_numpy + f"/*.npy")
        done = [int(file[-9:-4]) for file in files]
        todo = list(set(complete) - set(done))
        if len(todo) == 0:
            break
        chosen_ind = onp.random.choice(todo)
        print(f"\nSolving problem # {chosen_ind} on device = {args.device}, done = {len(done)}, todo = {len(todo)}, total = {len(complete)} ")
        sample_H_bar = samples[chosen_ind]
        sol_fluc, data = solve_rve_problem(problem, sample_H_bar)
        if np.any(np.isnan(data)):
            print(f"######################################### Failed solve, check why!")
            onp.savetxt(os.path.join(root_numpy, f"{chosen_ind:05d}.txt"), sample_H_bar)
        else:
            print(f"Saving data = {data}")
            onp.save(os.path.join(root_numpy, f"{chosen_ind:05d}.npy"), data)

        sol_disp = problem.fluc_to_disp(sol_fluc)
        jax_vtu_path = os.path.join(root_vtk, f"sol_disp_{chosen_ind:05d}.vtu")
        save_sol(problem, sol_disp, jax_vtu_path)


if __name__=="__main__":
    # exp()
    collect_data()
    # check_one_rve()
