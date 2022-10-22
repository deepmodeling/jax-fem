import numpy as onp
import jax
import jax.numpy as np
import time
import os
import glob
from functools import partial
from scipy.stats import qmc

from jax_am.fem.generate_mesh import box_mesh
from jax_am.fem.jax_fem import Mesh, Laplace
from jax_am.fem.solver import solver, assign_bc, get_A_fn_linear_fn
from jax_am.fem.utils import save_sol

from applications.fem.multi_scale.arguments import args
from applications.fem.multi_scale.utils import flat_to_tensor
from applications.fem.multi_scale.fem_model import HyperElasticity

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)


def aug_solve(problem, initial_guess=None):
    """Periodic solver with Lagrangian multiplier method
    The current implementation is SLOW.
    To make it faster, linearization by hand is needed instead of using jvp. Future work.
    """
    print(f"Start timing, H_bar = \n{problem.H_bar}")
    start = time.time()

    p_splits = np.cumsum(np.array([len(x) for x in problem.p_node_inds_list_B])).tolist()
    d_splits = np.cumsum(np.array([len(x) for x in problem.node_inds_list])).tolist()
 
    num_dofs = problem.num_total_nodes * problem.vec
    p_lmbda_len = p_splits[-1]  
    d_lmbda_len = d_splits[-1]

    def operator_to_matrix(operator_fn):
        """Only used for debugging purpose.
        Can be used to print the matrix, check the conditional number, etc.
        """
        J = jax.jacfwd(operator_fn)(np.zeros(num_dofs + p_lmbda_len + d_lmbda_len))
        return J

    def get_Lagrangian():
        def split_lamda(lmbda):
            p_lmbda = lmbda[:p_lmbda_len]
            d_lmbda = lmbda[p_lmbda_len:]
            p_lmbda_split = np.split(p_lmbda, p_splits)
            d_lmbda_split = np.split(d_lmbda, d_splits)
            return p_lmbda_split, d_lmbda_split

        @jax.jit
        def Lagrangian_fn(aug_dofs):
            dofs, lmbda = aug_dofs[:num_dofs], aug_dofs[num_dofs:]
            sol = dofs.reshape((problem.num_total_nodes, problem.vec))
            lag_1 = problem.compute_energy(sol)

            p_lmbda_split, d_lmbda_split = split_lamda(lmbda)
            lag_2 = 0.
            for i in range(len(problem.p_node_inds_list_B)):
                lag_2 += np.sum(p_lmbda_split[i] * (sol[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]] - 
                                                    sol[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]]))
            for i in range(len(problem.node_inds_list)):
                lag_2 += np.sum(d_lmbda_split[i] * (sol[problem.node_inds_list[i], problem.vec_inds_list[i]] - problem.vals_list[i]))

            return lag_1 + 1e2*lag_2

        return Lagrangian_fn

    print(f"num_dofs = {num_dofs}, p_lmbda_len = {p_lmbda_len}, d_lmbda_len = {d_lmbda_len}")

    if initial_guess is not None:
        aug_dofs = np.hstack((initial_guess.reshape(-1), np.zeros(p_lmbda_len + d_lmbda_len)))
    else:
        aug_dofs = np.zeros(num_dofs + p_lmbda_len + d_lmbda_len)

    Lagrangian_fn = get_Lagrangian()
    A_fn = jax.grad(Lagrangian_fn)

    linear_solve_step = 0

    b = -A_fn(aug_dofs)
    res_val = np.linalg.norm(b)
    print(f"Before calling Newton's method, res l_2 = {res_val}") 
    tol = 1e-6
    while res_val > tol:
        A_fn_linear = get_A_fn_linear_fn(aug_dofs, A_fn)
        debug = False
        if debug:
            # Check onditional number of the matrix
            A_dense = operator_to_matrix(A_fn_linear)
            print(f"conditional number = {np.linalg.cond(A_dense)}")
            # print(f"max A = {np.max(A_dense)}")
            # print(A_dense.shape)
            # print(A_dense)
            inc = jax.numpy.linalg.solve(A_dense, b)
        else:
            inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab

        linear_solve_step += 1
        aug_dofs = aug_dofs + inc
        b = -A_fn(aug_dofs)
        res_val = np.linalg.norm(b)
        print(f"step = {linear_solve_step}, res l_2 = {res_val}") 

    sol = aug_dofs[:num_dofs].reshape((problem.num_total_nodes, problem.vec))
    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s], finished in {linear_solve_step} linear solve steps")

    return sol


def compute_periodic_inds(location_fns_A, location_fns_B, mappings, vecs, mesh):
    p_node_inds_list_A = []
    p_node_inds_list_B = []
    p_vec_inds_list = []
    for i in range(len(location_fns_A)):
        node_inds_A = np.argwhere(jax.vmap(location_fns_A[i])(mesh.points)).reshape(-1)
        node_inds_B = np.argwhere(jax.vmap(location_fns_B[i])(mesh.points)).reshape(-1)
        points_set_A = mesh.points[node_inds_A]
        points_set_B = mesh.points[node_inds_B]

        EPS = 1e-8
        node_inds_B_ordered = []
        for node_ind in node_inds_A:
            point_A = mesh.points[node_ind]
            dist = np.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
            node_ind_B_ordered = node_inds_B[np.argwhere(dist < EPS)].reshape(-1)
            node_inds_B_ordered.append(node_ind_B_ordered)

        node_inds_B_ordered = np.array(node_inds_B_ordered).reshape(-1)
        vec_inds = np.ones_like(node_inds_A, dtype=np.int32)*vecs[i]

        p_node_inds_list_A.append(node_inds_A)
        p_node_inds_list_B.append(node_inds_B_ordered)
        p_vec_inds_list.append(vec_inds)

        # TODO: A better way needed
        assert len(node_inds_A) == len(node_inds_B_ordered)

    return p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list


def rve_problem(problem_name='rve'):
    args.units_x = 1
    args.units_y = 1
    args.units_z = 1
    L = args.L
    meshio_mesh = box_mesh(args.num_hex*args.units_x, args.num_hex*args.units_y, args.num_hex*args.units_z,
                           L*args.units_x, L*args.units_y, L*args.units_z)
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

    periodic_bc_info = compute_periodic_inds(location_fns_A, location_fns_B, mappings, vecs, jax_mesh)
    problem = HyperElasticity(f"{problem_name}", jax_mesh, mode='rve', dirichlet_bc_info=dirichlet_bc_info, periodic_bc_info=periodic_bc_info)

    return problem


def exp():
    """Do not delete. We use this to generate RVE demo.
    """
    problem = rve_problem('rve_debug')
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
    jax_vtu_path = f"applications/fem/multi_scale/data/vtk/{problem.name}/sol_disp_ini.vtu"
    save_sol(problem, sol_disp_ini, jax_vtu_path, [("E", problem.E.reshape((problem.num_cells, problem.num_quads))[:, 0])])

    sol_fluc = aug_solve(problem)

    # ratios = [1.5, 1.8, 2.]
    # for ratio in ratios:
    #     problem.H_bar = ratio * H_bar
    #     sol_fluc = aug_solve(problem, initial_guess=sol_fluc)

    energy = problem.compute_energy(sol_fluc)
    print(f"Final energy = {energy}")

    sol_disp = problem.fluc_to_disp(sol_fluc)
    jax_vtu_path = f"applications/fem/multi_scale/data/vtk/{problem.name}/sol_disp.vtu"
    save_sol(problem, sol_disp, jax_vtu_path)

    jax_vtu_path = f"applications/fem/multi_scale/data/vtk/{problem.name}/sol_fluc.vtu"
    save_sol(problem, sol_fluc, jax_vtu_path)

    p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list = problem.periodic_bc_info
    a = sol_fluc[p_node_inds_list_A[0], p_vec_inds_list[0]]
    b = sol_fluc[p_node_inds_list_B[0], p_vec_inds_list[0]]
    ap = problem.mesh.points[p_node_inds_list_A[0]]
    bp = problem.mesh.points[p_node_inds_list_B[0]]
    print(np.hstack((ap, bp, a[:, None], b[:, None]))[:10])


def solve_rve_problem(problem, sample_H_bar):
    base_H_bar = flat_to_tensor(sample_H_bar)
    problem.H_bar = base_H_bar
    sol_fluc = aug_solve(problem)
    energy = problem.compute_energy(sol_fluc)
    ratios = [0.25, 0.5, 0.75, 0.9, 1.]
    if np.any(np.isnan(energy)):
        print(f"Solve with quasi-static steps...")
        sol_fluc = np.zeros((problem.num_total_nodes, problem.vec))
        for ratio in ratios:
            problem.H_bar = ratio * base_H_bar
            sol_fluc = aug_solve(problem, initial_guess=sol_fluc)
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
    problem = rve_problem()
    date = f"10102022"
    root_numpy = os.path.join(f"applications/fem/multi_scale/data/numpy/training", date)
    if not os.path.exists(root_numpy):
        os.makedirs(root_numpy)

    root_vtk = os.path.join(f"applications/fem/multi_scale/data/vtk/training", date)
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
