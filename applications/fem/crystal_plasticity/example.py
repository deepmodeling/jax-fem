import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt

from jax_am.fem.solver import solver
from jax_am.fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_am.fem.utils import save_sol

from applications.fem.crystal_plasticity.models import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, 'numpy')
vtk_dir = os.path.join(data_dir, 'vtk')


def problem():

    # ele_type = 'tetrahedron'
    # lag_order = 2

    ele_type = 'hexahedron'
    lag_order = 1

    Nx, Ny, Nz = 10, 10, 10
    Lx, Ly, Lz = 10., 10., 10.

    cell_type = get_meshio_cell_type(ele_type, lag_order)
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir, ele_type, lag_order)

    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    disps = np.linspace(0., 0.07, 11)
    # disps = np.linspace(0., 0.001, 2)

    # forces = np.hstack((np.linspace(0., 120, 4), np.linspace(120, 200, 21)))
    # ts = np.hstack((np.linspace(0., 0.3, 4), np.linspace(0.3, 1., 21)))

    forces = np.linspace(0., 200, 41)
    ts = np.linspace(0., 1., 41)

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


    def get_neumann_val(val):
        def neumann_val(point):
            return np.array([0., 0., val])
        return neumann_val


    neumann_bc_info = [[top], [get_neumann_val(forces[0])]]


    # dirichlet_bc_info = [[bottom, bottom, bottom, top, top, top], 
    #                      [0, 1, 2, 0, 1, 2], 
    #                      [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
    #                       zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]

    # dirichlet_bc_info = [[bottom, top], 
    #                      [2, 2], 
    #                      [zero_dirichlet_val, get_dirichlet_top(disps[0])]]

    # dirichlet_bc_info = [[bottom, line1, line2], 
    #                      [2, 0, 1], 
    #                      [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]



    dirichlet_bc_info = [[bottom, corner, corner], 
                         [2, 0, 1], 
                         [zero_dirichlet_val]*3]


    # dirichlet_bc_info = [[bottom, bottom, bottom], 
    #                      [2, 0, 1], 
    #                      [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]


    # problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, lag_order=lag_order, dirichlet_bc_info=dirichlet_bc_info)

    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, lag_order=lag_order, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)

    avg_stresses = []
    max_sol = []

    sol = np.zeros((problem.num_total_nodes, problem.vec))

    # for i, disp in enumerate(disps):
    #     print(f"\nStep {i} in {len(disps)}, disp = {disp}")


    for i in range(len(ts) - 1):
        problem.dt = ts[i + 1] - ts[i]
        print(f"\nStep {i + 1} in {len(ts) - 1}, force = {forces[i + 1]}, dt = {problem.dt}")

        # dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
        # problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)

        problem.neumann_bc_info = [[top], [get_neumann_val(forces[i + 1])]]
        problem.neumann = problem.compute_Neumann_integral()

        sol = solver(problem, linear=False, initial_guess=sol)
        max_sol.append(np.max(sol))
        problem.update_int_vars_gp(sol)
        avg_stress = problem.compute_avg_stress(sol)
        print(f"avg_stress = \n{avg_stress}")
        avg_stresses.append(avg_stress)
        vtk_path = os.path.join(data_dir, f'vtk/u_{i:03d}.vtu')
        save_sol(problem, sol, vtk_path, cell_type=cell_type)

    max_sol = np.array(max_sol)
    avg_stresses = np.array(avg_stresses)

    os.makedirs(numpy_dir, exist_ok=True)
    onp.save(os.path.join(numpy_dir, 'strains.npy'), disps)
    onp.save(os.path.join(numpy_dir, 'forces.npy'), forces)
    onp.save(os.path.join(numpy_dir, 'stress.npy'), avg_stresses)
    onp.save(os.path.join(numpy_dir, 'max_sol.npy'), max_sol)


def plot_stress_strain():
    strains = onp.load(os.path.join(numpy_dir, 'strains.npy'))
    forces = onp.load(os.path.join(numpy_dir, 'forces.npy'))
    stresses = onp.load(os.path.join(numpy_dir, 'stress.npy'))[:, 2, 2]
    max_sol = onp.load(os.path.join(numpy_dir, 'max_sol.npy'))

    abaqus_data = onp.genfromtxt(os.path.join(numpy_dir, 'abaqus.csv'), delimiter=',')

    fig = plt.figure()
    
    # plt.plot(strains, stresses, marker='o', markersize=10, linestyle='-', linewidth=2)
    plt.plot(abaqus_data[:, 0], abaqus_data[:, 1], color='blue', marker='o', markersize=10, linestyle='-', linewidth=2)
    plt.plot(max_sol/10., forces[1:], color='red', marker='o', markersize=10, linestyle='-', linewidth=2)


if __name__ == "__main__":
    # problem()
    plot_stress_strain()
    plt.show()
