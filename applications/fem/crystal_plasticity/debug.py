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


def debug_problem():

    ele_type = 'tetrahedron'
    lag_order = 2

    # ele_type = 'hexahedron'
    # lag_order = 1

    Nx, Ny, Nz = 10, 10, 10
    Lx, Ly, Lz = 10., 10., 10.

    cell_type = get_meshio_cell_type(ele_type, lag_order)
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir, ele_type, lag_order)

    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, lag_order=lag_order, dirichlet_bc_info=[[],[],[]])

    tensor_map, update_int_vars_map = problem.get_maps()


    gss_initial = 60.8 
    num_slip_sys = 12
    slip_resistance_old = gss_initial*onp.ones(num_slip_sys)
    Fp_inv_old = onp.eye(problem.dim)


    u_grad = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.001]])
    S = tensor_map(u_grad, Fp_inv_old, slip_resistance_old)

    Fe = u_grad + np.eye(problem.dim)
    elastic_S = np.sum(problem.C * 1./2.*(Fe.T @ Fe - np.eye(problem.dim))[None, None, :, :], axis=(2, 3))

    print("\n")
    print(S)
    print(elastic_S)

    stresses = []
    strains = onp.linspace(0, 0.007, 11)
    for i in range(len(strains)):
        print(f"i = {i}")
        u_grad = onp.zeros((problem.dim, problem.dim))
        u_grad[2, 2] = strains[i]
        u_grad[0, 0] = -0.42*strains[i]
        u_grad[1, 1] = -0.42*strains[i]
        S = tensor_map(u_grad, Fp_inv_old, slip_resistance_old)
        stress = S[2, 2]
        Fp_inv_old, slip_resistance_old = update_int_vars_map(u_grad, Fp_inv_old, slip_resistance_old)
        print(f"Fp_inv_old = \n{Fp_inv_old}")
        print(f"slip_resistance_old = \n{slip_resistance_old}")
        stresses.append(stress)
    stresses = onp.array(stresses)
    print(stresses)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(strains, stresses, marker='o', markersize=10, linestyle='-', linewidth=2)


if __name__ == "__main__":
    debug_problem()
    plt.show()
