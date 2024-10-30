import jax
import jax.numpy as np
import os
import glob

from jax_fem.problem import Problem
from jax_fem.solver import solver, arc_length_solver_force_driven, get_q_vec
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

output_dir = os.path.join(os.path.dirname(__file__), 'output')
vtk_dir = os.path.join(output_dir, 'vtk')


class HyperElasticityMain(Problem):
    def get_tensor_map(self):
        def psi(F):
            E = 1e3
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 2.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 2.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x):
            # Some small noise to guide the arc-length solver
            return np.array([0., 1e-5])
        return [surface_map]


class HyperElasticityAux(Problem):
    def get_tensor_map(self):
        def first_PK_stress(u_grad):
            return np.zeros((self.dim, self.dim))
        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([10., 0])
        return [surface_map]


def example():
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 50., 2.

    meshio_mesh = rectangle_mesh(Nx=50, Ny=2, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def middle(point):
        return np.isclose(point[1], 0., atol=1e-5) & (point[0] > Lx/2. - 2.) & (point[0] < Lx/2. + 2.)

    location_fns_middle = [middle]

    dirichlet_bc_info = [[left]*2, [0, 1], [zero_dirichlet_val]*2]

    location_fns_right = [right]

    problem_main = HyperElasticityMain(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns_middle)
    problem_aux = HyperElasticityAux(mesh, vec=2, dim=2, ele_type=ele_type, location_fns=location_fns_right)

    q_vec = get_q_vec(problem_aux)
    u_vec = np.zeros(problem_main.num_total_dofs_all_vars)
    lamda = 0.
    Delta_u_vec_dir = np.zeros(problem_main.num_total_dofs_all_vars)
    Delta_lamda_dir = 0.

    for i in range(500):
        print(f"\n\nStep {i}, lamda = {lamda}")
        if i < 200:
            Delta_l = 0.1
            psi = 0.5
        else:
            Delta_l = 1.
            psi = 0.5

        u_vec, lamda, Delta_u_vec_dir, Delta_lamda_dir = arc_length_solver_force_driven(problem_main, u_vec, 
            lamda, Delta_u_vec_dir, Delta_lamda_dir, q_vec, Delta_l, psi)
        sol_list = problem_main.unflatten_fn_sol_list(u_vec)
        if i % 10 == 0:
            vtk_path = os.path.join(vtk_dir, f'u{i:05d}.vtu')
            sol = sol_list[0]
            save_sol(problem_main.fes[0], np.hstack((sol, np.zeros((len(sol), 1)))), vtk_path)

        if lamda > 1.:
            break


if __name__ == "__main__":
    example()
