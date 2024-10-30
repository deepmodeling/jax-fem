import jax
import jax.numpy as np
import numpy as onp
import os
import glob

from jax_fem.problem import Problem
from jax_fem.solver import solver, dynamic_relax_solve
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

output_dir = os.path.join(os.path.dirname(__file__), 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
onp.random.seed(0)


class HyperElasticity(Problem):
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
            # Some small noise to guide the dynamic relaxation solver
            return np.array([0., 1e-5])
        return [surface_map]


class HyperElasticityAux(Problem):
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

    def compressed_dirichlet_val(point):
        return -0.05*Lx

    def small_dirichlet_val(point):
        return 0.

    def middle(point):
        return np.isclose(point[1], 0., atol=1e-5) & (point[0] >= Lx/2. - 1.) & (point[0] <= Lx/2. + 1.)

    location_fns = [middle]

    dirichlet_bc_info = [[left]*2 + [right]*2, [0, 1, 0, 1], [zero_dirichlet_val]*2 + [compressed_dirichlet_val, small_dirichlet_val]]

    problem = HyperElasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    sol = dynamic_relax_solve(problem, tol=1e-8)
    vtk_path = os.path.join(output_dir, f'vtk/u.vtu')
    save_sol(problem.fes[0], np.hstack((sol, np.zeros((len(sol), 1)))), vtk_path)

    # The aux problem is to verify that our "noise" is actually very small.
    # The aux problem does not have any noise, but still converging to a similar configuration.
    problem_aux = HyperElasticityAux(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    sol_aux = dynamic_relax_solve(problem_aux, tol=1e-8, initial_guess=sol.reshape(-1))
    vtk_path_aux = os.path.join(output_dir, f'vtk/u_aux.vtu')
    save_sol(problem_aux.fes[0], np.hstack((sol_aux, np.zeros((len(sol), 1)))), vtk_path_aux)


if __name__ == "__main__":
    example()
