import jax
import jax.numpy as np
import numpy as onp
import os
import shutil

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'beam2d')
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

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def compressed_dirichlet_val(point):
        return -0.05 * Lx

    def location_midspan_bottom(point):
        return np.isclose(point[1], 0., atol=1e-5) & (point[0] > Lx / 2. - 2.) & (point[0] < Lx / 2. + 2.)

    location_fns = [location_midspan_bottom]
    dirichlet_bc_info = [
        [left] * 2 + [right] * 2,
        [0, 1, 0, 1],
        [zero_dirichlet_val] * 2 + [compressed_dirichlet_val, zero_dirichlet_val],
    ]

    solver_options = {'dynamic_relax': {'tol': 1e-8}}

    problem = HyperElasticity(
        mesh, vec=2, dim=2, ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns,
    )
    sol_list = solver(problem, solver_options)
    save_sol(problem.fes[0], np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))),
             os.path.join(OUTPUT_DIR, 'u.vtu'))

    # The aux problem is to verify that our "noise" (also used in the arc-length example) 
    # is actually very small.
    # The aux problem does not have any noise, but still converging to a similar configuration.
    problem_aux = HyperElasticityAux(
        mesh, vec=2, dim=2, ele_type=ele_type, 
        dirichlet_bc_info=dirichlet_bc_info,
    )
    sol_aux_list = solver(problem_aux, solver_options)
    save_sol(problem_aux.fes[0], np.hstack((sol_aux_list[0], np.zeros((len(sol_aux_list[0]), 1)))),
             os.path.join(OUTPUT_DIR, 'u_aux.vtu'))


if __name__ == '__main__':
    example()
