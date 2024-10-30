import jax
import jax.numpy as np
import os
import glob
import meshio

from jax_fem.problem import Problem
from jax_fem.solver import solver, dynamic_relax_solve
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh


class Elasticity(Problem):
    def get_tensor_map(self):
        def psi(F_2d):
            F = np.array([[F_2d[0, 0], F_2d[0, 1], 0.], 
                          [F_2d[1, 0], F_2d[1, 1], 0.],
                          [0., 0., 1.]])
            E = 70.e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            Jinv = J**(-2./3.)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        return first_PK_stress
 

def simulation():
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    files = glob.glob(os.path.join(output_dir, f'vtk/*'))
    for f in files:
        os.remove(f)

    ele_type = 'TRI6'
    cell_type = get_meshio_cell_type(ele_type)

    meshio_mesh = meshio.read(os.path.join(input_dir, f"abaqus/cellular_solid.inp"))
    meshio_mesh.points[:, 0] -= np.min(meshio_mesh.points[:, 0])
    meshio_mesh.points[:, 1] -= np.min(meshio_mesh.points[:, 1])
    meshio_mesh.write(os.path.join(output_dir, 'vtk/mesh.vtu'))

    Lx, Ly = np.max(meshio_mesh.points[:, 0]), np.max(meshio_mesh.points[:, 1])
    print(f"Lx={Lx}, Ly={Ly}")
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def dirichlet_val_bottom(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn

    disps = -0.15*Ly*np.linspace(1., 1., 1)

    location_fns = [bottom, bottom, top, top]
    value_fns = [dirichlet_val_bottom, dirichlet_val_bottom, dirichlet_val_bottom, get_dirichlet_top(disps[0])]
    vecs = [0, 1, 0, 1]

    dirichlet_bc_info = [location_fns, vecs, value_fns]

    problem = Elasticity(mesh, ele_type=ele_type, vec=2, dim=2, dirichlet_bc_info=dirichlet_bc_info)

    for i, disp in enumerate(disps):
        print(f"\nStep {i + 1} in {len(disps)}, disp = {disp}")
        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
        problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        sol = dynamic_relax_solve(problem, tol=1e-6)
        vtk_path = os.path.join(output_dir, f'vtk/u_{i + 1:03d}.vtu')
        save_sol(problem.fes[0], np.hstack((sol, np.zeros((len(sol), 1)))), vtk_path)


if __name__=="__main__":
    simulation()
