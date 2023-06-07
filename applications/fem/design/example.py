import jax
import jax.numpy as np
import os
import glob
import meshio

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver, DynamicRelaxSolve
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.common import rectangle_mesh


class Elasticity(FEM):
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

    # def get_tensor_map(self):
    #     """Override base class method.
    #     """
    #     def stress(u_grad, theta=1.):
    #         # Plane stress assumption
    #         # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
    #         Emax = 70.e3
    #         Emin = 1e-3*Emax
    #         nu = 0.3
    #         penal = 3.
    #         E = Emin + (Emax - Emin)*theta**penal
    #         epsilon = 0.5*(u_grad + u_grad.T)
    #         eps11 = epsilon[0, 0]
    #         eps22 = epsilon[1, 1]
    #         eps12 = epsilon[0, 1]
    #         sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
    #         sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
    #         sig12 = E/(1 + nu)*eps12
    #         sigma = np.array([[sig11, sig12], [sig12, sig22]])
    #         return sigma
    #     return stress

def simulation():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    files = glob.glob(os.path.join(data_dir, f'vtk/*'))
    for f in files:
        os.remove(f)

    ele_type = 'QUAD8'
    cell_type = get_meshio_cell_type(ele_type)

    mesh_file = os.path.join(data_dir, f"abaqus/beam.inp")
    meshio_mesh = meshio.read(mesh_file)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    Lx, Ly = np.max(meshio_mesh.points[:, 0]), np.max(meshio_mesh.points[:, 1])
    print(f"Lx={Lx}, Ly={Ly}")

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)



    def left_corner(point):
        return np.logical_and(np.isclose(point[0], 0., atol=1e-5), np.isclose(point[1], 0., atol=1e-5))

    def right_corner(point):
        return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=1e-5))


    def left_dirichlet_val_x1(point):
        return 0.

    def left_dirichlet_val_x2(point):
        return 0.

    def right_dirichlet_val_x1(point):
        return -0.01*Lx

    def right_dirichlet_val_x2(point):
        return 0.
 
    def body_force(point):
        return np.array([0., 0.])

    dirichlet_bc_info = [[left_corner]*2 + [right_corner]*2, 
                         [0, 1]*2, 
                         [left_dirichlet_val_x1, left_dirichlet_val_x2, right_dirichlet_val_x1, right_dirichlet_val_x2]]

    problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, source_info=body_force)
    sol = np.zeros((problem.num_total_nodes, problem.vec))
    sol = DynamicRelaxSolve(problem, sol)
    # sol = solver(problem, initial_guess=None, linear=False, use_petsc=True)
    vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
    save_sol(problem, np.hstack((sol, np.zeros((len(sol), 1)))), vtk_path)
    exit()



    # def left_dirichlet_val_x1(point):
    #     return 0.

    # def left_dirichlet_val_x2(point):
    #     return 0.
     
    # def right_dirichlet_val_x1(point):
    #     return -0.2*Lx

    # def right_dirichlet_val_x2(point):
    #     return 0.*Ly

    # dirichlet_bc_info = [[left]*2 + [right]*2, 
    #                      [0, 1]*2, 
    #                      [left_dirichlet_val_x1, left_dirichlet_val_x2, right_dirichlet_val_x1, right_dirichlet_val_x2]]

    # problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    # sol = np.zeros((problem.num_total_nodes, problem.vec))
    # # sol = DynamicRelaxSolve(problem, sol)
    # sol = solver(problem, initial_guess=None, linear=False, use_petsc=True)
    # vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
    # save_sol(problem, np.hstack((sol, np.zeros((len(sol), 1)))), vtk_path)
    # exit()


    def dirichlet_val(point):
        return 0.

 
    # rs = np.linspace(0.2, 1., 5)
    rs = np.linspace(0.2, 1., 10)

    # def get_neumann_val(r):
    #     def neumann_val(point):
    #         return r*np.array([1e3, 1e2])
    #     return neumann_val


    def get_neumann_val(r):
        def neumann_val(point):
            return -r*np.array([1e3, 1e2])
        return neumann_val



    dirichlet_bc_info = [[left]*2, [0, 1], [dirichlet_val]*2]
    neumann_bc_info = [[right], [get_neumann_val(0.)]]

    problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)

    sol = np.zeros((problem.num_total_nodes, problem.vec))
    for i in range(len(rs)):
        print(f"\nStep {i + 1} in {len(rs)}")
        problem.neumann_value_fns = [get_neumann_val(rs[i])]

        sol = solver(problem, initial_guess=sol, linear=False, use_petsc=True)

        # sol = DynamicRelaxSolve(problem, sol)

        vtk_path = os.path.join(data_dir, f'vtk/u_{i:03d}.vtu')
        save_sol(problem, np.hstack((sol, np.zeros((len(sol), 1)))), vtk_path)

        # exit()

if __name__=="__main__":
    simulation()
