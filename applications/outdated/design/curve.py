import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import meshio
import matplotlib.pyplot as plt

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.mma import optimize


class Elasticity(Problem):
    def custom_init(self):
        """Override base class method.
        Modify self.flex_inds so that location-specific TO can be realized.
        """
        self.fes[0].flex_inds = np.arange(len(self.fes[0].cells))

    def get_tensor_map(self):
        """Override base class method.
        """
        def stress(u_grad, theta=1.):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            Emax = 70.e3
            Emin = 1e-3*Emax
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin)*theta**penal
            epsilon = 0.5*(u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
            sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
            sig12 = E/(1 + nu)*eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
            return sigma
        return stress

    def set_params(self, params):
        """Override base class method.
        """
        full_params = np.ones((self.fes[0].num_cells, params.shape[1]))
        full_params = full_params.at[self.fes[0].flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fes[0].num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]


def simulation():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    input_dir = os.path.join(os.path.dirname(__file__), 'input')

    files = glob.glob(os.path.join(output_dir, f'vtk/*'))
    for f in files:
        os.remove(f)

    ele_type = 'QUAD8'
    cell_type = get_meshio_cell_type(ele_type)

    mesh_file = os.path.join(input_dir, f"abaqus/beam.inp")
    meshio_mesh = meshio.read(mesh_file)
    Lx, Ly = np.max(meshio_mesh.points[:, 0]), np.max(meshio_mesh.points[:, 1])
    meshio_mesh.points[:, 1] -= 0.4*Ly
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    print(f"Lx={Lx}, Ly={Ly}")


    def left_corner(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right_corner(point):
        return np.isclose(point[0], Lx, atol=1e-5)


    # def left_corner(point):
    #     return np.logical_and(np.isclose(point[0], 0., atol=1e-5), np.isclose(point[1], 0., atol=1e-5))

    # def right_corner(point):
    #     return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=1e-5))


    # def left_corner(point):
    #     return np.logical_and(np.isclose(point[0], 0., atol=1e-5), np.isclose(point[1], 0.1, atol=0.11))

    # def right_corner(point):
    #     return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0.1, atol=0.11))


    def left_dirichlet_val_x1(point):
        return 0.

    def left_dirichlet_val_x2(point):
        return 0.

    def right_dirichlet_val_x1(point):
        return -0.01*Lx

    def right_dirichlet_val_x2(point):
        return 0.
 
    dirichlet_bc_info = [[left_corner]*2 + [right_corner]*2, 
                         [0, 1]*2, 
                         [left_dirichlet_val_x1, left_dirichlet_val_x2, right_dirichlet_val_x1, right_dirichlet_val_x2]]

    problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    fwd_pred = ad_wrapper(problem, linear=True, use_petsc=True)

    theta = 0.5*onp.ones((problem.num_cells, 1))
    cell_centroids = np.mean(np.take(problem.fes[0].points, problem.fes[0].cells, axis=0), axis=1)

    theta[(cell_centroids[:, 0] > 0.5*Lx) & (cell_centroids[:, 1] > 0)] = 0.
    theta[(cell_centroids[:, 0] < 0.5*Lx) & (cell_centroids[:, 1] < 0)] = 0.
    theta[(cell_centroids[:, 0] > 0.5*Lx) & (cell_centroids[:, 1] < 0)] = 1.
    theta[(cell_centroids[:, 0] < 0.5*Lx) & (cell_centroids[:, 1] > 0)] = 1.

    sol_list = fwd_pred(theta)
    vtk_path = os.path.join(output_dir, f'vtk/u.vtu')
    save_sol(problem.fes[0], np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
        vtk_path, cell_infos=[('theta', problem.full_params[:, 0])])

    inds = onp.where(onp.isclose(mesh.points[:, 1], 0, atol=1e-5))
    target_y_vals = 0.15*np.sin(2*np.pi/Lx*mesh.points[inds, 0])


    def J_total(params):
        """J(u(theta), theta)
        """     
        sol_list = fwd_pred(params)
        error = np.sum((sol_list[0][inds, 1] - target_y_vals)**2)
        return error

    outputs = []
    def output_sol(params, obj_val):
        print(f"\nOutput solution - need to solve the forward problem again...")
        sol_list = fwd_pred(params)
        vtu_path = os.path.join(output_dir, f'vtk/sol_{output_sol.counter:03d}.vtu')
        save_sol(problem.fes[0], np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
            vtu_path, cell_infos=[('theta', problem.full_params[:, 0])])
        print(f"Objective = {obj_val} at step {output_sol.counter:03d}")
        outputs.append(obj_val)
        output_sol.counter += 1
    output_sol.counter = 0

    def objectiveHandle(rho):
        """MMA solver requires (J, dJ) as inputs
        J has shape ()
        dJ has shape (...) = rho.shape
        """
        J, dJ = jax.value_and_grad(J_total)(rho)
        output_sol(rho, J)
        return J, dJ

    def consHandle(rho, epoch):
        """MMA solver requires (c, dc) as inputs
        c should have shape (numConstraints,)
        gradc should have shape (numConstraints, ...)
        """
        def computeGlobalVolumeConstraint(rho):
            g = np.mean(rho)/vf - 1.
            return g
        c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
        c, gradc = c.reshape((1,)), gradc[None, ...]
        return c, gradc

    vf = 1.
    optimizationParams = {'maxIters':21, 'movelimit':0.01}
    rho_ini = 0.5*np.ones((len(problem.fes[0].flex_inds), 1))
    numConstraints = 1
    optimize(problem.fes[0], rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints)
    print(f"Objevtive values: {onp.array(outputs)}")


if __name__=="__main__":
    simulation()
