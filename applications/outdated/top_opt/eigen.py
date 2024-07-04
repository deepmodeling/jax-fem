import numpy as onp
import scipy
import jax
import jax.numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
import os
import glob
import os
import meshio
import time

from jax_fem.core import FEM
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import ad_wrapper, get_A_fn, get_flatten_fn, apply_bc
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import rectangle_mesh, box_mesh_gmsh

from applications.fem.top_opt.fem_model import Elasticity
from applications.fem.top_opt.mma import optimize


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
 

class Stiffness(FEM):
    def get_tensor_map(self):
        def stress(u_grad, theta):
            Emax = 70.e9
            Emin = 1e-3*Emax
            nu = 0.3
            penal = 5.
            E = Emin + (Emax - Emin)*theta[0]**penal
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
        thetas = np.repeat(params[:, None, :], self.num_quads, axis=1)
        self.internal_vars['laplace'] = [thetas]


class Mass(FEM):
    def get_mass_map(self):
        def mass_map(u, theta):
            density = 2.7e3
            return theta[0]*density*u
        return mass_map

    def set_params(self, params):
        thetas = np.repeat(params[:, None, :], self.num_quads, axis=1)
        self.internal_vars['mass'] = [thetas]


def topology_optimization():
    p_name = 'eigen'
    problem_name = p_name + '_w_cstr'
    data_path = os.path.join(os.path.dirname(__file__), 'data') 

    files1 = glob.glob(os.path.join(data_path, f'vtk/{problem_name}-TO/*'))
    files2 = glob.glob(os.path.join(data_path, f'vtk/{problem_name}-modes/*'))
    for f in files1 + files2:
        os.remove(f)

    L = 6
    W = 3
    N_L = 60
    N_W = 30
    meshio_mesh = rectangle_mesh(N_L, N_W, L, W)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'])

    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load_location(point):
        return np.logical_and(np.isclose(point[0], L, atol=1e-5), np.isclose(point[1], 0., atol=0.1*W+1e-5))

    def dirichlet_val(point):
        return 0.

    def neumann_val(point):
        return np.array([0., -1.e6])

    dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
    neumann_bc_info = [[load_location], [neumann_val]]
    problem = Elasticity(jax_mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=dirichlet_bc_info, 
        neumann_bc_info=neumann_bc_info, additional_info=(p_name,))
    fwd_pred = ad_wrapper(problem, linear=True, use_petsc=True)

    problem_K = Stiffness(jax_mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=None)
    problem_M = Mass(jax_mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=None)


    def eigen_analysis(params):
        sol = np.zeros((problem_K.num_total_nodes, problem_K.vec)) 
        
        def eigenval_grad_fn(problem):
            def vAv_fn(params, v):
                problem.set_params(params)
                res_fn = problem.compute_residual
                res_fn = get_flatten_fn(res_fn, problem)
                res_fn = apply_bc(res_fn, problem)
                return np.sum(v*res_fn(v.reshape(-1)))
            return vAv_fn

        vKv_fn = eigenval_grad_fn(problem_K)
        vMv_fn = eigenval_grad_fn(problem_M)

        problem_K.set_params(params)
        problem_K.newton_update(sol)
        K = get_A_fn(problem_K, use_petsc=True)

        problem_M.set_params(params)
        problem_M.newton_update(sol)
        M = get_A_fn(problem_M, use_petsc=True)

        Print = PETSc.Sys.Print

        # Create the results vectors
        xr, xi = K.createVecs()

        # Shift-and-invert
        F = SLEPc.ST().create()
        F.setType(F.Type.SINVERT)

        # Setup the eigensolver
        E = SLEPc.EPS().create()
        E.setST(F)
        E.setOperators(K, M)
        n_requested = 10
        E.setDimensions(n_requested)
        E.setTarget(10)
        E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        E.setTolerances(tol=1e-20, max_it=1000)

        E.setFromOptions()

        # Solve the eigensystem
        E.solve()

        Print("")
        its = E.getIterationNumber()
        Print("Number of iterations of the method: %i" % its)
        sol_type = E.getType()
        Print("Solution method: %s" % sol_type)
        nev, ncv, mpd = E.getDimensions()
        Print("Number of requested eigenvalues: %i" % nev)
        tol, maxit = E.getTolerances()
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
        nconv = E.getConverged()
        Print("Number of converged eigenpairs: %d" % nconv)
        eigen_vals = []
        d_eigen_vals = []

        if nconv > 0:
            Print("")
            Print("        k          ||Ax-kx||/||kx|| ")
            Print("----------------- ------------------")
            for i in range(n_requested):
                k = E.getEigenpair(i, xr, xi)
                error = E.computeError(i)
                if k.imag != 0.0:
                    print(f"### Warning: imaginary part exit!")
                    Print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
                else:
                    Print(" %12f       %12g" % (k.real, error))

                eigen_val = k.real
                eigen_vec = xr.getArray()

                if i > 2 and i < 6:
                    eigen_vals.append(eigen_val)
                    d_eigen_val = (jax.grad(vKv_fn)(params, eigen_vec) - eigen_val*jax.grad(vMv_fn)(params, eigen_vec))/vMv_fn(params, eigen_vec)
                    d_eigen_vals.append(d_eigen_val)

                vtu_path = os.path.join(data_path, f'vtk/{problem_name}-modes/sol_{i:03d}.vtu')
                sol3D = np.hstack((eigen_vec.reshape(sol.shape), np.zeros((len(sol), 1))))
                save_sol(problem_K, sol3D, vtu_path)

        eigen_vals = np.array(eigen_vals)
        d_eigen_vals = np.stack(d_eigen_vals)

        return eigen_vals, d_eigen_vals


    def J_fn(dofs, params):
        """J(u, p)
        """
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        compliance = problem.compute_compliance(neumann_val, sol)
        return compliance

    def J_total(params):
        """J(u(p), p)
        """     
        sol = fwd_pred(params)
        dofs = sol.reshape(-1)
        obj_val = J_fn(dofs, params)
        return obj_val

    outputs = []
    e_vals = []
    def output_sol(params, obj_val):
        print(f"\nOutput solution - need to solve the forward problem again...")
        sol = fwd_pred(params)
        vtu_path = os.path.join(data_path, f'vtk/{problem_name}-TO/sol_{output_sol.counter:03d}.vtu')
        save_sol(problem, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path, cell_infos=[('theta', problem.full_params[:, 0])])

        eigen_vals, _ = eigen_analysis(params)
        e_vals.append(eigen_vals)
        print(f"eigen vals = {eigen_vals}")
        print(f"compliance = {obj_val}")
        print(f"max theta = {np.max(params)}, min theta = {np.min(params)}, mean theta = {np.mean(params)}")
        outputs.append(obj_val)
        output_sol.counter += 1

    output_sol.counter = 0
        
    vf = 0.5
    min_freq = 1e6

    def objectiveHandle(rho):
        J, dJ = jax.value_and_grad(J_total)(rho)
        output_sol(rho, J)
        return J, dJ

    def computeConstraints(rho, epoch):
        def computeGlobalVolumeConstraint(rho):
            g1 = np.mean(rho)/vf - 1.
            return g1

        c1, gradc1 = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
        c1, gradc1 = c1.reshape((1,)), gradc1[None, ...]

        def min_eigen_vals_cstr(eigen_vals):
            alpha = 10.
            g2 = -1./alpha*jax.scipy.special.logsumexp(-alpha*eigen_vals)
            return 1. - g2/min_freq
            # If without constraint, adopt the following line.
            # return 0.

        eigen_vals, d_eigen_vals = eigen_analysis(rho)
        c2 = min_eigen_vals_cstr(eigen_vals)

        print(f"c2 = {c2}")

        gradc2 = np.einsum('ij,jkl->ikl', jax.grad(min_eigen_vals_cstr)(eigen_vals)[None, :], d_eigen_vals)
        print(f"min eigen val = {np.min(eigen_vals)}")

        c = np.hstack((c1, c2))
        gradc = np.concatenate((gradc1, gradc2), axis=0)

        return c, gradc

    optimizationParams = {'maxIters':101, 'minIters':101, 'relTol':0.05}
    rho_ini = vf*np.ones((len(problem.flex_inds), 1))

    optimize(problem, rho_ini, optimizationParams, objectiveHandle, computeConstraints, numConstraints=2, movelimit=0.2)
    onp.save(os.path.join(data_path, f"numpy/{problem_name}_outputs.npy"), onp.array(outputs))
    onp.save(os.path.join(data_path, f"numpy/{problem_name}_eigen_vals.npy"), onp.stack(e_vals))
    print(f"Compliance = {J_total(np.ones((len(problem.flex_inds), 1)))} for full material")


if __name__=="__main__":
    topology_optimization()
