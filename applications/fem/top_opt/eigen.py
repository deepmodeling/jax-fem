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

from jax_am.fem.core import FEM
from jax_am.fem.generate_mesh import Mesh
from jax_am.fem.solver import ad_wrapper, get_A_fn
from jax_am.fem.utils import save_sol
from jax_am.common import rectangle_mesh, box_mesh

from applications.fem.top_opt.fem_model import Elasticity
from applications.fem.top_opt.mma import optimize


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
 


def check_mesh_TET4(points, cells):

    def quality(pts):
        p1, p2, p3, p4 = pts
        v1 = p2 - p1
        v2 = p3 - p1
        v12 = np.cross(v1, v2)
        v3 = p4 - p1
        return np.dot(v12, v3)

    qlts = jax.vmap(quality)(points[cells])

    return qlts
 


class Stiffness(FEM):
    def get_tensor_map(self):
        def stress(u_grad, theta):
            Emax = 70.e9
            Emin = 1e-3*Emax
            nu = 0.23
            penal = 3.
            E = Emin + (Emax - Emin)*theta[0]**penal
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    def set_params(self, params):
        thetas = np.repeat(params[:, None, :], self.num_quads, axis=1)
        self.internal_vars['laplace'] = [thetas]


class Mass(FEM):
    def get_mass_map(self):
        def mass_map(u, theta):
            density = 2.5e3
            return theta[0]*density*u
        return mass_map

    def set_params(self, params):
        thetas = np.repeat(params[:, None, :], self.num_quads, axis=1)
        self.internal_vars['mass'] = [thetas]


def topology_optimization():
    problem_name = 'eigen'
    data_path = os.path.join(os.path.dirname(__file__), 'data') 

    files = glob.glob(os.path.join(data_path, f'vtk/{problem_name}/*'))
    for f in files:
        os.remove(f)

    # L = 0.6
    # W = 0.3
    # N_L = 60
    # N_W = 30
    # meshio_mesh = rectangle_mesh(N_L, N_W, L, W)
    # jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'])


    # def fixed_location(point):
    #     return np.isclose(point[0], 0., atol=1e-5)
        
    # def load_location(point):
    #     return np.logical_and(np.isclose(point[0], L, atol=1e-5), np.isclose(point[1], 0., atol=0.01+1e-5))

    # def dirichlet_val(point):
    #     return 0.

    # def neumann_val(point):
    #     return np.array([0., -1.e6])

    # dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
    # neumann_bc_info = [[load_location], [neumann_val]]
    # problem = Elasticity(jax_mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=dirichlet_bc_info, 
    #     neumann_bc_info=neumann_bc_info, additional_info=(problem_name,))
    # fwd_pred = ad_wrapper(problem, linear=True, use_petsc=True)


    # problem_K = Elasticity(jax_mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=None, additional_info=(problem_name,))
    # problem_M = Mass(jax_mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=None)


    # Lx, Ly, Lz = 1., 1., 0.01
    # Nx, Ny, Nz = 20, 20, 2

    # Lx, Ly, Lz = 10., 1., 1
    # Nx, Ny, Nz = 50, 5, 5

    # meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    # jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])


    meshio_mesh = meshio.read(os.path.join(data_path, f'xdmf/mesh000000.vtu'))

    meshio_mesh.write(os.path.join(data_path, f'xdmf/jax_mesh.vtu'))

    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['tetra'])

    qlts = check_mesh_TET4(jax_mesh.points, jax_mesh.cells)

    for i in range(len(qlts)):
        if qlts[i] < 0:
            tmp = jax_mesh.cells[i][1]
            jax_mesh.cells[i][1] = jax_mesh.cells[i][2]
            jax_mesh.cells[i][2] = tmp

 
    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[fixed_location]*3, [0, 1, 2], [dirichlet_val]*3]

    # problem_K = Stiffness(jax_mesh, vec=3, dim=3, ele_type='HEX8', dirichlet_bc_info=None)
    # problem_M = Mass(jax_mesh, vec=3, dim=3, ele_type='HEX8', dirichlet_bc_info=None)


    problem_K = Stiffness(jax_mesh, vec=3, dim=3, ele_type='TET4', dirichlet_bc_info=None)
    problem_M = Mass(jax_mesh, vec=3, dim=3, ele_type='TET4', dirichlet_bc_info=None)

 
    def eigen_analysis(params):
        sol = np.zeros((problem_K.num_total_nodes, problem_K.vec)) 

        problem_K.set_params(params)
        problem_K.newton_update(sol)
        K = get_A_fn(problem_K, use_petsc=True)


        problem_M.set_params(params)
        problem_M.newton_update(sol)
        M = get_A_fn(problem_M, use_petsc=True)

        K_scipy = problem_K.A_sp_scipy
        M_scipy = problem_M.A_sp_scipy
        print(K_scipy.todense()[:5, :5])
        print(M_scipy.todense()[:5, :5])
        print(M_scipy.todense().shape)
        print(np.max(M_scipy.todense()[:3]))    
        # vals, vecs = scipy.sparse.linalg.eigs(K_scipy, k=10, M=M_scipy, sigma=None, which='SM')
        # print(vals)
        # exit()

        Print = PETSc.Sys.Print

        # Create the results vectors
        xr, xi = K.createVecs()

        # pc = PETSc.PC().create()
        # # pc.setType(pc.Type.HYPRE)
        # pc.setType(pc.Type.BJACOBI)
        
        # ksp = PETSc.KSP().create()
        # ksp.setType(ksp.Type.PREONLY)
        # ksp.setPC(pc)
        
        F = SLEPc.ST().create()
        F.setType(F.Type.SINVERT)
        # F.setKSP(ksp)
        # F.setShift(100)

        # Setup the eigensolver
        E = SLEPc.EPS().create()
        E.setST(F)
        E.setOperators(K, M)
        # E.setType(E.Type.POWER) # LOBPCG POWER
        # E.setDimensions(10, PETSc.DECIDE)
        n_requested = 12
        E.setDimensions(n_requested)
        E.setTarget(100.)
        E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)  # TARGET_MAGNITUDE SMALLEST_MAGNITUDE
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


                vtu_path = os.path.join(data_path, f'vtk/{problem_name}/sol_{i:03d}.vtu')
                save_sol(problem_K, xr.getArray().reshape(sol.shape), vtu_path, cell_type='tetra')


        exit()



















        # E = SLEPc.EPS()
        # E.create()
        # E.setOperators(K, M)
        # E.setProblemType(SLEPc.EPS.ProblemType.GNHEP) # HEP # https://slepc.upv.es/slepc4py-current/docs/apiref/index.html
        # E.setDimensions(nev=10) # Number of eigenvalues to compute
        # E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE) # Take smallest eigenvalues

        # E.setFromOptions()

        # E.solve()

        # Print()
        # Print("******************************")
        # Print("*** SLEPc Solution Results ***")
        # Print("******************************")
        # Print()

        # its = E.getIterationNumber()
        # Print("Number of iterations of the method: %d" % its)

        # eps_type = E.getType()

        # Print("Solution method: %s" % eps_type)

        # nev, ncv, mpd = E.getDimensions()
        # Print("Number of requested eigenvalues: %d" % nev)

        # tol, maxit = E.getTolerances()
        # Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

        # nconv = E.getConverged()
        # Print("Number of converged eigenpairs %d" % nconv)

        # if nconv > 0:
        #     # Create the results vectors
        #     vr, wr = K.getVecs()
        #     vi, wi = K.getVecs()

        #     Print()
        #     Print("        k          ||Ax-kx||/||kx|| ")
        #     Print("----------------- ------------------")
        #     for i in range(nconv):
        #         k = E.getEigenpair(i, vr, vi)
        #         error = E.computeError(i)
        #         if k.imag != 0.0:
        #             Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
        #         else:
        #             Print(" %12f      %12g" % (k.real, error))
        #     Print()


    params = np.ones((len(problem_K.cells), 1))

    eigen_analysis(params)


    exit()



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
    def output_sol(params, obj_val):
        print(f"\nOutput solution - need to solve the forward problem again...")
        sol = fwd_pred(params)
        vtu_path = os.path.join(data_path, f'vtk/{problem_name}/sol_{output_sol.counter:03d}.vtu')
        save_sol(problem, sol, vtu_path, cell_infos=[('theta', problem.full_params[:, 0])], cell_type='quad')
        print(f"compliance = {obj_val}")
        print(f"max theta = {np.max(params)}, min theta = {np.min(params)}, mean theta = {np.mean(params)}")
        outputs.append(obj_val)
        output_sol.counter += 1

    output_sol.counter = 0
        
    vf = 0.5

    def objectiveHandle(rho):
        J, dJ = jax.value_and_grad(J_total)(rho)
        output_sol(rho, J)
        return J, dJ

    def computeConstraints(rho, epoch):
        def computeGlobalVolumeConstraint(rho):
            g = np.mean(rho)/vf - 1.
            return g
        c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
        c, gradc = c.reshape((1,)), gradc[None, ...]
        return c, gradc

    optimizationParams = {'maxIters':51, 'minIters':51, 'relTol':0.05}
    rho_ini = vf*np.ones((len(problem.flex_inds), 1))
    optimize(problem, rho_ini, optimizationParams, objectiveHandle, computeConstraints, numConstraints=1, movelimit=0.2)
    onp.save(os.path.join(data_path, f"numpy/{problem_name}_outputs.npy"), onp.array(outputs))
    print(f"Compliance = {J_total(np.ones((len(problem.flex_inds), 1)))} for full material")





    # A = get_A_fn(problem, use_petsc=True)

if __name__=="__main__":
    topology_optimization()


