import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import os
import meshio
import time

from jax_am.fem.generate_mesh import Mesh, box_mesh
from jax_am.fem.solver import ad_wrapper
from jax_am.fem.utils import save_sol

from applications.fem.top_opt.fem_model import Elasticity
from applications.fem.top_opt.mma import optimize



os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Plasticity(Elasticity):
    def custom_init(self):
        self.cell_centroids = onp.mean(onp.take(self.points, self.cells, axis=0), axis=1)
        self.flex_inds = np.arange(len(self.cells))

    def get_tensor_map(self):
        _, stress_return_map = self.get_maps()
        return stress_return_map

    def get_maps(self):
        def safe_sqrt(x):  
            safe_x = np.where(x > 0., np.sqrt(x), 0.)
            return safe_x

        def safe_divide(x, y):
            return np.where(y == 0., 0., x/y)

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def stress(epsilon, E):
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma

        def stress_return_map(u_grad, sigma_old, epsilon_old, theta):
            theta1, theta2 = theta
            Emax = 70.e3
            Emin = 70.
            penal = 3.
            E = Emin + (Emax - Emin)*theta1**penal

            sig0 = 250. + theta2*100
            # sig0 = 250. + 0.25*100

            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc, E) + sigma_old
            s_dev = sigma_trial - 1./self.dim*np.trace(sigma_trial)*np.eye(self.dim)
            s_norm = safe_sqrt(3./2.*np.sum(s_dev*s_dev))
            f_yield = s_norm - sig0
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            sigma = sigma_trial - safe_divide(f_yield_plus*s_dev, s_norm)
            return sigma

        return strain, stress_return_map

    def stress_strain_fns(self):
        strain, stress_return_map = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol, params):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        sigmas_old, epsilons_old, thetas = params
        sigmas_update = vmap_stress_rm(u_grads, sigmas_old, epsilons_old, thetas)
        epsilons_update = vmap_strain(u_grads)
        return sigmas_update, epsilons_update, thetas

    def set_params(self, params):
        self.internal_vars = {'laplace': params}

    def init_params(self, theta):
        epsilons_old = onp.zeros((len(self.cells), self.num_quads, self.vec, self.dim))
        sigmas_old = onp.zeros_like(epsilons_old)
        full_params = np.ones((self.num_cells, theta.shape[1]))
        full_params = full_params.at[self.flex_inds].set(theta)
        thetas = np.repeat(full_params[:, None, :], self.num_quads, axis=1)
        self.full_params = full_params
        return sigmas_old, epsilons_old, thetas

    def inspect(self):
        '''For debugging purpose
        '''
        sigmas_old = self.internal_vars['laplace'][0]
        return np.mean(np.sqrt(np.sum(sigmas_old*sigmas_old, axis=(2, 3))), axis=1)
        # return np.mean(sigmas_old, axis=1)

 
def topology_optimization():
    problem_name = 'plasticity'
    root_path = os.path.join(os.path.dirname(__file__), 'data') 

    files = glob.glob(os.path.join(root_path, f'vtk/{problem_name}/*'))
    for f in files:
        os.remove(f)

    meshio_mesh = box_mesh(50, 30, 1, 50., 30., 1., root_path)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load_location(point):
        return np.logical_and(np.isclose(point[0], 50., atol=1e-5), np.isclose(point[1], 15., atol=1.5))

    def dirichlet_val(point):
        return 0.

    # max_load = 5*1e2
    max_load = 4*1e2

    rs = np.linspace(0.2, 1., 5)

    def get_neumann_val(load):
        def neumann_val(point):
            return np.array([0., -load, 0.])
        return neumann_val

    dirichlet_bc_info = [[fixed_location]*3, [0, 1, 2], [dirichlet_val]*3]
    neumann_bc_info = [[load_location], [get_neumann_val(0.)]]
    problem = Plasticity(jax_mesh, vec=3, dim=3, neumann_bc_info=neumann_bc_info, dirichlet_bc_info=dirichlet_bc_info)
    fwd_pred = ad_wrapper(problem)


    def fwd_pred_seq(theta):
        params = problem.init_params(theta)
        for i in range(len(rs)):
            print(f"\nStep {i + 1} in {len(rs)}")
            problem.neumann_value_fns = [get_neumann_val(rs[i]*max_load)]
            sol = fwd_pred(params)
            params = problem.update_stress_strain(sol, params)  
        return sol     

    def J_total(theta):
        sol = fwd_pred_seq(theta)
        compliance = problem.compute_compliance(get_neumann_val(max_load), sol)
        return compliance


    outputs = []
    def output_sol(theta, obj_val):
        print(f"\nOutput solution - need to solve the forward problem again...")
        sol = fwd_pred_seq(theta)
        s = problem.inspect()
        vtu_path = os.path.join(root_path, f'vtk/{problem_name}/sol_{output_sol.counter:03d}.vtu')
        save_sol(problem, sol, vtu_path, cell_infos=[('theta1', problem.full_params[:, 0]), ('theta2', problem.full_params[:, 1]), ('s', s)])
        print(f"compliance = {obj_val}")
        print(f"max theta = {np.max(theta[:, 0])}, min theta = {np.min(theta[:, 0])}, mean theta = {np.mean(theta[:, 0])}")
        outputs.append(obj_val)
        output_sol.counter += 1

    output_sol.counter = 0
        

    vf = 0.50
    vy = 0.25
    num_flex = len(problem.flex_inds)

    # theta = np.hstack((vf*np.ones((num_flex, 1)), 0.5*np.ones((num_flex, 1))))
    # output_sol(theta, 0.)
    # exit()

    def objectiveHandle(rho):
        J, dJ = jax.value_and_grad(J_total)(rho)
        if objectiveHandle.counter % 5 == 0:
            output_sol(rho, J)
        objectiveHandle.counter += 1
        return J, dJ

    objectiveHandle.counter = 0

    def computeConstraints(rho, epoch):
        def computeGlobalVolumeConstraint(rho):
            rho1 = rho[:, 0]
            rho2 = rho[:, 1]
            g1 = np.sum(rho1)/num_flex/vf - 1.
            g2 = np.sum(rho2)/num_flex/vy - 1.
            return np.array([g1, g2])

        c = computeGlobalVolumeConstraint(rho)
        gradc = jax.jacrev(computeGlobalVolumeConstraint)(rho)
        return c, gradc

    optimizationParams = {'maxIters':51, 'minIters':51, 'relTol':0.05}
    rho_ini = np.hstack((vf*np.ones((num_flex, 1)), vy*np.ones((num_flex, 1))))  
    optimize(problem, rho_ini, optimizationParams, objectiveHandle, computeConstraints, numConstraints=2)
    onp.save(os.path.join(root_path, f"numpy/{problem_name}_outputs.npy"), onp.array(outputs))
    # print(f"Compliance = {J_total(np.ones((num_flex, 1)))} for full material")


if __name__=="__main__":
    topology_optimization()

