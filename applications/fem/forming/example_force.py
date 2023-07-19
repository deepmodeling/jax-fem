"""Reference
Simo, Juan C., and Thomas JR Hughes. Computational inelasticity. Vol. 7. Springer Science & Business Media, 2006.
Chapter 9: Phenomenological Plasticity Models

Force control is problematic since force can cause stress to exceed yield strength easily!
"""
import jax
import jax.numpy as np
import jax.flatten_util
import os
import glob
import matplotlib.pyplot as plt

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver, DynamicRelaxSolve
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh


def simulation():
    class Plasticity(FEM):
        def custom_init(self):
            self.F_old = np.repeat(np.repeat(np.eye(self.dim)[None, None, :, :], len(self.cells), axis=0), self.num_quads, axis=1)
            self.Be_old = np.array(self.F_old)
            self.alpha_old = np.zeros((len(self.cells), self.num_quads))
            self.internal_vars['laplace'] = [self.F_old, self.Be_old, self.alpha_old]

        def get_tensor_map(self):
            tensor_map, _, _ = self.get_maps()
            return tensor_map

        def get_maps(self):
            K = 164.e3
            G = 80.e3
            H1 = 18.
            sig0 = 400. 

            def get_partial_tensor_map(F_old, be_bar_old, alpha_old):
                def first_PK_stress(u_grad):
                    _, _, tau = return_map(u_grad)
                    F = u_grad + np.eye(self.dim)
                    P = tau @ np.linalg.inv(F).T 
                    return P    

                def update_int_vars(u_grad):
                    be_bar, alpha, _ = return_map(u_grad)
                    F = u_grad + np.eye(self.dim)
                    return F, be_bar, alpha

                def compute_cauchy_stress(u_grad):
                    F = u_grad + np.eye(self.dim)
                    J = np.linalg.det(F)
                    P = first_PK_stress(u_grad)
                    sigma = 1./J*P @ F.T
                    return sigma

                def get_tau(F, be_bar):
                    J = np.linalg.det(F)
                    tau = 0.5*K*(J**2 - 1)*np.eye(self.dim) + G*deviatoric(be_bar)
                    return tau

                def deviatoric(A):
                    return A - 1./self.dim*np.trace(A)*np.eye(self.dim)

                def return_map(u_grad):
                    F = u_grad + np.eye(self.dim)
                    F_inv = np.linalg.inv(F)
                    F_old_inv = np.linalg.inv(F_old)
                    f = F @ F_old_inv
                    f_bar =  np.linalg.det(f)**(-1./3.)*f
                    be_bar_trial = f @ be_bar_old @ f.T
                    s_trial = G*deviatoric(be_bar_trial)
                    yield_f_trial = np.linalg.norm(s_trial) - np.sqrt(2./3.)*(sig0 + H1*alpha_old)

                    def elastic_loading():
                        be_bar = be_bar_trial
                        alpha = alpha_old
                        tau = get_tau(F, be_bar)
                        return be_bar, alpha, tau

                    def plastic_loading():
                        Ie_bar = 1./3.*np.trace(be_bar_trial)
                        G_bar = Ie_bar*G
                        Delta_gamma = (yield_f_trial/(2.*G_bar))/(1. + H1/(3.*G_bar))
                        direction = s_trial/np.linalg.norm(s_trial)
                        s = s_trial - 2.*G_bar*Delta_gamma * direction
                        alpha = alpha_old + np.sqrt(2./3.)*Delta_gamma
                        be_bar = s/G + Ie_bar*np.eye(self.dim)
                        tau = get_tau(F, be_bar)
                        return be_bar, alpha, tau

                    return jax.lax.cond(yield_f_trial < 0., elastic_loading, plastic_loading)

                return first_PK_stress, update_int_vars, compute_cauchy_stress

            def tensor_map(u_grad, F_old, Be_old, alpha_old):
                first_PK_stress, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old)
                return first_PK_stress(u_grad)

            def update_int_vars_map(u_grad, F_old, Be_old, alpha_old):
                _, update_int_vars, _ = get_partial_tensor_map(F_old, Be_old, alpha_old)
                return update_int_vars(u_grad)

            def compute_cauchy_stress_map(u_grad, F_old, Be_old, alpha_old):
                _, _, compute_cauchy_stress = get_partial_tensor_map(F_old, Be_old, alpha_old)
                return compute_cauchy_stress(u_grad)

            return tensor_map, update_int_vars_map, compute_cauchy_stress_map

        def update_int_vars_gp(self, sol, int_vars):
            _, update_int_vars_map, _ = self.get_maps()
            vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
            # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, self.dim) -> (num_cells, num_quads, num_nodes, vec, self.dim) 
            u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
            u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, self.dim)
            updated_int_vars = vmap_update_int_vars_map(u_grads, *int_vars)
            return updated_int_vars

        def compute_stress(self, sol, int_vars):
            _, _, compute_cauchy_stress = self.get_maps()
            vmap_compute_cauchy_stress = jax.jit(jax.vmap(jax.vmap(compute_cauchy_stress)))
            # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, self.dim) -> (num_cells, num_quads, num_nodes, vec, self.dim) 
            u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
            u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, self.dim)
            sigma = vmap_compute_cauchy_stress(u_grads, *int_vars)
            return sigma

        def set_params(self, params):
            int_vars, scale = params
            self.internal_vars['laplace'] = int_vars
            self.neumann_value_fns[0] = get_neumann_top(scale)


    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    vtk_dir = os.path.join(data_dir, 'vtk')

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    Lx, Ly, Lz = 10., 10., 0.25
    meshio_mesh = box_mesh(Nx=40, Ny=40, Nz=1, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def walls(point):
        left = np.isclose(point[0], 0., atol=1e-5)
        right = np.isclose(point[0], Lx, atol=1e-5)
        front = np.isclose(point[1], 0., atol=1e-5)
        back = np.isclose(point[1], Ly, atol=1e-5)
        return left | right | front | back

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def dirichlet_val(point):
        return 0.

    def get_neumann_top(scale):
        def neumann_top(point):
            x = point[0]
            y = point[1]
            xc = 0.5*Lx
            yc = 0.5*Ly
            lx = 0.25*Lx
            ly = 0.25*Ly
            traction_z = scale * np.exp(-((x - xc)**2 + (y - yc)**2)/(lx**2 + ly**2))
            return np.array([0., 0., -traction_z])
        return neumann_top

    scales = 40*np.hstack((np.linspace(0., 1., 5), np.linspace(1, 0., 5)))

    # scales = 50*np.hstack((np.linspace(0., 1., 5), np.linspace(1, 0., 5)))


    location_fns = [walls]*3
    value_fns = [dirichlet_val]*3
    vecs = [0, 1, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    neumann_bc_info = [[top], [None]]

    problem = Plasticity(mesh, ele_type=ele_type, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)
    sol = np.zeros(((problem.num_total_nodes, problem.vec)))
 
    int_vars = problem.internal_vars['laplace']

    int_vars_initial = problem.internal_vars['laplace']

    for i, scale in enumerate(scales):
        print(f"\nStep {i} in {len(scales)}, scale = {scale}")


        # problem.set_params([int_vars_initial, scale])

        # if i == 4 or False:
        #     sol = DynamicRelaxSolve(problem, sol, tol=0.1)
        # else:
        #     sol = solver(problem, initial_guess=None, use_petsc=False)


        problem.set_params([int_vars, scale])


 
        sol = solver(problem, initial_guess=sol, use_petsc=False)


        int_vars_copy = int_vars
        int_vars = problem.update_int_vars_gp(sol, int_vars)
        sigmas = problem.compute_stress(sol, int_vars_copy).mean(axis=1)
        print(f"max alpha = \n{np.max(int_vars[-1])}")
        print(sigmas[0])
        vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
        save_sol(problem, sol, vtk_path, cell_infos=[('s_norm', np.linalg.norm(sigmas, axis=(1, 2)))])

if __name__=="__main__":
    simulation()
