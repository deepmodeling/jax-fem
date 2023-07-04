import jax
import jax.numpy as np
import jax.flatten_util
import os
import matplotlib.pyplot as plt

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
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

            def safe_sqrt(x):  
                safe_x = np.where(x > 0., np.sqrt(x), 0.)
                return safe_x

            def safe_divide(x, y):
                return np.where(y == 0., 0., x/y)

            def to_vector(A):
                return np.array([A[0, 0], A[1, 1,], A[2, 2], A[0, 1], A[0, 2], A[1, 2]])

            def to_tensor(A_vec):
                return np.array([[A_vec[0], A_vec[3], A_vec[4]], 
                                 [A_vec[3], A_vec[1], A_vec[5]], 
                                 [A_vec[4], A_vec[5], A_vec[2]]])

            def get_partial_tensor_map(F_old, Be_old, alpha_old):
                y0 = to_vector(Be_old)
                _, unflatten_fn_x = jax.flatten_util.ravel_pytree([F_old, F_old, Be_old, alpha_old]) # u_grad, F_old, Be_old, alpha_old
                _, unflatten_fn_Be_alpha = jax.flatten_util.ravel_pytree([Be_old, alpha_old])

                def first_PK_stress(u_grad):
                    x, _ = jax.flatten_util.ravel_pytree([u_grad, F_old, Be_old, alpha_old])
                    Be, alpha = unflatten_fn_Be_alpha(plastic_or_elastic_loading(x))
                    tau = get_tau(Be)
                    F = u_grad + np.eye(self.dim)
                    P = tau @ np.linalg.inv(F).T 
                    return P    

                def update_int_vars(u_grad):
                    x, _ = jax.flatten_util.ravel_pytree([u_grad, F_old, Be_old, alpha_old])
                    Be, alpha = unflatten_fn_Be_alpha(plastic_or_elastic_loading(x))
                    F = u_grad + np.eye(self.dim)
                    return F, Be, alpha

                def compute_cauchy_stress(u_grad):
                    F = u_grad + np.eye(self.dim)
                    J = np.linalg.det(F)
                    P = first_PK_stress(u_grad)
                    sigma = 1./J*P @ F.T
                    return sigma

                def get_tau(Be):
                    J_Be = np.linalg.det(Be)
                    be_bar = J_Be**(-1./3.) * Be
                    be_bar_dev = be_bar - 1./self.dim*np.trace(be_bar)*np.eye(self.dim)
                    tau = 0.5*K*(J_Be - 1)*np.eye(self.dim) + G*be_bar_dev
                    return tau

                def get_tau_dev_norm(tau):
                    tau_dev = tau - 1./self.dim*np.trace(tau)*np.eye(self.dim)
                    tau_dev_norm = safe_sqrt(np.sum(tau_dev*tau_dev))
                    return tau_dev_norm    

                def plastic_or_elastic_loading(x):
                    u_grad, F_old, Be_old, alpha_old = unflatten_fn_x(x)
                    F = u_grad + np.eye(self.dim)
                    F_old_inv = np.linalg.inv(F_old)
                    Cp_old_inv = F_old_inv @ Be_old @ F_old_inv.T
                    Be_trial = F @ Cp_old_inv @ F.T
                    tau_trial = get_tau(Be_trial)
                    tau_trial_dev_norm = get_tau_dev_norm(tau_trial)
                    yield_f = tau_trial_dev_norm - np.sqrt(2./3.)*(sig0 + H1*alpha_old)

                    def implicit_residual(x, y):
                        u_grad, F_old, Be_old, alpha_old = unflatten_fn_x(x)
                        Be_vec = y
                        Be = to_tensor(Be_vec)
                        F = u_grad + np.eye(self.dim)
                        F_inv = np.linalg.inv(F)
                        F_old_inv = np.linalg.inv(F_old)
                        Cp_inv = F_inv @ Be @ F_inv.T
                        Cp_old_inv = F_old_inv @ Be_old @ F_old_inv.T
                        J_Be = np.linalg.det(Be)
                        be_bar = J_Be**(-1./3.) * Be
                        tau = get_tau(Be)
                        tau_dev = tau - 1./self.dim*np.trace(tau)*np.eye(self.dim)
                        tau_dev_norm = get_tau_dev_norm(tau)
                        direction = safe_divide(tau_dev, tau_dev_norm)
                        alpha_solved = (tau_dev_norm/np.sqrt(2./3.) - sig0)/H1
                        C1 = (Cp_inv - Cp_old_inv) + (alpha_solved - alpha_old)*np.sqrt(2./3.)*np.trace(be_bar)*F_inv @ direction @ F_inv.T
                        res = to_vector(C1)
                        return res

                    @jax.custom_jvp
                    def newton_solver(x):
                        step = 0
                        res_vec = implicit_residual(x, y0)
                        tol = 1e-8

                        def cond_fun(state):
                            step, res_vec, y = state
                            return np.linalg.norm(res_vec) > tol

                        def body_fun(state):
                            step, res_vec, y = state
                            f_partial = lambda y: implicit_residual(x, y)
                            jac = jax.jacfwd(f_partial)(y)
                            y_inc = np.linalg.solve(jac, -res_vec)
                            y_new = y + y_inc
                            res_vec_new = f_partial(y_new)
                            return step + 1, res_vec_new, y_new

                        step_f, res_vec_f, y_f = jax.lax.while_loop(cond_fun, body_fun, (step, res_vec, y0))

                        return y_f

                    @newton_solver.defjvp
                    def f_jvp(primals, tangents):
                        x, = primals
                        v, = tangents
                        y = newton_solver(x)
                        jac_x = jax.jacfwd(implicit_residual, argnums=0)(x, y)
                        jac_y = jax.jacfwd(implicit_residual, argnums=1)(x, y)
                        jvp_result = np.linalg.solve(jac_y, -(jac_x @ v[:, None]).reshape(-1))
                        return y, jvp_result

                    def elastic_loading(x):
                        Be = Be_trial
                        alpha = alpha_old
                        return transform(Be, alpha)

                    def plastic_loading(x):
                        y = newton_solver(x)
                        Be = to_tensor(y)
                        tau = get_tau(Be)
                        tau_dev_norm = get_tau_dev_norm(tau)
                        alpha = (tau_dev_norm/np.sqrt(2./3.) - sig0)/H1
                        return transform(Be, alpha)

                    def transform(Be, alpha):
                        result, _ = jax.flatten_util.ravel_pytree([Be, alpha])
                        return result

                    # return jax.lax.cond(yield_f < 0., elastic_loading, plastic_loading, x)
                    return np.where(yield_f < 0, elastic_loading(x), plastic_loading(x))
                    
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
            int_vars, disp = params
            self.dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
            self.update_Dirichlet_boundary_conditions(self.dirichlet_bc_info)
            self.internal_vars['laplace'] = int_vars


    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    Lx, Ly, Lz = 10., 10., 10.
    meshio_mesh = box_mesh(Nx=10, Ny=10, Nz=10, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def dirichlet_val_bottom(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn

    disps = np.hstack((np.linspace(0., 0.1, 11), np.linspace(0.09, 0., 10)))
    # disps = np.linspace(0., 2., 11)

    location_fns = [bottom, top]
    value_fns = [dirichlet_val_bottom, get_dirichlet_top(disps[0])]
    vecs = [2, 2]

    dirichlet_bc_info = [location_fns, vecs, value_fns]

    problem = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    sol = np.zeros(((problem.num_total_nodes, problem.vec)))
 
    int_vars = problem.internal_vars['laplace']
    for i, disp in enumerate(disps):
        print(f"\nStep {i} in {len(disps)}, disp = {disp}")
        problem.set_params([int_vars, disp])
        sol = solver(problem, initial_guess=None, use_petsc=False)
        int_vars_copy = int_vars
        int_vars = problem.update_int_vars_gp(sol, int_vars)
        sigmas = problem.compute_stress(sol, int_vars_copy)
        print(f"alpha = {int_vars[-1]}")
        print(sigmas[0])
        vtk_path = os.path.join(data_dir, f'vtk/u_{i:03d}.vtu')
        save_sol(problem, sol, vtk_path)

if __name__=="__main__":
    simulation()
