import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
import os
import sys

from jax_am.fem.models import Mechanics


from jax.config import config
config.update("jax_enable_x64", True)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=10)


crt_dir = os.path.dirname(__file__)


class CrystalPlasticity(Mechanics):
    def custom_init(self):
        r = 1.
        self.gss_initial = 60.8 

        input_slip_sys = onp.loadtxt(os.path.join(crt_dir, 'input_slip_sys.txt'))

        num_slip_sys = len(input_slip_sys)
        slip_directions = input_slip_sys[:, self.dim:]
        slip_directions = slip_directions/onp.linalg.norm(slip_directions, axis=1)[:, None]
        slip_normals = input_slip_sys[:, :self.dim]
        slip_normals = slip_normals/onp.linalg.norm(slip_normals, axis=1)[:, None]
        self.Schmid_tensors = jax.vmap(np.outer)(slip_directions, slip_normals)

        self.q = r*onp.ones((num_slip_sys, num_slip_sys))

        num_directions_per_normal = 3
        for i in range(num_slip_sys):
            for j in range(num_directions_per_normal):
                self.q[i, i//num_directions_per_normal*num_directions_per_normal + j] = 1.

        self.Fp_inv_old_gp = onp.repeat(onp.repeat(onp.eye(self.dim)[None, None, :, :], len(self.cells), axis=0), self.num_quads, axis=1)
        self.slip_resistance_old_gp = self.gss_initial*onp.ones((len(self.cells), self.num_quads, num_slip_sys))
        self.slip_old_gp = onp.zeros_like(self.slip_resistance_old_gp)

        self.C = onp.zeros((self.dim, self.dim, self.dim, self.dim))

        C11 = 1.684e5
        C12 = 1.214e5
        C44 = 0.754e5

        # E = 1.25e5
        # nu = 0.36
        # C11 = E*(1-nu)/((1+nu)*(1-2*nu))
        # C12 = E*nu/((1+nu)*(1-2*nu))
        # C44 = E/(2.*(1. + nu))

        self.C[0, 0, 0, 0] = C11
        self.C[1, 1, 1, 1] = C11
        self.C[2, 2, 2, 2] = C11

        self.C[0, 0, 1, 1] = C12
        self.C[1, 1, 0, 0] = C12

        self.C[0, 0, 2, 2] = C12
        self.C[2, 2, 0, 0] = C12

        self.C[1, 1, 2, 2] = C12
        self.C[2, 2, 1, 1] = C12

        self.C[1, 2, 1, 2] = C44
        self.C[1, 2, 2, 1] = C44
        self.C[2, 1, 1, 2] = C44
        self.C[2, 1, 2, 1] = C44

        self.C[2, 0, 2, 0] = C44
        self.C[2, 0, 0, 2] = C44
        self.C[0, 2, 2, 0] = C44
        self.C[0, 2, 0, 2] = C44

        self.C[0, 1, 0, 1] = C44
        self.C[0, 1, 1, 0] = C44
        self.C[1, 0, 0, 1] = C44
        self.C[1, 0, 1, 0] = C44

    def get_tensor_map(self):
        tensor_map, _ = self.get_maps()
        return tensor_map

    def newton_update(self, sol):
        return self.newton_vars(sol, laplace=[self.Fp_inv_old_gp, self.slip_resistance_old_gp, self.slip_old_gp])

    def compute_residual(self, sol):
        return self.compute_residual_vars(sol, laplace=[self.Fp_inv_old_gp, self.slip_resistance_old_gp, self.slip_old_gp])

    def get_maps(self):
        h = 541.5
        t_sat = 109.8
        gss_a = 2.5
        ao = 0.001
        xm = 0.1

        def get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old):
            _, unflatten_fn_x = jax.flatten_util.ravel_pytree(Fp_inv_old)
            tmp_y, unflatten_fn_y = jax.flatten_util.ravel_pytree((Fp_inv_old, slip_resistance_old, slip_old))

            def first_PK_stress(u_grad):
                y = newton_solver(u_grad.reshape(-1))
                S, gamma_inc, g_inc = unflatten_fn_y(y)
                _, _, _, Fe, F = helper(u_grad, S, gamma_inc, g_inc)
                sigma = 1./np.linalg.det(Fe)*Fe @ S @ Fe.T
                P = np.linalg.det(F)*sigma @ np.linalg.inv(F).T 
                return P    

            def update_int_vars(u_grad):
                y = newton_solver(u_grad.reshape(-1))
                S, gamma_inc, g_inc = unflatten_fn_y(y)
                Fp_inv_new, slip_resistance_new, slip_new, Fe, F = helper(u_grad, S, gamma_inc, g_inc)
                return Fp_inv_new, slip_resistance_new, slip_new

            def helper(u_grad, S, gamma_inc, g_inc):
                slip_resistance_new = slip_resistance_old + g_inc
                slip_new = slip_old + gamma_inc
                F = u_grad + np.eye(self.dim)
                L_plastic_inc = np.sum(gamma_inc[:, None, None] * self.Schmid_tensors, axis=0)
                Fp_inv_new = Fp_inv_old @ (np.eye(self.dim) - L_plastic_inc)
                Fe = F @ Fp_inv_new 
                return Fp_inv_new, slip_resistance_new, slip_new, Fe, F

            def implicit_residual(x, y):
                u_grad = unflatten_fn_x(x)
                S, gamma_inc, g_inc = unflatten_fn_y(y)
                _, slip_resistance_new, slip_new, Fe, _ = helper(u_grad, S, gamma_inc, g_inc)

                slip_resistance = slip_resistance_old

                S_ = np.sum(self.C * 1./2.*(Fe.T @ Fe - np.eye(self.dim))[None, None, :, :], axis=(2, 3))           

                tau = np.sum(S[None, :, :] * self.Schmid_tensors, axis=(1, 2))
                gamma_inc_ = ao*self.dt*np.absolute(tau/slip_resistance)**(1./xm)*np.sign(tau)

                tmp = h*np.absolute(gamma_inc) * np.absolute(1 - slip_resistance/t_sat)**gss_a * np.sign(1 - slip_resistance/t_sat)
                g_inc_ = (self.q @ tmp[:, None]).reshape(-1)

                # tmp = h*np.absolute(gamma_inc) / np.cosh(h*np.sum(slip_new)/(t_sat - self.gss_initial))**2
                # g_inc_ = (self.q @ tmp[:, None]).reshape(-1)

                res, _ = jax.flatten_util.ravel_pytree((S - S_, gamma_inc - gamma_inc_, g_inc - g_inc_))

                return res

            @jax.custom_jvp
            def newton_solver(x):
                y0 = np.zeros_like(tmp_y)

                def cond_fun(y):
                    tol = 1e-8
                    res = implicit_residual(x, y)
                    return np.linalg.norm(res) > tol

                def body_fun(y):
                    # TODO: would jvp + cg solver be faster?
                    f_partial = lambda y: implicit_residual(x, y)
                    jac = jax.jacfwd(f_partial)(y)
                    res = f_partial(y)
                    y_inc = np.linalg.solve(jac, -res)
                    y = y + y_inc
                    return y

                return jax.lax.while_loop(cond_fun, body_fun, y0)

            @newton_solver.defjvp
            def f_jvp(primals, tangents):
                x, = primals
                v, = tangents
                y = newton_solver(x)
                jac_x = jax.jacfwd(implicit_residual, argnums=0)(x, y)
                jac_y = jax.jacfwd(implicit_residual, argnums=1)(x, y)
                jvp_result = np.linalg.solve(jac_y, -(jac_x @ v[:, None]).reshape(-1))
                return y, jvp_result

            return first_PK_stress, update_int_vars
     
        def tensor_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old):
            first_PK_stress, _ = get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old)
            return first_PK_stress(u_grad)

        def update_int_vars_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old):
            _, update_int_vars = get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old)
            return update_int_vars(u_grad)

        return tensor_map, update_int_vars_map

    def update_int_vars_gp(self, sol):
        _, update_int_vars_map = self.get_maps()
        vmap_update_int_vars_map = jax.vmap(jax.vmap(update_int_vars_map))

        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
  
        Fp_inv_new_gp, slip_resistance_new_gp, slip_new_gp = \
            vmap_update_int_vars_map(u_grads, self.Fp_inv_old_gp, self.slip_resistance_old_gp, self.slip_old_gp)

        slip_inc_dt_index_0 = (slip_new_gp[0, 0, 0] - self.slip_old_gp[0, 0, 0])/self.dt
        print(f"slip inc dt index 0 = {slip_inc_dt_index_0}")

        self.Fp_inv_old_gp, self.slip_resistance_old_gp, self.slip_old_gp = Fp_inv_new_gp, slip_resistance_new_gp, slip_new_gp

        F_p = np.linalg.inv(self.Fp_inv_old_gp[0, 0])
        print(f"Fp = \n{F_p}")
        slip_resistance_0 = self.slip_resistance_old_gp[0, 0, 0]
        print(f"slip_resistance index 0 = {slip_resistance_0}")

        return F_p[2, 2], slip_resistance_0, slip_inc_dt_index_0 

    def compute_avg_stress(self, sol):
        """For post-processing only
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)

        partial_tensor_map, _ = self.get_maps()
        vmap_partial_tensor_map = jax.vmap(jax.vmap(partial_tensor_map))
        P = vmap_partial_tensor_map(u_grads, self.Fp_inv_old_gp, self.slip_resistance_old_gp, self.slip_old_gp)

        def P_to_sigma(P, F):
            return 1./np.linalg.det(F) * P @ F.T

        vvmap_P_to_sigma = jax.vmap(jax.vmap(P_to_sigma))        
        F = u_grads + np.eye(self.dim)[None, None, :, :]
        sigma = vvmap_P_to_sigma(P, F)[0, 0]
        
        # num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1)
        avg_P = np.sum(P.reshape(-1, self.vec, self.dim) * self.JxW.reshape(-1)[:, None, None], 0) / np.sum(self.JxW)
  
        return sigma[2, 2]
