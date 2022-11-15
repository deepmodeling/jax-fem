import numpy as onp
import jax
import jax.numpy as np
import os
import sys
from jax_am.fem.models import Mechanics


from jax.config import config
config.update("jax_enable_x64", True)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=5)


crt_dir = os.path.dirname(__file__)


class CrystalPlasticity(Mechanics):
    def custom_init(self):
        r = 1.
        gss_initial = 60.8 

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
        self.slip_resistance_old_gp = gss_initial*onp.ones((len(self.cells), self.num_quads, num_slip_sys))

        self.C = onp.zeros((self.dim, self.dim, self.dim, self.dim))
        C11 = 1.684e5
        C12 = 1.214e5
        C44 = 0.754e5
 
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
        return self.newton_vars(sol, laplace=[self.Fp_inv_old_gp, self.slip_resistance_old_gp])

    def compute_residual(self, sol):
        return self.compute_residual_vars(sol, laplace=[self.Fp_inv_old_gp, self.slip_resistance_old_gp])

    def get_maps(self):

        h = 541.5
        t_sat = 109.8
        gss_a = 2.5
        ao = 0.001
        xm = 0.1

        def get_partial_tensor_map(Fp_inv_old, slip_resistance_old):

            def helper(u_grad, S):
                F = u_grad + np.eye(self.dim)
                tau = np.sum(S[None, :, :] * self.Schmid_tensors, axis=(1, 2))
                slip_inc = ao*self.dt*np.absolute(tau/slip_resistance_old)**(1./xm)*np.sign(tau)
                L_plastic_inc = np.sum(slip_inc[:, None, None] * self.Schmid_tensors, axis=0)
                Fp_inv_new = Fp_inv_old @ (np.eye(self.dim) - L_plastic_inc)
                Fe = F @ Fp_inv_new 
                return Fp_inv_new, slip_inc, Fe, F

            def implicit_residual(u_grad, S):
                _, _, Fe, _ = helper(u_grad, S)
                S_ = np.sum(self.C * 1./2.*(Fe.T @ Fe - np.eye(self.dim))[None, None, :, :], axis=(2, 3))
                return S - S_

            def update_int_vars(u_grad, S):
                Fp_inv_new, slip_inc, _, _ = helper(u_grad, S)
                tmp = h*np.absolute(slip_inc) * np.absolute(1 - slip_resistance_old/t_sat)**gss_a * np.sign(1 - slip_resistance_old/t_sat)
                slip_resistance_new = slip_resistance_old + (self.q @ tmp[:, None]).reshape(-1)
                return Fp_inv_new, slip_resistance_new

            @jax.custom_jvp
            def newton_solver(x):
                y0 = np.zeros_like(x)

                def cond_fun(y):
                    tol = 1e-5
                    res = implicit_residual(x, y)
                    return np.linalg.norm(res) > tol

                def body_fun(y):
                    # TODO: would jvp + cg solver be faster?
                    f_partial = lambda y: implicit_residual(x, y)
                    jac = jax.jacfwd(f_partial)(y)
                    res = f_partial(y)
                    res_flat = res.reshape(-1)
                    jac_flat = jac.reshape(len(res_flat), len(res_flat))
                    y_inc = np.linalg.solve(jac_flat, -res_flat).reshape(res.shape)
                    y = y + y_inc
                    return y

                return jax.lax.while_loop(cond_fun, body_fun, y0)

            @newton_solver.defjvp
            def f_jvp(primals, tangents):
                x, = primals
                v, = tangents
                v_flat = v.reshape(-1)
                y = newton_solver(x)
                jac_x = jax.jacfwd(implicit_residual, argnums=0)(x, y)
                jac_y = jax.jacfwd(implicit_residual, argnums=1)(x, y)
                jac_x_flat = jac_x.reshape((len(v_flat), len(v_flat)))
                jac_y_flat = jac_y.reshape((len(v_flat), len(v_flat)))
                jvp_result = np.linalg.solve(jac_y_flat, -(jac_x_flat @ v_flat[:, None]).reshape(-1)).reshape(y.shape)
                return y, jvp_result

            return newton_solver, update_int_vars, helper
     
        def tensor_map(u_grad, Fp_inv_old, slip_resistance_old):
            partial_tensor_map, _, helper = get_partial_tensor_map(Fp_inv_old, slip_resistance_old)
            S = partial_tensor_map(u_grad)
            _, _, Fe, F = helper(u_grad, S)
            sigma = 1./np.linalg.det(Fe)*Fe @ S @ Fe.T
            P = np.linalg.det(F)*sigma @ np.linalg.inv(F).T
            return P

        def update_int_vars_map(u_grad, Fp_inv_old, slip_resistance_old):
            partial_tensor_map, update_int_vars, _ = get_partial_tensor_map(Fp_inv_old, slip_resistance_old)
            S = partial_tensor_map(u_grad)
            return update_int_vars(u_grad, S)

        return tensor_map, update_int_vars_map

    def update_int_vars_gp(self, sol):
        _, update_int_vars_map = self.get_maps()
        vmap_update_int_vars_map = jax.vmap(jax.vmap(update_int_vars_map))

        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
  
        self.Fp_inv_old_gp, self.slip_resistance_old_gp = vmap_update_int_vars_map(u_grads, self.Fp_inv_old_gp, self.slip_resistance_old_gp)

        print(f"max Fp_inv_old_gp = {np.max(self.Fp_inv_old_gp)}")
        print(f"max slip_resistance_old_gp = {np.max(self.slip_resistance_old_gp)}")


    # def debug_gp(self, sol)

    #     _, update_int_vars_map = self.get_maps()
    #     vmap_update_int_vars_map = jax.vmap(jax.vmap(update_int_vars_map))

    #     # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
    #     u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
    #     u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
  
    #     self.Fp_inv_old_gp, self.slip_resistance_old_gp = vmap_update_int_vars_map(u_grads, self.Fp_inv_old_gp, self.slip_resistance_old_gp)

      

    def compute_avg_stress(self, sol):
        """For post-processing only
        """
        # TODO: duplicated code
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)

        partial_tensor_map, _ = self.get_maps()
        vmap_partial_tensor_map = jax.vmap(jax.vmap(partial_tensor_map))
        P = vmap_partial_tensor_map(u_grads, self.Fp_inv_old_gp, self.slip_resistance_old_gp)

        # num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1)
        P = np.sum(P.reshape(-1, self.vec, self.dim) * self.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.JxW)
        avg_P = P/vol
        return avg_P


