import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
import os
import sys
from functools import partial

from jax_am.fem.models import Mechanics


from jax.config import config
config.update("jax_enable_x64", True)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=10)


crt_dir = os.path.dirname(__file__)


def rotate_tensor_rank_4(R, T):
    R0 = R[:, :, None, None, None, None, None, None]
    R1 = R[None, None, :, :, None, None, None, None]
    R2 = R[None, None, None, None, :, :, None, None]
    R3 = R[None, None, None, None, None, None, :, :]
    return np.sum(R0 * R1 * R2 * R3 * T[None, :, None, :, None, :, None, :], axis=(1, 3, 5, 7))


def rotate_tensor_rank_2(R, T):
    R0 = R[:, :, None, None]
    R1 = R[None, None, :, :]
    return np.sum(R0 * R1 * T[None, :, None, :], axis=(1, 3))


# def rotate_tensor_rank_4(R, T):
#     return T

# def rotate_tensor_rank_2(R, T):
#     return T


rotate_tensor_rank_2_vmap = jax.vmap(rotate_tensor_rank_2, in_axes=(None, 0))


def get_rot_mat(q):
    '''
    Transformation from quaternion to the corresponding rotation matrix.
    Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    '''
    return np.array([[q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                     [2*q[1]*q[2] + 2*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2*q[2]*q[3] - 2*q[0]*q[1]],
                     [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]]])


get_rot_mat_vmap = jax.vmap(get_rot_mat)


class CrystalPlasticity(Mechanics):
    def custom_init(self, quat, cell_ori_inds):
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

        rot_mats = onp.array(get_rot_mat_vmap(quat)[cell_ori_inds])

        self.Fp_inv_old_gp = onp.repeat(onp.repeat(onp.eye(self.dim)[None, None, :, :], len(self.cells), axis=0), self.num_quads, axis=1)
        self.slip_resistance_old_gp = self.gss_initial*onp.ones((len(self.cells), self.num_quads, num_slip_sys))
        self.slip_old_gp = onp.zeros_like(self.slip_resistance_old_gp)
        self.rot_mats_gp = onp.repeat(rot_mats[:, None, :, :], self.num_quads, axis=1)
        self.tol_gp = 1e-5*onp.ones((len(self.cells), self.num_quads))
        self.y_ini_gp = onp.zeros((len(self.cells), self.num_quads, 9)) # TODO

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
        return self.newton_vars(sol, laplace=[self.Fp_inv_old_gp, self.slip_resistance_old_gp, 
            self.slip_old_gp, self.rot_mats_gp, self.tol_gp, self.y_ini_gp])

    def compute_residual(self, sol):
        return self.compute_residual_vars(sol, laplace=[self.Fp_inv_old_gp, self.slip_resistance_old_gp, 
            self.slip_old_gp, self.rot_mats_gp, self.tol_gp, self.y_ini_gp])

    def get_maps(self):
        h = 541.5
        t_sat = 109.8
        gss_a = 2.5
        ao = 0.001
        xm = 0.1

        # def get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat, tol, y_ini):
        #     _, unflatten_fn_x = jax.flatten_util.ravel_pytree(Fp_inv_old)
        #     tmp_y, unflatten_fn_y = jax.flatten_util.ravel_pytree((Fp_inv_old, slip_resistance_old, slip_old))

        #     def first_PK_stress(u_grad):
        #         y = newton_solver(u_grad.reshape(-1))
        #         S, gamma_inc, g_inc = unflatten_fn_y(y)
        #         _, _, _, Fe, F = helper(u_grad, S, gamma_inc, g_inc)
        #         sigma = 1./np.linalg.det(Fe)*Fe @ S @ Fe.T
        #         P = np.linalg.det(F)*sigma @ np.linalg.inv(F).T 
        #         return P    

        #     def update_int_vars(u_grad):
        #         y = newton_solver(u_grad.reshape(-1))
        #         S, gamma_inc, g_inc = unflatten_fn_y(y)
        #         Fp_inv_new, slip_resistance_new, slip_new, Fe, F = helper(u_grad, S, gamma_inc, g_inc)
        #         return Fp_inv_new, slip_resistance_new, slip_new, y

        #     def helper(u_grad, S, gamma_inc, g_inc):
        #         slip_resistance_new = slip_resistance_old + g_inc
        #         slip_new = slip_old + gamma_inc
        #         F = u_grad + np.eye(self.dim)
        #         L_plastic_inc = np.sum(gamma_inc[:, None, None] * rotate_tensor_rank_2_vmap(rot_mat, self.Schmid_tensors), axis=0)
        #         Fp_inv_new = Fp_inv_old @ (np.eye(self.dim) - L_plastic_inc)
        #         Fe = F @ Fp_inv_new 
        #         return Fp_inv_new, slip_resistance_new, slip_new, Fe, F

        #     def implicit_residual(x, y):
        #         u_grad = unflatten_fn_x(x)
        #         S, gamma_inc, g_inc = unflatten_fn_y(y)
        #         _, slip_resistance_new, slip_new, Fe, _ = helper(u_grad, S, gamma_inc, g_inc)

        #         slip_resistance = slip_resistance_old
        #         # slip_resistance = slip_resistance_new

        #         S_ = np.sum(rotate_tensor_rank_4(rot_mat, self.C) * 1./2.*(Fe.T @ Fe - np.eye(self.dim))[None, None, :, :], axis=(2, 3))           

        #         tau = np.sum(S[None, :, :] * rotate_tensor_rank_2_vmap(rot_mat, self.Schmid_tensors), axis=(1, 2))
        #         gamma_inc_ = ao*self.dt*np.absolute(tau/slip_resistance)**(1./xm)*np.sign(tau)

        #         tmp = h*np.absolute(gamma_inc) * np.absolute(1 - slip_resistance/t_sat)**gss_a * np.sign(1 - slip_resistance/t_sat)
        #         g_inc_ = (self.q @ tmp[:, None]).reshape(-1)

        #         # tmp = h*np.absolute(gamma_inc) / np.cosh(h*np.sum(slip_new)/(t_sat - self.gss_initial))**2
        #         # g_inc_ = (self.q @ tmp[:, None]).reshape(-1)

        #         res, _ = jax.flatten_util.ravel_pytree((S - S_, gamma_inc - gamma_inc_, g_inc - g_inc_))

        #         return res

        #     @jax.custom_jvp
        #     def newton_solver(x):

        #         # y0 = np.zeros_like(tmp_y)

        #         y0 = y_ini

        #         # y0 = 1e-1*np.ones_like(tmp_y)

        #         def cond_fun(y):
        #             # # tol = 1e-8
        #             # tol = 1e-9
        #             res = implicit_residual(x, y)
        #             return np.linalg.norm(res) > tol

        #         def body_fun(y):
        #             # TODO: would jvp + cg solver be faster?
        #             f_partial = lambda y: implicit_residual(x, y)
        #             jac = jax.jacfwd(f_partial)(y)
        #             res = f_partial(y)
        #             y_inc = np.linalg.solve(jac, -res)

        #             res_norm = np.linalg.norm(res)

        #             # relax_param = np.where(res_norm > 1e-6, 1, 0.8)
        #             relax_param = 1.

        #             y = y + relax_param*y_inc
        #             return y

        #             # y1 = y + y_inc
        #             # y2 = y + 0.8*y_inc
        #             # res1 = f_partial(y1)
        #             # res2 = f_partial(y2)

        #             # return np.where(res1 < res2, y1, y2)


        #         return jax.lax.while_loop(cond_fun, body_fun, y0)

        #     @newton_solver.defjvp
        #     def f_jvp(primals, tangents):
        #         x, = primals
        #         v, = tangents
        #         y = newton_solver(x)
        #         jac_x = jax.jacfwd(implicit_residual, argnums=0)(x, y)
        #         jac_y = jax.jacfwd(implicit_residual, argnums=1)(x, y)
        #         jvp_result = np.linalg.solve(jac_y, -(jac_x @ v[:, None]).reshape(-1))
        #         return y, jvp_result

        #     return first_PK_stress, update_int_vars
     


        def get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat, tol, y_ini):
            _, unflatten_fn = jax.flatten_util.ravel_pytree(Fp_inv_old)
    
            def first_PK_stress(u_grad):
                y = newton_solver(u_grad.reshape(-1))
                S = unflatten_fn(y)
                _, _, _, Fe, F = helper(u_grad, S)
                sigma = 1./np.linalg.det(Fe)*Fe @ S @ Fe.T
                P = np.linalg.det(F)*sigma @ np.linalg.inv(F).T 
                return P    

            def update_int_vars(u_grad):
                y = newton_solver(u_grad.reshape(-1))
                S = unflatten_fn(y)
                Fp_inv_new, slip_resistance_new, slip_new, Fe, F = helper(u_grad, S)
                return Fp_inv_new, slip_resistance_new, slip_new, y

            def helper(u_grad, S):
                tau = np.sum(S[None, :, :] * rotate_tensor_rank_2_vmap(rot_mat, self.Schmid_tensors), axis=(1, 2))
                gamma_inc = ao*self.dt*np.absolute(tau/slip_resistance_old)**(1./xm)*np.sign(tau)

                tmp = h*np.absolute(gamma_inc) * np.absolute(1 - slip_resistance_old/t_sat)**gss_a * np.sign(1 - slip_resistance_old/t_sat)
                g_inc = (self.q @ tmp[:, None]).reshape(-1)

                # tmp = h*np.absolute(gamma_inc) / np.cosh(h*np.sum(slip_new)/(t_sat - self.gss_initial))**2
                # g_inc = (self.q @ tmp[:, None]).reshape(-1)

                slip_resistance_new = slip_resistance_old + g_inc
                slip_new = slip_old + gamma_inc
                F = u_grad + np.eye(self.dim)
                L_plastic_inc = np.sum(gamma_inc[:, None, None] * rotate_tensor_rank_2_vmap(rot_mat, self.Schmid_tensors), axis=0)
                Fp_inv_new = Fp_inv_old @ (np.eye(self.dim) - L_plastic_inc)
                Fe = F @ Fp_inv_new 
                return Fp_inv_new, slip_resistance_new, slip_new, Fe, F

            def implicit_residual(x, y):
                u_grad = unflatten_fn(x)
                S = unflatten_fn(y)
                _, _, _, Fe, _ = helper(u_grad, S)
                S_ = np.sum(rotate_tensor_rank_4(rot_mat, self.C) * 1./2.*(Fe.T @ Fe - np.eye(self.dim))[None, None, :, :], axis=(2, 3))           
                res, _ = jax.flatten_util.ravel_pytree(S - S_)
                return res

            @jax.custom_jvp
            def newton_solver(x):

                y0 = np.zeros_like(Fp_inv_old.reshape(-1))
                # y0 = y_ini

                step = 0
                res_vec = implicit_residual(x, y0)

                # tol1 = 1e-5*np.linalg.norm(y_ini)
                # cus_tol = np.where(tol1 > tol, tol1, tol)

                def cond_fun(state):
                    step, res_vec, y = state
                    # # tol = 1e-8
                    # tol = 1e-9

                    # res = implicit_residual(x, y)

                    # return np.logical_and(np.linalg.norm(res_vec) > 1e-8, np.logical_or(step < 100, np.linalg.norm(res_vec) > 1e-5))

                    return np.linalg.norm(res_vec) > 1e-8

                # def body_fun(state):
                #     step, res_vec, y = state
                #     # TODO: would jvp + cg solver be faster?
                #     f_partial = lambda y: implicit_residual(x, y)
                #     jac = jax.jacfwd(f_partial)(y)
                    
                #     y_inc = np.linalg.solve(jac, -res_vec)

                #     # relax_param = np.where(np.linalg.norm(res) > 1e-2*cus_tol, 1, 0.8)
                #     # relax_param = 1.
                #     # y = y + relax_param*y_inc
                #     # return y

                #     y1 = y + y_inc
                #     y2 = y + 0.5*y_inc

                #     def true_fun():
                #         return y1

                #     def false_fun():
                #         res1 = f_partial(y1)
                #         res2 = f_partial(y2)
                #         return np.where(res1 < res2, y1, y2)

                #     y_update = jax.lax.cond(True, true_fun, false_fun)
                #     res_vec_update = f_partial(y_update)
                #     step_update = step + 1

                #     return jax.lax.cond(np.linalg.norm(res_vec_update) < np.linalg.norm(res_vec), lambda :(step_update, res_vec_update, y_update), 
                #         lambda : (step_update, f_partial(y2), y2))


                def body_fun(state):
                    # https://github.com/idaholab/moose/blob/next/modules/tensor_mechanics/src/materials/
                    # crystal_plasticity/ComputeMultipleCrystalPlasticityStress.C#L634
                    step, res_vec, y = state
                    # TODO: would jvp + cg solver be faster?
                    f_partial = lambda y: implicit_residual(x, y)
                    jac = jax.jacfwd(f_partial)(y)
                    y_inc = np.linalg.solve(jac, -res_vec)

                    relax_param_ini = 1.
                    sub_step_ini = 0

                    def sub_cond_fun(state):
                        _, crt_res_vec, sub_step = state
                        return np.logical_and(np.linalg.norm(crt_res_vec) >= np.linalg.norm(res_vec), sub_step < 5)

                    def sub_body_fun(state):
                        relax_param, res_vec, sub_step = state
                        res_vec = f_partial(y + relax_param*y_inc)
                        return 0.5*relax_param, res_vec, sub_step + 1

                    relax_param_f, res_vec_f, _ = jax.lax.while_loop(sub_cond_fun, sub_body_fun, (relax_param_ini, res_vec, sub_step_ini))
                    step_update = step + 1

                    return step_update, res_vec_f, y + 2.*relax_param_f*y_inc

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

            return first_PK_stress, update_int_vars

        def tensor_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, tol, y_ini):
            first_PK_stress, _ = get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat, tol, y_ini)
            return first_PK_stress(u_grad)

        def update_int_vars_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, tol, y_ini):
            _, update_int_vars = get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat, tol, y_ini)
            return update_int_vars(u_grad)

        return tensor_map, update_int_vars_map

    def update_int_vars_gp(self, sol):
        _, update_int_vars_map = self.get_maps()
        vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))

        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
  
        Fp_inv_new_gp, slip_resistance_new_gp, slip_new_gp, y_ini_gp = \
            vmap_update_int_vars_map(u_grads, self.Fp_inv_old_gp, self.slip_resistance_old_gp, self.slip_old_gp, self.rot_mats_gp, self.tol_gp, self.y_ini_gp)

        slip_inc_dt_index_0 = (slip_new_gp[0, 0, 0] - self.slip_old_gp[0, 0, 0])/self.dt
        print(f"slip inc dt index 0 = {slip_inc_dt_index_0}, max slip = {np.max(np.absolute(slip_new_gp))}")


        self.Fp_inv_old_gp, self.slip_resistance_old_gp, self.slip_old_gp, self.y_ini_gp = Fp_inv_new_gp, slip_resistance_new_gp, slip_new_gp, y_ini_gp

        # print(y_ini_gp[:10])

        F_p = np.linalg.inv(self.Fp_inv_old_gp[0, 0])
        print(f"Fp = \n{F_p}")
        slip_resistance_0 = self.slip_resistance_old_gp[0, 0, 0]
        print(f"slip_resistance index 0 = {slip_resistance_0}, max slip_resistance = {np.max(self.slip_resistance_old_gp)}")

        return F_p[2, 2], slip_resistance_0, slip_inc_dt_index_0 

    def compute_avg_stress(self, sol):
        """For post-processing only
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)

        partial_tensor_map, _ = self.get_maps()
        vmap_partial_tensor_map = jax.jit(jax.vmap(jax.vmap(partial_tensor_map)))
        P = vmap_partial_tensor_map(u_grads, self.Fp_inv_old_gp, self.slip_resistance_old_gp, self.slip_old_gp, self.rot_mats_gp, self.tol_gp, self.y_ini_gp)

        def P_to_sigma(P, F):
            return 1./np.linalg.det(F) * P @ F.T

        vvmap_P_to_sigma = jax.vmap(jax.vmap(P_to_sigma))        
        F = u_grads + np.eye(self.dim)[None, None, :, :]
        sigma = vvmap_P_to_sigma(P, F)

        sigma_cell_data = np.sum(sigma * self.JxW[:, :, None, None], 1) / np.sum(self.JxW, axis=1)[:, None, None]

        # num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1)
        avg_P = np.sum(P.reshape(-1, self.vec, self.dim) * self.JxW.reshape(-1)[:, None, None], 0) / np.sum(self.JxW)
  
        return sigma_cell_data
