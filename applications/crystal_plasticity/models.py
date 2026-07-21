import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
import os
import sys
from functools import partial

from jax_fem.problem import Problem


from jax import config
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


class CrystalPlasticity(Problem):
    def custom_init(self, quat, cell_ori_inds):
        r = 1.
        self.gss_initial = 60.8 

        input_slip_sys = onp.loadtxt(os.path.join(crt_dir, 'data/csv/input_slip_sys.txt'))

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

        Fp_inv_gp = onp.repeat(onp.repeat(onp.eye(self.dim)[None, None, :, :], len(self.fes[0].cells), axis=0), self.fes[0].num_quads, axis=1)
        slip_resistance_gp = self.gss_initial*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads, num_slip_sys))
        slip_gp = onp.zeros_like(slip_resistance_gp)
        rot_mats_gp = onp.repeat(rot_mats[:, None, :, :], self.fes[0].num_quads, axis=1)
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

        self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp]

    def get_tensor_map(self):
        tensor_map, _ = self.get_maps()
        return tensor_map

    def get_maps(self):
        h = 541.5
        t_sat = 109.8
        gss_a = 2.5
        ao = 0.001
        xm = 0.1

        def get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat):
            _, unflatten_fn = jax.flatten_util.ravel_pytree(Fp_inv_old)
            _, unflatten_fn_params = jax.flatten_util.ravel_pytree([Fp_inv_old, Fp_inv_old, slip_resistance_old, slip_old, rot_mat])
    
            def first_PK_stress(u_grad):
                x, _ = jax.flatten_util.ravel_pytree([u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat])
                y = newton_solver(x)
                S = unflatten_fn(y)
                _, _, _, Fe, F = helper(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, S)
                sigma = 1./np.linalg.det(Fe)*Fe @ S @ Fe.T
                P = np.linalg.det(F)*sigma @ np.linalg.inv(F).T 
                return P    

            def update_int_vars(u_grad):
                x, _ = jax.flatten_util.ravel_pytree([u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat])
                y = newton_solver(x)
                S = unflatten_fn(y)
                Fp_inv_new, slip_resistance_new, slip_new, Fe, F = helper(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, S)
                return Fp_inv_new, slip_resistance_new, slip_new, rot_mat

            def helper(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, S):
                tau = np.sum(S[None, :, :] * rotate_tensor_rank_2_vmap(rot_mat, self.Schmid_tensors), axis=(1, 2))
                gamma_inc = ao*self.dt*np.absolute(tau/slip_resistance_old)**(1./xm)*np.sign(tau)

                tmp = h*np.absolute(gamma_inc) * np.absolute(1 - slip_resistance_old/t_sat)**gss_a * np.sign(1 - slip_resistance_old/t_sat)
                g_inc = (self.q @ tmp[:, None]).reshape(-1)

                # tmp = h*np.absolute(gamma_inc) / np.cosh(h*np.sum(slip_old)/(t_sat - self.gss_initial))**2
                # g_inc = (self.q @ tmp[:, None]).reshape(-1)

                slip_resistance_new = slip_resistance_old + g_inc
                slip_new = slip_old + gamma_inc
                F = u_grad + np.eye(self.dim)
                L_plastic_inc = np.sum(gamma_inc[:, None, None] * rotate_tensor_rank_2_vmap(rot_mat, self.Schmid_tensors), axis=0)
                Fp_inv_new = Fp_inv_old @ (np.eye(self.dim) - L_plastic_inc)
                Fe = F @ Fp_inv_new 
                return Fp_inv_new, slip_resistance_new, slip_new, Fe, F

            def implicit_residual(x, y):
                u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat = unflatten_fn_params(x)
                S = unflatten_fn(y)
                _, _, _, Fe, _ = helper(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, S)
                S_ = np.sum(rotate_tensor_rank_4(rot_mat, self.C) * 1./2.*(Fe.T @ Fe - np.eye(self.dim))[None, None, :, :], axis=(2, 3))           
                res, _ = jax.flatten_util.ravel_pytree(S - S_)
                return res

            @jax.custom_jvp
            def newton_solver(x):
                # Critical change: The following line causes JAX (version 0.4.13) tracer error
                # y0 = np.zeros_like(Fp_inv_old.reshape(-1))
                y0 = np.zeros(self.dim*self.dim)

                step = 0
                res_vec = implicit_residual(x, y0)
                tol = 1e-8

                def cond_fun(state):
                    step, res_vec, y = state
                    return np.linalg.norm(res_vec) > tol

                def body_fun(state):
                    # Line search with decaying relaxation parameter (the "cut half" method).
                    # This is necessary since vanilla Newton's method may sometimes not converge.
                    # MOOSE has an implementation in C++, see the following link
                    # https://github.com/idaholab/moose/blob/next/modules/tensor_mechanics/src/materials/
                    # crystal_plasticity/ComputeMultipleCrystalPlasticityStress.C#L634
                    step, res_vec, y = state
                    f_partial = lambda y: implicit_residual(x, y)
                    jac = jax.jacfwd(f_partial)(y)
                    y_inc = np.linalg.solve(jac, -res_vec)

                    relax_param_ini = 1.
                    sub_step_ini = 0
                    max_sub_step = 5

                    def sub_cond_fun(state):
                        _, crt_res_vec, sub_step = state
                        return np.logical_and(np.linalg.norm(crt_res_vec) >= np.linalg.norm(res_vec), sub_step < max_sub_step)

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

        def tensor_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat):
            first_PK_stress, _ = get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat)
            return first_PK_stress(u_grad)

        def update_int_vars_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat):
            _, update_int_vars = get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat)
            return update_int_vars(u_grad)

        return tensor_map, update_int_vars_map

    def update_int_vars_gp(self, sol, params):
        _, update_int_vars_map = self.get_maps()
        vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp = \
            vmap_update_int_vars_map(u_grads, *params)
        # TODO
        return [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp]

    def set_params(self, params):
        self.internal_vars = params

    def inspect_interval_vars(self, params):
        """For post-processing only
        """
        Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp = params
        F_p = np.linalg.inv(Fp_inv_gp[0, 0])
        print(f"Fp = \n{F_p}")
        slip_resistance_0 = slip_resistance_gp[0, 0, 0]
        print(f"slip_resistance index 0 = {slip_resistance_0}, max slip_resistance = {np.max(slip_resistance_gp)}")
        return F_p[2, 2], slip_resistance_0, slip_gp[0, 0, 0]

    def compute_avg_stress(self, sol, params):
        """For post-processing only
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)

        partial_tensor_map, _ = self.get_maps()
        vmap_partial_tensor_map = jax.jit(jax.vmap(jax.vmap(partial_tensor_map)))
        P = vmap_partial_tensor_map(u_grads, *params)

        def P_to_sigma(P, F):
            return 1./np.linalg.det(F) * P @ F.T

        vvmap_P_to_sigma = jax.vmap(jax.vmap(P_to_sigma))        
        F = u_grads + np.eye(self.dim)[None, None, :, :]
        sigma = vvmap_P_to_sigma(P, F)

        sigma_cell_data = np.sum(sigma * self.fes[0].JxW[:, :, None, None], 1) / np.sum(self.fes[0].JxW, axis=1)[:, None, None]

        # num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1)
        avg_P = np.sum(P.reshape(-1, self.fes[0].vec, self.dim) * self.fes[0].JxW.reshape(-1)[:, None, None], 0) / np.sum(self.fes[0].JxW)
        return sigma_cell_data
