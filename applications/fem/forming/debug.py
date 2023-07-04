import jax
import jax.numpy as np
import jax.flatten_util
import os
import matplotlib.pyplot as plt

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh


def debug():

    def f():
        return a

    a = 1.
    print(jax.jit(f)())

    a = 2.
    print(jax.jit(f)())


def safe_sqrt(x):  
    safe_x = np.where(x > 0., np.sqrt(x), 0.)
    return safe_x

def safe_divide(x, y):
    return np.where(y == 0., 0., x/y)


def simulation():

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

        def first_PK_stress(u_grad):
            x, _ = jax.flatten_util.ravel_pytree([u_grad, F_old, Be_old, alpha_old])
            Be, alpha = plastic_or_elastic_loading(x)
            tau = get_tau(Be)
            F = u_grad + np.eye(dim)
            P = tau @ np.linalg.inv(F).T 
            return P    

        def update_int_vars(u_grad):
            x, _ = jax.flatten_util.ravel_pytree([u_grad, F_old, Be_old, alpha_old])
            Be, alpha = plastic_or_elastic_loading(x)
            F = u_grad + np.eye(dim)
            return F, Be, alpha

        def compute_cauchy_stress(u_grad):
            F = u_grad + np.eye(dim)
            J = np.linalg.det(F)
            P = first_PK_stress(u_grad)
            sigma = 1./J*P @ F.T
            return sigma

        def get_tau(Be):
            J_Be = np.linalg.det(Be)
            be_bar = J_Be**(-1./3.) * Be
            be_bar_dev = be_bar - 1./dim*np.trace(be_bar)*np.eye(dim)
            tau = 0.5*K*(J_Be - 1)*np.eye(dim) + G*be_bar_dev
            return tau

        def get_tau_dev_norm(tau):
            tau_dev = tau - 1./dim*np.trace(tau)*np.eye(dim)
            tau_dev_norm = safe_sqrt(np.sum(tau_dev*tau_dev))
            return tau_dev_norm    

        def plastic_or_elastic_loading(x):
            u_grad, F_old, Be_old, alpha_old = unflatten_fn_x(x)
            F = u_grad + np.eye(dim)
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

                F = u_grad + np.eye(dim)
                F_inv = np.linalg.inv(F)
                F_old_inv = np.linalg.inv(F_old)

                Cp_inv = F_inv @ Be @ F_inv.T
                Cp_old_inv = F_old_inv @ Be_old @ F_old_inv.T

                J_Be = np.linalg.det(Be)
                be_bar = J_Be**(-1./3.) * Be

                tau = get_tau(Be)
                tau_dev = tau - 1./dim*np.trace(tau)*np.eye(dim)
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

                f_partial = lambda y: implicit_residual(x, y)
                jac = jax.jacfwd(f_partial)(y0)
                y_inc = np.linalg.solve(jac, -res_vec)

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

                print(f"step_f = {step_f}")

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
                return Be, alpha

            def plastic_loading(x):
                y = newton_solver(x)
                Be = to_tensor(y)
                tau = get_tau(Be)
                tau_dev_norm = get_tau_dev_norm(tau)
                alpha = (tau_dev_norm/np.sqrt(2./3.) - sig0)/H1

                print(tau)

                print(f"alpha = {alpha}")
                print(f"tau_dev_norm = {tau_dev_norm}")

                return Be, alpha


            print(f"yield_f = {yield_f}")


            return jax.lax.cond(yield_f < 0., elastic_loading, plastic_loading, x)

            # return np.where(yield_f < 0, elastic_loading(x), plastic_loading(x))
            
        return first_PK_stress, update_int_vars, compute_cauchy_stress

    def tensor_map(u_grad, F_old, Be_old, alpha_old):
        first_PK_stress, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old)
        return first_PK_stress(u_grad)

    def compute_cauchy_stress_map(u_grad, F_old, Fe_old, alpha_old):
        _, _, compute_cauchy_stress = get_partial_tensor_map(F_old, Fe_old, alpha_old)
        return compute_cauchy_stress(u_grad)
  
    dim = 3
    F_old = np.eye(dim)
    Be_old = np.eye(dim)
    alpha_old = 0.

    u_grad = np.array([[-0.005, 0., 0.],
                       [0., -0.005, 0.],
                       [0., 0., 0.01]])

    # u_grad = u_grad*0.1

    # P = jax.jit(jax.jacfwd(tensor_map))(u_grad, F_old, Be_old, alpha_old)

    sigma = compute_cauchy_stress_map(u_grad, F_old, Be_old, alpha_old)

    print(sigma)


if __name__=="__main__":
    # simulation()
    simulation()