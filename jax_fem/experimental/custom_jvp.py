import jax
import jax.numpy as np
import os


def implicit_residual(x, y):
    A = np.diag(np.array([1., 2., 3]))
    # Assume that Ay - x = 0
    return A @ y - x

@jax.custom_jvp
def newton_solver(x):
    y_0 = np.zeros(3)
    step_0 = 0
    res_vec_0 = implicit_residual(x, y_0)
    tol = 1e-8

    def cond_fun(state):
        step, res_vec, y = state
        return np.linalg.norm(res_vec) > tol

    def body_fun(state):
        step, res_vec, y = state
        f_partial = lambda y: implicit_residual(x, y)
        jac = jax.jacfwd(f_partial)(y) # Works for small system
        y_inc = np.linalg.solve(jac, -res_vec) # Works for small system
        res_vec = f_partial(y + y_inc)
        step_update = step + 1
        return step_update, res_vec, y + y_inc

    step_f, res_vec_f, y_f = jax.lax.while_loop(cond_fun, body_fun, (step_0, res_vec_0, y_0))
    return y_f

@newton_solver.defjvp
def f_jvp(primals, tangents):
    x, = primals
    v, = tangents
    y = newton_solver(x)
    jac_x = jax.jacfwd(implicit_residual, argnums=0)(x, y) # Works for small system
    jac_y = jax.jacfwd(implicit_residual, argnums=1)(x, y) # Works for small system
    jvp_result = np.linalg.solve(jac_y, -(jac_x @ v[:, None]).reshape(-1)) # Works for small system
    return y, jvp_result


x = np.ones(3)
y = newton_solver(x)
print(f"\ny = {y}")

jac_y_over_x_fwd = jax.jacfwd(newton_solver)(x)
jac_y_over_x_rev = jax.jacrev(newton_solver)(x)

print(f"\njac_y_over_x_fwd = \n{jac_y_over_x_fwd}")
print(f"\njac_y_over_x_rev = \n{jac_y_over_x_rev}")