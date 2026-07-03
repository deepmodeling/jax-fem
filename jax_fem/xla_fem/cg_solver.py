import jax
import jax.numpy as jnp


def cg(A_fn, b, x0, maxiter=20):

    def body(carry):
        x, r, p, k = carry
        Ap = A_fn(p)

        alpha = jnp.dot(r, r) / (jnp.dot(p, Ap) + 1e-12)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        beta = jnp.dot(r_new, r_new) / (jnp.dot(r, r) + 1e-12)
        p_new = r_new + beta * p

        return (x_new, r_new, p_new, k + 1)

    def cond(carry):
        return carry[-1] < maxiter

    r0 = b - A_fn(x0)
    p0 = r0

    init = (x0, r0, p0, 0)

    return jax.lax.while_loop(cond, body, init)[0]