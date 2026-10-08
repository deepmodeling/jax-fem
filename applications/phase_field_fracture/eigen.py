import jax
import jax.numpy as np

jax.config.update("jax_enable_x64", True)


def get_eigen_f_jax(fn):
    fn_vmap = jax.vmap(fn)
    def eigen_f_jax(x):
        evals, evecs = np.linalg.eigh(x)
        evecs = evecs.T
        M = np.einsum('bi,bj->bij', evecs, evecs)
        # [batch, dim, dim] * [batch, 1, 1] -> [dim, dim]
        result = np.sum(M * fn_vmap(evals)[:, None, None], axis=0)
        return result
    return eigen_f_jax


def get_eigen_f_custom(fn):
    grad_fn = jax.grad(fn)
    fn_vmap = jax.vmap(fn)
    grad_fn_vmap = jax.vmap(grad_fn)

    @jax.custom_jvp
    def eigen_f(x):
        evals, evecs = np.linalg.eigh(x)
        evecs = evecs.T
        M = np.einsum('bi,bj->bij', evecs, evecs)
        # [batch, dim, dim] * [batch, 1, 1] -> [dim, dim]
        result = np.sum(M * fn_vmap(evals)[:, None, None], axis=0)
        return result

    @eigen_f.defjvp
    def f_jvp(primals, tangents):
        """Impelemtation of Miehe's paper (https://doi.org/10.1002/cnm.404) Eq. (19)
        """
        x, = primals
        v, = tangents

        evals, evecs = np.linalg.eigh(x)
        fvals = fn_vmap(evals)
        grads = grad_fn_vmap(evals)
        evecs = evecs.T

        M = np.einsum('bi,bj->bij', evecs, evecs)

        result = np.sum(M * fvals[:, None, None], axis=0)

        MM = np.einsum('bij,bkl->bijkl', M, M)
        # [batch, dim, dim, dim, dim] * [batch, 1, 1, 1, 1] -> [dim, dim, dim, dim]
        term1 = np.sum(MM * grads[:, None, None, None, None], axis=0)

        G = np.einsum('aik,bjl->abijkl', M, M) + np.einsum('ail,bjk->abijkl', M, M)

        diff_evals = evals[:, None] - evals[None, :]
        diff_fvals = fvals[:, None] - fvals[None, :]
        diff_grads = grads[:, None]

        theta = np.where(diff_evals == 0., diff_grads, diff_fvals/diff_evals)

        tmp = G * theta[:, :, None, None, None, None]
        tmp1 = np.sum(tmp, axis=(0, 1))
        tmp2 = np.einsum('aa...->...', tmp)
        term2 = 0.5*(tmp1 - tmp2)

        P = term1 + term2
        jvp_result = np.einsum('ijkl,kl', P, v)

        return result, jvp_result

    return eigen_f


def test_eigen_f():
    """When repeated eigenvalues occur, JAX fail to find out the correct derivative (returning NaN)
    See the discussion: 
    https://github.com/google/jax/issues/669
    Also see the basic derivation for the case of distinct eigenvalues: 
    https://mathoverflow.net/questions/229425/derivative-of-eigenvectors-of-a-matrix-with-respect-to-its-components
    
    Here, we followed Miehe's approach to provide a solution to repeated eigenvalue case:
    https://doi.org/10.1002/cnm.404

    You will see how get_eigen_f_jax works for the variables a and b, but fails for c.
    Yet, our implementation of get_eigen_f_custom works for all the variables a, b and c.
    """
    a = np.array([[1., -2., 3.],
                  [-2., 5., 7.],
                  [3.,  7., 10.]])

    key = jax.random.PRNGKey(0)
    b = jax.random.uniform(key, shape=(5, 5), minval=-0.1, maxval=0.1)
    b = 0.5*(b + b.T)

    c = np.zeros((3, 3))

    fn = lambda x: x

    input_vars = [
        ('fixed 3x3', a),
        ('random symmetric 5x5', b),
        ('repeated zero 3x3', c),
    ]

    eigen_f_jax = get_eigen_f_jax(fn)
    eigen_f_custom = get_eigen_f_custom(fn)

    print("\nEigenfunction derivative check")
    print(f"{'Case':<24} {'Native JAX':<12} {'Custom':<12} {'Max difference'}")
    for name, x in input_vars:
        jax_result = jax.jacfwd(eigen_f_jax)(x)
        custom_result = jax.jacfwd(eigen_f_custom)(x)
        jax_finite = bool(np.all(np.isfinite(jax_result)))
        custom_finite = bool(np.all(np.isfinite(custom_result)))
        if jax_finite and custom_finite:
            max_diff = f"{float(np.max(np.abs(jax_result - custom_result))):.3e}"
        else:
            max_diff = "n/a"
        print(f"{name:<24} {str(jax_finite):<12} "
              f"{str(custom_finite):<12} {max_diff}")


def f1(x):
    unsafe_plus = lambda x: np.maximum(x, 0.)
    unsafe_minus = lambda x: np.minimum(x, 0.)
    tr_x_plus = unsafe_plus(np.trace(x))
    tr_x_minus = unsafe_minus(np.trace(x))
    return 0.5*tr_x_plus**2 + 0.5*tr_x_minus**2

def f2(x):
    safe_plus = lambda x: 0.5*(x + np.abs(x))
    safe_minus = lambda x: 0.5*(x - np.abs(x))
    tr_x_plus = safe_plus(np.trace(x))
    tr_x_minus = safe_minus(np.trace(x))
    return 0.5*tr_x_plus**2 + 0.5*tr_x_minus**2

def f_gold(x):
    tr_x = np.trace(x)
    return 0.5*tr_x**2

def test_bracket_operator():
    # Different behaviors observed when derivative is taken at x=0.
    # The "abs" way of implemetation is preferred
    unsafe_grad = jax.grad(lambda x: np.maximum(x, 0.))(0.)
    safe_grad = jax.grad(lambda x: 0.5*(x + np.abs(x)))(0.)

    # Further tests
    a = np.zeros((3, 3))
    unsafe_hessian = jax.jacfwd(jax.grad(f1))(a)
    safe_hessian = jax.jacfwd(jax.grad(f2))(a)
    exact_hessian = jax.jacfwd(jax.grad(f_gold))(a)

    indices = np.arange(a.shape[0])
    unsafe_block = unsafe_hessian[indices[:, None], indices[:, None],
                                    indices[None, :], indices[None, :]]
    safe_block = safe_hessian[indices[:, None], indices[:, None],
                              indices[None, :], indices[None, :]]
    exact_block = exact_hessian[indices[:, None], indices[:, None],
                                indices[None, :], indices[None, :]]

    print("\nBracket operator check at zero")
    print(f"Derivative of np.maximum(x, 0): {float(unsafe_grad):.1f}")
    print(f"Derivative of 0.5*(x + abs(x)): {float(safe_grad):.1f}")
    print(f"\nHessian block H[ii,jj] using np.maximum/minimum:\n{unsafe_block}")
    print(f"\nHessian block H[ii,jj] using abs-based brackets:\n{safe_block}")
    print(f"\nGround-truth Hessian block H[ii,jj]:\n{exact_block}")


if __name__ == "__main__":
    test_eigen_f()
    test_bracket_operator()
