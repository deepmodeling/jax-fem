import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
import scipy

from jax_fem.hessian import incremental_forward_and_adjoint

logger.setLevel(logging.DEBUG)


class HessVecProductSimple:
    def __init__(self, u_init, J_fn):
        self.internal_vars = {'θ': None, 'u': None, 'λ': None, 'F_fn': None, 'A': None}
        self.J_fn = J_fn
        self.state_linear_solver = lambda A, b: np.linalg.solve(A, b)
        self.adjoint_linear_solver = lambda A, b: np.linalg.solve(A, b)
        
    def J(self, θ):
        u, F_fn = forward_step_simple(θ, u_init)
        return self.J_fn(u, θ)

    def grad(self, θ):
        u, λ, F_fn, A = forward_and_adjoint_simple(θ, self.J_fn, u_init)
        primals_out, f_vjp = jax.vjp(lambda θ: F_fn(u, θ), θ) # λ_i * (∂/∂θ_k)F_i
        vjp_θ, = f_vjp(λ)
        dJ_dθ = jax.grad(self.J_fn, argnums=1)(u, θ) # ∂J/∂θ_k
        vjp_result = jax.tree_util.tree_map(lambda x, y: x + y, vjp_θ, dJ_dθ) # dJ/dθ_k
        return vjp_result

    def hessp(self, θ, θ_hat):
        print("Calling hessp...")
        tol = 1e-8
        if (self.internal_vars['θ'] is None) or tree_l2_norm_error(self.internal_vars['θ'], θ) > tol:
            print(f"hessp needs to solve forward and adjoint problem...")
            u, λ, F_fn, A = forward_and_adjoint_simple(θ, self.J_fn, u_init)
            self.internal_vars['θ'] = θ
            self.internal_vars['u'] = u
            self.internal_vars['λ'] = λ
            self.internal_vars['F_fn'] = F_fn
            self.internal_vars['A'] = A
        else:
            print(f"hessp does NOT need to solve forward and adjoint problem...")
            θ = self.internal_vars['θ']
            u = self.internal_vars['u']
            λ = self.internal_vars['λ']
            F_fn = self.internal_vars['F_fn']
            A = self.internal_vars['A']

        dθ_dθ_J_θ_hat = incremental_forward_and_adjoint(u, θ, λ, θ_hat, self.J_fn, F_fn, A, self.state_linear_solver, self.adjoint_linear_solver)
        return dθ_dθ_J_θ_hat


def forward_step_simple(θ, u_init):
    def F_fn(u, θ):
        return np.array([θ[0]**2 * u[0] + θ[1] - 1, θ[1]**2 * u[0]**2 + θ[1] * u[1]**2 - 2])

    _, unflatten = jax.flatten_util.ravel_pytree(u_init)

    def u_fn(θ):
        # Newton solve
        tol = 1e-8
        max_iter = 1000
        u_flat, _ = jax.flatten_util.ravel_pytree(u_init)

        def flat_F_fn(u_flat):
            u = unflatten(u_flat)
            F = F_fn(u, θ)
            return jax.flatten_util.ravel_pytree(F)[0]

        # Main Newton loop
        for _ in range(max_iter):
            # Compute current residual
            F_flat = flat_F_fn(u_flat)
            residual_norm = np.linalg.norm(F_flat)

            print(f"res = {residual_norm}")
            if residual_norm < tol:
                break

            # Compute dense Jacobian
            J = jax.jacfwd(flat_F_fn)(u_flat)
            
            # Solve linear system (with small regularization for stability)
            Δu_flat = np.linalg.solve(J, -F_flat)
            
            # Update solution
            u_flat += Δu_flat

        return unflatten(u_flat)

    # Solve forward problem
    print(f"\n################## Solve forward problem...")
    u = u_fn(θ)

    return u, F_fn


def adjoint_step_simple(u, θ, J_fn, F_fn):
    # Solve adjoint problem
    _, unflatten = jax.flatten_util.ravel_pytree(u)

    def flat_F_fn(u_flat):
        u = unflatten(u_flat)
        F = F_fn(u, θ)
        return jax.flatten_util.ravel_pytree(F)[0]

    print(f"\n################## Solve adjoint problem...")
    u_flat, _ = jax.flatten_util.ravel_pytree(u)

    A = jax.jacfwd(flat_F_fn)(u_flat)

    λ_rhs = jax.grad(J_fn)(u, θ)
    λ_rhs_vec = jax.flatten_util.ravel_pytree(λ_rhs)[0]
    λ_vec = np.linalg.solve(A.transpose(), -λ_rhs_vec)
    λ = unflatten(λ_vec)

    return λ, A


def forward_and_adjoint_simple(θ, J_fn, u_init):
    u, F_fn = forward_step_simple(θ, u_init)
    λ, A = adjoint_step_simple(u, θ, J_fn, F_fn)
    return u, λ, F_fn, A



def analytical_hessp_simple(θ, θ_hat):
    def u_fn(θ):
        return np.array([(1. - θ[1])/θ[0]**2, np.sqrt( (2. -  ((1. - θ[1])/θ[0]**2)**2 * θ[1]**2) / θ[1] )])

    def J(θ):
        return J_fn(u_fn(θ), θ)

    dθ_dθ_J_θ_hat = jax.jacrev(jax.grad(J))(θ) @ θ_hat
    print(f"AN = {dθ_dθ_J_θ_hat}")
    return dθ_dθ_J_θ_hat


def finite_difference_hessp_simple(hess_vec_prod_simple, θ, θ_hat):
    h = 1e-6
    θ_minus = jax.tree_util.tree_map(lambda x, y: x - h*y, θ, θ_hat)
    θ_plus  = jax.tree_util.tree_map(lambda x, y: x + h*y, θ, θ_hat)
    value_plus = hess_vec_prod_simple.grad(θ_plus)
    value_minus = hess_vec_prod_simple.grad(θ_minus)
    dθ_dθ_J_θ_hat = jax.tree_util.tree_map(lambda x, y: (x - y)/(2*h), value_plus, value_minus)
    print(f"FD = {dθ_dθ_J_θ_hat}")
    return dθ_dθ_J_θ_hat


def J_fn(u, θ):
    u_vec = jax.flatten_util.ravel_pytree(u)[0]
    θ_vec = jax.flatten_util.ravel_pytree(θ)[0]
    return np.sum(u_vec**3) + np.sum(θ_vec**3) + np.sum(u_vec**2) * np.sum(np.exp(θ_vec))


θ = np.array([3., .2])
θ_hat = np.array([0.2, 0.3])
u_init = np.array([0.1, 0.1])

hess_vec_prod_simple = HessVecProductSimple(u_init, J_fn)

hess_vec_prod_simple.hessp(θ, θ_hat)
finite_difference_hessp_simple(hess_vec_prod_simple, θ, θ_hat)
analytical_hessp_simple(θ, θ_hat)


