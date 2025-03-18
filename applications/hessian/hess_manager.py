import jax
import jax.numpy as np
from jax_fem.solver import ad_wrapper, linear_solver
from jax_fem.hessian import forward_and_adjoint, incremental_forward_and_adjoint


def finite_difference_hessp(hess_vec_prod, θ, θ_hat):
    h = 1e-6
    θ_minus = jax.tree_util.tree_map(lambda x, y: x - h*y, θ, θ_hat)
    θ_plus  = jax.tree_util.tree_map(lambda x, y: x + h*y, θ, θ_hat)
    value_plus = hess_vec_prod.grad(θ_plus)
    value_minus = hess_vec_prod.grad(θ_minus)
    dθ_dθ_J_θ_hat = jax.tree_util.tree_map(lambda x, y: (x - y)/(2*h), value_plus, value_minus)
    print(f"FD = {dθ_dθ_J_θ_hat}")
    return dθ_dθ_J_θ_hat


class HessVecProduct:
    def __init__(self, problem, J_fn, solver_options, adjoint_solver_options):
        self.internal_vars = {'θ': None, 'u': None, 'λ': None, 'F_fn': None, 'A': None}
        self.problem = problem
        self.J_fn = J_fn
        self.solver_options = solver_options
        self.adjoint_solver_options = adjoint_solver_options
        self.state_linear_solver = lambda A, b: linear_solver(A, b, None, solver_options)
        self.adjoint_linear_solver = lambda A, b: linear_solver(A, b, None, adjoint_solver_options)
        self.fwd_pred = ad_wrapper(problem, solver_options, adjoint_solver_options)

    def J(self, θ):
        u, F_fn = forward_step(self.problem, θ, self.solver_options)
        return self.J_fn(u, θ)

    def grad(self, θ):
        def J(θ):
            return self.J_fn(self.fwd_pred(θ), θ)
        return jax.grad(J)(θ)

    def hessp(self, θ, θ_hat):
        print("Calling hessp...")
        tol = 1e-8
        if (self.internal_vars['θ'] is None) or tree_l2_norm_error(self.internal_vars['θ'], θ) > tol:
            print(f"hessp needs to solve forward and adjoint problem...")
            u, λ, F_fn, A = forward_and_adjoint(self.problem, θ, self.J_fn, self.solver_options, self.adjoint_solver_options)
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