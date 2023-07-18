from jax_am.fem.solver import (apply_bc, get_flatten_fn, solver,
                               petsc_solve, jax_solve,
                               get_jacobi_precond, jacobi_preconditioner)
import jax
import jax.numpy as np
import numpy as onp
from functools import partial
# https://stackoverflow.com/questions/7811247/how-to-fill-specific-positional-arguments-with-partial-in-python
# ToDo: incorporate sparsity
# Backward solve in custom_linear_solve ?
# Wrap Petsc in pure_function


def implicit_jvp_helper(problem, sol0, params0, params_dot0, use_petsc):

    # create residual function

    def residual(dofs, params):
        """r(u, p) = 0, at u = sol
        """
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    problem.set_params(params0)
    problem.newton_update(sol0) # ---------------------------> Should be autodiff capable

    # Construct terms for JVP calculation
    partial_fn_of_params = partial(residual, sol0)  # r(u=sol, rho)
    change_arg_order_fn = lambda params, dofs: residual(dofs, params)
    partial_fn_of_u = partial(change_arg_order_fn, params0)  # r(u, rho=params)
    # dr/du . x
    _, jvp_fn = jax.linearize(partial_fn_of_u, sol0)
    backward_matvec = lambda v: jvp_fn(v.reshape(-1, problem.vec))
    # dr/drho . v --> need a negative sign here - will provide later in actual func
    _, backward_rhs = jax.jvp(partial_fn_of_params, (params0, ), (params_dot0, ))

    # Call a black-box solver
    if use_petsc:
        # I may need to wrap the petsc one in pure_callback ?
        petsc_solver_modified = lambda matvec, v : petsc_solve(matvec, v.reshape(-1),
                                                                'minres', 'ilu')
        chosen_bb_solver = petsc_solver_modified
    else:
        precond_matrix = get_jacobi_precond(jacobi_preconditioner(problem))
        jax_solver_modified = lambda matvec, v : jax_solve(problem, matvec,
                                                            v.reshape(-1), None, False,
                                                            precond_matrix)
        chosen_bb_solver = jax_solver_modified
    # Find adjoint value
    tangent_out = jax.lax.custom_linear_solve(backward_matvec, -1*backward_rhs,
                                          chosen_bb_solver,
                                          transpose_solve=chosen_bb_solver)
    return tangent_out


def ad_wrapper_jvp(problem, linear=False, use_petsc=True):
    @jax.custom_jvp
    def forward_solve(params):
        problem.set_params(params)
        # solver is not JITTable
        sol = solver(problem, linear=linear, use_petsc=use_petsc)
        return sol

    @forward_solve.defjvp
    def forward_solve_jvp(primals, tangents):
        params, = primals
        params_dot, = tangents
        sol = forward_solve(params)
        sol_dot = implicit_jvp_helper(problem, sol, params, params_dot, use_petsc)
        return sol.reshape(sol.shape), sol_dot.reshape(sol.shape)

    return forward_solve




@jax.custom_jvp
def jax_array_list_to_numpy_diff(jax_array_list):
    # Convert jax array to numpy array
    numpy_array = onp.vstack(jax_array_list)
    return numpy_array

@jax_array_list_to_numpy_diff.defjvp
def jax_array_list_to_numpy_diff_jvp(primals, tangents):
    jax_array_list, = primals
    jax_array_list_dot, = tangents
    numpy_array = jax_array_list_to_numpy_diff(jax_array_list)
    # Reroute the tangents
    numpy_array_dot = jax_array_list_to_numpy_diff(jax_array_list_dot)
    return numpy_array, numpy_array_dot
















# def optimality_fn_wrapper(problem):
#     def residual(dofs, params):
#         """r(u, p) = 0"""
#         problem.set_params(params)
#         res_fn = problem.compute_residual
#         res_fn = get_flatten_fn(res_fn, problem)
#         res_fn = apply_bc(res_fn, problem)
#         return res_fn(dofs) # need to adjust the shapes!!!!!!!!!!!!!!!!
#     return residual


# def solver_wrapper(problem, linear=True, use_petsc=False):

#     @implicit_diff.custom_root(optimality_fn_wrapper(problem))
#     def solve_for_u(dofs, params): # Ku = F
#         """Solve for u given dofs and params"""
#         del dofs
#         problem.set_params(params)
#         sol = solver(problem, linear=linear, use_petsc=use_petsc)
#         return sol

#     return solve_for_u
