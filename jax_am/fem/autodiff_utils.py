from jax_am.fem.solver import (apply_bc, get_flatten_fn, solver,
                               petsc_solve, jax_solve,
                               get_jacobi_precond, jacobi_preconditioner)
import jax
import numpy as onp
from functools import partial
# https://stackoverflow.com/questions/7811247/how-to-fill-specific-positional-arguments-with-partial-in-python
# ToDo:
# 1. Make PETSc work for backward solve [Not too important]
#   a. petsc_solve should be made as a callback
#   b. JVP has to be defined for this callback
# 2. Working with sparsity [Might be important]
#   a. Matrix creation function needs to be a callback
#   b. JVP has to be defined for this callback


def implicit_jvp_helper(problem, sol0, params0, params_dot0):

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
    problem.newton_update(sol0)

    # Construct terms for JVP calculation
    partial_fn_of_params = partial(residual, sol0)  # r(u=sol, rho)
    change_arg_order_fn = lambda params, dofs: residual(dofs, params)
    partial_fn_of_u = partial(change_arg_order_fn, params0)  # r(u, rho=params)
    # dr/du . x
    _, jvp_fn = jax.linearize(partial_fn_of_u, sol0)
    backward_matvec = lambda v: jvp_fn(v.reshape(-1, problem.vec))
    # dr/drho . v --> need a negative sign here - will provide later in actual func
    _, backward_rhs = jax.jvp(partial_fn_of_params, (params0, ), (params_dot0, ))

    # Call a JAX solver
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


def petsc_solve_pure(A, b, ksp_type, pc_type):
    result_dtype_shape = jax.ShapeDtypeStruct(shape=b.shape, dtype=b.dtype)
    solution = jax.pure_callback(petsc_solve, result_dtype_shape, A, b, ksp_type, pc_type)
    return solution




@jax.custom_jvp
def jax_array_list_to_numpy_diff(jax_array_list):
    # For compatibility with JIT
    _numpy_vstack = lambda x: onp.vstack(x).astype(x[0].dtype)
    out_shape = list(jax_array_list[0].shape)
    out_shape[0] *= len(jax_array_list)
    output_shape_type = jax.ShapeDtypeStruct(shape=tuple(out_shape),
                                             dtype=jax_array_list[0].dtype)
    # Convert jax array to numpy array
    #numpy_array = onp.vstack(jax_array_list)
    numpy_array = jax.pure_callback(_numpy_vstack, output_shape_type,
                                    jax_array_list)
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
