"""This module contains functions to compute derivatives.

Functions
---------
ad_wrapper_jvp
    Wrapper for forward solve with a custom JVP rule
implicit_jvp_helper
    Helper function to compute JVP of FEA forward solve
jax_array_list_to_numpy_diff
    Convert a list of JAX arrays to a single numpy array. This function
    is JITable. However, reverse-mode differentiation is not supported,
    as it uses pure_callback functionality.
jax_array_list_to_numpy_diff_jvp
    JVP of jax_array_list_to_numpy_diff

Todo:
1. Make PETSc work for backward solve
2. Working with sparsity [Might be important]
3. Use saved tangent matrices for backward solve
4. Create Primitive for all external calls
"""
#                                                                       Modules
# =============================================================================
# Standard
from functools import partial
from typing import List
# Third-party
import jax
from jax import Array
import numpy as onp
# Local
from jax_fem.solver import (apply_bc, get_flatten_fn, solver,
                            jax_solve,
                            get_jacobi_precond, jacobi_preconditioner)
# =============================================================================


def ad_wrapper_jvp(problem, linear: bool = False,
                   use_petsc: bool = True) -> callable:
    """Wrapper for forward solve with a custom JVP rule.
    Both forward and backward autodiffs are supported.
    Works well to find Hessian-vector products as well
    However, this function is not JITable.

    Parameters
    ----------
    problem
        FEA problem object (of type FEM)
    linear
        If True, use linear solver. Otherwise, use nonlinear solver
    use_petsc
        If True, use PETSc solver. Otherwise, use JAX solver
        Note: PETSc solvers are not supported for backward solve

    Returns
    -------
    The output of the forward solve ie. the solution to FE problem
    """
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
        sol_dot = implicit_jvp_helper(
            problem, sol, params, params_dot)
        return sol.reshape(sol.shape), sol_dot.reshape(sol.shape)

    return forward_solve


def implicit_jvp_helper(problem, sol0: Array,
                        params0: Array, params_dot0: Array) -> Array:
    """Helper function to compute JVP of FEA forward solve.

    The forward solve is setup such that it can use either
    PETSc or JAX solvers. However, the backward solve is setup such
    that only JAX solvers can be used.

    Parameters
    ----------
    problem
        FEA problem object
    sol0
        Solution of the forward solve
    params0
        Parameters of the forward solve
    params_dot0
        Parameters of the backward solve

    Returns
    -------
    The output tangents of the forward solve
    """
    # create residual function

    def residual(dofs, params):
        """Function calculates the r(u(p), p) based on weak form.
        r should be equal to 0 at the solution.
        """
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    # For time-dependent problems
    problem.set_params(params0)
    problem.newton_update(sol0)
    # Construct terms for JVP calculation
    partial_fn_of_params = partial(residual, sol0)  # r(u=sol, p)

    def change_arg_order_fn(params, dofs): return residual(dofs, params)
    partial_fn_of_u = partial(change_arg_order_fn, params0)  # r(u, p=params)
    # dr/du . x
    _, jvp_fn = jax.linearize(partial_fn_of_u, sol0)

    def backward_matvec(v): return jvp_fn(v.reshape(-1, problem.vec))
    # dr/drho . v --> need a negative sign here - will provide later
    _, backward_rhs = jax.jvp(partial_fn_of_params,
                              (params0, ), (params_dot0, ))
    # Call a JAX solver
    precond_matrix = get_jacobi_precond(jacobi_preconditioner(problem))

    def jax_solver_modified(matvec, v): return jax_solve(problem, matvec,
                                                         v.reshape(-1), None,
                                                         False,
                                                         precond_matrix)
    chosen_bb_solver = jax_solver_modified
    # Find adjoint value
    tangent_out = jax.lax.custom_linear_solve(backward_matvec, -1*backward_rhs,
                                              chosen_bb_solver,
                                              transpose_solve=chosen_bb_solver)
    return tangent_out


@jax.custom_jvp
def jax_array_list_to_numpy_diff(jax_array_list:
                                 List[Array]) -> onp.ndarray:
    """Convert a list of JAX arrays to a single numpy array.
    This function is JITable. However, reverse-mode differentiation
    is not supported. This is used in the split_and_compute_cell
    function in the jax_fem.core module as well as in the JVP rule
    for the forward solve.

    Parameters
    ----------
    jax_array_list
        List of jax.numpy arrays

    Returns
    -------
        numpy_array that vertically stacks the jax_array_list
    """
    # For compatibility with JIT

    def _numpy_vstack(x): return onp.vstack(x).astype(x[0].dtype)
    out_shape = list(jax_array_list[0].shape)
    out_shape[0] *= len(jax_array_list)
    output_shape_type = jax.ShapeDtypeStruct(shape=tuple(out_shape),
                                             dtype=jax_array_list[0].dtype)
    # Convert jax array to numpy array
    # numpy_array = onp.vstack(jax_array_list)
    numpy_array = jax.pure_callback(_numpy_vstack, output_shape_type,
                                    jax_array_list)
    return numpy_array


@jax_array_list_to_numpy_diff.defjvp
def jax_array_list_to_numpy_diff_jvp(primals, tangents):
    """JVP of jax_array_list_to_numpy_diff"""
    jax_array_list, = primals
    jax_array_list_dot, = tangents
    numpy_array = jax_array_list_to_numpy_diff(jax_array_list)
    # Reroute the tangents
    numpy_array_dot = jax_array_list_to_numpy_diff(jax_array_list_dot)
    return numpy_array, numpy_array_dot


# USING JAXOPT - only supports custom_vjp
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
