import jax
import jax.numpy as np
import numpy as onp
import time
from functools import partial


def apply_bc(res_fn, problem):
    def A_fn(dofs):
        """Apply Dirichlet boundary conditions
        """
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        res = res_fn(dofs).reshape(sol.shape)
        for i in range(len(problem.node_inds_list)):
            res = (res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set
                   (sol[problem.node_inds_list[i], problem.vec_inds_list[i]], unique_indices=True))
            res = res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].add(-problem.vals_list[i])
        return res.reshape(-1)
    return A_fn


def row_elimination(res_fn, problem):
    def fn_dofs_row(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        res = res_fn(dofs).reshape(sol.shape)
        for i in range(len(problem.node_inds_list)):
            res = (res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set
                   (sol[problem.node_inds_list[i], problem.vec_inds_list[i]], unique_indices=True))
        return res.reshape(-1)
    return fn_dofs_row


def assign_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(problem.vals_list[i])
    return sol.reshape(-1)


def assign_zero_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(0.)
    return sol.reshape(-1)

 
def assign_ones_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(1.)
    return sol.reshape(-1)


def get_flatten_fn(fn_sol, problem):
    def fn_dofs(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        val_sol = fn_sol(sol)
        return val_sol.reshape(-1)
    return fn_dofs


def get_A_fn_linear_fn(dofs, fn):
    def A_fn_linear_fn(inc):
        primals, tangents = jax.jvp(fn, (dofs,), (inc,))
        return tangents
    return A_fn_linear_fn


def get_A_fn_linear_fn_JFNK(dofs, fn):
    """Jacobian-free Newton–Krylov (JFNK) method. 
    Not quite used since we have auto diff to compute exact JVP.
    Knoll, Dana A., and David E. Keyes. 
    "Jacobian-free Newton–Krylov methods: a survey of approaches and applications." 
    Journal of Computational Physics 193.2 (2004): 357-397.
    """
    def A_fn_linear_fn(inc):
        EPS = 1e-3
        return (fn(dofs + EPS*inc) - fn(dofs))/EPS
    return A_fn_linear_fn


def operator_to_matrix(operator_fn, problem):
    """Only used for debugging purpose.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_nodes*problem.vec))
    return J


def jacobi_preconditioner(problem, dofs):
    print(f"Compute and use jacobi preconditioner")
    jacobi = np.array(problem.A_sp_scipy.diagonal())
    jacobi = assign_ones_bc(jacobi.reshape(-1), problem) 
    return jacobi


def get_jacobi_precond(jacobi):
    def jacobi_precond(x):
        return x * (1./jacobi)
    return jacobi_precond


def test_jacobi_precond(problem, jacobi, A_fn):
    num_total_dofs = problem.num_total_nodes*problem.vec
    for ind in range(500):
        test_vec = np.zeros(num_total_dofs)
        test_vec = test_vec.at[ind].set(1.)
        print(f"{A_fn(test_vec)[ind]}, {jacobi[ind]}, ratio = {A_fn(test_vec)[ind]/jacobi[ind]}")

    print(f"test jacobi preconditioner")
    print(f"np.min(jacobi) = {np.min(jacobi)}, np.max(jacobi) = {np.max(jacobi)}")
    print(f"finish jacobi preconditioner")
 

def linear_full_solve(problem, A_fn, precond, dofs):
    b = np.zeros((problem.num_total_nodes, problem.vec))
    b = assign_bc(b, problem).reshape(-1)
    pc = get_jacobi_precond(jacobi_preconditioner(problem, dofs)) if precond else None
    dofs, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=b, M=pc, tol=1e-10, atol=1e-10, maxiter=10000)
    return dofs


def linear_incremental_solver(problem, res_fn, A_fn, dofs, precond):
    """
    Lift solver
    dofs must already satisfy Dirichlet boundary conditions
    """
    b = -res_fn(dofs)
    pc = get_jacobi_precond(jacobi_preconditioner(problem, dofs)) if precond else None
    inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=None, M=pc, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
    dofs = dofs + inc
    return dofs


def compute_residual_val(res_fn, dofs):
   res_vec = res_fn(dofs)
   res_val = np.linalg.norm(res_vec)
   return res_val


def solver(problem, initial_guess=None, linear=False, precond=True):
    print("Start timing")
    start = time.time()

    if initial_guess is not None:
        sol = initial_guess
    else:
        sol = np.zeros((problem.num_total_nodes, problem.vec))

    dofs = sol.reshape(-1)

    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem) 

    problem.newton_update(dofs.reshape(sol.shape))
    A_fn = problem.compute_linearized_residual
    A_fn = row_elimination(A_fn, problem)

    # TODO: more notes here
    # TODO: detect np.nan and assert
    if linear:
        # If we know the problem is linear, this way of solving seems faster.
        dofs = assign_bc(dofs, problem).reshape(-1)
        dofs = linear_incremental_solver(problem, res_fn, A_fn, dofs, precond)
    else:
        dofs = linear_full_solve(problem, A_fn, precond, dofs)
        res_val = compute_residual_val(res_fn, dofs)
        print(f"Before, res l_2 = {res_val}") 
        tol = 1e-6
        while res_val > tol:
            problem.newton_update(dofs.reshape(sol.shape))
            dofs = linear_incremental_solver(problem, res_fn, A_fn, dofs, precond)
            # test_jacobi_precond(problem, jacobi_preconditioner(problem, dofs), A_fn)
            res_val = compute_residual_val(res_fn, dofs)
            print(f"res l_2 = {res_val}") 

    sol = dofs.reshape(sol.shape)
    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s]")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    return sol


def adjoint_method(problem, J_fn, output_sol, linear=False):
    """The forward problem can be linear or nonlinear.
    But only linear forward problems have been tested.
    We need to test nonlinear forward problems.
    """
    def fn(params):
        """J(u(p), p)
        """
        print(f"\nStep {fn.counter}")
        problem.params = params
        sol = solver(problem, linear=linear)
        dofs = sol.reshape(-1)
        obj_val = J_fn(dofs, params)
        fn.dofs = dofs
        output_sol(params, dofs, obj_val)
        fn.counter += 1
        return obj_val

    fn.counter = 0

    def constraint_fn(dofs, params):
        """c(u, p)
        """
        problem.params = params
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    def get_partial_dofs_c_fn(params):
        """c(u, p=p)
        """
        def partial_dofs_c_fn(dofs):
            return constraint_fn(dofs, params)
        return partial_dofs_c_fn

    def get_partial_params_c_fn(dofs):
        """c(u=u, p)
        """
        def partial_params_c_fn(params):
            return constraint_fn(dofs, params)
        return partial_params_c_fn

    def get_vjp_contraint_fn_dofs_slow(params, dofs):
        """v*(partial dc/du)
        This version is slow.
        Linearization from "problem.compute_residual" with vjp (or jvp) is slow!
        """
        partial_c_fn = get_partial_dofs_c_fn(params)
        def adjoint_linear_fn(adjoint):
            primals, f_vjp = jax.vjp(partial_c_fn, dofs)
            val, = f_vjp(adjoint)
            return val
        return adjoint_linear_fn

    def get_vjp_contraint_fn_dofs(params, dofs):
        """v*(partial dc/du)
        This version should be fast even for nonlinear problem.
        If not, consider implementing the adjoint version of "problem.compute_linearized_residual" directly.
        """
        # The following two lines may not be needed
        problem.params = params
        problem.D = problem.newton_update(dofs.reshape((problem.num_total_nodes, problem.vec)))
        A_fn = problem.compute_linearized_residual
        A_fn = row_elimination(A_fn, problem)
        def adjoint_linear_fn(adjoint):
            primals, f_vjp = jax.vjp(A_fn, dofs)
            val, = f_vjp(adjoint)
            return val
        return adjoint_linear_fn

    def get_vjp_contraint_fn_params(params, dofs):
        """v*(partial dc/dp)
        """
        partial_c_fn = get_partial_params_c_fn(dofs)
        def vjp_linear_fn(v):
            primals, f_vjp = jax.vjp(partial_c_fn, params)
            val, = f_vjp(v)
            return val
        return vjp_linear_fn

    def fn_grad(params):
        """total dJ/dp
        """
        dofs = fn.dofs
        partial_dJ_du = jax.grad(J_fn, argnums=0)(dofs, params)
        partial_dJ_dp = jax.grad(J_fn, argnums=1)(dofs, params)
        adjoint_linear_fn = get_vjp_contraint_fn_dofs(params, dofs)
        vjp_linear_fn = get_vjp_contraint_fn_params(params, dofs)
        # test_jacobi_precond(problem, jacobi_preconditioner(problem, dofs), adjoint_linear_fn)
        problem.newton_update(dofs.reshape((problem.num_total_nodes, problem.vec)))
        pc = get_jacobi_precond(jacobi_preconditioner(problem, dofs))
        start = time.time()
        adjoint, info = jax.scipy.sparse.linalg.bicgstab(adjoint_linear_fn, partial_dJ_du, x0=None, M=pc, tol=1e-10, atol=1e-10, maxiter=10000)
        end = time.time()
        print(f"Adjoint solve took {end - start} [s]")
        total_dJ_dp = -vjp_linear_fn(adjoint) + partial_dJ_dp
        return total_dJ_dp

    return fn, fn_grad
