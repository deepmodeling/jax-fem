import jax
import jax.numpy as np
import numpy as onp
from jax.experimental.sparse import BCOO
import scipy
import time
from functools import partial


################################################################################
# "row elimination" solver

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


def jacobi_preconditioner(problem):
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
 

def linear_guess_solve(problem, A_fn, precond):
    b = np.zeros((problem.num_total_nodes, problem.vec))
    b = assign_bc(b, problem)
    pc = get_jacobi_precond(jacobi_preconditioner(problem)) if precond else None
    dofs, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=b, M=pc, tol=1e-10, atol=1e-10, maxiter=10000)
    return dofs


def linear_incremental_solver(problem, res_fn, A_fn, dofs, precond):
    """
    Lift solver
    dofs must already satisfy Dirichlet boundary conditions
    """
    b = -res_fn(dofs)
    pc = get_jacobi_precond(jacobi_preconditioner(problem)) if precond else None
    inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=None, M=pc, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
    dofs = dofs + inc
    return dofs


def compute_residual_val(res_fn, dofs):
   res_vec = res_fn(dofs)
   res_val = np.linalg.norm(res_vec)
   return res_val


def get_A_fn(problem):
    print(f"Creating sparse matrix with scipy...")
    A_sp_scipy = scipy.sparse.csc_array((problem.V, (problem.I, problem.J)), shape=(problem.num_total_dofs, problem.num_total_dofs))
    print(f"Creating sparse matrix from scipy using JAX BCOO...")
    A_sp = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    print(f"self.A_sp.data.shape = {A_sp.data.shape}")
    print(f"Global sparse matrix takes about {A_sp.data.shape[0]*8*3/2**30} G memory to store.")
    problem.A_sp_scipy = A_sp_scipy

    def compute_linearized_residual(dofs):
        return A_sp @ dofs

    return compute_linearized_residual


def solver_row_elimination(problem, linear=False, precond=True):
    """Imposing Dirichlet B.C. with "row elimination" method.
    """
    print(f"Calling the row elimination solver for imposing Dirichlet B.C.")
    print("Start timing")
    start = time.time()

    sol = np.zeros((problem.num_total_nodes, problem.vec))
    dofs = sol.reshape(-1)

    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem) 

    problem.newton_update(dofs.reshape(sol.shape))
    A_fn = get_A_fn(problem)
    A_fn = row_elimination(A_fn, problem)

    # TODO: more notes here
    # TODO: detect np.nan and assert
    if linear:
        dofs = assign_bc(dofs, problem)
        dofs = linear_incremental_solver(problem, res_fn, A_fn, dofs, precond)
    else:
        dofs = linear_guess_solve(problem, A_fn, precond)
        res_val = compute_residual_val(res_fn, dofs)
        print(f"Before, res l_2 = {res_val}") 
        tol = 1e-6
        while res_val > tol:
            problem.newton_update(dofs.reshape(sol.shape))
            A_fn = get_A_fn(problem)
            A_fn = row_elimination(A_fn, problem)            
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


################################################################################
# Lagrangian multiplier solver

def aug_dof_w_zero_bc(problem, dofs):
    aug_size = 0
    for i in range(len(problem.node_inds_list)):
        aug_size += len(problem.node_inds_list[i])
    for i in range(len(problem.p_node_inds_list_A)):
        aug_size += len(problem.p_node_inds_list_A[i])
    return np.hstack((dofs, np.zeros(aug_size)))


def aug_dof_w_bc(problem, dofs, p_num_eps):
    aug_d = np.array([])
    for i in range(len(problem.node_inds_list)):
        aug_d = np.hstack((aug_d, p_num_eps*problem.vals_list[i]))
    for i in range(len(problem.p_node_inds_list_A)):
        aug_d = np.hstack((aug_d, np.zeros(len(problem.p_node_inds_list_A[i]))))
    return np.hstack((dofs, aug_d))


def linear_guess_solve_lm(problem, A_fn_aug, p_num_eps):
    x0 = np.zeros((problem.num_total_nodes, problem.vec))
    x0 = assign_bc(x0, problem)
    x0 = aug_dof_w_zero_bc(problem, x0)
    b = np.zeros(problem.num_total_dofs)
    b_aug = aug_dof_w_bc(problem, b, p_num_eps)
    dofs_aug, info = jax.scipy.sparse.linalg.bicgstab(A_fn_aug, b_aug, x0=x0, M=None, tol=1e-10, atol=1e-10, maxiter=10000)
    return dofs_aug


def linear_incremental_solver_lm(problem, res_fn, A_fn_aug, dofs_aug, p_num_eps):
    """
    Lift solver
    dofs must already satisfy Dirichlet boundary conditions
    """
    b_aug = -compute_residual_lm(problem, res_fn, dofs_aug, p_num_eps)
    inc_aug, info = jax.scipy.sparse.linalg.bicgstab(A_fn_aug, b_aug, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
    dofs_aug = dofs_aug + inc_aug
    return dofs_aug


def compute_residual_lm(problem, res_fn, dofs_aug, p_num_eps):
    d_splits = np.cumsum(np.array([len(x) for x in problem.node_inds_list])).tolist()
    p_splits = np.cumsum(np.array([len(x) for x in problem.p_node_inds_list_A])).tolist()

    d_lmbda_len = d_splits[-1] if len(d_splits) > 0 else 0
    p_lmbda_len = p_splits[-1] if len(p_splits) > 0 else 0

    def get_Lagrangian():
        def split_lamda(lmbda):
            d_lmbda = lmbda[:d_lmbda_len]
            p_lmbda = lmbda[d_lmbda_len:]
            d_lmbda_split = np.split(d_lmbda, d_splits)
            p_lmbda_split = np.split(p_lmbda, p_splits)
            return d_lmbda_split, p_lmbda_split

        # @jax.jit
        def Lagrangian_fn(dofs_aug):
            dofs, lmbda = dofs_aug[:problem.num_total_dofs], dofs_aug[problem.num_total_dofs:]
            sol = dofs.reshape((problem.num_total_nodes, problem.vec))
            d_lmbda_split, p_lmbda_split = split_lamda(lmbda)
            lag = 0.
            for i in range(len(problem.node_inds_list)):
                lag += np.sum(d_lmbda_split[i] * (sol[problem.node_inds_list[i], problem.vec_inds_list[i]] - problem.vals_list[i]))

            for i in range(len(problem.p_node_inds_list_A)):
                lag += np.sum(p_lmbda_split[i] * (sol[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]] - 
                                                    sol[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]]))

            return p_num_eps*lag

        return Lagrangian_fn

    Lagrangian_fn = get_Lagrangian()
    A_fn = jax.grad(Lagrangian_fn)
    res_vec_1 = A_fn(dofs_aug)

    dofs = dofs_aug[:problem.num_total_dofs]
    res_vec = res_fn(dofs)
    res_vec_2 = aug_dof_w_zero_bc(problem, res_vec)

    res_vec_aug = res_vec_1 + res_vec_2

    return res_vec_aug


def get_A_fn_aug(problem, p_num_eps):
    def symmetry(I, J, V):
        I_sym = onp.hstack((I, J))
        J_sym = onp.hstack((J, I))
        V_sym = onp.hstack((V, V))
        return I_sym, J_sym, V_sym

    I_d = onp.array([])
    J_d = onp.array([])
    V_d = onp.array([])
    group_index = problem.num_total_dofs
    for i in range(len(problem.node_inds_list)):
        group_size = len(problem.node_inds_list[i])
        I_d = onp.hstack((I_d, problem.vec*problem.node_inds_list[i] + problem.vec_inds_list[i]))
        J_d = onp.hstack((J_d, group_index + onp.arange(group_size)))
        V_d = onp.hstack((V_d, p_num_eps*onp.ones(group_size)))
        group_index += group_size
    I_d_sym, J_d_sym, V_d_sym = symmetry(I_d, J_d, V_d)

    I_p = onp.array([])
    J_p = onp.array([])
    V_p = onp.array([])
    for i in range(len(problem.p_node_inds_list_A)):
        group_size = len(problem.p_node_inds_list_A[i])
        I_p = onp.hstack((I_p, problem.vec*problem.p_node_inds_list_A[i] + problem.p_vec_inds_list[i]))
        J_p = onp.hstack((J_p, group_index + onp.arange(group_size)))
        V_p = onp.hstack((V_p, p_num_eps*onp.ones(group_size)))
        I_p = onp.hstack((I_p, problem.vec*problem.p_node_inds_list_B[i] + problem.p_vec_inds_list[i]))
        J_p = onp.hstack((J_p, group_index + onp.arange(group_size)))
        V_p = onp.hstack((V_p, -p_num_eps*onp.ones(group_size)))
        group_index += group_size
    I_p_sym, J_p_sym, V_p_sym = symmetry(I_p, J_p, V_p)

    I = onp.hstack((problem.I, I_d_sym, I_p_sym))
    J = onp.hstack((problem.J, J_d_sym, J_p_sym))
    V = onp.hstack((problem.V, V_d_sym, V_p_sym))

    print(f"Aug - Creating sparse matrix with scipy...")
    A_sp_scipy_aug = scipy.sparse.csc_array((V, (I, J)), shape=(group_index, group_index))
    print(f"Aug - Creating sparse matrix from scipy using JAX BCOO...")
    A_sp_aug = BCOO.from_scipy_sparse(A_sp_scipy_aug).sort_indices()
    print(f"Aug - self.A_sp.data.shape = {A_sp_aug.data.shape}")
    print(f"Aug - Global sparse matrix takes about {A_sp_aug.data.shape[0]*8*3/2**30} G memory to store.")
    problem.A_sp_scipy_aug = A_sp_scipy_aug

    def compute_linearized_residual(dofs_aug):
        return A_sp_aug @ dofs_aug

    return compute_linearized_residual


def solver_lagrange_multiplier(problem, linear=False):
    """Imposing Dirichlet B.C. and periodic B.C. with lagrangian multiplier method.

    The global matrix is of the form 
    [A   B 
     B^T 0]
    and a good preconditioner is needed.
    The problem is well studied, though.
    However, in the current code, there is no trick applied. 

    Reference:
    https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femI/Presentation.pdf
    """
    print(f"Calling the lagrange multiplier solver for imposing Dirichlet B.C. and periodic B.C.")
    print("Start timing")
    start = time.time()

    # Ad-hoc parameter to get a better conditioned global matrix.
    # Currently, this parameter needs to be manually tuned.
    # We need a (much) better way to deal with this type of saddle-point problem.
    # Will interfacing with PETSc be a good idea?
    if hasattr(problem, 'p_num_eps'):
        p_num_eps = problem.p_num_eps
    else:
        p_num_eps = 1.

    print(f"Setting p_num_eps = {p_num_eps}. If periodic B.C. fails to be applied, consider modifying this parameter.")

    sol = np.zeros((problem.num_total_nodes, problem.vec))
    dofs = sol.reshape(-1)

    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)

    problem.newton_update(dofs.reshape(sol.shape))
    A_fn_aug = get_A_fn_aug(problem, p_num_eps)

    if linear:
        # If we know the problem is linear, this way of solving seems faster.
        dofs = assign_bc(dofs, problem)
        dofs_aug = aug_dof_w_zero_bc(problem, dofs)
        dofs_aug = linear_incremental_solver_lm(problem, res_fn, A_fn_aug, dofs_aug, p_num_eps)
        print(f"Linear problem res l_2 = {np.linalg.norm(compute_residual_lm(problem, res_fn, dofs_aug, p_num_eps))}")
    else:
        dofs_aug = linear_guess_solve_lm(problem, A_fn_aug, p_num_eps)
        res_val = np.linalg.norm(compute_residual_lm(problem, res_fn, dofs_aug, p_num_eps))
        print(f"Before, res l_2 = {res_val}") 
        tol = 1e-6
        while res_val > tol:
            problem.newton_update(dofs_aug[:problem.num_total_dofs].reshape(sol.shape))
            A_fn_aug = get_A_fn_aug(problem, p_num_eps)
            dofs_aug = linear_incremental_solver_lm(problem, res_fn, A_fn_aug, dofs_aug, p_num_eps)
            res_val = np.linalg.norm(compute_residual_lm(problem, res_fn, dofs_aug, p_num_eps))
            print(f"res l_2 dofs_aug = {res_val}") 
 
    sol = dofs_aug[:problem.num_total_dofs].reshape(sol.shape)
    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s]")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    return sol


################################################################################
# General solver

def solver(problem, linear=False, precond=True):
    """periodic B.C. is a special form of adding a linear constraint. 
    Lagrange multiplier seems to be convenient to impose this constraint.
    """
    if problem.periodic_bc_info is None:
        return solver_row_elimination(problem, linear, precond)
    else:
        return solver_lagrange_multiplier(problem, linear)


################################################################################
# Adjoint method for inverse problem

def adjoint_method(problem, J_fn, output_sol, linear=False):
    """Adjoint method with automatic differentiation.

    Currently, the function cannot deal with periodic B.C.,
    but it should not be easy to add.
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
        A_fn = get_A_fn(problem)
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
        # test_jacobi_precond(problem, jacobi_preconditioner(problem), adjoint_linear_fn)
        problem.newton_update(dofs.reshape((problem.num_total_nodes, problem.vec)))
        pc = get_jacobi_precond(jacobi_preconditioner(problem))
        start = time.time()
        adjoint, info = jax.scipy.sparse.linalg.bicgstab(adjoint_linear_fn, partial_dJ_du, x0=None, M=pc, tol=1e-10, atol=1e-10, maxiter=10000)
        end = time.time()
        print(f"Adjoint solve took {end - start} [s]")
        total_dJ_dp = -vjp_linear_fn(adjoint) + partial_dJ_dp
        return total_dJ_dp

    return fn, fn_grad
