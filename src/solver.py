import jax
import jax.numpy as np
import numpy as onp
from jax.experimental.sparse import BCOO
import scipy
import time

# petsc4py.init()
from petsc4py import PETSc

from jax_am import logger

################################################################################
# PETSc linear solver or JAX linear solver


def petsc_solve(A, b, ksp_type, pc_type):
    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), onp.array(b))
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType(ksp_type)
    ksp.pc.setType(pc_type)
    logger.debug(
        f'PETSc - Solving with ksp_type = {ksp.getType()}, '
        f'pc = {ksp.pc.getType()}'
    )
    x = PETSc.Vec().createSeq(len(b))
    ksp.solve(rhs, x)

    # Verify convergence
    y = PETSc.Vec().createSeq(len(b))
    A.mult(x, y)

    err = np.linalg.norm(y.getArray() - rhs.getArray())
    logger.debug(f"PETSc linear solve res = {err}")
    # assert err < 0.1, f"PETSc linear solver failed to converge, err = {err}"

    return x.getArray()


def jax_solve(problem, A_fn, b, x0, precond: bool, pc_matrix=None):
    """Solves the equilibrium equation using a JAX solver.
    Is fully traceable and runs on GPU.

    Parameters
    ----------
    precond
        Whether to calculate the preconditioner or not
    pc_matrix
        The matrix to use as preconditioner
    """
    pc = get_jacobi_precond(
        jacobi_preconditioner(problem)) if precond else None
    x, info = jax.scipy.sparse.linalg.bicgstab(A_fn,
                                               b,
                                               x0=x0,
                                               M=pc,
                                               tol=1e-10,
                                               atol=1e-10,
                                               maxiter=10000)

    # Verify convergence
    err = np.linalg.norm(A_fn(x) - b)
    logger.debug(f"JAX scipy linear solve res = {err}")

    # Remarks(Tianju): assert seems to unexpectedly change the behavior of bicgstab (on my Linux machine).
    # Sometimes the solver simply fails without converging (it does converge without assert)
    # Particularly happening in topology optimization examples.
    # Don't know why yet.

    # assert err < 0.1, f"JAX linear solver failed to converge with err = {err}"
    # x = np.where(err < 0.1, x, np.nan) # For assert purpose, some how this also affects bicgstab.

    return x


################################################################################
# "row elimination" solver


def apply_bc_vec(res_vec, dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    res = res_vec.reshape(sol.shape)
    for i in range(len(problem.node_inds_list)):
        res = (res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(
            sol[problem.node_inds_list[i], problem.vec_inds_list[i]],
            unique_indices=True))
        res = res.at[problem.node_inds_list[i],
                     problem.vec_inds_list[i]].add(-problem.vals_list[i])
    return res.reshape(-1)


def apply_bc(res_fn, problem):

    def A_fn(dofs):
        """Apply Dirichlet boundary conditions
        """
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem)

    return A_fn


def row_elimination(fn, problem):

    def fn_dofs_row(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        res = fn(dofs).reshape(sol.shape)
        for i in range(len(problem.node_inds_list)):
            res = (res.at[problem.node_inds_list[i],
                          problem.vec_inds_list[i]].set(
                              sol[problem.node_inds_list[i],
                                  problem.vec_inds_list[i]],
                              unique_indices=True))
        return res.reshape(-1)

    return fn_dofs_row


def assign_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i],
                     problem.vec_inds_list[i]].set(problem.vals_list[i])
    return sol.reshape(-1)


def assign_ones_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i],
                     problem.vec_inds_list[i]].set(1.)
    return sol.reshape(-1)


def assign_zeros_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i],
                     problem.vec_inds_list[i]].set(0.)
    return sol.reshape(-1)


def copy_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    new_sol = np.zeros_like(sol)
    for i in range(len(problem.node_inds_list)):
        new_sol = (new_sol.at[problem.node_inds_list[i],
                              problem.vec_inds_list[i]].set(
                                  sol[problem.node_inds_list[i],
                                      problem.vec_inds_list[i]]))
    return new_sol.reshape(-1)


def get_flatten_fn(fn_sol, problem):

    def fn_dofs(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        val_sol = fn_sol(sol)
        return val_sol.reshape(-1)

    return fn_dofs


def get_A_fn_linear_fn(dofs, fn):
    """Not quite used.
    """

    def A_fn_linear_fn(inc):
        primals, tangents = jax.jvp(fn, (dofs, ), (inc, ))
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
        return (fn(dofs + EPS * inc) - fn(dofs)) / EPS

    return A_fn_linear_fn


def operator_to_matrix(operator_fn, problem):
    """Only used for when debugging.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_nodes *
                                         problem.vec))
    return J


def jacobi_preconditioner(problem):
    logger.debug(f"Compute and use jacobi preconditioner")
    jacobi = np.array(problem.A_sp_scipy.diagonal())
    jacobi = assign_ones_bc(jacobi.reshape(-1), problem)
    return jacobi


def get_jacobi_precond(jacobi):

    def jacobi_precond(x):
        return x * (1. / jacobi)

    return jacobi_precond


def test_jacobi_precond(problem, jacobi, A_fn):
    num_total_dofs = problem.num_total_nodes * problem.vec
    for ind in range(500):
        test_vec = np.zeros(num_total_dofs)
        test_vec = test_vec.at[ind].set(1.)
        logger.debug(
            f"{A_fn(test_vec)[ind]}, {jacobi[ind]}, ratio = {A_fn(test_vec)[ind]/jacobi[ind]}"
        )

    logger.debug(f"test jacobi preconditioner")
    logger.debug(
        f"np.min(jacobi) = {np.min(jacobi)}, np.max(jacobi) = {np.max(jacobi)}"
    )
    logger.debug(f"finish jacobi preconditioner")


def linear_guess_solve(problem, A_fn, precond, use_petsc):
    logger.debug(f"Linear guess solve...")
    # b = np.zeros((problem.num_total_nodes, problem.vec))
    b = problem.body_force + problem.neumann
    b = assign_bc(b, problem)
    if use_petsc:
        dofs = petsc_solve(A_fn, b, 'bcgsl', 'ilu')
    else:
        dofs = jax_solve(problem, A_fn, b, b, precond)
    return dofs


def linear_incremental_solver(problem, res_vec, A_fn, dofs, precond,
                              use_petsc):
    """Lift solver
    """
    logger.debug(f"Solving linear system with lift solver...")
    b = -res_vec

    if use_petsc:
        inc = petsc_solve(A_fn, b, 'bcgsl', 'ilu')
    else:
        x0_1 = assign_bc(np.zeros_like(b), problem)
        x0_2 = copy_bc(dofs, problem)
        x0 = x0_1 - x0_2
        inc = jax_solve(problem, A_fn, b, x0, precond)

    dofs = dofs + inc

    # dofs = line_search(problem, dofs, inc)

    return dofs


def line_search(problem, dofs, inc):
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha*inc)
        return np.linalg.norm(res_vec)

    # grad_res_norm_fn = jax.grad(res_norm_fn)
    # hess_res_norm_fn = jax.hessian(res_norm_fn)

    # tol = 1e-3
    # alpha = 1.
    # lr = 1.
    # grad_alpha = 1.
    # while np.abs(grad_alpha) > tol:
    #     grad_alpha = grad_res_norm_fn(alpha)
    #     hess_alpha = hess_res_norm_fn(alpha)
    #     alpha = alpha - 1./hess_alpha*grad_alpha
    #     print(f"alpha = {alpha}, grad_alpha = {grad_alpha}, hess_alpha = {hess_alpha}")

    alpha = 1.
    res_norm = res_norm_fn(alpha)
    for i in range(3):
        alpha *= 0.5
        res_norm_half = res_norm_fn(alpha)
        print(f"i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.
            break
        res_norm = res_norm_half


    return dofs + alpha*inc


def get_A_fn(problem, use_petsc):
    logger.debug(f"Creating sparse matrix with scipy...")
    A_sp_scipy = scipy.sparse.csr_array(
        (onp.array(problem.V), (problem.I, problem.J)),
        shape=(problem.num_total_dofs, problem.num_total_dofs))
    # logger.debug(f"Creating sparse matrix from scipy using JAX BCOO...")
    A_sp = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    # logger.info(f"Global sparse matrix takes about {A_sp.data.shape[0]*8*3/2**30} G memory to store.")
    problem.A_sp_scipy = A_sp_scipy

    def compute_linearized_residual(dofs):
        return A_sp @ dofs

    if use_petsc:
        # https://scicomp.stackexchange.com/questions/2355/32bit-64bit-issue-when-working-with-numpy-and-petsc4py/2356#2356
        A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape, csr=(A_sp_scipy.indptr.astype(PETSc.IntType, copy=False),
                                                       A_sp_scipy.indices.astype(PETSc.IntType, copy=False), A_sp_scipy.data))
        for i in range(len(problem.node_inds_list)):
            row_inds = onp.array(problem.node_inds_list[i] * problem.vec +
                                 problem.vec_inds_list[i],
                                 dtype=onp.int32)
            A.zeroRows(row_inds)
    else:
        A = row_elimination(compute_linearized_residual, problem)

    return A


def solver_row_elimination(problem, linear, precond, initial_guess, use_petsc):
    """The solver imposes Dirichlet B.C. with "row elimination" method.

    Some memo:

    res(u) = D*r(u) + (I - D)u - u_b
    D = [[1 0 0 0]
         [0 1 0 0]
         [0 0 0 0]
         [0 0 0 1]]
    I = [[1 0 0 0]
         [0 1 0 0]
         [0 0 1 0]
         [0 0 0 1]
    A_fn = d(res)/d(u) = D*dr/du + (I - D)

    The function newton_update computes r(u) and dr/du
    """
    logger.debug(
        f"Calling the row elimination solver for imposing Dirichlet B.C.")
    logger.debug("Start timing")
    start = time.time()
    sol_shape = (problem.num_total_nodes, problem.vec)
    dofs = np.zeros(sol_shape).reshape(-1)

    def newton_update_helper(dofs):
        res_vec = problem.newton_update(dofs.reshape(sol_shape)).reshape(-1)
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A_fn = get_A_fn(problem, use_petsc)
        return res_vec, A_fn

    if linear:
        dofs = assign_bc(dofs, problem)
        res_vec, A_fn = newton_update_helper(dofs)

        dofs = linear_incremental_solver(problem, res_vec, A_fn, dofs, precond,
                                         use_petsc)

        res_vec, A_fn = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        logger.debug(f"Linear solve, res l_2 = {res_val}")

    else:
        if initial_guess is None:
            res_vec, A_fn = newton_update_helper(dofs)
            dofs = linear_guess_solve(problem, A_fn, precond, use_petsc)
        else:
            dofs = initial_guess.reshape(-1)

        res_vec, A_fn = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        logger.debug(f"Before, res l_2 = {res_val}")
        tol = 1e-6
        while res_val > tol:
            dofs = linear_incremental_solver(problem, res_vec, A_fn, dofs,
                                             precond, use_petsc)
            res_vec, A_fn = newton_update_helper(dofs)
            # test_jacobi_precond(problem, jacobi_preconditioner(problem, dofs), A_fn)
            res_val = np.linalg.norm(res_vec)
            logger.debug(f"res l_2 = {res_val}")

    assert np.all(
        np.isfinite(res_val)), f"res_val contains NaN, stop the program!"

    assert np.all(np.isfinite(dofs)), f"dofs contains NaN, stop the program!"

    sol = dofs.reshape(sol_shape)
    end = time.time()
    solve_time = end - start
    logger.info(f"Solve took {solve_time} [s]")
    logger.debug(f"max of sol = {np.max(sol)}")
    logger.debug(f"min of sol = {np.min(sol)}")

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
        aug_d = np.hstack((aug_d, p_num_eps * problem.vals_list[i]))
    for i in range(len(problem.p_node_inds_list_A)):
        aug_d = np.hstack(
            (aug_d, np.zeros(len(problem.p_node_inds_list_A[i]))))
    return np.hstack((dofs, aug_d))


def linear_guess_solve_lm(problem, A_aug, p_num_eps, use_petsc):
    b = (problem.body_force + problem.neumann).reshape(-1)
    b_aug = aug_dof_w_bc(problem, b, p_num_eps)
    if use_petsc:
        dofs_aug = petsc_solve(A_aug, b_aug, 'minres', 'none')
    else:
        x0 = np.zeros((problem.num_total_nodes, problem.vec))
        x0 = assign_bc(x0, problem)
        x0 = aug_dof_w_zero_bc(problem, x0)
        dofs_aug = jax_solve(problem, A_aug, b_aug, x0, None)
    return dofs_aug


def linear_incremental_solver_lm(problem, res_vec_aug, A_aug, dofs_aug,
                                 p_num_eps, use_petsc):
    b_aug = -res_vec_aug
    if use_petsc:
        inc_aug = petsc_solve(A_aug, b_aug, 'minres', 'none')
    else:
        inc_aug = jax_solve(problem, A_aug, b_aug, None, None)
    dofs_aug = dofs_aug + inc_aug
    return dofs_aug


def compute_residual_lm(problem, res_vec, dofs_aug, p_num_eps):
    """Some memo here
    Saddle point problem energy function: L(u, lmbda) = E(u) + lmbda*(u - u0)
    with dL/d(u, lmbda) = res_vec_aug and dE/du = res_vec
    """
    d_splits = np.cumsum(np.array([len(x)
                                   for x in problem.node_inds_list])).tolist()
    p_splits = np.cumsum(np.array([len(x) for x in problem.p_node_inds_list_A
                                   ])).tolist()

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
            dofs, lmbda = dofs_aug[:problem.num_total_dofs], dofs_aug[
                problem.num_total_dofs:]
            sol = dofs.reshape((problem.num_total_nodes, problem.vec))
            d_lmbda_split, p_lmbda_split = split_lamda(lmbda)
            lag = 0.
            for i in range(len(problem.node_inds_list)):
                lag += np.sum(
                    d_lmbda_split[i] *
                    (sol[problem.node_inds_list[i], problem.vec_inds_list[i]] -
                     problem.vals_list[i]))

            for i in range(len(problem.p_node_inds_list_A)):
                lag += np.sum(p_lmbda_split[i] *
                              (sol[problem.p_node_inds_list_A[i],
                                   problem.p_vec_inds_list[i]] -
                               sol[problem.p_node_inds_list_B[i],
                                   problem.p_vec_inds_list[i]]))
            return p_num_eps * lag

        return Lagrangian_fn

    Lagrangian_fn = get_Lagrangian()
    A_fn = jax.grad(Lagrangian_fn)
    res_vec_1 = A_fn(dofs_aug)
    res_vec_2 = aug_dof_w_zero_bc(problem, res_vec)
    res_vec_aug = res_vec_1 + res_vec_2

    return res_vec_aug


def get_A_fn_and_res_aug(problem, dofs_aug, res_vec, p_num_eps, use_petsc):

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
        I_d = onp.hstack((I_d, problem.vec * problem.node_inds_list[i] +
                          problem.vec_inds_list[i]))
        J_d = onp.hstack((J_d, group_index + onp.arange(group_size)))
        V_d = onp.hstack((V_d, p_num_eps * onp.ones(group_size)))
        group_index += group_size
    I_d_sym, J_d_sym, V_d_sym = symmetry(I_d, J_d, V_d)

    I_p = onp.array([])
    J_p = onp.array([])
    V_p = onp.array([])
    for i in range(len(problem.p_node_inds_list_A)):
        group_size = len(problem.p_node_inds_list_A[i])
        I_p = onp.hstack((I_p, problem.vec * problem.p_node_inds_list_A[i] +
                          problem.p_vec_inds_list[i]))
        J_p = onp.hstack((J_p, group_index + onp.arange(group_size)))
        V_p = onp.hstack((V_p, p_num_eps * onp.ones(group_size)))
        I_p = onp.hstack((I_p, problem.vec * problem.p_node_inds_list_B[i] +
                          problem.p_vec_inds_list[i]))
        J_p = onp.hstack((J_p, group_index + onp.arange(group_size)))
        V_p = onp.hstack((V_p, -p_num_eps * onp.ones(group_size)))
        group_index += group_size
    I_p_sym, J_p_sym, V_p_sym = symmetry(I_p, J_p, V_p)

    I = onp.hstack((problem.I, I_d_sym, I_p_sym))
    J = onp.hstack((problem.J, J_d_sym, J_p_sym))
    V = onp.hstack((problem.V, V_d_sym, V_p_sym))

    logger.debug(f"Aug - Creating sparse matrix with scipy...")
    A_sp_scipy_aug = scipy.sparse.csc_array((V, (I, J)),
                                            shape=(group_index, group_index))
    # logger.info(f"Aug - Creating sparse matrix from scipy using JAX BCOO...")
    A_sp_aug = BCOO.from_scipy_sparse(A_sp_scipy_aug).sort_indices()

    # logger.info(f"Aug - Global sparse matrix takes about {A_sp_aug.data.shape[0]*8*3/2**30} G memory to store.")

    # TODO: Potential bug
    # Used only in jacobi_preconditioner
    # problem.A_sp_scipy = A_sp_scipy_aug

    def compute_linearized_residual(dofs_aug):
        return A_sp_aug @ dofs_aug

    if use_petsc:

        A = PETSc.Mat().createAIJ(size=A_sp_scipy_aug.shape,
                                  csr=(A_sp_scipy_aug.indptr.astype(PETSc.IntType, copy=False),
                                       A_sp_scipy_aug.indices.astype(PETSc.IntType, copy=False),
                                       A_sp_scipy_aug.data))

        # A_aug = PETSc.Mat().createAIJ(size=A_sp_scipy_aug.shape,
        #                               csr=(A_sp_scipy_aug.indptr,
        #                                    A_sp_scipy_aug.indices,
        #                                    A_sp_scipy_aug.data))
    else:
        A_aug = compute_linearized_residual

    res_vec_aug = compute_residual_lm(problem, res_vec, dofs_aug, p_num_eps)

    return A_aug, res_vec_aug


def solver_lagrange_multiplier(problem, linear, use_petsc=True):
    """The solver imposes Dirichlet B.C. and periodic B.C. with lagrangian multiplier method.

    The global matrix is of the form
    [A   B
     B^T 0]
    JAX built solver gmres and bicgstab sometimes fail to solve such a system.
    PESTc solver minres seems to work.
    TODO: explore which solver in PESTc is the best, and which preconditioner should be used.

    Reference:
    https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femI/Presentation.pdf
    """
    logger.info(
        f"Calling the lagrange multiplier solver for imposing Dirichlet B.C. and periodic B.C."
    )
    logger.info("Start timing")
    start = time.time()
    sol_shape = (problem.num_total_nodes, problem.vec)
    dofs = np.zeros(sol_shape).reshape(-1)

    # Ad-hoc parameter to get a better conditioned global matrix. Not useful for PETSc solver.
    if hasattr(problem, 'p_num_eps'):
        p_num_eps = problem.p_num_eps
    else:
        p_num_eps = 1.

    if not use_petsc:
        logger.info(
            f"Setting p_num_eps = {p_num_eps}. If periodic B.C. fails to be applied, consider modifying this parameter."
        )

    def newton_update_helper(dofs_aug):
        res_vec = problem.newton_update(
            dofs_aug[:problem.num_total_dofs].reshape(sol_shape)).reshape(-1)
        A_aug, res_vec_aug = get_A_fn_and_res_aug(problem, dofs_aug, res_vec,
                                                  p_num_eps, use_petsc)
        return res_vec_aug, A_aug

    if linear:
        # If we know the problem is linear, this way of solving seems faster.
        dofs = assign_bc(dofs, problem)
        dofs_aug = aug_dof_w_zero_bc(problem, dofs)
        res_vec_aug, A_aug = newton_update_helper(dofs_aug)
        dofs_aug = linear_incremental_solver_lm(problem, res_vec_aug, A_aug,
                                                dofs_aug, p_num_eps, use_petsc)
    else:
        dofs_aug = aug_dof_w_zero_bc(problem, dofs)
        res_vec_aug, A_aug = newton_update_helper(dofs_aug)
        dofs_aug = linear_guess_solve_lm(problem, A_aug, p_num_eps, use_petsc)

        res_vec_aug, A_aug = newton_update_helper(dofs_aug)
        res_val = np.linalg.norm(res_vec_aug)
        logger.debug(f"Before, res l_2 = {res_val}")
        tol = 1e-6
        while res_val > tol:
            dofs_aug = linear_incremental_solver_lm(problem, res_vec_aug,
                                                    A_aug, dofs_aug, p_num_eps,
                                                    use_petsc)
            res_vec_aug, A_aug = newton_update_helper(dofs_aug)
            res_val = np.linalg.norm(res_vec_aug)
            logger.debug(f"res l_2 dofs_aug = {res_val}")

    sol = dofs_aug[:problem.num_total_dofs].reshape(sol_shape)
    end = time.time()
    solve_time = end - start
    logger.info(f"Solve took {solve_time} [s]")
    logger.debug(f"max of sol = {np.max(sol)}")
    logger.debug(f"min of sol = {np.min(sol)}")

    return sol


################################################################################
# Dynamic relaxation solver


def assembleCSR(problem, dofs):
    problem.newton_update(dofs.reshape(
        (problem.num_total_nodes, problem.vec))).reshape(-1)
    A_sp_scipy = scipy.sparse.csr_array(
        (problem.V, (problem.I, problem.J)),
        shape=(problem.num_total_dofs, problem.num_total_dofs))

    A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape,
                              csr=(A_sp_scipy.indptr, A_sp_scipy.indices,
                                   A_sp_scipy.data))
    for i in range(len(problem.node_inds_list)):
        row_inds = onp.array(problem.node_inds_list[i] * problem.vec +
                             problem.vec_inds_list[i],
                             dtype=onp.int32)
        A.zeroRows(row_inds)

    row, col, val = A.getValuesCSR()
    A_sp_scipy.data = val
    A_sp_scipy.indices = col
    A_sp_scipy.indptr = row

    return A_sp_scipy


def calC(t, cmin, cmax):

    if t < 0.: t = 0.

    c = 2. * onp.sqrt(t)
    if (c < cmin): c = cmin
    if (c > cmax): c = cmax

    return c


def printInfo(error, t, c, tol, eps, qdot, qdotdot, nIters, nPrint, info_force,
              info):

    ## printing control
    if nIters % nPrint == 1:
        #logger.info('\t------------------------------------')
        if info_force == True:
            print(('\nDR Iteration %d: Max force (residual error) = %g (tol = %g)' +
                   'Max velocity = %g') % (nIters, error, tol,
                                            np.max(np.absolute(qdot))))
        if info == True:
            print('\nDamping t: ',t, );
            print('Damping coefficient: ', c)
            print('Max epsilon: ',np.max(eps))
            print('Max acceleration: ',np.max(np.absolute(qdotdot)))


def dynamic_relax_solve(problem, tol=1e-6, nKMat=1000, nPrint=500, info=True, info_force=True):
    """
    Implementation of

    Luet, David Joseph. Bounding volume hierarchy and non-uniform rational B-splines for contact enforcement
    in large deformation finite element analysis of sheet metal forming. Diss. Princeton University, 2016.
    Chapter 4.3 Nonlinear System Solution

    Particularly good for handling buckling behavior.
    There is a FEniCS version of this dynamic relaxation algorithm.
    The code below is a direct translation from the FEniCS version.
    """
    def newton_update_helper(dofs):
        res_vec = problem.newton_update(dofs.reshape(sol_shape)).reshape(-1)
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A_fn = get_A_fn(problem, use_petsc=False)
        return res_vec, A_fn

    res_vec, A_fn = newton_update_helper(dofs)
    dofs = linear_guess_solve(problem, A_fn, precond=True, use_petsc=False)

    # parameters not to change
    cmin = 1e-3
    cmax = 3.9
    h_tilde = 1.1
    h = 1.

    # initialize all arrays
    N = len(dofs)  #print("--------num of DOF's: %d-----------" % N)
    #initialize displacements, velocities and accelerations
    q, qdot, qdotdot = onp.zeros(N), onp.zeros(N), onp.zeros(N)
    #initialize displacements, velocities and accelerations from a previous time step
    q_old, qdot_old, qdotdot_old = onp.zeros(N), onp.zeros(N), onp.zeros(N)
    #initialize the M, eps, R_old arrays
    eps, M, R, R_old = onp.zeros(N), onp.zeros(N), onp.zeros(N), onp.zeros(N)

    @jax.jit
    def assembleVec(dofs):
        res_fn = get_flatten_fn(problem.compute_residual, problem)
        res_vec = res_fn(dofs)
        res_vec = assign_zeros_bc(res_vec, problem)
        return res_vec

    R = onp.array(assembleVec(dofs))
    KCSR = assembleCSR(problem, dofs)

    M[:] = h_tilde * h_tilde / 4. * onp.array(
        onp.absolute(KCSR).sum(axis=1)).squeeze()
    q[:] = dofs
    qdot[:] = -h / 2. * R / M
    # set the counters for iterations and
    nIters, iKMat = 0, 0
    error = 1.0
    timeZ = time.time() #Measurement of loop time.

    assert onp.all(onp.isfinite(M)), f"M not finite"
    assert onp.all(onp.isfinite(q)), f"q not finite"
    assert onp.all(onp.isfinite(qdot)), f"qdot not finite"

    error = onp.max(onp.absolute(R))

    while error > tol:

        # print(f"error = {error}")
        # marching forward
        q_old[:] = q[:]; R_old[:] = R[:]
        q[:] += h*qdot; dofs = np.array(q)

        R = onp.array(assembleVec(dofs))

        nIters += 1
        iKMat += 1
        error = onp.max(onp.absolute(R))

        # damping calculation
        S0 = onp.dot((R - R_old) / h, qdot)
        t = S0 / onp.einsum('i,i,i', qdot, M, qdot)
        c = calC(t, cmin, cmax)

        # determine whether to recal KMat
        eps = h_tilde * h_tilde / 4. * onp.absolute(
            onp.divide((qdotdot - qdotdot_old), (q - q_old),
                       out=onp.zeros_like((qdotdot - qdotdot_old)),
                       where=(q - q_old) != 0))

        # calculating the jacobian matrix

        if ((onp.max(eps) > 1) and (iKMat > nKMat)): #SPR JAN max --> min
            if info == True:
                print('\nRecalculating the tangent matrix: ', nIters)

            iKMat = 0
            KCSR = assembleCSR(problem, dofs)
            M[:] = h_tilde * h_tilde / 4. * onp.array(
                onp.absolute(KCSR).sum(axis=1)).squeeze()

        #compute new velocities and accelerations
        # qdot_old[:] = qdot[:]; qdotdot_old[:] = qdotdot[:];
        # qdot = (2.- c*h)/(2 + c*h) * qdot_old - 2.*h/(2.+c*h)* R / M
        qdot_old[:] = qdot[:]
        qdotdot = qdot - qdot_old

        # output on screen
        printInfo(error, t, c, tol, eps, qdot, qdotdot, nIters, nPrint)

    # check if converged
    convergence = True
    if onp.isnan(onp.max(onp.absolute(R))):
        convergence = False

    # print final info
    if convergence:
        print("DRSolve finished in %d iterations and %fs" % \
              (nIters, time.time() - timeZ))
    else:
        print("FAILED to converged")

    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    return sol


################################################################################
# General
def solver(problem,
           linear=False,
           precond=True,
           initial_guess=None,
           use_petsc=False):
    """periodic B.C. is a special form of adding a linear constraint.
    Lagrange multiplier seems to be convenient to impose this constraint.
    """
    # TODO: print platform jax.lib.xla_bridge.get_backend().platform
    # and suggest PETSc or jax solver
    if problem.periodic_bc_info is None:
        return solver_row_elimination(problem, linear, precond, initial_guess,
                                      use_petsc)
    else:
        return solver_lagrange_multiplier(problem, linear, use_petsc)


################################################################################
# Implicit differentiation with the adjoint method


def implicit_vjp(problem, sol, params, v, use_petsc):

    def constraint_fn(dofs, params):
        """c(u, p)
        """
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    def constraint_fn_sol_to_sol(sol, params):
        return constraint_fn(sol.reshape(-1), params).reshape(sol.shape)

    def get_vjp_contraint_fn_dofs(dofs):
        # Just a transpose of A_fn
        def adjoint_linear_fn(adjoint):
            primals, f_vjp = jax.vjp(A_fn, dofs)
            val, = f_vjp(adjoint)
            return val

        return adjoint_linear_fn

    def get_partial_params_c_fn(sol):
        """c(u=u, p)
        """

        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol, params)

        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, sol):
        """v*(partial dc/dp)
        """
        partial_c_fn = get_partial_params_c_fn(sol)

        def vjp_linear_fn(v):
            primals, f_vjp = jax.vjp(partial_c_fn, params)
            val, = f_vjp(v)
            return val

        return vjp_linear_fn

    problem.set_params(params)
    problem.newton_update(sol)
    A_fn = get_A_fn(problem, use_petsc)

    if use_petsc:
        A_transpose = A_fn.transpose()

        # Remark: Eliminating rows seems to make A better conditioned.
        # If Dirichlet B.C. is part of the design variable, the following should NOT be implemented.
        # for i in range(len(problem.node_inds_list)):
        #     row_inds = onp.array(problem.node_inds_list[i]*problem.vec + problem.vec_inds_list[i], dtype=onp.int32)
        #     A_transpose.zeroRows(row_inds)
        # v = assign_zeros_bc(v, problem)

        adjoint = petsc_solve(A_transpose, v.reshape(-1), 'minres', 'ilu')

    else:
        adjoint_linear_fn = get_vjp_contraint_fn_dofs(sol.reshape(-1))
        adjoint = jax_solve(problem, adjoint_linear_fn, v.reshape(-1), None,
                            True)

    vjp_linear_fn = get_vjp_contraint_fn_params(params, sol)
    vjp_result = vjp_linear_fn(adjoint.reshape(sol.shape))
    vjp_result = jax.tree_map(lambda x: -x, vjp_result)

    return vjp_result


def ad_wrapper(problem, linear=False, use_petsc=False):

    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        sol = solver(problem, linear=linear, use_petsc=use_petsc)
        return sol

    def f_fwd(params):
        sol = fwd_pred(params)
        return sol, (params, sol)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol = res
        vjp_result = implicit_vjp(problem, sol, params, v, use_petsc)
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred
