################################################################################
# Lagrangian multiplier solver

# TODO: Old version, not working for multi-physics problems. Requires refactoring. 

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
                lag += np.sum(d_lmbda_split[i] *
                    (sol[problem.node_inds_list[i], problem.vec_inds_list[i]] - problem.vals_list[i]))

            for i in range(len(problem.p_node_inds_list_A)):
                lag += np.sum(p_lmbda_split[i] *
                              (sol[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]] -
                               sol[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]]))
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
    A_sp_scipy_aug = scipy.sparse.csc_array((V, (I, J)), shape=(group_index, group_index))
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
    Update: It seems that petsc_options={'ksp_type': 'tfqmr', 'pc_type': 'ilu'} works better.

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
