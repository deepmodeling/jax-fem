import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
from jax.experimental.sparse import BCOO
import scipy
import time
from petsc4py import PETSc
from jax_fem import logger
from jax import config
config.update("jax_enable_x64", True)


try:
    import pyamgx
    PYAMGX_AVAILABLE = True
except ImportError:
    PYAMGX_AVAILABLE = False
    logger.info("pyamgx not installed. AMGX solver disabled.")


################################################################################
# JAX solver or scipy solver or PETSc solver

def jax_solve(A, b, x0, precond):
    """Solves the equilibrium equation using a JAX solver.

    Parameters
    ----------
    precond
        Whether to calculate the preconditioner or not
    """
    logger.debug(f"JAX Solver - Solving linear system")
    indptr, indices, data = A.getValuesCSR()
    A_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=A.getSize())
    A = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    jacobi = np.array(A_sp_scipy.diagonal())
    pc = lambda x: x * (1. / jacobi) if precond else None
    
    if issubclass(PETSc.ScalarType, np.complexfloating):
        logger.debug("JAX Solver - Using PETSc with complex number support")
        A = A.astype(complex)
        b = b.astype(complex)
        if x0 is not None:
            x0 = x0.astype(complex)

    x, info = jax.scipy.sparse.linalg.bicgstab(A,
                                               b,
                                               x0=x0,
                                               M=pc,
                                               tol=1e-10,
                                               atol=1e-10,
                                               maxiter=10000)

    # Verify convergence
    err = np.linalg.norm(A @ x - b)
    logger.debug(f"JAX Solver - Finshed solving, res = {err}")
    assert err < 0.1, f"JAX linear solver failed to converge with err = {err}"
    x = np.where(err < 0.1, x, np.nan) # For assert purpose, some how this also affects bicgstab.

    return x

def umfpack_solve(A, b):
    # logger.debug(f"Scipy Solver - Solving linear system with UMFPACK")
    indptr, indices, data = A.getValuesCSR()
    # Asp = scipy.sparse.csr_matrix((data, indices, indptr))
    # x = scipy.sparse.linalg.spsolve(Asp, onp.array(b))

    logger.debug(f"Scipy Solver - Solving linear system with jax spsolve")
    # TODO: try https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
    x = jax.experimental.sparse.linalg.spsolve(data, indices, indptr, b, tol=1e-3)

    # logger.debug(f'Scipy Solver - Finished solving, linear solve res = {np.linalg.norm(Asp @ x - b)}')
    return x

def petsc_solve(A, b, ksp_type, pc_type):
    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), onp.array(b))
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType(ksp_type)
    ksp.pc.setType(pc_type)

    # TODO: This works better. Do we need to generalize the code a little bit?
    if ksp_type == 'tfqmr':
        ksp.pc.setFactorSolverType('mumps')

    logger.debug(f'PETSc Solver - Solving linear system with ksp_type = {ksp.getType()}, pc = {ksp.pc.getType()}')
    x = PETSc.Vec().createSeq(len(b))
    ksp.solve(rhs, x)

    # Verify convergence
    y = PETSc.Vec().createSeq(len(b))
    A.mult(x, y)

    err = np.linalg.norm(y.getArray() - rhs.getArray())
    logger.debug(f"PETSc Solver - Finished solving, linear solve res = {err}")
    assert err < 0.1, f"PETSc linear solver failed to converge, err = {err}"

    return x.getArray()


def AMGX_solve_host(A, x, b):
    dtype, shape = b.dtype, b.shape
    A = scipy.sparse.csr_matrix((A.data, (A.indices[:, 0], A.indices[:, 1])), shape=A.shape)
    b = onp.array(b)
    x_guess = onp.array(x)
    # setup AmgX solver
    # Initialize PyAMGX
    pyamgx.initialize()
    ## For solver options: https://github.com/NVIDIA/AMGX/tree/main/src/configs
    # Create resources
    cfg = pyamgx.Config().create_from_dict({
         "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "BICGSTAB",  # "CG", BICGSTAB
            #change to PBICGSTAB to use preconditioners
            "use_scalar_norm": 1,
            "norm": "L2",
            "tolerance": 1e-10,
            "monitor_residual": 1,
            "max_iters": 10000,
            "convergence": "ABSOLUTE",  # RELATIVE_INI_CORE
            "monitor_residual": 1,
            # "print_solve_stats": 1,
            "preconditioner": { 
                "scope": "amg",
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "smoother": "JACOBI",
                "cycle": "V",
                "max_levels": 10,
                "max_iters": 2
            }
        }
    })

    resources = pyamgx.Resources().create_simple(cfg)
    solver = pyamgx.Solver().create(resources, cfg)
    # Create matrix and vector objects
    A_amg = pyamgx.Matrix().create(resources)
    b_amg = pyamgx.Vector().create(resources)
    x_amg = pyamgx.Vector().create(resources)
    # ======
    # Upload data to PyAMGX objects
    A_amg.upload_CSR(A)
    b_amg.upload(b)
    x_amg.upload(x_guess)
    # logger.debug(f"Setting up the AMGx solver...")
    solver.setup(A_amg)
    solver.solve(b_amg, x_amg)
    # Download the result
    result = x_amg.download()
    # Cleanup
    x_amg.destroy()
    b_amg.destroy()
    A_amg.destroy()
    solver.destroy()
    cfg.destroy()
    resources.destroy()
    # Finalize PyAMGX
    pyamgx.finalize()
    logger.info(f'AMGX Solver - Finished solving, linear solve res = {np.linalg.norm(A @ result - b)}')
    return result.astype(dtype).reshape(shape)

def AMGX_solve(A, b, x0):
    if not PYAMGX_AVAILABLE:
        raise RuntimeError("AMGX disabled: 'pyamgx' not installed")

    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(AMGX_solve_host, result_shape, A,x0,b)


def linear_solver(A, b, x0, solver_options):
    # If user does not specify any solver, set jax_solver as the default one.
    if  len(solver_options.keys() & {'jax_solver','amgx_solver', 'umfpack_solver', 'petsc_solver', 'custom_solver'}) == 0: 
        solver_options['jax_solver'] = {}

    if 'jax_solver' in solver_options:      
        precond = solver_options['jax_solver']['precond'] if 'precond' in solver_options['jax_solver'] else True
        x = jax_solve(A, b, x0, precond)
    elif 'amgx_solver' in solver_options:
        x = AMGX_solve(A, b, x0)
    elif 'umfpack_solver' in solver_options:
        x = umfpack_solve(A, b)
    elif 'petsc_solver' in solver_options:   
        ksp_type = solver_options['petsc_solver']['ksp_type'] if 'ksp_type' in solver_options['petsc_solver'] else 'bcgsl' 
        pc_type = solver_options['petsc_solver']['pc_type'] if 'pc_type' in solver_options['petsc_solver'] else 'ilu'
        x = petsc_solve(A, b, ksp_type, pc_type)
    elif 'custom_solver' in solver_options:
        # Users can define their own solver
        custom_solver = solver_options['custom_solver']
        x = custom_solver(A, b, x0, solver_options)
    else:
        raise NotImplementedError(f"Unknown linear solver.")

    return x


################################################################################
# "row elimination" solver

def apply_bc_vec(res_vec, dofs, problem, scale=1.):
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = (res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True))
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(-fe.vals_list[i]*scale)

        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def apply_bc(res_fn, problem, scale=1.):
    def res_fn_bc(dofs):
        """Apply Dirichlet boundary conditions
        """
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem, scale)
    return res_fn_bc


def assign_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_ones_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(1.)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_zeros_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(0.)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def copy_bc(dofs, problem):
    new_dofs = np.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)
  
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = (new_sol.at[fe.node_inds_list[i],
                                  fe.vec_inds_list[i]].set(sol[fe.node_inds_list[i],
                                          fe.vec_inds_list[i]]))
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


def get_flatten_fn(fn_sol_list, problem):

    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


def operator_to_matrix(operator_fn, problem):
    """Only used for when debugging.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_dofs_all_vars))
    return J


def linear_incremental_solver(problem, res_vec, A, dofs, solver_options):
    """
    Linear solver at each Newton's iteration
    """
    logger.debug(f"Solving linear system...")
    b = -res_vec

    # x0 will always be correct at boundary locations
    x0_1 = assign_bc(np.zeros(problem.num_total_dofs_all_vars), problem)
    if hasattr(problem, 'P_mat'):
        x0_2 = copy_bc(problem.P_mat @ dofs, problem)
        x0 = problem.P_mat.T @ (x0_1 - x0_2)
    else:
        x0_2 = copy_bc(dofs, problem)
        x0 = x0_1 - x0_2

    inc = linear_solver(A, b, x0, solver_options)

    line_search_flag = solver_options['line_search_flag'] if 'line_search_flag' in solver_options else False
    if line_search_flag:
        dofs = line_search(problem, dofs, inc)
    else:
        dofs = dofs + inc

    return dofs


def line_search(problem, dofs, inc):
    """
    TODO: This is useful for finite deformation plasticity.
    """
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


def get_A(problem):
    logger.debug(f"Creating sparse matrix with scipy...")
    A_sp_scipy = scipy.sparse.csr_array((onp.array(problem.V), (problem.I, problem.J)),
        shape=(problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars))
    # logger.info(f"Global sparse matrix takes about {A_sp_scipy.data.shape[0]*8*3/2**30} G memory to store.")

    A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape, 
                              csr=(A_sp_scipy.indptr.astype(PETSc.IntType, copy=False),
                                   A_sp_scipy.indices.astype(PETSc.IntType, copy=False), 
                                   A_sp_scipy.data))

    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = onp.array(fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind], dtype=onp.int32)
            A.zeroRows(row_inds)

    # Linear multipoint constraints
    if hasattr(problem, 'P_mat'):
        P = PETSc.Mat().createAIJ(size=problem.P_mat.shape, csr=(problem.P_mat.indptr.astype(PETSc.IntType, copy=False),
                                                   problem.P_mat.indices.astype(PETSc.IntType, copy=False), problem.P_mat.data))

        tmp = A.matMult(P)
        P_T = P.transpose()
        A = P_T.matMult(tmp)

    return A


################################################################################
# The "row elimination" solver

def solver(problem, solver_options={}):
    r"""Solve the nonlinear problem using Newton's method with configurable linear solvers.

    The solver imposes Dirichlet B.C. with "row elimination" method. Conceptually,

    .. math::
        r(u) = D \, r_{\text{unc}}(u) + (I - D)u - u_b \\
        A = \frac{\text{d}r}{\text{d}u} = D \frac{\text{d}r}{\text{d}u} + (I - D)

    where:

    - :math:`r_{\text{unc}}: \mathbb{R}^N\rightarrow\mathbb{R}^N` is the residual function without considering Dirichlet boundary conditions.

    - :math:`u\in\mathbb{R}^N` is the FE solution vector.

    - :math:`u_b\in\mathbb{R}^N` is the vector for Dirichlet boundary conditions, e.g.,

      .. math::
            u_b = \begin{bmatrix}
                  0 \\
                  0 \\
                  2 \\
                  3
                  \end{bmatrix}
    
    - :math:`D\in\mathbb{R}^{N\times N}` is the auxiliary matrix for masking, e.g.,

      .. math::
            D = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0
            \end{bmatrix}
    
    - :math:`I\in\mathbb{R}^{N\times N}` is the ientity matrix, e.g., 

      .. math::
            I = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}

    - :math:`A\in\mathbb{R}^{N\times N}` is the tangent stiffness matrix (the global Jacobian matrix).

    Notes
    -----
    - TODO: Show some comments for linear multipoint constraint handling.

    Parameters
    ----------
    problem : Problem
        The nonlinear problem to solve
    solver_options : dict
        Configuration dictionary for solver parameters and algorithms.
        Three solvers are currently available:

        - `JAX solver <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.bicgstab.html>`_
        - `UMFPACK solver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html>`_
        - `PETSc solver <https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/index.html>`_
    
        The empty choice ::

            solver_options = {}

        will default to JAX solver as ::
        
            solver_options = {'jax_solver': {}}

        which will further default to ::

            solver_options = {'jax_solver': {'precond': True}}

        The UMFPACK solver can be specified as ::

            solver_options = {'umfpack_solver': {}}

        The PETSc solver can be specified as ::

            solver_options = {
                 'petsc_solver': {},
            }        
    
        which will default to ::

            solver_options = {
                 'petsc_solver': {
                     'ksp_type': 'bcgsl', # other choices can be, e.g., 'minres', 'gmres', 'tfqmr'
                     'pc_type': 'ilu', # other choices can be, e.g., 'jacobi'
                 }
            }  
    
        Other available options are :: 

            solver_options = {
                'line_search_flag': False, # Line search method
                'initial_guess': initial_guess, # Same shape as sol_list
                'tol': 1e-5, # Absolute tolerance for residual vector (l2 norm), used in Newton's method
                'rel_tol': 1e-8, # Relative tolerance for residual vector (l2 norm), used in Newton's method
            }

    Returns
    -------
    sol_list : list

    """
    logger.debug(f"Calling the row elimination solver for imposing Dirichlet B.C.")
    logger.debug("Start timing")
    start = time.time()

    if 'initial_guess' in solver_options:
        # We dont't want inititual guess to play a role in the differentiation chain.
        initial_guess = jax.lax.stop_gradient(solver_options['initial_guess'])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
    else:
        if hasattr(problem, 'P_mat'):
            dofs = np.zeros(problem.P_mat.shape[1]) # reduced dofs
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars)

    rel_tol = solver_options['rel_tol'] if 'rel_tol' in solver_options else 1e-8
    tol = solver_options['tol'] if 'tol' in solver_options else 1e-6

    def newton_update_helper(dofs):
        if hasattr(problem, 'P_mat'):
            dofs = problem.P_mat @ dofs

        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        if hasattr(problem, 'P_mat'):
            res_vec = problem.P_mat.T @ res_vec

        A = get_A(problem)
        return res_vec, A

    res_vec, A = newton_update_helper(dofs)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = res_val
    rel_res_val = res_val/res_val_initial
    logger.debug(f"Before, l_2 res = {res_val}, relative l_2 res = {rel_res_val}")
    while (rel_res_val > rel_tol) and (res_val > tol):
        dofs = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)
        res_vec, A = newton_update_helper(dofs)
        # logger.debug(f"DEBUG: l_2 res = {np.linalg.norm(apply_bc_vec(A @ dofs, dofs, problem))}")
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val/res_val_initial

        logger.debug(f"l_2 res = {res_val}, relative l_2 res = {rel_res_val}")

    assert np.all(np.isfinite(res_val)), f"res_val contains NaN, stop the program!"
    assert np.all(np.isfinite(dofs)), f"dofs contains NaN, stop the program!"

    if hasattr(problem, 'P_mat'):
        dofs = problem.P_mat @ dofs

    # If sol_list = [[[u1x, u1y], 
    #                 [u2x, u2y], 
    #                 [u3x, u3y], 
    #                 [u4x, u4y]], 
    #                [[p1], 
    #                 [p2]]],
    # the flattend DOF vector will be [u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y, p1, p2]
    sol_list = problem.unflatten_fn_sol_list(dofs)

    end = time.time()
    solve_time = end - start
    logger.info(f"Solve took {solve_time} [s]")
    logger.info(f"max of dofs = {np.max(dofs)}")
    logger.info(f"min of dofs = {np.min(dofs)}")

    return sol_list


################################################################################
# The "arc length" solver
# Reference: Vasios, Nikolaos. "Nonlinear analysis of structures." The Arc-Length method. Harvard (2015).
# Our implementation follows the Crisfeld's formulation

# TODO: Do we want to merge displacement-control and force-control codes?

def arc_length_solver_disp_driven(problem, prev_u_vec, prev_lamda, prev_Delta_u_vec, prev_Delta_lamda, Delta_l=0.1, psi=1.):
    """
    TODO: Does not support periodic B.C., need some work here.
    """
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem, lamda)
        A = get_A(problem)
        return res_vec, A

    def u_lamda_dot_product(Delta_u_vec1, Delta_lamda1, Delta_u_vec2, Delta_lamda2):
        return np.sum(Delta_u_vec1*Delta_u_vec2) + psi**2.*Delta_lamda1*Delta_lamda2*np.sum(u_b**2.)
 
    u_vec = prev_u_vec
    lamda = prev_lamda

    u_b = assign_bc(np.zeros_like(prev_u_vec), problem)

    Delta_u_vec_dir = prev_Delta_u_vec
    Delta_lamda_dir = prev_Delta_lamda

    tol = 1e-6
    res_val = 1.
    while res_val > tol:

        res_vec, A = newton_update_helper(u_vec)
        res_val = np.linalg.norm(res_vec)
        logger.debug(f"Arc length solver: res_val = {res_val}")
  
        delta_u_bar = umfpack_solve(A, -res_vec)
        delta_u_t = umfpack_solve(A, u_b)

        Delta_u_vec = u_vec - prev_u_vec
        Delta_lamda = lamda - prev_lamda
        a1 = np.sum(delta_u_t**2.) + psi**2.*np.sum(u_b**2.)
        a2 = 2.* np.sum((Delta_u_vec + delta_u_bar)*delta_u_t) + 2.*psi**2.*Delta_lamda*np.sum(u_b**2.)
        a3 = np.sum((Delta_u_vec + delta_u_bar)**2.) + psi**2.*Delta_lamda**2.*np.sum(u_b**2.) - Delta_l**2.

        delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)
        delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)

        logger.debug(f"Arc length solver: delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")
        assert np.isfinite(delta_lamda1) and np.isfinite(delta_lamda2), f"No valid solutions for delta lambda, a1 = {a1}, a2 = {a2}, a3 = {a3}"

        delta_u_vec1 = delta_u_bar + delta_lamda1 * delta_u_t
        delta_u_vec2 = delta_u_bar + delta_lamda2 * delta_u_t

        Delta_u_vec_dir1 = u_vec + delta_u_vec1 - prev_u_vec
        Delta_lamda_dir1 = lamda + delta_lamda1 - prev_lamda
        dot_prod1 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir1, Delta_lamda_dir1)

        Delta_u_vec_dir2 = u_vec + delta_u_vec2 - prev_u_vec
        Delta_lamda_dir2 = lamda + delta_lamda2 - prev_lamda
        dot_prod2 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir2, Delta_lamda_dir2)

        if np.abs(dot_prod1) < 1e-10 and np.abs(dot_prod2) < 1e-10:
            # At initial step, (Delta_u_vec_dir, Delta_lamda_dir) is zero, so both dot_prod1 and dot_prod2 are zero.
            # We simply select the larger value for delta_lamda.
            delta_lamda = np.maximum(delta_lamda1, delta_lamda2)
        elif dot_prod1 > dot_prod2:
            delta_lamda = delta_lamda1
        else:
            delta_lamda = delta_lamda2

        lamda = lamda + delta_lamda
        delta_u = delta_u_bar + delta_lamda * delta_u_t
        u_vec = u_vec + delta_u

        Delta_u_vec_dir = u_vec - prev_u_vec
        Delta_lamda_dir = lamda - prev_lamda

    logger.debug(f"Arc length solver: finished for one step, with Delta lambda = {lamda - prev_lamda}")
 
    return u_vec, lamda, Delta_u_vec_dir, Delta_lamda_dir


def arc_length_solver_force_driven(problem, prev_u_vec, prev_lamda, prev_Delta_u_vec, prev_Delta_lamda, q_vec, Delta_l=0.1, psi=1.):
    """
    TODO: Does not support periodic B.C., need some work here.
    """
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A = get_A(problem)
        return res_vec, A

    def u_lamda_dot_product(Delta_u_vec1, Delta_lamda1, Delta_u_vec2, Delta_lamda2):
        return np.sum(Delta_u_vec1*Delta_u_vec2) + psi**2.*Delta_lamda1*Delta_lamda2*np.sum(q_vec_mapped**2.)
 
    u_vec = prev_u_vec
    lamda = prev_lamda
    q_vec_mapped = assign_zeros_bc(q_vec, problem)

    Delta_u_vec_dir = prev_Delta_u_vec
    Delta_lamda_dir = prev_Delta_lamda

    tol = 1e-6
    res_val = 1.
    while res_val > tol:
        res_vec, A = newton_update_helper(u_vec)
        res_val = np.linalg.norm(res_vec + lamda*q_vec_mapped)
        logger.debug(f"Arc length solver: res_val = {res_val}")

        # TODO: the scipy umfpack solver seems to be far better than the jax linear solver, so we use umfpack solver here.
        # x0_1 = assign_bc(np.zeros_like(u_vec), problem)
        # x0_2 = copy_bc(u_vec, problem)
        # delta_u_bar = jax_solve(problem, A, -(res_vec + lamda*q_vec_mapped), x0=x0_1 - x0_2, precond=True)   
        # delta_u_t = jax_solve(problem, A, -q_vec_mapped, x0=np.zeros_like(u_vec), precond=True)   

        delta_u_bar = umfpack_solve(A, -(res_vec + lamda*q_vec_mapped))
        delta_u_t = umfpack_solve(A, -q_vec_mapped)

        Delta_u_vec = u_vec - prev_u_vec
        Delta_lamda = lamda - prev_lamda
        a1 = np.sum(delta_u_t**2.) + psi**2.*np.sum(q_vec_mapped**2.)
        a2 = 2.* np.sum((Delta_u_vec + delta_u_bar)*delta_u_t) + 2.*psi**2.*Delta_lamda*np.sum(q_vec_mapped**2.)
        a3 = np.sum((Delta_u_vec + delta_u_bar)**2.) + psi**2.*Delta_lamda**2.*np.sum(q_vec_mapped**2.) - Delta_l**2.

        delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)
        delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)

        logger.debug(f"Arc length solver: delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")
        assert np.isfinite(delta_lamda1) and np.isfinite(delta_lamda2), f"No valid solutions for delta lambda, a1 = {a1}, a2 = {a2}, a3 = {a3}"

        delta_u_vec1 = delta_u_bar + delta_lamda1 * delta_u_t
        delta_u_vec2 = delta_u_bar + delta_lamda2 * delta_u_t

        Delta_u_vec_dir1 = u_vec + delta_u_vec1 - prev_u_vec
        Delta_lamda_dir1 = lamda + delta_lamda1 - prev_lamda
        dot_prod1 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir1, Delta_lamda_dir1)

        Delta_u_vec_dir2 = u_vec + delta_u_vec2 - prev_u_vec
        Delta_lamda_dir2 = lamda + delta_lamda2 - prev_lamda
        dot_prod2 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir2, Delta_lamda_dir2)

        if np.abs(dot_prod1) < 1e-10 and np.abs(dot_prod2) < 1e-10:
            # At initial step, (Delta_u_vec_dir, Delta_lamda_dir) is zero, so both dot_prod1 and dot_prod2 are zero.
            # We simply select the larger value for delta_lamda.
            delta_lamda = np.maximum(delta_lamda1, delta_lamda2)
        elif dot_prod1 > dot_prod2:
            delta_lamda = delta_lamda1
        else:
            delta_lamda = delta_lamda2

        lamda = lamda + delta_lamda
        delta_u = delta_u_bar + delta_lamda * delta_u_t
        u_vec = u_vec + delta_u

        Delta_u_vec_dir = u_vec - prev_u_vec
        Delta_lamda_dir = lamda - prev_lamda

    logger.debug(f"Arc length solver: finished for one step, with Delta lambda = {lamda - prev_lamda}")
 
    return u_vec, lamda, Delta_u_vec_dir, Delta_lamda_dir


def get_q_vec(problem):
    """
    Used in the arc length method only, to get the external force vector q_vec
    """
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    res_list = problem.newton_update(sol_list)
    q_vec = jax.flatten_util.ravel_pytree(res_list)[0]
    return q_vec


################################################################################
# Dynamic relaxation solver

def assembleCSR(problem, dofs):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    problem.newton_update(sol_list)
    A_sp_scipy = scipy.sparse.csr_array((problem.V, (problem.I, problem.J)),
        shape=(problem.fes[0].num_total_dofs, problem.fes[0].num_total_dofs))

    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = onp.array(fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind], dtype=onp.int32)
            for row_ind in row_inds:
                A_sp_scipy.data[A_sp_scipy.indptr[row_ind]: A_sp_scipy.indptr[row_ind + 1]] = 0.
                A_sp_scipy[row_ind, row_ind] = 1.

    return A_sp_scipy


def calC(t, cmin, cmax):

    if t < 0.: t = 0.

    c = 2. * onp.sqrt(t)
    if (c < cmin): c = cmin
    if (c > cmax): c = cmax

    return c


def printInfo(error, t, c, tol, eps, qdot, qdotdot, nIters, nPrint, info, info_force):

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


def dynamic_relax_solve(problem, tol=1e-6, nKMat=50, nPrint=500, info=True, info_force=True, initial_guess=None):
    """
    Implementation of

    Luet, David Joseph. Bounding volume hierarchy and non-uniform rational B-splines for contact enforcement
    in large deformation finite element analysis of sheet metal forming. Diss. Princeton University, 2016.
    Chapter 4.3 Nonlinear System Solution

    Particularly good for handling buckling behavior.
    There is a FEniCS version of this dynamic relaxation algorithm.
    The code below is a direct translation from the FEniCS version.

 
    TODO: Does not support periodic B.C., need some work here.
    """
    solver_options = {'umfpack_solver': {}}

    # TODO: consider these in initial guess
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A = get_A(problem)
        return res_vec, A
 
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    res_vec, A = newton_update_helper(dofs)
    dofs = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)

    if initial_guess is not None:
        dofs = initial_guess
        dofs = assign_bc(dofs, problem)

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

        print(f"error = {error}")
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

        # compute new velocities and accelerations
        qdot_old[:] = qdot[:]; qdotdot_old[:] = qdotdot[:];
        qdot = (2.- c*h)/(2 + c*h) * qdot_old - 2.*h/(2.+c*h)* R / M
        qdot_old[:] = qdot[:]
        qdotdot = qdot - qdot_old

        # output on screen
        printInfo(error, t, c, tol, eps, qdot, qdotdot, nIters, nPrint, info, info_force)

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

    sol_list = problem.unflatten_fn_sol_list(dofs)

    return sol_list[0]


################################################################################
# Implicit differentiation with the adjoint method

def implicit_vjp(problem, sol_list, params, v_list, adjoint_solver_options):

    def constraint_fn(dofs, params):
        """c(u, p)
        """
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = constraint_fn(dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def get_partial_params_c_fn(sol_list):
        """c(u=u, p)
        """
        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol_list, params)

        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, sol_list):
        """v*(partial dc/dp)
        """
        partial_c_fn = get_partial_params_c_fn(sol_list)
        def vjp_linear_fn(v_list):
            primals_output, f_vjp = jax.vjp(partial_c_fn, params)
            val, = f_vjp(v_list)
            return val
        return vjp_linear_fn

    problem.set_params(params)
    problem.newton_update(sol_list)

    A = get_A(problem)
    v_vec = jax.flatten_util.ravel_pytree(v_list)[0]

    if hasattr(problem, 'P_mat'):
        v_vec = problem.P_mat.T @ v_vec

    # Be careful that A.transpose() does in-place change to A
    adjoint_vec = linear_solver(A.transpose(), v_vec, None, adjoint_solver_options)

    if hasattr(problem, 'P_mat'):
        adjoint_vec = problem.P_mat @ adjoint_vec

    vjp_linear_fn = get_vjp_contraint_fn_params(params, sol_list)
    vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)

    return vjp_result


def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}):
    """Automatic differentiation wrapper for the forward problem.

    Parameters
    ----------
    problem : Problem
    solver_options : dictionary
    adjoint_solver_options : dictionary

    Returns
    -------
    fwd_pred : callable
    """
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        sol_list = solver(problem, solver_options)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options)
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred
