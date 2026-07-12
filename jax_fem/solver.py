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
    pyamgx.initialize()
    PYAMGX_AVAILABLE = True
except ImportError:
    PYAMGX_AVAILABLE = False


def _timing_record(timing, name, dt):
    timing[name] += dt


def _log_newton_iter_start(iter_num):
    print()
    logger.info("  iter %d", iter_num)


def _log_newton_iter_summary(iter_num, local_s, global_s, res_val, rel_res_val, linear_s=None):
    logger.info("           nonlinear residual: L2 norm = %.3g (relative to initial = %.3g)",
                res_val, rel_res_val)
    if linear_s is None:
        logger.info("           timing: local assembly %6.3f s, global matrix %6.3f s",
                    local_s, global_s)
    else:
        logger.info("           timing: linear solve %6.3f s, local assembly %6.3f s, global matrix %6.3f s",
                    linear_s, local_s, global_s)


def _log_timing_table(n_iters, parts, wall_s):
    rows = (
        ('local_assembly', 'local'),
        ('global_matrix', 'global'),
        ('linear', 'linear'),
    )
    print()
    logger.info("Timing summary — %d Newton iter, %.3f s wall", n_iters, wall_s)
    for key, label in rows:
        dt = parts[key]
        pct = 100. * dt / wall_s if wall_s > 0 else 0.
        logger.info("  %-8s %7.3f s  %5.1f%%", label, dt, pct)
    other = wall_s - sum(parts.values())
    if other >= 0.01:
        pct = 100. * other / wall_s if wall_s > 0 else 0.
        logger.info("  %-8s %7.3f s  %5.1f%%", "other", other, pct)


################################################################################
# Linear solvers (JAX / SciPy / PETSc / AMGX)

def jax_solve(A, b, x0, precond):
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
    logger.debug("JAX Solver - Finished solving, linear solve res = %.3g", err)
    assert err < 0.1, f"JAX linear solver failed to converge with err = {err}"
    x = np.where(err < 0.1, x, np.nan) # For assert purpose, somehow this also affects bicgstab.

    return x

def scipy_spsolve(A, b):
    logger.debug("Scipy Solver - Solving linear system with scipy.sparse.linalg.spsolve")
    indptr, indices, data = A.getValuesCSR()
    Asp = scipy.sparse.csr_matrix((data, indices, indptr))
    # SciPy's spsolve uses UMFPACK only when scikits.umfpack is installed and
    # applicable; otherwise it falls back to SuperLU.
    x = scipy.sparse.linalg.spsolve(Asp, onp.array(b))

    # TODO: try https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
    # x = jax.experimental.sparse.linalg.spsolve(av, aj, ai, b)

    logger.debug("Scipy Solver - Finished solving, linear solve res = %.3g",
                 np.linalg.norm(Asp @ x - b))
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
    logger.debug("PETSc Solver - Finished solving, linear solve res = %.3g", err)
    assert err < 0.1, f"PETSc linear solver failed to converge, err = {err}"

    return x.getArray()

def AMGX_solve_host(indptr, indices, data, shape_arr, x, b, cfg_path):
    dtype, shape_b = b.dtype, b.shape

    n_rows = int(shape_arr[0])
    n_cols = int(shape_arr[1])

    A_csr = scipy.sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_rows, n_cols)
    )

    b_host = onp.asarray(b)
    x_guess = onp.zeros_like(b_host) if x is None else onp.asarray(x)

    cfg = None
    resources = None
    solver = None
    A_amg = None
    b_amg = None
    x_amg = None

    try:
        ## See: https://github.com/NVIDIA/AMGX/tree/main/src/configs
        if cfg_path is not None:
            cfg = pyamgx.Config().create_from_file(cfg_path)
        else:
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
        
        # Create resources
        resources = pyamgx.Resources().create_simple(cfg)
        solver = pyamgx.Solver().create(resources, cfg)

        A_amg = pyamgx.Matrix().create(resources)
        b_amg = pyamgx.Vector().create(resources)
        x_amg = pyamgx.Vector().create(resources)

        A_amg.upload_CSR(A_csr)
        b_amg.upload(b_host)
        x_amg.upload(x_guess)

        solver.setup(A_amg)
        solver.solve(b_amg, x_amg)

        result = x_amg.download()
        result = onp.asarray(result)

        res = onp.linalg.norm(A_csr @ result - b_host)
        logger.info("AMGX Solver - Finished solving, linear solve res = %.3g", res)

        return result.astype(dtype).reshape(shape_b)

    finally:
        if x_amg is not None:
            x_amg.destroy()
        if b_amg is not None:
            b_amg.destroy()
        if A_amg is not None:
            A_amg.destroy()
        if solver is not None:
            solver.destroy()
        if resources is not None:
            resources.destroy()
        if cfg is not None:
            cfg.destroy()
        
        # pyamgx.finalize()

def AMGX_solve(A, b, x0, cfg_path):

    if not PYAMGX_AVAILABLE:
        raise RuntimeError("pyamgx not installed. AMGX solver disabled.")

    # A is PETSc.Mat here.
    indptr, indices, data = A.getValuesCSR()
    n_rows, n_cols = A.getSize()

    # Convert to numpy arrays directly to avoid JAX device memory copy overhead
    indptr = onp.asarray(indptr, dtype=onp.int32)
    indices = onp.asarray(indices, dtype=onp.int32)

    # Keep matrix data dtype consistent with b.
    data = onp.asarray(data, dtype=onp.asarray(b).dtype)

    shape_arr = onp.array([n_rows, n_cols], dtype=onp.int64)

    if x0 is None:
        x0 = np.zeros_like(b)

    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)

    def amgx_solve_callback(x, b_in):
        return AMGX_solve_host(indptr, indices, data, shape_arr, x, b_in, cfg_path)

    return jax.pure_callback(
        amgx_solve_callback,
        result_shape,
        x0,
        b
    )

def linear_solver(A, b, x0, linear_options):
    # If user does not specify any solver, set jax_solver as the default one.
    if len(linear_options.keys() & {'jax_solver', 'amgx_solver', 'spsolve_solver', 'petsc_solver', 'custom_solver'}) == 0:
        linear_options['jax_solver'] = {}

    if 'jax_solver' in linear_options:
        precond = linear_options['jax_solver']['precond'] if 'precond' in linear_options['jax_solver'] else True
        x = jax_solve(A, b, x0, precond)
    elif 'amgx_solver' in linear_options:
        cfg_path = linear_options['amgx_solver']['cfg_path'] if 'cfg_path' in linear_options['amgx_solver'] else None
        x = AMGX_solve(A, b, x0, cfg_path)
    elif 'spsolve_solver' in linear_options:
        x = scipy_spsolve(A, b)
    elif 'petsc_solver' in linear_options:
        ksp_type = linear_options['petsc_solver']['ksp_type'] if 'ksp_type' in linear_options['petsc_solver'] else 'bcgsl'
        pc_type = linear_options['petsc_solver']['pc_type'] if 'pc_type' in linear_options['petsc_solver'] else 'ilu'
        x = petsc_solve(A, b, ksp_type, pc_type)
    elif 'custom_solver' in linear_options:
        custom_solver = linear_options['custom_solver']
        x = custom_solver(A, b, x0, linear_options)
    else:
        raise NotImplementedError(f"Unknown linear solver.")

    return x


################################################################################
# Dirichlet boundary conditions ("row elimination")

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


################################################################################
# Newton helpers: flattening and tangent probe

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


################################################################################
# Newton step (linear increment + optional line search)

def newton_step(problem, res_vec, A, dofs, newton_cfg, timing):
    """One Newton correction: solve :math:`A\\,\\Delta u = -R`, then update ``dofs``.

    Returns
    -------
    dofs : ndarray
    linear_s : float
        Linear solve wall time (also accumulated in ``timing``).
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

    t0 = time.perf_counter()
    inc = linear_solver(A, b, x0, newton_cfg.get('linear', {}))
    linear_s = time.perf_counter() - t0
    _timing_record(timing, 'linear', linear_s)

    if newton_cfg.get('line_search_flag', False):
        dofs = line_search(problem, dofs, inc)
    else:
        dofs = dofs + inc

    return dofs, linear_s


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
        logger.debug(f"i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.
            break
        res_norm = res_norm_half

    return dofs + alpha*inc


################################################################################
# Tangent stiffness matrix (PETSc cache)

class _PetscTangentCache:
    """Reusable full-space PETSc tangent built from fixed ``problem.I/J`` COO pattern."""

    def __init__(self, problem):
        n = problem.num_total_dofs_all_vars
        coo_i = onp.asarray(problem.I, dtype=PETSc.IntType)
        coo_j = onp.asarray(problem.J, dtype=PETSc.IntType)
        self.mat = PETSc.Mat().createAIJ(size=(n, n))
        self.mat.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        self.mat.setPreallocationCOO(coo_i, coo_j)
        self.bc_row_inds_list = []
        for ind, fe in enumerate(problem.fes):
            for i in range(len(fe.node_inds_list)):
                row_inds = onp.array(
                    fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
                    dtype=onp.int32,
                )
                self.bc_row_inds_list.append(row_inds)

    def update(self, problem):
        values = onp.asarray(problem.V, dtype=onp.float64)
        self.mat.setValuesCOO(values)
        self.mat.assemble()
        for row_inds in self.bc_row_inds_list:
            self.mat.zeroRows(row_inds)
        return self.mat


def _get_petsc_tangent_cache(problem):
    cache = getattr(problem, '_petsc_tangent_cache', None)
    if cache is None:
        cache = _PetscTangentCache(problem)
        problem._petsc_tangent_cache = cache
    return cache


def get_A(problem):
    logger.debug("Updating cached PETSc tangent from COO values...")
    A = _get_petsc_tangent_cache(problem).update(problem)

    # Linear multipoint constraints
    if hasattr(problem, 'P_mat'):
        P = PETSc.Mat().createAIJ(size=problem.P_mat.shape, csr=(problem.P_mat.indptr.astype(PETSc.IntType, copy=False),
                                                   problem.P_mat.indices.astype(PETSc.IntType, copy=False), problem.P_mat.data))

        tmp = A.matMult(P)
        P_T = P.transpose()
        A = P_T.matMult(tmp)

    return A


################################################################################
# Arc-length: Crisfeld formulation (displacement / force control)

def arc_length_solver_disp_driven(problem, prev_u_vec, prev_lamda, prev_Delta_u_vec,
                                  prev_Delta_lamda, Delta_l=0.1, psi=1.):
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem, lamda)
        A = get_A(problem)
        return res_vec, A

    def u_lamda_dot_product(Delta_u_vec1, Delta_lamda1, Delta_u_vec2, Delta_lamda2):
        return (np.sum(Delta_u_vec1 * Delta_u_vec2)
                + psi**2. * Delta_lamda1 * Delta_lamda2 * np.sum(u_b**2.))

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

        delta_u_bar = scipy_spsolve(A, -res_vec)
        delta_u_t = scipy_spsolve(A, u_b)

        Delta_u_vec = u_vec - prev_u_vec
        Delta_lamda = lamda - prev_lamda
        a1 = np.sum(delta_u_t**2.) + psi**2. * np.sum(u_b**2.)
        a2 = (2. * np.sum((Delta_u_vec + delta_u_bar) * delta_u_t)
              + 2. * psi**2. * Delta_lamda * np.sum(u_b**2.))
        a3 = (np.sum((Delta_u_vec + delta_u_bar)**2.)
              + psi**2. * Delta_lamda**2. * np.sum(u_b**2.) - Delta_l**2.)

        delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4. * a1 * a3)) / (2. * a1)
        delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4. * a1 * a3)) / (2. * a1)

        logger.debug(f"Arc length solver: delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")
        assert np.isfinite(delta_lamda1) and np.isfinite(delta_lamda2), (
            f"No valid solutions for delta lambda, a1 = {a1}, a2 = {a2}, a3 = {a3}")

        delta_u_vec1 = delta_u_bar + delta_lamda1 * delta_u_t
        delta_u_vec2 = delta_u_bar + delta_lamda2 * delta_u_t

        Delta_u_vec_dir1 = u_vec + delta_u_vec1 - prev_u_vec
        Delta_lamda_dir1 = lamda + delta_lamda1 - prev_lamda
        dot_prod1 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir,
                                        Delta_u_vec_dir1, Delta_lamda_dir1)

        Delta_u_vec_dir2 = u_vec + delta_u_vec2 - prev_u_vec
        Delta_lamda_dir2 = lamda + delta_lamda2 - prev_lamda
        dot_prod2 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir,
                                        Delta_u_vec_dir2, Delta_lamda_dir2)

        if np.abs(dot_prod1) < 1e-10 and np.abs(dot_prod2) < 1e-10:
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
    """Load vector at ``u=0`` for force-controlled arc-length (``arc_length`` cfg ``q_vec_aux``)."""
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    res_list = problem.newton_update(sol_list)
    return jax.flatten_util.ravel_pytree(res_list)[0]


def arc_length_solver_force_driven(problem, prev_u_vec, prev_lamda, prev_Delta_u_vec,
                                   prev_Delta_lamda, q_aux, Delta_l=0.1, psi=1.):
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A = get_A(problem)
        return res_vec, A

    def u_lamda_dot_product(Delta_u_vec1, Delta_lamda1, Delta_u_vec2, Delta_lamda2):
        return (np.sum(Delta_u_vec1 * Delta_u_vec2)
                + psi**2. * Delta_lamda1 * Delta_lamda2 * np.sum(q_aux_mapped**2.))

    u_vec = prev_u_vec
    lamda = prev_lamda
    q_aux_mapped = assign_zeros_bc(q_aux, problem)

    Delta_u_vec_dir = prev_Delta_u_vec
    Delta_lamda_dir = prev_Delta_lamda

    tol = 1e-6
    res_val = 1.
    while res_val > tol:
        res_vec, A = newton_update_helper(u_vec)
        load_term = (1. - lamda) * q_aux_mapped
        res_val = np.linalg.norm(res_vec + load_term)
        logger.debug(f"Arc length solver: res_val = {res_val}")

        delta_u_bar = scipy_spsolve(A, -(res_vec + load_term))
        delta_u_t = scipy_spsolve(A, q_aux_mapped)

        Delta_u_vec = u_vec - prev_u_vec
        Delta_lamda = lamda - prev_lamda
        a1 = np.sum(delta_u_t**2.) + psi**2. * np.sum(q_aux_mapped**2.)
        a2 = (2. * np.sum((Delta_u_vec + delta_u_bar) * delta_u_t)
              + 2. * psi**2. * Delta_lamda * np.sum(q_aux_mapped**2.))
        a3 = (np.sum((Delta_u_vec + delta_u_bar)**2.)
              + psi**2. * Delta_lamda**2. * np.sum(q_aux_mapped**2.) - Delta_l**2.)

        delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4. * a1 * a3)) / (2. * a1)
        delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4. * a1 * a3)) / (2. * a1)

        logger.debug(f"Arc length solver: delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")
        assert np.isfinite(delta_lamda1) and np.isfinite(delta_lamda2), (
            f"No valid solutions for delta lambda, a1 = {a1}, a2 = {a2}, a3 = {a3}")

        delta_u_vec1 = delta_u_bar + delta_lamda1 * delta_u_t
        delta_u_vec2 = delta_u_bar + delta_lamda2 * delta_u_t

        Delta_u_vec_dir1 = u_vec + delta_u_vec1 - prev_u_vec
        Delta_lamda_dir1 = lamda + delta_lamda1 - prev_lamda
        dot_prod1 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir,
                                        Delta_u_vec_dir1, Delta_lamda_dir1)

        Delta_u_vec_dir2 = u_vec + delta_u_vec2 - prev_u_vec
        Delta_lamda_dir2 = lamda + delta_lamda2 - prev_lamda
        dot_prod2 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir,
                                        Delta_u_vec_dir2, Delta_lamda_dir2)

        if np.abs(dot_prod1) < 1e-10 and np.abs(dot_prod2) < 1e-10:
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


def _arc_length_newton_polish(problem, sol_list, cfg, lam_continuation):
    logger.info(
        "Arc-length continuation ended at lambda=%.6f (target=%.6f); "
        "standard Newton polish at full load",
        lam_continuation, _LAMBDA_TARGET)
    polish = dict(cfg.get('newton', {}))
    polish['initial_guess'] = sol_list
    if cfg.get('linear'):
        polish['linear'] = cfg['linear']
    return solver(problem, {'newton': polish})


def _finish_arc_length(problem, u_vec, lam, cfg, max_steps, history, control):
    sol_list = problem.unflatten_fn_sol_list(onp.asarray(u_vec))
    lam_continuation = float(lam)
    reached_target = lam_continuation >= _LAMBDA_TARGET
    if not reached_target:
        logger.warning(
            "Arc-length stopped at lambda=%.6f after %d continuation steps "
            "(max_continuation_steps=%d); lambda=1 was not reached — "
            "the intended forward problem was not solved. "
            "Increase max_continuation_steps or adjust arc-length settings.",
            lam_continuation, len(history), max_steps)
    if reached_target:
        sol_list = _arc_length_newton_polish(
            problem, sol_list, cfg, lam_continuation)
    return sol_list, {
        'lam': lam_continuation,
        'lambda_target': _LAMBDA_TARGET,
        'polished': reached_target,
        'history': history,
        'control': control,
    }


def _solve_arc_length_disp(problem, cfg):
    """Displacement-controlled arc-length (Crisfeld outer loop)."""
    psi = cfg.get('psi', 1.)
    delta_l = cfg.get('Delta_l', 0.1)
    max_steps = cfg.get('max_continuation_steps', 600)
    step_callback = cfg.get('step_callback')

    u_vec = onp.zeros(problem.num_total_dofs_all_vars)
    lam = 0.
    delta_u_dir = onp.zeros_like(u_vec)
    delta_lam_dir = 0.
    history = []

    logger.info("Arc-length solve started (displacement control, Crisfeld).")
    start = time.time()
    for step in range(max_steps):
        u_vec, lam, delta_u_dir, delta_lam_dir = arc_length_solver_disp_driven(
            problem, u_vec, lam, delta_u_dir, delta_lam_dir, Delta_l=delta_l, psi=psi)
        record = {
            'step': step,
            'lam': float(lam),
            'u': onp.asarray(u_vec, dtype=onp.float64),
        }
        history.append(record)
        if step_callback is not None:
            step_callback(step, record['u'], lam)
        if lam >= _LAMBDA_TARGET:
            break

    elapsed = time.time() - start
    logger.info("Arc-length finished in %.3f s, %d continuation steps, final lambda=%.6f",
                elapsed, len(history), lam)

    return _finish_arc_length(
        problem, u_vec, lam, cfg, max_steps, history, 'displacement')


def _solve_arc_length_force(problem, cfg):
    """Force-controlled arc-length (Crisfeld outer loop)."""
    q_aux = cfg.get('q_vec_aux')
    if q_aux is None:
        raise ValueError("arc_length force control requires cfg['q_vec_aux'].")

    psi = cfg.get('psi', 0.5)
    delta_l = cfg.get('Delta_l', 0.1)
    delta_l_late = cfg.get('Delta_l_late', 1.0)
    switch_step = cfg.get('Delta_l_switch_step', 200)
    max_steps = cfg.get('max_continuation_steps', 500)
    step_callback = cfg.get('step_callback')

    u_vec = onp.zeros(problem.num_total_dofs_all_vars)
    lam = 0.
    delta_u_dir = onp.zeros_like(u_vec)
    delta_lam_dir = 0.
    history = []

    logger.info("Arc-length solve started (force control, Crisfeld).")
    start = time.time()
    for step in range(max_steps):
        dl = delta_l if step < switch_step else delta_l_late
        u_vec, lam, delta_u_dir, delta_lam_dir = arc_length_solver_force_driven(
            problem, u_vec, lam, delta_u_dir, delta_lam_dir, q_aux, Delta_l=dl, psi=psi)
        record = {
            'step': step,
            'lam': float(lam),
            'u': onp.asarray(u_vec, dtype=onp.float64),
        }
        history.append(record)
        if step_callback is not None:
            step_callback(step, record['u'], lam)
        if lam >= _LAMBDA_TARGET:
            break

    elapsed = time.time() - start
    logger.info("Arc-length finished in %.3f s, %d continuation steps, final lambda=%.6f",
                elapsed, len(history), lam)

    return _finish_arc_length(
        problem, u_vec, lam, cfg, max_steps, history, 'force')


def _solve_arc_length(problem, cfg):
    """
    Reference: Vasios, Nikolaos. "Nonlinear analysis of structures." The Arc-Length method.
    """
    if 'control' not in cfg:
        raise ValueError(
            "arc_length requires cfg['control']; use 'displacement' or 'force'.")
    control = cfg['control']
    if control == 'displacement':
        return _solve_arc_length_disp(problem, cfg)
    if control == 'force':
        return _solve_arc_length_force(problem, cfg)
    raise ValueError(f"Unknown arc_length control={control!r}; use 'displacement' or 'force'.")


################################################################################
# Dynamic relaxation

def _solve_dynamic_relax(problem, cfg):
    flat_guess = None
    if 'initial_guess' in cfg:
        initial_guess = jax.lax.stop_gradient(cfg['initial_guess'])
        flat_guess = jax.flatten_util.ravel_pytree(initial_guess)[0]
    return dynamic_relax_solve(
        problem,
        tol=cfg.get('tol', 1e-6),
        nKMat=cfg.get('nKMat', 1000),
        nPrint=cfg.get('nPrint', 500),
        info=cfg.get('info', True),
        info_force=cfg.get('info_force', True),
        initial_guess=flat_guess,
        linear_options=cfg.get('linear', {}),
    )


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
                   ' Max velocity = %g') % (nIters, error, tol,
                                            np.max(np.absolute(qdot))))
        if info == True:
            print('\nDamping t: ',t, );
            print('Damping coefficient: ', c)
            print('Max epsilon: ',np.max(eps))
            print('Max acceleration: ',np.max(np.absolute(qdotdot)))


def dynamic_relax_solve(problem, tol=1e-6, nKMat=1000, nPrint=500, info=True, info_force=True,
                        initial_guess=None, linear_options=None):
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
    linear_options = linear_options or {'spsolve_solver': {}}

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
    dofs, _ = newton_step(problem, res_vec, A, dofs, {'linear': linear_options},
                          {'local_assembly': 0., 'global_matrix': 0., 'linear': 0.})

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
        qdotdot = (qdot - qdot_old) / h

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

    return sol_list


################################################################################
# solver_options registry and dispatch
#
# Layout:
#
# Top level: at most ONE method key. Omit for Newton; legacy flat dicts
# (petsc_solver, tol, initial_guess, ...) are auto-wrapped as newton.
#
#   {'newton': {
#       'tol': 1e-6, 'rel_tol': 1e-8, 'line_search_flag': False,
#       'initial_guess': sol_list,
#       'linear': {'petsc_solver': {}},
#   }}
#
#   {'arc_length': {
#       'control': 'displacement' | 'force',
#       'return_info': True,
#       'q_vec_aux': ..., 'Delta_l': 0.1, 'step_callback': fn, ...
#       'linear': {'petsc_solver': {}},          # polish + inner solves
#       'newton': {'tol': 1e-6},                 # polish only
#   }}
#
#   {'dynamic_relax': {
#       'tol': 1e-8, 'nKMat': 1000, 'initial_guess': sol_list, ...
#       'linear': {'spsolve_solver': {}},
#   }}

_METHOD_KEYS = frozenset({'newton', 'arc_length', 'dynamic_relax'})
_LINEAR_OPTION_KEYS = frozenset({
    'jax_solver', 'amgx_solver', 'spsolve_solver', 'petsc_solver', 'custom_solver',
})
_NEWTON_OPTION_KEYS = frozenset({'tol', 'rel_tol', 'line_search_flag', 'initial_guess'})

_LAMBDA_TARGET = 1.


def _resolve_solver_options(solver_options):
    """Return (nonlinear_method, method_cfg). Legacy flat dicts become Newton."""
    opts = solver_options or {}
    methods = [m for m in _METHOD_KEYS if m in opts]

    if not methods:
        linear = {k: opts[k] for k in _LINEAR_OPTION_KEYS if k in opts}
        cfg = {k: opts[k] for k in _NEWTON_OPTION_KEYS if k in opts}
        if linear:
            cfg['linear'] = linear
        return 'newton', cfg

    if len(methods) > 1:
        raise ValueError(f"Pick one nonlinear method, got {methods}.")

    method = methods[0]
    if not isinstance(opts[method], dict):
        raise ValueError(f"solver_options['{method}'] must be a dict.")
    return method, opts[method]


################################################################################
# Nonlinear solver entry point (``solver()``)

def solver(problem, solver_options={}):
    r"""Solve a nonlinear problem (Newton by default, or arc-length / dynamic relaxation).

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
        Configuration for the nonlinear solve. Use exactly one top-level method key—
        ``newton``, ``arc_length``, or ``dynamic_relax``. Nest the linear
        solver and method-specific options inside that block.

        **Newton** (default nonlinear backend)::

            solver_options = {
                'newton': {
                    'tol': 1e-5,
                    'rel_tol': 1e-8,
                    'line_search_flag': False,
                    'initial_guess': initial_guess,
                    'linear': {'petsc_solver': {}},
                },
            }

        **Linear solvers** (keys under ``linear`` in any method block).
        Four backends are currently available:

        - `JAX solver <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.bicgstab.html>`_
        - `SciPy solver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html>`_
        - `PETSc solver <https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/index.html>`_
        - `AMGX solver <https://github.com/NVIDIA/AMGX>`_ (requires ``pyamgx``)

        Examples nested under ``newton``::

            solver_options = {'newton': {'linear': {'jax_solver': {}}}}

            solver_options = {'newton': {'linear': {'spsolve_solver': {}}}}

            solver_options = {
                'newton': {
                    'linear': {
                        'petsc_solver': {
                            'ksp_type': 'bcgsl',  # e.g. 'minres', 'gmres', 'tfqmr'
                            'pc_type': 'ilu',     # e.g. 'jacobi'
                        },
                    },
                },
            }

            solver_options = {'newton': {'linear': {'amgx_solver': {'cfg_path': 'path/to/amgx.json'}}}}

        **Defaults.** Omitted keys are filled in as follows.

        Newton (inside a ``newton`` block, or implied when no method key is
        given):

        - ``tol`` → ``1e-6`` (absolute residual :math:`\ell_2` norm)
        - ``rel_tol`` → ``1e-8`` (relative to the initial residual)
        - ``line_search_flag`` → ``False``
        - ``initial_guess`` → zero displacement vector
        - ``linear``: The following are all equivalent for the linear solve::

            solver_options = {}
            solver_options = {'newton': {}}
            solver_options = {'newton': {'linear': {}}}
            solver_options = {'newton': {'linear': {'jax_solver': {}}}}
            solver_options = {'newton': {'linear': {'jax_solver': {'precond': True}}}}

        - ``{'jax_solver': {}}`` → ``precond`` → ``True``
        - ``{'petsc_solver': {}}`` → ``ksp_type`` → ``'bcgsl'``; ``pc_type`` → ``'ilu'``
        - ``{'amgx_solver': {}}`` → ``cfg_path`` → ``None`` (built-in BICGSTAB + AMG)

        **Arc-length** (Crisfeld; ``control`` is required; set
        ``return_info`` to obtain continuation metadata)::

            solver_options = {
                'arc_length': {
                    'control': 'displacement',  # or 'force' (needs q_vec_aux)
                    'return_info': True,
                    'Delta_l': 0.1,
                    'linear': {'petsc_solver': {}},
                    'newton': {'tol': 1e-6},  # optional polish at lambda=1
                },
            }

        **Dynamic relaxation** (useful for buckling paths)::

            solver_options = {
                'dynamic_relax': {
                    'tol': 1e-8,
                    'linear': {'spsolve_solver': {}},
                },
            }

        **Legacy flat dict.** For backward compatibility, a dict with *no*
        method key is still accepted and interpreted as Newton. Linear and
        Newton keys may appear at the top level, e.g.::

            solver_options = {'petsc_solver': {}, 'tol': 1e-5}

        is equivalent to specifying::

            solver_options = {'newton': {'linear': {'petsc_solver': {}}, 'tol': 1e-5}}


    Returns
    -------
    sol_list : list

    """
    method, cfg = _resolve_solver_options(solver_options)
    if method == 'arc_length':
        sol_list, arc_info = _solve_arc_length(problem, cfg)
        if cfg.get('return_info', False):
            return sol_list, arc_info
        return sol_list

    if method == 'dynamic_relax':
        return _solve_dynamic_relax(problem, cfg)

    print()
    logger.info("Solving the nonlinear problem...")
    timing = {'local_assembly': 0., 'global_matrix': 0., 'linear': 0.}
    wall_start = time.perf_counter()

    if 'initial_guess' in cfg:
        # We don't want inititual guess to play a role in the differentiation chain.
        initial_guess = jax.lax.stop_gradient(cfg['initial_guess'])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
    else:
        if hasattr(problem, 'P_mat'):
            dofs = np.zeros(problem.P_mat.shape[1]) # reduced dofs
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars)

    rel_tol = cfg.get('rel_tol', 1e-8)
    tol = cfg.get('tol', 1e-6)

    def newton_update_helper(dofs):
        if hasattr(problem, 'P_mat'):
            dofs = problem.P_mat @ dofs

        sol_list = problem.unflatten_fn_sol_list(dofs)
        t0 = time.perf_counter()
        res_list = problem.newton_update(sol_list)
        local_s = time.perf_counter() - t0
        _timing_record(timing, 'local_assembly', local_s)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        if hasattr(problem, 'P_mat'):
            res_vec = problem.P_mat.T @ res_vec

        t0 = time.perf_counter()
        A = get_A(problem)
        global_s = time.perf_counter() - t0
        _timing_record(timing, 'global_matrix', global_s)
        return res_vec, A, local_s, global_s

    _log_newton_iter_start(0)
    res_vec, A, local_s, global_s = newton_update_helper(dofs)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = res_val
    rel_res_val = res_val/res_val_initial
    _log_newton_iter_summary(0, local_s, global_s, res_val, rel_res_val)
    n_iters = 0
    while (rel_res_val > rel_tol) and (res_val > tol):
        n_iters += 1
        _log_newton_iter_start(n_iters)
        dofs, linear_s = newton_step(problem, res_vec, A, dofs, cfg, timing)
        res_vec, A, local_s, global_s = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val/res_val_initial
        _log_newton_iter_summary(n_iters, local_s, global_s, res_val, rel_res_val, linear_s)

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

    _log_timing_table(n_iters, timing, time.perf_counter() - wall_start)

    print()
    logger.info(f"max of dofs = {np.max(dofs)}")
    logger.info(f"min of dofs = {np.min(dofs)}")

    return sol_list


################################################################################
# Implicit differentiation (adjoint method)

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
    # However, A.transpose(A_T) does not do in-place change to A
    A_T = PETSc.Mat()
    A.transpose(A_T)
    adjoint_vec = linear_solver(A_T, v_vec, None, adjoint_solver_options)

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
    solver_options : dict
        Same layout as :func:`solver` (nonlinear method + nested ``linear``).
    adjoint_solver_options : dict
        Linear solver options for the adjoint solve only (flat dict, e.g.
        ``{'petsc_solver': {}}``).

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
        print()
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options)
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred
