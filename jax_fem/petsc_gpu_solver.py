"""Single-GPU PETSc Newton solver with device-resident bulk arrays.

The standard solver intentionally supports several CPU and GPU backends.  This
module is a narrow, opt-in path for a CUDA-enabled PETSc build.  Element
residuals and tangents stay as JAX CUDA arrays, PETSc assembles an
``AIJCUSPARSE`` matrix directly from device COO values, and vectors cross the
JAX/PETSc boundary through DLPack without host staging.

Only small convergence scalars are synchronized to the host.  Mesh creation,
the one-time COO sparsity setup, and optional output are outside the device
solve pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as np
import numpy as onp
from petsc4py import PETSc

from jax_fem import logger


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class _LinearConfig:
    mat_type: str = PETSc.Mat.Type.AIJCUSPARSE
    vec_type: str = PETSc.Vec.Type.CUDA
    ksp_type: str = PETSc.KSP.Type.CG
    pc_type: str = PETSc.PC.Type.ICC
    factor_solver_type: str | None = PETSc.Mat.SolverType.CUSPARSE
    factor_ordering_type: str = "natural"
    factor_reuse_ordering: bool = True
    factor_levels: int = 0
    cg_single_reduction: bool = False
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    max_it: int = 10_000
    options_prefix: str = "jax_fem_gpu_"
    require_cuda: bool = True


def _device_platform(array) -> str | None:
    devices = array.devices() if hasattr(array, "devices") else ()
    if not devices:
        return None
    return next(iter(devices)).platform


def _block_until_ready(*arrays) -> None:
    for array in arrays:
        if hasattr(array, "block_until_ready"):
            array.block_until_ready()


def _sync_petsc_device() -> None:
    """Make PETSc phase timings include queued device work."""
    PETSc.DeviceContext.getCurrent().synchronize()


def _boundary_rows(problem) -> onp.ndarray:
    rows = []
    for ind, fe in enumerate(problem.fes):
        for node_inds, vec_inds in zip(fe.node_inds_list, fe.vec_inds_list):
            rows.append(onp.asarray(
                node_inds * fe.vec + vec_inds + problem.offset[ind],
                dtype=PETSc.IntType,
            ))
    if not rows:
        return onp.empty(0, dtype=PETSc.IntType)
    return onp.unique(onp.concatenate(rows)).astype(PETSc.IntType, copy=False)


def _apply_bc_vec(res_vec, dofs, problem):
    """Device-only Dirichlet residual replacement used by Newton."""
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for node_inds, vec_inds, vals in zip(
            fe.node_inds_list, fe.vec_inds_list, fe.vals_list
        ):
            res = res.at[node_inds, vec_inds].set(
                sol[node_inds, vec_inds] - vals,
                unique_indices=True,
            )
        res_list[ind] = res
    return jax.flatten_util.ravel_pytree(res_list)[0]


def _validate_runtime(config: _LinearConfig) -> None:
    if PETSc.COMM_WORLD.getSize() != 1:
        raise RuntimeError(
            "petsc_gpu_solver supports one process and one GPU; do not launch it with mpiexec."
        )

    if onp.dtype(PETSc.ScalarType) != onp.dtype(onp.float64):
        raise RuntimeError(
            f"PETSc must use real float64 scalars, got {PETSc.ScalarType}."
        )

    if not config.require_cuda:
        return

    if not PETSc.Sys.hasExternalPackage("cuda"):
        raise RuntimeError("This PETSc build has no CUDA support.")

    cuda_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(cuda_devices) != 1:
        raise RuntimeError(
            "Exactly one JAX CUDA device must be visible. Set CUDA_VISIBLE_DEVICES "
            "to one device before importing JAX."
        )


class _PetscGpuLinearSystem:
    """Reusable CUDA matrix/KSP for a fixed JAX-FEM COO sparsity pattern."""

    def __init__(self, problem, config: _LinearConfig):
        self.config = config
        self.num_dofs = problem.num_total_dofs_all_vars
        self.bc_rows = _boundary_rows(problem)
        self.bc_rows_device = np.asarray(self.bc_rows, dtype=np.int32)

        coo_i = onp.asarray(problem.I, dtype=PETSc.IntType)
        coo_j = onp.asarray(problem.J, dtype=PETSc.IntType)
        if coo_i.shape != coo_j.shape:
            raise ValueError("problem.I and problem.J must have the same shape.")

        self.num_coo = coo_i.size
        self.mat = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        self.mat.setSizes((self.num_dofs, self.num_dofs))
        self.mat.setType(config.mat_type)
        self.mat.setVecType(config.vec_type)
        self.mat.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        self.mat.setPreallocationCOO(coo_i, coo_j)

        # Symmetric row-and-column elimination below preserves the SPD tangent
        # and makes CG + ICC mathematically appropriate.
        self.mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.mat.setOption(PETSc.Mat.Option.SPD, True)

        self.ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp.setOptionsPrefix(config.options_prefix)
        self.ksp.setOperators(self.mat)
        self.ksp.setType(config.ksp_type)
        self.ksp.setTolerances(
            rtol=config.rtol,
            atol=config.atol,
            max_it=config.max_it,
        )
        self.ksp.setInitialGuessNonzero(False)

        pc = self.ksp.getPC()
        pc.setType(config.pc_type)
        if config.pc_type in (PETSc.PC.Type.ICC, PETSc.PC.Type.ILU):
            if config.factor_solver_type is not None:
                pc.setFactorSolverType(config.factor_solver_type)
            pc.setFactorOrdering(
                config.factor_ordering_type,
                reuse=config.factor_reuse_ordering,
            )
            pc.setFactorLevels(config.factor_levels)

        # Only prefixed options can override the defaults, e.g.
        # -jax_fem_gpu_ksp_monitor or -jax_fem_gpu_ksp_view.
        temporary_option = None
        if config.cg_single_reduction and config.ksp_type == PETSc.KSP.Type.CG:
            option_name = config.options_prefix + "ksp_cg_single_reduction"
            petsc_options = PETSc.Options()
            if not petsc_options.hasName(option_name):
                petsc_options.setValue(option_name, True)
                temporary_option = (petsc_options, option_name)
        try:
            self.ksp.setFromOptions()
        finally:
            if temporary_option is not None:
                temporary_option[0].delValue(temporary_option[1])
        self._validate_petsc_types()

    def _validate_petsc_types(self) -> None:
        mat_type = self.mat.getType().lower()
        vec_type = self.mat.getVecType().lower()
        if self.config.require_cuda:
            if "cusparse" not in mat_type:
                raise RuntimeError(
                    f"PETSc matrix is {mat_type!r}, expected AIJCUSPARSE."
                )
            if "cuda" not in vec_type:
                raise RuntimeError(
                    f"PETSc matrix creates {vec_type!r} vectors, expected CUDA vectors."
                )

        pc = self.ksp.getPC()
        if (
            self.config.require_cuda
            and pc.getType() in (PETSc.PC.Type.ICC, PETSc.PC.Type.ILU)
            and pc.getFactorSolverType().lower() != PETSc.Mat.SolverType.CUSPARSE
        ):
            raise RuntimeError(
                "ICC/ILU was not configured with the cuSPARSE factor solver."
            )

    def update_matrix(self, values) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != self.num_coo:
            raise ValueError(
                f"Tangent has {values.size} COO values; expected {self.num_coo}."
            )
        if self.config.require_cuda and _device_platform(values) != "gpu":
            raise RuntimeError("Tangent COO values left the GPU before PETSc assembly.")

        # Synchronization is required for ownership/stream safety, but this is
        # not a host copy. petsc4py passes the CUDA array pointer to
        # MatSetValuesCOO for AIJCUSPARSE.
        _block_until_ready(values)
        self.mat.setValuesCOO(values, addv=PETSc.InsertMode.INSERT_VALUES)
        _sync_petsc_device()

    def _copy_jax_to_petsc(self, array, destination: PETSc.Vec) -> PETSc.Vec:
        array = np.asarray(array, dtype=np.float64).reshape(-1)
        if self.config.require_cuda and _device_platform(array) != "gpu":
            raise RuntimeError("A linear vector left the GPU before the PETSc solve.")
        _block_until_ready(array)
        source = PETSc.Vec().createWithDLPack(array, comm=PETSc.COMM_SELF)
        if self.config.require_cuda and "cuda" not in source.getType().lower():
            raise RuntimeError(
                f"DLPack produced PETSc vector type {source.getType()!r}, expected CUDA."
            )
        source.copy(destination)  # device-to-device; destination is mutable
        return source  # keep the shared JAX allocation alive until solve() returns

    def solve(self, rhs_array) -> tuple[Any, dict[str, Any]]:
        rhs_array = np.asarray(rhs_array, dtype=np.float64).reshape(-1)
        rhs = self.mat.createVecLeft()
        rhs_source = self._copy_jax_to_petsc(rhs_array, rhs)

        # MatZeroRowsColumns needs a mutable RHS.  The known constrained
        # increment equals rhs_array on the boundary and zero elsewhere.
        if self.bc_rows.size:
            boundary_increment = np.zeros_like(rhs_array)
            boundary_increment = boundary_increment.at[self.bc_rows_device].set(
                rhs_array[self.bc_rows_device]
            )
            _block_until_ready(boundary_increment)
            boundary_vec = PETSc.Vec().createWithDLPack(
                boundary_increment, comm=PETSc.COMM_SELF
            )
            self.mat.zeroRowsColumns(
                self.bc_rows, diag=1.0, x=boundary_vec, b=rhs
            )
        else:
            boundary_increment = None
            boundary_vec = None

        solution = self.mat.createVecRight()
        solution.set(0.0)
        setup_start = time.perf_counter()
        self.ksp.setUp()
        _sync_petsc_device()
        setup_seconds = time.perf_counter() - setup_start

        krylov_start = time.perf_counter()
        self.ksp.solve(rhs, solution)
        _sync_petsc_device()
        krylov_seconds = time.perf_counter() - krylov_start

        reason = int(self.ksp.getConvergedReason())
        linear_info = {
            "iterations": self.ksp.getIterationNumber(),
            "residual_norm": self.ksp.getResidualNorm(),
            "reason": reason,
            "setup_seconds": setup_seconds,
            "krylov_seconds": krylov_seconds,
        }
        if reason <= 0:
            raise RuntimeError(
                "PETSc GPU KSP failed: "
                f"reason={reason}, iterations={linear_info['iterations']}, "
                f"residual={linear_info['residual_norm']:.3e}."
            )

        # PETSc Vec implements __dlpack__ and __dlpack_device__.  JAX owns the
        # resulting read-only view after the handoff; there is no host staging.
        increment = jax.dlpack.from_dlpack(solution)
        _block_until_ready(increment)

        # References are intentionally kept through this point for DLPack
        # lifetime safety.
        del rhs_source, boundary_vec, boundary_increment
        return increment, linear_info

    def describe(self) -> dict[str, Any]:
        pc = self.ksp.getPC()
        description = {
            "mat_type": self.mat.getType(),
            "vec_type": self.mat.getVecType(),
            "ksp_type": self.ksp.getType(),
            "pc_type": pc.getType(),
            "cg_single_reduction": self.config.cg_single_reduction,
            "factor_reuse_ordering": self.config.factor_reuse_ordering,
        }
        if pc.getType() in (PETSc.PC.Type.ICC, PETSc.PC.Type.ILU):
            description["factor_solver_type"] = pc.getFactorSolverType()
        return description


def _linear_config(options: dict[str, Any]) -> _LinearConfig:
    linear = options.get("linear", {})
    require_cuda = linear.get("require_cuda", True)
    return _LinearConfig(
        mat_type=linear.get(
            "mat_type",
            PETSc.Mat.Type.AIJCUSPARSE if require_cuda else PETSc.Mat.Type.AIJ,
        ),
        vec_type=linear.get(
            "vec_type",
            PETSc.Vec.Type.CUDA if require_cuda else PETSc.Vec.Type.STANDARD,
        ),
        ksp_type=linear.get("ksp_type", PETSc.KSP.Type.CG),
        pc_type=linear.get("pc_type", PETSc.PC.Type.ICC),
        factor_solver_type=linear.get(
            "factor_solver_type",
            PETSc.Mat.SolverType.CUSPARSE if require_cuda else PETSc.Mat.SolverType.PETSC,
        ),
        factor_ordering_type=linear.get("factor_ordering_type", "natural"),
        factor_reuse_ordering=linear.get("factor_reuse_ordering", True),
        factor_levels=linear.get("factor_levels", 0),
        cg_single_reduction=linear.get("cg_single_reduction", False),
        rtol=linear.get("rtol", 1.0e-10),
        atol=linear.get("atol", 1.0e-12),
        max_it=linear.get("max_it", 10_000),
        options_prefix=linear.get("options_prefix", "jax_fem_gpu_"),
        require_cuda=require_cuda,
    )


def _get_linear_system(problem, config: _LinearConfig) -> _PetscGpuLinearSystem:
    cache = getattr(problem, "_petsc_gpu_linear_system", None)
    if cache is None or cache.config != config:
        cache = _PetscGpuLinearSystem(problem, config)
        problem._petsc_gpu_linear_system = cache
    return cache


def petsc_gpu_solver(problem, solver_options=None, return_info=False):
    """Solve a nonlinear problem through the single-GPU PETSc pipeline.

    Parameters
    ----------
    problem : Problem
        JAX-FEM problem with a fixed COO sparsity pattern.
    solver_options : dict, optional
        Newton keys ``tol``, ``rel_tol``, ``initial_guess`` and ``max_newton_it``.
        PETSc keys live under ``linear``. Defaults are CUDA AIJ, CG, and
        cuSPARSE ICC(0). ``linear.require_cuda=False`` exists only for CPU CI
        smoke tests and never activates automatically.
    return_info : bool
        Return ``(sol_list, info)`` when true.

    Notes
    -----
    Multipoint constraints and line search are intentionally not included in
    this narrow path. They can be added without changing the device linear
    interface once a GPU-native formulation is needed.
    """
    options = dict(solver_options or {})
    if hasattr(problem, "P_mat"):
        raise NotImplementedError(
            "petsc_gpu_solver does not support host SciPy multipoint constraints."
        )
    if options.get("line_search_flag", False):
        raise NotImplementedError("petsc_gpu_solver does not yet support line search.")

    config = _linear_config(options)
    _validate_runtime(config)

    previous_device_assembly = getattr(
        problem, "_jax_fem_device_assembly", None
    )
    had_device_assembly = hasattr(problem, "_jax_fem_device_assembly")
    problem._jax_fem_device_assembly = True

    timing = {"local_assembly": 0.0, "matrix_update": 0.0, "linear_solve": 0.0}
    linear_history = []
    wall_start = time.perf_counter()

    try:
        linear_system = _get_linear_system(problem, config)

        if "initial_guess" in options:
            initial_guess = jax.lax.stop_gradient(options["initial_guess"])
            dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars, dtype=np.float64)

        tol = options.get("tol", 1.0e-6)
        rel_tol = options.get("rel_tol", 1.0e-8)
        max_newton_it = options.get("max_newton_it", 50)

        def assemble(current_dofs):
            t0 = time.perf_counter()
            sol_list = problem.unflatten_fn_sol_list(current_dofs)
            res_list = problem.newton_update(sol_list)
            res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
            res_vec = _apply_bc_vec(res_vec, current_dofs, problem)
            tangent_values = problem.V
            _block_until_ready(res_vec, tangent_values)
            timing["local_assembly"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            linear_system.update_matrix(tangent_values)
            timing["matrix_update"] += time.perf_counter() - t0
            return res_vec

        res_vec = assemble(dofs)
        res_norm = float(np.linalg.norm(res_vec))
        initial_res_norm = res_norm
        relative_res_norm = 0.0 if initial_res_norm == 0.0 else 1.0
        logger.info(
            "PETSc GPU Newton iter 0: residual=%.6e, relative=%.6e",
            res_norm,
            relative_res_norm,
        )

        newton_it = 0
        while relative_res_norm > rel_tol and res_norm > tol:
            if newton_it >= max_newton_it:
                raise RuntimeError(
                    f"PETSc GPU Newton failed to converge in {max_newton_it} iterations."
                )
            newton_it += 1

            t0 = time.perf_counter()
            increment, linear_info = linear_system.solve(-res_vec)
            dofs = dofs + increment
            _block_until_ready(dofs)
            timing["linear_solve"] += time.perf_counter() - t0
            linear_history.append(linear_info)

            res_vec = assemble(dofs)
            res_norm = float(np.linalg.norm(res_vec))
            relative_res_norm = (
                0.0 if initial_res_norm == 0.0 else res_norm / initial_res_norm
            )
            logger.info(
                "PETSc GPU Newton iter %d: residual=%.6e, relative=%.6e, KSP iters=%d",
                newton_it,
                res_norm,
                relative_res_norm,
                linear_info["iterations"],
            )

        if not onp.isfinite(res_norm):
            raise FloatingPointError("PETSc GPU Newton residual is not finite.")
        _block_until_ready(dofs)
        if not bool(np.all(np.isfinite(dofs))):
            raise FloatingPointError("PETSc GPU Newton solution is not finite.")

        sol_list = problem.unflatten_fn_sol_list(dofs)
        wall_time = time.perf_counter() - wall_start
        info = {
            "wall_time": wall_time,
            "newton_iterations": newton_it,
            "residual_norm": res_norm,
            "relative_residual_norm": relative_res_norm,
            "timing": timing,
            "linear_history": linear_history,
            **linear_system.describe(),
        }
        logger.info(
            "PETSc GPU solve: %.3f s wall (assembly %.3f, matrix %.3f, linear %.3f)",
            wall_time,
            timing["local_assembly"],
            timing["matrix_update"],
            timing["linear_solve"],
        )
        if return_info:
            return sol_list, info
        return sol_list
    finally:
        if had_device_assembly:
            problem._jax_fem_device_assembly = previous_device_assembly
        else:
            del problem._jax_fem_device_assembly
