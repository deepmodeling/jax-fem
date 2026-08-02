"""Single-GPU bridge between JAX device arrays and PETSc CUDA objects.

This module is intentionally isolated from the regular CPU PETSc path in
``jax_fem.solver``.  The matrix sparsity pattern is registered once from the
host-side COO indices, while every subsequent coefficient update is passed
directly from JAX device memory to ``MatSetValuesCOO``.

The bridge currently supports serial PETSc (one MPI rank), CUDA, and forward
Newton solves.  It does not silently fall back to host assembly.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as np
import numpy as onp
from petsc4py import PETSc

_MAT_SET_VALUES_COO = None
_PETSC_CUDA_CSR_API = None
_CUDA_RUNTIME = None
_AMGX_API = None


def _mat_set_values_coo():
    """Load PETSc's public C ``MatSetValuesCOO`` symbol lazily."""
    global _MAT_SET_VALUES_COO
    if _MAT_SET_VALUES_COO is None:
        petsc_lib = ctypes.CDLL(PETSc.__file__)
        fn = petsc_lib.MatSetValuesCOO
        fn.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
        fn.restype = ctypes.c_int
        _MAT_SET_VALUES_COO = fn
    return _MAT_SET_VALUES_COO


def _petsc_cuda_csr_api():
    """Load PETSc's public device-CSR and CUDA-Vec accessors lazily."""
    global _PETSC_CUDA_CSR_API
    if _PETSC_CUDA_CSR_API is None:
        lib = ctypes.CDLL(PETSc.__file__)

        get_ij = lib.MatSeqAIJCUSPARSEGetIJ
        get_ij.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        )
        get_ij.restype = ctypes.c_int

        restore_ij = lib.MatSeqAIJCUSPARSERestoreIJ
        restore_ij.argtypes = get_ij.argtypes
        restore_ij.restype = ctypes.c_int

        get_values = lib.MatSeqAIJCUSPARSEGetArrayRead
        get_values.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        get_values.restype = ctypes.c_int

        restore_values = lib.MatSeqAIJCUSPARSERestoreArrayRead
        restore_values.argtypes = get_values.argtypes
        restore_values.restype = ctypes.c_int

        get_vec = lib.VecCUDAGetArrayWrite
        get_vec.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        get_vec.restype = ctypes.c_int

        restore_vec = lib.VecCUDARestoreArrayWrite
        restore_vec.argtypes = get_vec.argtypes
        restore_vec.restype = ctypes.c_int

        _PETSC_CUDA_CSR_API = (
            get_ij,
            restore_ij,
            get_values,
            restore_values,
            get_vec,
            restore_vec,
        )
    return _PETSC_CUDA_CSR_API


def _cuda_runtime():
    """Load the CUDA runtime used only for cross-library synchronization."""
    global _CUDA_RUNTIME
    if _CUDA_RUNTIME is None:
        cuda_root = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        candidates = [
            ctypes.util.find_library("cudart"),
            str(Path(cuda_root) / "lib64" / "libcudart.so"),
            str(Path(cuda_root) / "lib64" / "libcudart.so.12"),
            "libcudart.so.12",
            "libcudart.so",
        ]
        errors = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                lib = ctypes.CDLL(candidate)
                synchronize = lib.cudaDeviceSynchronize
                synchronize.argtypes = ()
                synchronize.restype = ctypes.c_int
                _CUDA_RUNTIME = synchronize
                break
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        if _CUDA_RUNTIME is None:
            raise RuntimeError(
                "Could not load libcudart for JAX/PETSc/AMGX synchronization: "
                + "; ".join(errors)
            )
    return _CUDA_RUNTIME


def _cuda_synchronize():
    error = _cuda_runtime()()
    if error:
        raise RuntimeError(f"cudaDeviceSynchronize failed with error code {error}.")


def _petsc_check(error):
    if error:
        raise PETSc.Error(error)


def _find_amgx_library():
    explicit = os.environ.get("JAX_FEM_AMGX_LIBRARY")
    candidates = [explicit] if explicit else []

    petsc_dir = os.environ.get("PETSC_DIR")
    if petsc_dir:
        candidates.append(str(Path(petsc_dir) / "lib" / "libamgxsh.so"))

    candidates.extend([ctypes.util.find_library("amgxsh"), "libamgxsh.so"])
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "Could not load libamgxsh.so. Set JAX_FEM_AMGX_LIBRARY to its full "
        "path. Attempts: " + "; ".join(errors)
    )


class _AmgxApi:
    """Small ctypes binding to the stable single-GPU AMGX C API."""

    MODE_DDDI = 8193
    SOLVE_SUCCESS = 0

    def __init__(self):
        self.lib, self.library_path = _find_amgx_library()

        self.get_error_string = self._bind(
            "AMGX_get_error_string",
            (ctypes.c_int, ctypes.c_void_p, ctypes.c_int),
        )
        self.initialize = self._bind("AMGX_initialize", ())
        self.config_create = self._bind(
            "AMGX_config_create",
            (ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p),
        )
        self.config_create_from_file = self._bind(
            "AMGX_config_create_from_file",
            (ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p),
        )
        self.resources_create_simple = self._bind(
            "AMGX_resources_create_simple",
            (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p),
        )
        self.matrix_create = self._bind(
            "AMGX_matrix_create",
            (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int),
        )
        self.vector_create = self._bind(
            "AMGX_vector_create",
            (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int),
        )
        self.solver_create = self._bind(
            "AMGX_solver_create",
            (
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ),
        )
        self.matrix_upload_all = self._bind(
            "AMGX_matrix_upload_all",
            (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ),
        )
        self.matrix_replace_coefficients = self._bind(
            "AMGX_matrix_replace_coefficients",
            (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ),
        )
        self.vector_upload = self._bind(
            "AMGX_vector_upload",
            (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p),
        )
        self.vector_download = self._bind(
            "AMGX_vector_download",
            (ctypes.c_void_p, ctypes.c_void_p),
        )
        self.solver_setup = self._bind(
            "AMGX_solver_setup", (ctypes.c_void_p, ctypes.c_void_p)
        )
        self.solver_resetup = self._bind(
            "AMGX_solver_resetup", (ctypes.c_void_p, ctypes.c_void_p)
        )
        self.solver_solve = self._bind(
            "AMGX_solver_solve",
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
        )
        self.solver_get_status = self._bind(
            "AMGX_solver_get_status",
            (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)),
        )
        self.solver_get_iterations_number = self._bind(
            "AMGX_solver_get_iterations_number",
            (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)),
        )

        self.check(self.initialize(), "AMGX_initialize")

    def _bind(self, name, argtypes):
        function = getattr(self.lib, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        return function

    def check(self, error, operation):
        if not error:
            return
        message = ctypes.create_string_buffer(4096)
        self.get_error_string(error, message, len(message))
        detail = message.value.decode(errors="replace")
        raise RuntimeError(f"{operation} failed (AMGX error {error}): {detail}")


def _amgx_api():
    global _AMGX_API
    if _AMGX_API is None:
        _AMGX_API = _AmgxApi()
    return _AMGX_API


def _require_single_gpu_environment():
    if PETSc.COMM_WORLD.getSize() != 1:
        raise RuntimeError(
            "petsc_gpu_solver currently supports exactly one PETSc MPI rank."
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "petsc_gpu_solver requires JAX to use a GPU backend; "
            f"JAX selected {jax.default_backend()!r}."
        )
    if not PETSc.Sys.hasExternalPackage("cuda"):
        raise RuntimeError(
            "petsc_gpu_solver requires a PETSc build configured with CUDA."
        )


def _require_device_vector(values, expected_size):
    values = np.asarray(values).reshape(-1)
    if values.size != expected_size:
        raise ValueError(
            "PETSc COO value count does not match the registered sparsity "
            f"pattern: got {values.size}, expected {expected_size}."
        )
    if onp.dtype(values.dtype) != onp.dtype(PETSc.ScalarType):
        raise TypeError(
            "JAX/PETSc scalar dtype mismatch: "
            f"JAX has {values.dtype}, PETSc expects {onp.dtype(PETSc.ScalarType)}."
        )

    devices = values.devices()
    if len(devices) != 1 or next(iter(devices)).platform != "gpu":
        raise RuntimeError(
            "MatSetValuesCOO device update requires one JAX GPU array; "
            f"got devices {devices}."
        )
    return values


def set_values_coo_device(mat, values, expected_size):
    """Update a preallocated PETSc matrix from a JAX GPU array without D2H.

    ``petsc4py.Mat.setValuesCOO`` currently converts its input through NumPy.
    Calling the same public PETSc API through its C symbol avoids that Python
    conversion.  JAX is synchronized before PETSc consumes the device pointer.
    """
    values = _require_device_vector(values, expected_size)
    values.block_until_ready()

    ierr = _mat_set_values_coo()(
        ctypes.c_void_p(mat.handle),
        ctypes.c_void_p(values.unsafe_buffer_pointer()),
        int(PETSc.InsertMode.INSERT_VALUES),
    )
    if ierr:
        raise PETSc.Error(ierr)


class PetscGpuTangentCache:
    """Reusable ``MATSEQAIJCUSPARSE`` tangent with a fixed COO pattern."""

    def __init__(self, problem):
        _require_single_gpu_environment()
        if hasattr(problem, "P_mat"):
            raise NotImplementedError(
                "petsc_gpu_solver does not yet support problem.P_mat "
                "(periodic or multipoint constraints)."
            )

        self.num_dofs = problem.num_total_dofs_all_vars
        self.coo_i = onp.asarray(problem.I, dtype=PETSc.IntType)
        self.coo_j = onp.asarray(problem.J, dtype=PETSc.IntType)
        self.num_coo_values = self.coo_i.size

        self.mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self.mat.setSizes((self.num_dofs, self.num_dofs))
        self.mat.setType("seqaijcusparse")
        self.mat.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        self.mat.setPreallocationCOO(self.coo_i, self.coo_j)

        self.bc_row_inds_list = []
        for ind, fe in enumerate(problem.fes):
            for i in range(len(fe.node_inds_list)):
                row_inds = onp.asarray(
                    fe.node_inds_list[i] * fe.vec
                    + fe.vec_inds_list[i]
                    + problem.offset[ind],
                    dtype=PETSc.IntType,
                )
                self.bc_row_inds_list.append(row_inds)
        self.update_seconds = 0.0
        self.update_calls = 0

    def update(self, problem):
        start = time.perf_counter()
        set_values_coo_device(self.mat, problem.V, self.num_coo_values)
        for row_inds in self.bc_row_inds_list:
            self.mat.zeroRows(row_inds)
        # MatSetValuesCOO and MatZeroRows may enqueue CUDA work.  Completing it
        # here gives both a safe hand-off to native AMGX and useful timings.
        _cuda_synchronize()
        self.update_seconds += time.perf_counter() - start
        self.update_calls += 1
        return self.mat


def get_petsc_gpu_tangent(problem):
    """Return the problem's cached device-resident PETSc tangent."""
    cache = getattr(problem, "_petsc_gpu_tangent_cache", None)
    if cache is None:
        cache = PetscGpuTangentCache(problem)
        problem._petsc_gpu_tangent_cache = cache
    return cache.update(problem)


def _solver_config(options):
    options = options or {}
    allowed = {
        "rtol",
        "max_it",
        "residual_check_tol",
        "backend",
        "amgx_config_path",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            "Unknown petsc_gpu_solver option(s): "
            + ", ".join(sorted(unknown))
        )
    backend = str(options.get("backend", "native_amgx")).lower()
    if backend != "native_amgx":
        raise ValueError(
            "petsc_gpu_solver only supports backend='native_amgx'."
        )
    amgx_config_path = options.get("amgx_config_path")
    if amgx_config_path is not None:
        amgx_config_path = str(amgx_config_path)
    return {
        "backend": backend,
        "rtol": options.get("rtol", 1.0e-10),
        "max_it": options.get("max_it", 10_000),
        "residual_check_tol": options.get("residual_check_tol", 1.0e-8),
        "amgx_config_path": amgx_config_path,
    }


def validate_petsc_gpu_environment(options=None):
    """Fail before finite-element assembly if the requested GPU stack is absent."""
    _solver_config(options)
    _require_single_gpu_environment()
    if onp.dtype(PETSc.IntType) != onp.dtype(onp.int32):
        raise RuntimeError("native_amgx requires 32-bit PETSc indices.")
    if onp.dtype(PETSc.ScalarType) != onp.dtype(onp.float64):
        raise RuntimeError("native_amgx currently requires PETSc float64 scalars.")
    _amgx_api()


def _default_native_amgx_config(config):
    """Match the long-standing ``AMGX_solve_host`` defaults."""
    return {
        "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "BICGSTAB",
            "use_scalar_norm": 1,
            "norm": "L2",
            "tolerance": config["rtol"],
            "monitor_residual": 1,
            "max_iters": config["max_it"],
            "convergence": "ABSOLUTE",
            "preconditioner": {
                "scope": "amg",
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "smoother": "JACOBI",
                "cycle": "V",
                "max_levels": 10,
                "max_iters": 2,
            },
        },
    }


class PetscGpuNativeAmgxCache:
    """Native AMGX solve fed by PETSc's device-resident CSR arrays.

    PETSc owns the CSR storage. AMGX receives the structure once and only the
    changing coefficient array on later Newton steps. No matrix/vector array
    is converted through NumPy in this class.
    """

    def __init__(self, mat, options):
        self.config = _solver_config(options)
        self.api = _amgx_api()
        self.mat = mat
        self.n = int(mat.getSize()[0])
        if mat.getSize()[1] != self.n:
            raise ValueError("native_amgx requires a square matrix.")
        self.nnz = int(mat.getInfo(PETSc.Mat.InfoType.LOCAL)["nz_used"])
        if self.n > onp.iinfo(onp.int32).max or self.nnz > onp.iinfo(onp.int32).max:
            raise OverflowError("native_amgx requires 32-bit row and nonzero counts.")

        self.amgx_config = ctypes.c_void_p()
        self.resources = ctypes.c_void_p()
        self.amgx_matrix = ctypes.c_void_p()
        self.rhs = ctypes.c_void_p()
        self.solution = ctypes.c_void_p()
        self.solver = ctypes.c_void_p()

        config_path = self.config["amgx_config_path"]
        if config_path:
            self.api.check(
                self.api.config_create_from_file(
                    ctypes.byref(self.amgx_config), os.fsencode(config_path)
                ),
                "AMGX_config_create_from_file",
            )
        else:
            config_json = json.dumps(_default_native_amgx_config(self.config))
            self.api.check(
                self.api.config_create(
                    ctypes.byref(self.amgx_config), config_json.encode()
                ),
                "AMGX_config_create",
            )

        self.api.check(
            self.api.resources_create_simple(
                ctypes.byref(self.resources), self.amgx_config
            ),
            "AMGX_resources_create_simple",
        )
        self.api.check(
            self.api.matrix_create(
                ctypes.byref(self.amgx_matrix),
                self.resources,
                self.api.MODE_DDDI,
            ),
            "AMGX_matrix_create",
        )
        for handle, name in ((self.rhs, "rhs"), (self.solution, "solution")):
            self.api.check(
                self.api.vector_create(
                    ctypes.byref(handle), self.resources, self.api.MODE_DDDI
                ),
                f"AMGX_vector_create({name})",
            )
        self.api.check(
            self.api.solver_create(
                ctypes.byref(self.solver),
                self.resources,
                self.api.MODE_DDDI,
                self.amgx_config,
            ),
            "AMGX_solver_create",
        )

        self.x = mat.createVecRight()
        if self.x.getType() != "seqcuda":
            raise RuntimeError(
                "Expected a PETSc CUDA solution vector, "
                f"but Mat.createVecRight() returned {self.x.getType()!r}."
            )

        self.matrix_uploaded = False
        self.solver_is_setup = False
        self.last_iterations = 0
        self.last_status = None
        self.last_relative_residual = None
        self.timings = {
            "matrix_upload": 0.0,
            "setup": 0.0,
            "vector_upload": 0.0,
            "solve": 0.0,
            "vector_download": 0.0,
        }
        self.solve_calls = 0

    def _update_amgx_matrix(self):
        get_ij, restore_ij, get_values, restore_values, _, _ = (
            _petsc_cuda_csr_api()
        )
        row_ptr = ctypes.POINTER(ctypes.c_int)()
        col_ind = ctypes.POINTER(ctypes.c_int)()
        values = ctypes.c_void_p()

        _petsc_check(
            get_ij(
                ctypes.c_void_p(self.mat.handle),
                0,
                ctypes.byref(row_ptr),
                ctypes.byref(col_ind),
            )
        )
        try:
            _petsc_check(
                get_values(
                    ctypes.c_void_p(self.mat.handle), ctypes.byref(values)
                )
            )
            try:
                if not self.matrix_uploaded:
                    error = self.api.matrix_upload_all(
                        self.amgx_matrix,
                        self.n,
                        self.nnz,
                        1,
                        1,
                        ctypes.cast(row_ptr, ctypes.c_void_p),
                        ctypes.cast(col_ind, ctypes.c_void_p),
                        values,
                        None,
                    )
                    operation = "AMGX_matrix_upload_all"
                else:
                    error = self.api.matrix_replace_coefficients(
                        self.amgx_matrix,
                        self.n,
                        self.nnz,
                        values,
                        None,
                    )
                    operation = "AMGX_matrix_replace_coefficients"
                self.api.check(error, operation)
                _cuda_synchronize()
                self.matrix_uploaded = True
            finally:
                _petsc_check(
                    restore_values(
                        ctypes.c_void_p(self.mat.handle), ctypes.byref(values)
                    )
                )
        finally:
            _petsc_check(
                restore_ij(
                    ctypes.c_void_p(self.mat.handle),
                    0,
                    ctypes.byref(row_ptr),
                    ctypes.byref(col_ind),
                )
            )

    def _download_solution(self):
        *_, get_vec, restore_vec = _petsc_cuda_csr_api()
        output = ctypes.c_void_p()
        _petsc_check(
            get_vec(ctypes.c_void_p(self.x.handle), ctypes.byref(output))
        )
        try:
            self.api.check(
                self.api.vector_download(self.solution, output),
                "AMGX_vector_download",
            )
            _cuda_synchronize()
        finally:
            _petsc_check(
                restore_vec(ctypes.c_void_p(self.x.handle), ctypes.byref(output))
            )

    def solve(self, b, x0=None):
        rhs_array = _require_device_vector(b, self.n)
        if x0 is None:
            x0_array = np.zeros_like(rhs_array)
        else:
            x0_array = _require_device_vector(x0, self.n)
        rhs_array.block_until_ready()
        x0_array.block_until_ready()
        _cuda_synchronize()

        start = time.perf_counter()
        self._update_amgx_matrix()
        self.timings["matrix_upload"] += time.perf_counter() - start

        start = time.perf_counter()
        if self.solver_is_setup:
            error = self.api.solver_resetup(self.solver, self.amgx_matrix)
            operation = "AMGX_solver_resetup"
        else:
            error = self.api.solver_setup(self.solver, self.amgx_matrix)
            operation = "AMGX_solver_setup"
        self.api.check(error, operation)
        _cuda_synchronize()
        self.solver_is_setup = True
        self.timings["setup"] += time.perf_counter() - start

        start = time.perf_counter()
        self.api.check(
            self.api.vector_upload(
                self.rhs,
                self.n,
                1,
                ctypes.c_void_p(rhs_array.unsafe_buffer_pointer()),
            ),
            "AMGX_vector_upload(rhs)",
        )
        self.api.check(
            self.api.vector_upload(
                self.solution,
                self.n,
                1,
                ctypes.c_void_p(x0_array.unsafe_buffer_pointer()),
            ),
            "AMGX_vector_upload(solution)",
        )
        _cuda_synchronize()
        self.timings["vector_upload"] += time.perf_counter() - start

        start = time.perf_counter()
        self.api.check(
            self.api.solver_solve(self.solver, self.rhs, self.solution),
            "AMGX_solver_solve",
        )
        _cuda_synchronize()
        self.timings["solve"] += time.perf_counter() - start

        status = ctypes.c_int()
        iterations = ctypes.c_int()
        self.api.check(
            self.api.solver_get_status(self.solver, ctypes.byref(status)),
            "AMGX_solver_get_status",
        )
        self.api.check(
            self.api.solver_get_iterations_number(
                self.solver, ctypes.byref(iterations)
            ),
            "AMGX_solver_get_iterations_number",
        )
        self.last_status = status.value
        self.last_iterations = iterations.value
        if status.value != self.api.SOLVE_SUCCESS:
            raise RuntimeError(
                "Native AMGX linear solve did not converge: "
                f"status={status.value}, iterations={iterations.value}."
            )

        start = time.perf_counter()
        self._download_solution()
        self.timings["vector_download"] += time.perf_counter() - start

        rhs_view = PETSc.Vec().createWithDLPack(
            rhs_array, size=self.n, comm=PETSc.COMM_WORLD
        )
        residual = self.mat.createVecLeft()
        try:
            self.mat.mult(self.x, residual)
            residual.axpy(-1.0, rhs_view)
            relative_residual = residual.norm() / max(rhs_view.norm(), 1.0)
        finally:
            residual.destroy()
            rhs_view.destroy()
        self.last_relative_residual = relative_residual
        if relative_residual > self.config["residual_check_tol"]:
            raise RuntimeError(
                "Native AMGX residual check failed: "
                f"relative residual={relative_residual:.3e}."
            )

        shared = jax.dlpack.from_dlpack(self.x)
        result = np.array(shared, copy=True)
        result.block_until_ready()
        self.solve_calls += 1
        return result


def solve_petsc_gpu(problem, mat, b, x0, options):
    """Solve on CUDA and return an owning JAX GPU array."""
    config = _solver_config(options)
    cache_name = "_petsc_gpu_native_amgx_cache"
    cache = getattr(problem, cache_name, None)
    cache_key = (mat.handle, tuple(sorted(config.items())))
    if cache is None or cache[0] != cache_key:
        tangent_cache = getattr(problem, "_petsc_gpu_tangent_cache", None)
        if tangent_cache is None or tangent_cache.mat.handle != mat.handle:
            raise RuntimeError(
                "PETSc GPU tangent cache is missing or does not match the "
                "matrix passed to the linear solver."
            )
        solver_cache = PetscGpuNativeAmgxCache(mat, config)
        cache = (cache_key, solver_cache)
        setattr(problem, cache_name, cache)
    return cache[1].solve(b, x0)
