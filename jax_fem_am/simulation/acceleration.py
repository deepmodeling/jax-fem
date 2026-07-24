#!/usr/bin/env python3
"""XLA / GPU linear-solver wrapper for the mech100 macro thermal-stress solver.

This module wraps ``jax_fem_am.simulation.stepper`` (the CPU
reference implementation) and lets the user swap the *linear* solver used
inside the Newton loop without touching the physics driver:

    spsolve  -- SciPy sparse direct solve (CPU reference / fallback)
    jax      -- JAX BiCGStab/CG on GPU (optional Jacobi preconditioner)
    petsc    -- petsc4py KSP (optionally GPU-aware, ksp/pc configurable)
    amgx     -- NVIDIA AMGX (via pyamgx) with persistent resources
    pardiso  -- MKL PARDISO via pypardiso (CPU multithreaded direct solve)
    keep     -- do not rewrite; use whatever the base config specifies

Design constraints (see docs/XLA_UPGRADE_ROADMAP.md):

* Pure-Python option plumbing (`linear_options_from_args`,
  `rewrite_solver_options`) is importable WITHOUT jax/petsc/pyamgx being
  installed, so unit tests and CPU-only CI can exercise it.
* Every run is instrumented with a `ProfilingReport` that separates
  solver time, matrix-conversion time, Python main-loop overhead and I/O
  time. Claims about GPU speedup must come from these numbers on real
  meshes, never from dry runs.
* If the GPU path is slower than spsolve, the wrapper reports *why*
  (conversion-dominated vs solve-dominated) instead of forcing GPU use.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager

from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODULE_NAME = "jax_fem_am.simulation.stepper"
# Equals ``Path(jax_fem_am.simulation.stepper.__file__)``: stepper.py lives in
# the same package directory as this module. Kept as a path constant (instead
# of importing the stepper here) so importing this wrapper stays lightweight
# and ``apply_runtime_env`` can still set JAX_PLATFORM_NAME & friends before
# JAX is imported. Used for ``report.meta["base_solver"]``.
BASE_SOLVER_PATH = SCRIPT_DIR / "stepper.py"
# Measured on the real 197k-cell h60 mesh (RTX 5080 16GB): 2048 splits the
# jacobian assembly into 96 chunks whose per-chunk onp.vstack device->host
# copies dominate the kernel cost (cell_jacobian 6.4s -> 0.09s per 8-step
# slice when assembled as a single chunk). 262144 keeps meshes up to ~262k
# cells in one fixed-shape kernel launch; larger meshes still chunk. Mechanics
# jacobians at this batch are ~300MB device-side, well within 16GB.
DEFAULT_CELL_TARGET_BATCH_SIZE = 262144
DOF_TO_QUAD_CACHE_MAX_ENTRIES = 8
THERMAL_TABLE_KEYS = (
    "k_solid",
    "cp_solid",
    "k_powder",
    "cp_powder",
    "k_liquid",
    "cp_liquid",
)
MECHANICAL_TABLE_KEYS = (
    "E",
    "alpha",
    "poisson",
    "yield",
    "hardening",
)
_LOOP_KERNEL_JIT_THERMAL_CACHE: Dict[tuple[Any, ...], Any] = {}
_LOOP_KERNEL_JIT_MECHANICS_CACHE: Dict[tuple[Any, ...], Any] = {}
_LOOP_KERNEL_JIT_HISTORY_CACHE: Dict[tuple[Any, ...], Any] = {}

# ---------------------------------------------------------------------------
# Linear-solver option plumbing (pure Python, no heavy imports)
# ---------------------------------------------------------------------------

#: Keys that identify a linear-solver block inside a solver-options dict.
#: The legacy (flat) config format puts one of these at the top level of the
#: nonlinear-solver options; the current format nests it under
#: ``options["newton"]["linear"]``.
LINEAR_SOLVER_KEYS = frozenset(
    {
        "spsolve_solver",
        "splu_solver",
        "jax_solver",
        "petsc_solver",
        "amgx_solver",
        "cg_solver",
        "bicgstab_solver",
        "gmres_solver",
        "custom_solver",
    }
)

METHOD_KEYS = frozenset({"newton", "arc_length", "dynamic_relax"})

#: argparse choice -> canonical linear-solver key
_SOLVER_CHOICE_TO_KEY = {
    "spsolve": "spsolve_solver",
    "jax": "jax_solver",
    "petsc": "petsc_solver",
    "amgx": "amgx_solver",
    "pardiso": "custom_solver",
}


class _PardisoCustomSolver:
    """`custom_solver` adapter: PETSc AIJ -> SciPy CSR -> MKL PARDISO.

    Direct solve like spsolve (same accuracy class), but factorization runs
    multithreaded via MKL. pypardiso is imported lazily so the option
    plumbing stays importable without it.
    """

    label = "pardiso_solver(mkl multithreaded direct)"

    def __init__(self, mode: Optional[str] = None) -> None:
        valid_modes = {None, "base", "nocmp", "cache-idx", "phase23", "fp32ir"}
        if mode not in valid_modes:
            raise ValueError(f"unsupported PARDISO mode: {mode!r}")
        self._solver = None
        self._v07_variant = None
        self._requested_mode = mode
        if mode not in (None, "base"):
            self.label = f"pardiso_v07({mode})"

    def __deepcopy__(self, memo):
        # Shared instance keeps the PARDISO handle alive across the
        # option-rewrite deep copies done for every solve.
        return self

    def _maybe_v07_variant(self):
        # V07 ablation hook: V07_PARDISO_MODE selects an experimental
        # solver ladder from jax_fem_am/solvers/pardiso.py. Unset (or
        # "base") keeps this class's behaviour untouched.
        if self._v07_variant is None:
            import os

            mode = self._requested_mode
            if mode is None:
                mode = os.environ.get("V07_PARDISO_MODE", "").strip()
            if not mode or mode == "base":
                self._v07_variant = False
            else:
                from jax_fem_am.solvers.pardiso import VariantSolver

                self._v07_variant = VariantSolver(mode)
                self.label = self._v07_variant.label
        return self._v07_variant

    def __call__(self, A, b, x0, linear_options):
        import numpy as onp
        import scipy.sparse
        import pypardiso

        variant = self._maybe_v07_variant()
        if variant is not False:
            return variant(A, b, x0, linear_options)

        if self._solver is None:
            self._solver = pypardiso.PyPardisoSolver()
        indptr, indices, data = A.getValuesCSR()
        Asp = scipy.sparse.csr_matrix(
            (
                data,
                indices.astype(onp.int32, copy=False),
                indptr.astype(onp.int32, copy=False),
            )
        )
        rhs = onp.asarray(b, dtype=onp.float64)
        return pypardiso.spsolve(Asp, rhs, solver=self._solver)


def linear_options_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Build the replacement linear-solver options block from CLI args.

    Returns ``None`` when the user asked to keep the base configuration
    (``--xla-linear-solver keep``), otherwise a single-key dict such as
    ``{"jax_solver": {"precond": False}}`` suitable for
    :func:`rewrite_solver_options`.
    """
    choice = getattr(args, "xla_linear_solver", "keep")
    if choice in (None, "keep", "preserve"):
        return None
    if choice not in _SOLVER_CHOICE_TO_KEY:
        raise ValueError(
            f"unknown --xla-linear-solver value {choice!r}; "
            f"expected one of {sorted(_SOLVER_CHOICE_TO_KEY)} or 'keep'"
        )

    key = _SOLVER_CHOICE_TO_KEY[choice]
    inner: Dict[str, Any] = {}

    if choice == "jax":
        inner["precond"] = bool(getattr(args, "xla_jax_precond", False))
        method = getattr(args, "xla_jax_method", None)
        if method:
            inner["method"] = method
        tol = getattr(args, "xla_jax_tol", None)
        if tol is not None:
            inner["tol"] = float(tol)
        atol = getattr(args, "xla_jax_atol", None)
        if atol is not None:
            inner["atol"] = float(atol)
        maxiter = getattr(args, "xla_jax_maxiter", None)
        if maxiter is not None:
            inner["maxiter"] = int(maxiter)
        restart = getattr(args, "xla_jax_gmres_restart", None)
        if restart is not None:
            inner["restart"] = int(restart)
        solve_method = getattr(args, "xla_jax_gmres_solve_method", None)
        if solve_method:
            inner["solve_method"] = solve_method
        if getattr(args, "xla_jax_skip_residual_check", False):
            inner["check_residual"] = False
    elif choice == "petsc":
        inner["ksp_type"] = getattr(args, "xla_petsc_ksp_type", "gmres")
        inner["pc_type"] = getattr(args, "xla_petsc_pc_type", "jacobi")
        if getattr(args, "xla_petsc_gpu", False):
            # GPU-aware Mat/Vec types; requires a CUDA-enabled PETSc build.
            inner["mat_type"] = "aijcusparse"
            inner["vec_type"] = "cuda"
    elif choice == "amgx":
        cfg = getattr(args, "xla_amgx_config", None)
        if cfg:
            inner["cfg_path"] = str(cfg)
        # AMGX resources are expensive to create; keep them alive across
        # the whole scan (persistent handle managed by the solver adapter).
        inner["persistent_resources"] = True
    elif choice == "pardiso":
        # jax_fem's custom_solver hook expects the callable itself as the
        # option value, not a nested options dict.
        return {
            key: _PardisoCustomSolver(
                getattr(args, "xla_pardiso_mode", None)
            )
        }

    return {key: inner}


def _replace_linear_block(node: Dict[str, Any], replacement: Dict[str, Any]) -> bool:
    """Recursively rewrite linear-solver blocks in-place on a deep copy.

    Handles both layouts:

    * nested:  ``{"newton": {"linear": {"spsolve_solver": {}}, ...}}``
      -> the whole ``linear`` value is replaced, siblings (tol, rel_tol,
      max_iter, ...) are preserved.
    * legacy flat: ``{"tol": ..., "spsolve_solver": {}}``
      -> the old linear-solver key(s) are removed and the replacement
      key/value pair is merged at the same level.

    Returns True if at least one block was rewritten.
    """
    rewrote = False

    # Nested layout: a dict-valued "linear" entry that itself holds a
    # linear-solver key is swapped wholesale.
    linear = node.get("linear")
    if isinstance(linear, dict) and (
        not linear or LINEAR_SOLVER_KEYS.intersection(linear)
    ):
        node["linear"] = copy.deepcopy(replacement)
        rewrote = True

    # Legacy flat layout: linear-solver key(s) live next to the tolerances.
    flat_keys = LINEAR_SOLVER_KEYS.intersection(node)
    if flat_keys:
        for key in flat_keys:
            del node[key]
        for key, value in replacement.items():
            node[key] = copy.deepcopy(value)
        rewrote = True

    # Recurse into sub-dicts (e.g. "newton", "line_search", ...) that we
    # have not just replaced.
    for key, value in node.items():
        if key == "linear" and rewrote:
            continue
        if isinstance(value, dict) and key not in replacement:
            if _replace_linear_block(value, replacement):
                rewrote = True

    return rewrote


def rewrite_solver_options(
    options: Dict[str, Any], replacement: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a deep copy of ``options`` with its linear solver rewritten.

    ``replacement`` is a block like ``{"jax_solver": {"precond": True}}``
    (typically produced by :func:`linear_options_from_args`). When it is
    ``None`` the options are preserved verbatim -- but still deep-copied,
    so callers can mutate the result without corrupting the base config.

    The original ``options`` dict is never modified.
    """
    rewritten = copy.deepcopy(options)
    if replacement is None:
        return rewritten
    if not isinstance(replacement, dict) or len(replacement) == 0:
        raise ValueError(f"invalid linear-solver replacement: {replacement!r}")

    method_keys = [
        key for key in METHOD_KEYS if isinstance(rewritten.get(key), dict)
    ]
    if method_keys:
        for method in method_keys:
            method_options = dict(rewritten.get(method) or {})
            method_options["linear"] = copy.deepcopy(replacement)
            rewritten[method] = method_options
        return rewritten

    _replace_linear_block(rewritten, replacement)
    return rewritten


def inject_newton_option(
    options: Optional[Dict[str, Any]], key: str, value: Any
) -> Dict[str, Any]:
    """Return a copy of ``options`` with a Newton-level option defaulted.

    Handles both the nested ``{"newton": {...}}`` layout and the legacy flat
    layout that :func:`jax_fem.solver._resolve_solver_options` auto-wraps as
    Newton. Caller-provided values are never overridden. Explicit
    ``arc_length`` / ``dynamic_relax`` blocks are left untouched.
    """
    opts = copy.deepcopy(options) if isinstance(options, dict) else {}
    method_blocks = [m for m in METHOD_KEYS if isinstance(opts.get(m), dict)]
    if method_blocks:
        if isinstance(opts.get("newton"), dict):
            opts["newton"].setdefault(key, value)
        return opts
    opts.setdefault(key, value)
    return opts


# ---------------------------------------------------------------------------
# Phase 1: profiling report -- solver / conversion / python / io breakdown
# ---------------------------------------------------------------------------

STAGE_SETUP = "setup"              # mesh/problem setup before first solve
STAGE_ACTIVATION = "activation"    # layer/window activation masks
STAGE_QUAD_STATE = "quad_state"    # cell/quad state broadcast helpers
STAGE_MATERIAL = "material"        # temperature-dependent material updates
STAGE_HISTORY = "history"          # phase/reference/plastic history updates
STAGE_POSTPROCESS = "postprocess"  # cell fields and output-state reductions
STAGE_DOF_TO_QUAD = "dof_to_quad"  # nodal DOF fields projected to quadrature
STAGE_NONLINEAR_SOLVE = "nonlinear_solve"  # full jax-fem Newton solve call
STAGE_NONLINEAR_SOLVE_OVERHEAD = "nonlinear_solve_overhead"
STAGE_BC_INITIAL_GUESS = "bc_initial_guess"  # Newton linear-solve x0 BC setup
STAGE_RESIDUAL_VECTOR = "residual_vector"  # residual flatten/BC/projection
STAGE_RESIDUAL_FLATTEN = "residual_flatten"  # residual pytree/array flatten
STAGE_RESIDUAL_BC = "residual_bc"  # Dirichlet residual row replacement
STAGE_RESIDUAL_PROJECTION = "residual_projection"  # optional P_mat.T projection
STAGE_SOLVER = "solver"            # time inside the linear solve itself
STAGE_LOCAL_ASSEMBLY = "local_assembly"  # residual/Jacobian local assembly
STAGE_GLOBAL_MATRIX = "global_matrix"    # PETSc/global sparse matrix build
STAGE_ASSEMBLY = "assembly"        # derived local_assembly + global_matrix
STAGE_CELL_JACOBIAN = "cell_jacobian"  # volume residual/Jacobian kernel
STAGE_CELL_RESIDUAL = "cell_residual"  # volume residual-only kernel
STAGE_FACE_JACOBIAN = "face_jacobian"  # surface residual/Jacobian kernel
STAGE_FACE_RESIDUAL = "face_residual"  # surface residual-only kernel
STAGE_RESIDUAL_SCATTER = "residual_scatter"  # element/face residual scatter
STAGE_CONVERSION = "conversion"    # PETSc<->SciPy<->JAX BCOO format shuffling
STAGE_TRANSFER = "transfer"        # host<->device copies outside conversion
STAGE_IO = "io"                    # VTU / checkpoint / log writes
STAGE_PYTHON = "python_overhead"   # main-loop bookkeeping not covered above

ALL_STAGES = (
    STAGE_SETUP,
    STAGE_ACTIVATION,
    STAGE_QUAD_STATE,
    STAGE_MATERIAL,
    STAGE_HISTORY,
    STAGE_POSTPROCESS,
    STAGE_DOF_TO_QUAD,
    STAGE_NONLINEAR_SOLVE,
    STAGE_NONLINEAR_SOLVE_OVERHEAD,
    STAGE_BC_INITIAL_GUESS,
    STAGE_RESIDUAL_VECTOR,
    STAGE_RESIDUAL_FLATTEN,
    STAGE_RESIDUAL_BC,
    STAGE_RESIDUAL_PROJECTION,
    STAGE_SOLVER,
    STAGE_LOCAL_ASSEMBLY,
    STAGE_GLOBAL_MATRIX,
    STAGE_ASSEMBLY,
    STAGE_CELL_JACOBIAN,
    STAGE_CELL_RESIDUAL,
    STAGE_FACE_JACOBIAN,
    STAGE_FACE_RESIDUAL,
    STAGE_RESIDUAL_SCATTER,
    STAGE_CONVERSION,
    STAGE_TRANSFER,
    STAGE_IO,
    STAGE_PYTHON,
)

LOCAL_ASSEMBLY_DETAIL_STAGES = frozenset(
    {
        STAGE_CELL_JACOBIAN,
        STAGE_CELL_RESIDUAL,
        STAGE_FACE_JACOBIAN,
        STAGE_FACE_RESIDUAL,
        STAGE_RESIDUAL_SCATTER,
    }
)

RESIDUAL_VECTOR_DETAIL_STAGES = frozenset(
    {
        STAGE_RESIDUAL_FLATTEN,
        STAGE_RESIDUAL_BC,
        STAGE_RESIDUAL_PROJECTION,
    }
)


def _setup_detail_seconds(meta: Mapping[str, Any]) -> float:
    total = 0.0
    for key, value in meta.items():
        if not key.startswith("setup_detail_") or not key.endswith("_seconds"):
            continue
        if key == "setup_detail_total_seconds":
            continue
        total += float(value or 0.0)
    return total


class ProfilingReport:
    """Accumulates per-stage wall time and per-step counts for one run.

    Deliberately a plain class (not a dataclass): this module is loaded via
    ``importlib.util.spec_from_file_location`` in the unit tests without
    being registered in ``sys.modules``, and dataclass processing on
    Python >= 3.12 requires the defining module to be resolvable there.
    """

    def __init__(self, label: str = "run") -> None:
        self.label = label
        self.stage_seconds: Dict[str, float] = {s: 0.0 for s in ALL_STAGES}
        self.stage_calls: Dict[str, int] = {s: 0 for s in ALL_STAGES}
        self.steps: int = 0
        self.linear_iterations: int = 0
        self.wall_start: float = time.perf_counter()
        self.wall_seconds: float = 0.0
        self.meta: Dict[str, Any] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self.stage_seconds:
            self.stage_seconds[name] = 0.0
            self.stage_calls[name] = 0
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] += time.perf_counter() - t0
            self.stage_calls[name] += 1

    def add(self, name: str, seconds: float, calls: int = 1) -> None:
        self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + seconds
        self.stage_calls[name] = self.stage_calls.get(name, 0) + calls

    def _accounting_excluded_stages(self) -> set[str]:
        excluded = {STAGE_PYTHON, STAGE_NONLINEAR_SOLVE}
        has_assembly_substages = (
            self.stage_seconds.get(STAGE_LOCAL_ASSEMBLY, 0.0) > 0.0
            or self.stage_seconds.get(STAGE_GLOBAL_MATRIX, 0.0) > 0.0
        )
        if has_assembly_substages:
            excluded.add(STAGE_ASSEMBLY)
        has_local_assembly_details = (
            self.stage_seconds.get(STAGE_LOCAL_ASSEMBLY, 0.0) > 0.0
            and any(
                self.stage_seconds.get(stage, 0.0) > 0.0
                for stage in LOCAL_ASSEMBLY_DETAIL_STAGES
            )
        )
        if has_local_assembly_details:
            excluded.update(LOCAL_ASSEMBLY_DETAIL_STAGES)
        has_residual_vector_details = (
            self.stage_seconds.get(STAGE_RESIDUAL_VECTOR, 0.0) > 0.0
            and any(
                self.stage_seconds.get(stage, 0.0) > 0.0
                for stage in RESIDUAL_VECTOR_DETAIL_STAGES
            )
        )
        if has_residual_vector_details:
            excluded.update(RESIDUAL_VECTOR_DETAIL_STAGES)
        return excluded

    def finish(self) -> None:
        self.wall_seconds = time.perf_counter() - self.wall_start
        excluded = self._accounting_excluded_stages()
        accounted = sum(
            seconds
            for stage, seconds in self.stage_seconds.items()
            if stage not in excluded
        )
        # Everything not attributed to an explicit stage is Python overhead:
        # path-step bookkeeping, layer-activation checks, output-step
        # predicates, dict churn, etc. This is exactly the per-step fixed
        # cost that dominates at 826k scan steps.
        self.stage_seconds[STAGE_PYTHON] = max(
            0.0, self.wall_seconds - accounted
        )

    def record_setup_before_first_solve(self) -> None:
        """Close setup timing at the first thermal/mechanical solve call."""
        if self.meta.get("setup_recorded_before_first_solve"):
            return
        elapsed = time.perf_counter() - self.wall_start
        excluded = self._accounting_excluded_stages()
        excluded.add(STAGE_SETUP)
        accounted = sum(
            seconds
            for stage, seconds in self.stage_seconds.items()
            if stage not in excluded
        )
        setup_seconds = max(0.0, elapsed - accounted)
        self.stage_seconds[STAGE_SETUP] += setup_seconds
        self.stage_calls[STAGE_SETUP] += 1
        self.meta["setup_recorded_before_first_solve"] = True
        self.meta["setup_seconds_before_first_solve"] = setup_seconds
        detail_seconds = _setup_detail_seconds(self.meta)
        self.meta["setup_detail_total_seconds"] = detail_seconds
        self.meta["setup_unattributed_seconds"] = max(
            0.0,
            setup_seconds - detail_seconds,
        )
        self.meta["setup_timing_source"] = "first solver call boundary"

    def per_step(self, name: str) -> float:
        if self.steps == 0:
            return 0.0
        return self.stage_seconds.get(name, 0.0) / self.steps

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "wall_seconds": self.wall_seconds,
            "steps": self.steps,
            "linear_iterations": self.linear_iterations,
            "stage_seconds": dict(self.stage_seconds),
            "stage_calls": dict(self.stage_calls),
            "per_step_seconds": {
                s: self.per_step(s) for s in self.stage_seconds
            },
            "meta": dict(self.meta),
        }

    def dump(self, path: os.PathLike | str) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    def summary(self) -> str:
        lines = [
            f"[{self.label}] wall={self.wall_seconds:.3f}s "
            f"steps={self.steps} lin_iters={self.linear_iterations}"
        ]
        for s in sorted(self.stage_seconds, key=self.stage_seconds.get,
                        reverse=True):
            sec = self.stage_seconds[s]
            if sec <= 0.0:
                continue
            pct = 100.0 * sec / self.wall_seconds if self.wall_seconds else 0.0
            lines.append(
                f"  {s:<16} {sec:10.3f}s  {pct:5.1f}%  "
                f"calls={self.stage_calls.get(s, 0)}  "
                f"per_step={self.per_step(s) * 1e3:.3f} ms"
            )
        return "\n".join(lines)


def explain_gpu_vs_cpu(gpu: ProfilingReport, cpu: ProfilingReport) -> str:
    """Human-readable verdict for the acceptance rule: if jax_solver is
    slower than spsolve, say *why* rather than forcing GPU usage."""
    fallbacks = int(gpu.meta.get("solver_fallbacks", 0) or 0)
    if fallbacks:
        reason = gpu.meta.get("last_solver_fallback", "reason not recorded")
        return (
            f"GPU path fell back to spsolve {fallbacks} time(s); no GPU "
            f"speedup claim is valid for this run. Last fallback: {reason}"
        )
    if gpu.wall_seconds <= cpu.wall_seconds:
        gpu_solver = gpu.stage_seconds.get(STAGE_SOLVER, 0.0)
        cpu_solver = cpu.stage_seconds.get(STAGE_SOLVER, 0.0)
        if cpu_solver > 0.0 and gpu_solver > cpu_solver:
            return (
                "GPU path overall wall time is lower "
                f"({gpu.wall_seconds:.2f}s vs {cpu.wall_seconds:.2f}s), "
                "but the linear solve is slower "
                f"({gpu_solver:.3f}s vs {cpu_solver:.3f}s). Treat this as "
                "warm-cache / non-solver overhead evidence, not a GPU "
                "linear-solver speedup."
            )
        speedup = cpu.wall_seconds / max(gpu.wall_seconds, 1e-12)
        return (f"GPU path is {speedup:.2f}x faster overall "
                f"({gpu.wall_seconds:.2f}s vs {cpu.wall_seconds:.2f}s).")
    conv = gpu.stage_seconds.get(STAGE_CONVERSION, 0.0)
    solve = gpu.stage_seconds.get(STAGE_SOLVER, 0.0)
    xfer = gpu.stage_seconds.get(STAGE_TRANSFER, 0.0)
    dominant = max(
        ((STAGE_CONVERSION, conv), (STAGE_SOLVER, solve),
         (STAGE_TRANSFER, xfer)),
        key=lambda kv: kv[1],
    )[0]
    hints = {
        STAGE_CONVERSION: (
            "matrix conversion dominates: cache the sparsity pattern / "
            "BCOO index mapping and update values in place (Phase 4)"
        ),
        STAGE_SOLVER: _solver_stage_hint(gpu),
        STAGE_TRANSFER: (
            "host<->device copies dominate: keep state GPU-resident and "
            "only materialize output fields on VTU steps (Phase 2/3)"
        ),
    }
    return (
        f"GPU path is SLOWER ({gpu.wall_seconds:.2f}s vs "
        f"{cpu.wall_seconds:.2f}s); dominant GPU-side cost is "
        f"'{dominant}' -> {hints[dominant]}"
    )


def _solver_stage_hint(report: ProfilingReport) -> str:
    label = str(report.meta.get("linear_solver_label", ""))
    calls = int(report.meta.get("jax_spsolve_calls", 0) or 0)
    if "method=spsolve" in label or calls > 0:
        return (
            "the JAX sparse direct solve is slower than the direct CPU "
            "factorization at this problem size; keep spsolve as default "
            "for this tier and re-evaluate on the representative mesh"
        )
    return (
        "the iterative solve itself is slower than the direct CPU "
        "factorization at this problem size; keep spsolve as default "
        "for this tier and re-evaluate on the representative mesh"
    )


# ---------------------------------------------------------------------------
# v03 base-solver loading, CLI wiring, and runtime instrumentation
# ---------------------------------------------------------------------------


def _argv(argv: Sequence[str] | None) -> list[str]:
    return list(sys.argv[1:] if argv is None else argv)


def load_base_solver(
    module_name: str = BASE_MODULE_NAME,
    module_path: Path = BASE_SOLVER_PATH,
):
    """Load the v03-derived CPU reference solver lazily.

    v04 is intentionally a wrapper around the v03 production loop (now
    ``jax_fem_am.simulation.stepper``): it exposes ``read_config()``,
    ``build_parser()`` and ``main()``, but not the speculative
    ``run()/default_solver_options()`` interface that the first v04 skeleton
    assumed.

    ``module_name``/``module_path`` are kept for signature compatibility with
    the historical file-path loader. The stepper is a regular module now, so
    repeated calls return the same ``sys.modules`` entry.
    """
    try:
        import jax_fem_am.simulation.stepper as stepper_module
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"could not import base solver module {module_name!r} from "
            f"{module_path}; activate the jax-fem runtime before running "
            "the physics driver"
        ) from exc
    return stepper_module


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--xla-platform",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Set JAX_PLATFORM_NAME before importing jax-fem.",
    )
    parser.add_argument(
        "--xla-preallocate",
        choices=("preserve", "on", "off"),
        default="preserve",
        help="Control XLA_PYTHON_CLIENT_PREALLOCATE before importing JAX.",
    )
    parser.add_argument(
        "--xla-mem-fraction",
        type=float,
        default=None,
        help="Set XLA_PYTHON_CLIENT_MEM_FRACTION before importing JAX.",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _cell_chunking_options(
    num_cuts: Optional[int],
    target_batch_size: Optional[int],
) -> Dict[str, Any]:
    if num_cuts is not None:
        return {
            "cell_assembly_num_cuts": int(num_cuts),
            "cell_assembly_num_cuts_source": "cli",
            "cell_assembly_target_batch_size": None,
            "cell_assembly_chunking": "fixed_num_cuts",
        }
    if target_batch_size is not None:
        return {
            "cell_assembly_num_cuts": None,
            "cell_assembly_num_cuts_source": "auto",
            "cell_assembly_target_batch_size": int(target_batch_size),
            "cell_assembly_chunking": "auto_target_batch_size",
        }
    return {
        "cell_assembly_num_cuts": 20,
        "cell_assembly_num_cuts_source": "legacy",
        "cell_assembly_target_batch_size": None,
        "cell_assembly_chunking": "legacy_num_cuts",
    }


def preparse_runtime_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    _add_runtime_args(parser)
    args, _ = parser.parse_known_args(_argv(argv))
    return args


def apply_runtime_env(args: argparse.Namespace) -> None:
    if getattr(args, "xla_platform", "auto") != "auto":
        os.environ["JAX_PLATFORM_NAME"] = args.xla_platform

    preallocate = getattr(args, "xla_preallocate", "preserve")
    if preallocate == "on":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    elif preallocate == "off":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    mem_fraction = getattr(args, "xla_mem_fraction", None)
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)


def configure_jax_fem_logging(
    quiet: bool,
    profiler: Optional[ProfilingReport] = None,
) -> bool:
    logger = logging.getLogger("jax_fem")
    if not hasattr(logger, "_v04_original_level"):
        logger._v04_original_level = logger.level

    if quiet:
        logger.setLevel(logging.WARNING)
        if profiler is not None:
            profiler.meta["quiet_jax_fem_logs_enabled"] = True
            profiler.meta["jax_fem_log_level"] = "WARNING"
        return True

    logger.setLevel(logger._v04_original_level)
    if profiler is not None:
        profiler.meta["quiet_jax_fem_logs_enabled"] = False
        profiler.meta["jax_fem_log_level"] = "preserve"
    return False


def build_arg_parser(parser: Optional[argparse.ArgumentParser] = None
                     ) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        description="mech100 thermal-stress solver, XLA/GPU linear-solver "
                    "wrapper")
    g = parser.add_argument_group("XLA / GPU linear solver")
    _add_runtime_args(g)
    g.add_argument("--xla-linear-solver", default="keep",
                   choices=["keep", "preserve", *sorted(_SOLVER_CHOICE_TO_KEY)],
                   help="linear solver used inside the Newton loop")
    g.add_argument("--xla-jax-precond", action="store_true",
                   help="enable Jacobi preconditioning for the JAX solver")
    g.add_argument("--xla-jax-method", default=None,
                   choices=["bicgstab", "cg", "gmres", "spsolve"],
                   help=("JAX linear method (iterative bicgstab/cg/gmres or "
                         "experimental sparse direct spsolve)"))
    g.add_argument("--xla-jax-tol", type=float, default=None)
    g.add_argument("--xla-jax-atol", type=float, default=None)
    g.add_argument("--xla-jax-maxiter", type=int, default=None)
    g.add_argument("--xla-jax-gmres-restart", type=int, default=None)
    g.add_argument("--xla-jax-gmres-solve-method",
                   choices=["batched", "incremental"],
                   default=None)
    g.add_argument("--xla-jax-skip-residual-check", action="store_true",
                   help="skip the extra JAX A@x-b convergence check after solve")
    g.add_argument("--xla-petsc-ksp-type", default="gmres")
    g.add_argument("--xla-petsc-pc-type", default="jacobi")
    g.add_argument("--xla-petsc-gpu", action="store_true",
                   help="use PETSc aijcusparse/cuda Mat/Vec types")
    g.add_argument("--xla-pardiso-mode",
                   choices=["base", "nocmp", "cache-idx", "phase23", "fp32ir"],
                   default=None,
                   help=("PARDISO optimization ladder. phase23 reuses CSR indices and "
                         "symbolic factorization while recomputing numeric factors when "
                         "matrix values change. Default preserves the base path (or the "
                         "legacy V07_PARDISO_MODE environment override)."))
    g.add_argument("--xla-amgx-config", default=None,
                   help="path to an AMGX JSON config")
    g.add_argument("--xla-fallback-to-spsolve", action="store_true",
                   default=True,
                   help="retry a failed experimental solve with spsolve")
    g.add_argument("--no-xla-fallback-to-spsolve",
                   dest="xla_fallback_to_spsolve",
                   action="store_false")
    g.add_argument("--xla-show-devices", action="store_true",
                   help="print JAX devices before solving")
    g.add_argument("--xla-dry-run", action="store_true",
                   help="parse arguments, print settings, and exit")
    g.add_argument("--xla-thermal-warm-start", action="store_true",
                   default=False,
                   help=("seed each thermal Newton solve with the previous "
                         "temperature field"))
    g.add_argument("--no-xla-thermal-warm-start",
                   dest="xla_thermal_warm_start",
                   action="store_false",
                   help="disable v04 thermal Newton warm-start injection")
    g.add_argument("--xla-jit-loop-kernels",
                   dest="xla_jit_loop_kernels",
                   action="store_true",
                   default=True,
                   help=("JIT safe v04 loop-side material/history kernels "
                         "when the v03 run has no property tables"))
    g.add_argument("--no-xla-jit-loop-kernels",
                   dest="xla_jit_loop_kernels",
                   action="store_false",
                   help="disable v04 loop-side material/history JIT patches")
    g.add_argument("--xla-residual-only-check",
                   dest="xla_residual_only_check",
                   action="store_true",
                   default=True,
                   help=("thermal Newton convergence checks assemble the "
                         "residual only; the tangent Jacobian (and its "
                         "device->host transfer) is rebuilt only when "
                         "another Newton step actually runs. Mechanics "
                         "solves keep the v03 residual+Jacobian check"))
    g.add_argument("--no-xla-residual-only-check",
                   dest="xla_residual_only_check",
                   action="store_false",
                   help=("restore v03 behavior: every convergence check "
                         "assembles residual and Jacobian together"))
    g.add_argument("--xla-dof-to-quad-cache",
                   dest="xla_dof_to_quad_cache",
                   action="store_true",
                   default=True,
                   help=("cache repeated JAX-array DOF-to-quadrature "
                         "projections by FiniteElement and solution identity"))
    g.add_argument("--no-xla-dof-to-quad-cache",
                   dest="xla_dof_to_quad_cache",
                   action="store_false",
                   help="disable v04 DOF-to-quadrature identity cache")
    g.add_argument("--xla-step-predicate-cache",
                   dest="xla_step_predicate_cache",
                   action="store_true",
                   default=True,
                   help=("precompute per-step activation/mechanics/output "
                         "predicates after path generation"))
    g.add_argument("--no-xla-step-predicate-cache",
                   dest="xla_step_predicate_cache",
                   action="store_false",
                   help="disable v04 per-step predicate cache")
    g.add_argument("--xla-skip-unused-mechanics-material",
                   dest="xla_skip_unused_mechanics_material",
                   action="store_true",
                   default=True,
                   help=("skip mechanics material arrays when mechanics is "
                         "fully disabled and no release solve needs them"))
    g.add_argument("--no-xla-skip-unused-mechanics-material",
                   dest="xla_skip_unused_mechanics_material",
                   action="store_false",
                   help="disable v04 thermal-only mechanics material skip")
    g.add_argument("--xla-quiet-jax-fem-logs",
                   dest="xla_quiet_jax_fem_logs",
                   action="store_true",
                   default=True,
                   help=("raise the jax_fem package logger to WARNING during "
                         "v04 performance runs"))
    g.add_argument("--no-xla-quiet-jax-fem-logs",
                   dest="xla_quiet_jax_fem_logs",
                   action="store_false",
                   help="preserve the existing jax_fem logger verbosity")
    g.add_argument("--xla-lazy-output-postprocess",
                   dest="xla_lazy_output_postprocess",
                   action="store_true",
                   default=False,
                   help=("skip output-only material-state reductions on "
                         "non-save steps when step predicates are cached"))
    g.add_argument("--no-xla-lazy-output-postprocess",
                   dest="xla_lazy_output_postprocess",
                   action="store_false",
                   help="disable v04 lazy output postprocess patch")
    g.add_argument("--xla-thermal-only-mechanics-surrogate",
                   dest="xla_thermal_only_mechanics_surrogate",
                   action="store_true",
                   default=True,
                   help=("reuse the thermal FiniteElement instead of building "
                         "a full mechanics Problem when mechanics and release "
                         "are both disabled"))
    g.add_argument("--no-xla-thermal-only-mechanics-surrogate",
                   dest="xla_thermal_only_mechanics_surrogate",
                   action="store_false",
                   help="disable v04 thermal-only mechanics Problem surrogate")
    g.add_argument("--xla-cell-num-cuts", type=_positive_int, default=None,
                   help=("override Problem.split_and_compute_cell chunk count "
                         "(overrides target-batch auto chunking)"))
    g.add_argument("--xla-cell-target-batch-size",
                   type=_positive_int,
                   default=DEFAULT_CELL_TARGET_BATCH_SIZE,
                   help=("auto-select cell assembly cuts so each chunk has at "
                         "most this many cells; ignored when "
                         "--xla-cell-num-cuts is set"))

    b = parser.add_argument_group("profiling")
    b.add_argument("--profile-json", default=None,
                   help="write the ProfilingReport to this JSON file")
    b.add_argument("--profile-label", default="run")
    return parser


def parse_args(base_module, argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv_list = _argv(argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv_list)
    config = base_module.read_config(config_args.config)

    parser = base_module.build_parser(config)
    build_arg_parser(parser)
    args = parser.parse_args(argv_list)
    args.config = config_args.config
    return args


def _solver_label(linear_options: Mapping[str, Any] | None) -> str:
    if linear_options is None:
        return "preserve original solver_options"
    if "jax_solver" in linear_options:
        opts = linear_options["jax_solver"]
        parts = [
            f"method={opts.get('method', 'bicgstab')}",
            f"precond={opts.get('precond', False)}",
        ]
        if "tol" in opts:
            parts.append(f"tol={opts['tol']}")
        if "atol" in opts:
            parts.append(f"atol={opts['atol']}")
        if "maxiter" in opts:
            parts.append(f"maxiter={opts['maxiter']}")
        if opts.get("check_residual") is False:
            parts.append("check_residual=False")
        return f"jax_solver({', '.join(parts)})"
    if "amgx_solver" in linear_options:
        cfg_path = linear_options["amgx_solver"].get("cfg_path")
        return f"amgx_solver(cfg_path={cfg_path or 'built-in'})"
    if "petsc_solver" in linear_options:
        opts = linear_options["petsc_solver"]
        return (
            f"petsc_solver(ksp_type={opts.get('ksp_type')}, "
            f"pc_type={opts.get('pc_type')})"
        )
    if "spsolve_solver" in linear_options:
        return "spsolve_solver(cpu scipy baseline)"
    if "custom_solver" in linear_options:
        custom = linear_options["custom_solver"]
        return getattr(custom, "label", f"custom_solver({custom!r})")
    return str(linear_options)


def print_acceleration_summary(
    args: argparse.Namespace,
    linear_options: Mapping[str, Any] | None,
) -> None:
    print("============================================================")
    print("mech100 XLA/GPU upgrade wrapper (v04)")
    print(f"original_solver_module = {BASE_SOLVER_PATH}")
    print(f"linear_solver_override = {_solver_label(linear_options)}")
    print(f"xla_platform           = {args.xla_platform}")
    print(f"xla_preallocate       = {args.xla_preallocate}")
    print(f"xla_mem_fraction      = {args.xla_mem_fraction}")
    print(f"fallback_to_spsolve   = {args.xla_fallback_to_spsolve}")
    print(f"thermal_warm_start    = {args.xla_thermal_warm_start}")
    print(f"loop_kernel_jit       = {args.xla_jit_loop_kernels}")
    print(
        "skip_unused_mech_mat = "
        f"{args.xla_skip_unused_mechanics_material}"
    )
    print(f"dof_to_quad_cache    = {args.xla_dof_to_quad_cache}")
    print(f"quiet_jax_fem_logs = {args.xla_quiet_jax_fem_logs}")
    print(f"lazy_output_postprocess = {args.xla_lazy_output_postprocess}")
    print(
        "thermal_only_mech_surrogate = "
        f"{args.xla_thermal_only_mechanics_surrogate}"
    )
    chunking = _cell_chunking_options(
        args.xla_cell_num_cuts,
        args.xla_cell_target_batch_size,
    )
    if chunking["cell_assembly_chunking"] == "fixed_num_cuts":
        print(f"cell_assembly_cuts    = {chunking['cell_assembly_num_cuts']}")
    elif chunking["cell_assembly_chunking"] == "auto_target_batch_size":
        print(
            "cell_assembly_chunking = auto "
            f"target_batch_size={chunking['cell_assembly_target_batch_size']}"
        )
    else:
        print("cell_assembly_cuts    = 20")
    print("full_loop_xla         = disabled; v03 Python/jax-fem loop is preserved")
    print("============================================================")


def show_jax_devices() -> None:
    import jax

    print("JAX devices:", jax.devices())
    print("JAX default backend:", jax.default_backend())
    print("JAX enable x64:", jax.config.read("jax_enable_x64"))


@contextmanager
def _profile_stage(
    profiler: Optional[ProfilingReport],
    stage: str,
) -> Iterator[None]:
    if profiler is None:
        yield
    else:
        with profiler.stage(stage):
            yield


def install_solver_patch(
    base_module,
    linear_options: Mapping[str, Any] | None,
    fallback_to_spsolve: bool,
    profiler: Optional[ProfilingReport] = None,
    profile_solver_call: bool = True,
    thermal_warm_start: bool = False,
    residual_only_check: bool = False,
) -> None:
    original_solver = base_module.solver
    fallback_options = {"spsolve_solver": {}}
    solve_internal_stages = (
        STAGE_LOCAL_ASSEMBLY,
        STAGE_GLOBAL_MATRIX,
        STAGE_BC_INITIAL_GUESS,
        STAGE_RESIDUAL_VECTOR,
        STAGE_SOLVER,
        STAGE_CONVERSION,
        STAGE_TRANSFER,
    )

    def run_original_solver(problem, patched_options):
        if profiler is None:
            if profile_solver_call:
                with _profile_stage(profiler, STAGE_SOLVER):
                    return original_solver(problem, solver_options=patched_options)
            return original_solver(problem, solver_options=patched_options)

        before = {
            stage: float(profiler.stage_seconds.get(stage, 0.0))
            for stage in solve_internal_stages
        }
        t0 = time.perf_counter()
        try:
            if profile_solver_call:
                with _profile_stage(profiler, STAGE_SOLVER):
                    return original_solver(problem, solver_options=patched_options)
            return original_solver(problem, solver_options=patched_options)
        finally:
            elapsed = time.perf_counter() - t0
            internal_seconds = sum(
                float(profiler.stage_seconds.get(stage, 0.0)) - seconds
                for stage, seconds in before.items()
            )
            profiler.add(STAGE_NONLINEAR_SOLVE, elapsed)
            profiler.add(
                STAGE_NONLINEAR_SOLVE_OVERHEAD,
                max(0.0, elapsed - internal_seconds),
            )

    def _wants_residual_only_check(problem):
        # Scope to the thermal problem: its convergence decisions are far from
        # the tolerance boundary, so skipping the tangent on the converged
        # check leaves the temperature field bit-identical. Mechanics solves
        # sit near tol and may stop one Newton iteration earlier/later, so
        # they keep the v03 residual+Jacobian check semantics.
        if not residual_only_check:
            return False
        # Other v04 patches may replace base_module.TransientThermal with a
        # wrapper factory, so isinstance against the module attribute is not
        # reliable; fall back to matching the problem's class name.
        thermal_cls = getattr(base_module, "TransientThermal", None)
        if isinstance(thermal_cls, type) and isinstance(problem, thermal_cls):
            return True
        return any(
            klass.__name__ == "TransientThermal"
            for klass in type(problem).__mro__
        )

    def accelerated_solver(problem, solver_options=None):
        if profiler is not None:
            profiler.record_setup_before_first_solve()
        patched_options = (
            rewrite_solver_options(solver_options or {}, dict(linear_options))
            if linear_options is not None
            else solver_options
        )
        if _wants_residual_only_check(problem):
            patched_options = inject_newton_option(
                patched_options, "residual_only_check", True
            )
            if profiler is not None:
                profiler.meta["residual_only_check_injections"] = (
                    int(profiler.meta.get("residual_only_check_injections", 0))
                    + 1
                )
        if thermal_warm_start:
            patched_options = maybe_inject_thermal_initial_guess(
                problem,
                patched_options,
                profiler,
            )
        try:
            return run_original_solver(problem, patched_options)
        except Exception as exc:
            # A Newton stall is a property of the nonlinear problem, not of the
            # linear backend: SciPy spsolve reproduces pardiso stall residuals
            # bit-identically (both are direct solvers), so retrying burns a
            # full Newton budget for nothing. Re-raise so callers (e.g. the
            # mechanics increment cutback) can subdivide the load instead.
            newton_stall = (
                isinstance(exc, RuntimeError)
                and "Newton solver did not converge" in str(exc)
            )
            if (
                linear_options is None
                or not fallback_to_spsolve
                or "spsolve_solver" in linear_options
                or newton_stall
            ):
                raise
            if profiler is not None:
                profiler.meta["solver_fallbacks"] = (
                    int(profiler.meta.get("solver_fallbacks", 0)) + 1
                )
                profiler.meta["last_solver_fallback"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            print(
                "WARNING: experimental linear solver failed; retrying this solve "
                f"with SciPy spsolve. Error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            retry_options = rewrite_solver_options(
                solver_options or {}, fallback_options
            )
            if _wants_residual_only_check(problem):
                retry_options = inject_newton_option(
                    retry_options, "residual_only_check", True
                )
            if thermal_warm_start:
                retry_options = maybe_inject_thermal_initial_guess(
                    problem,
                    retry_options,
                    profiler,
                )
            return run_original_solver(problem, retry_options)

    base_module.solver = accelerated_solver


def _shape_dtype_label(value: Any) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return f"shape={shape}, dtype={dtype}"


def _inject_initial_guess(
    solver_options: Optional[Mapping[str, Any]],
    initial_guess: Any,
) -> tuple[Mapping[str, Any], bool]:
    """Return solver options with a Newton initial guess added if absent."""
    if solver_options is None:
        return {"newton": {"initial_guess": initial_guess}}, True

    options = dict(solver_options)
    method_keys = [
        key for key in METHOD_KEYS if isinstance(options.get(key), dict)
    ]
    if "newton" in method_keys:
        newton_options = dict(options["newton"])
        if "initial_guess" in newton_options:
            return solver_options, False
        newton_options["initial_guess"] = initial_guess
        options["newton"] = newton_options
        return options, True

    if method_keys:
        return solver_options, False

    if "initial_guess" in options:
        return solver_options, False
    options["initial_guess"] = initial_guess
    return options, True


def maybe_inject_thermal_initial_guess(
    problem: Any,
    solver_options: Optional[Mapping[str, Any]],
    profiler: Optional[ProfilingReport] = None,
) -> Optional[Mapping[str, Any]]:
    initial_guess = getattr(problem, "_v04_thermal_initial_guess", None)
    if initial_guess is None:
        return solver_options

    patched_options, injected = _inject_initial_guess(
        solver_options,
        initial_guess,
    )
    if injected and profiler is not None:
        profiler.meta["thermal_warm_start_injections"] = (
            int(profiler.meta.get("thermal_warm_start_injections", 0) or 0) + 1
        )
        profiler.meta["thermal_warm_start_last_guess"] = (
            _shape_dtype_label(initial_guess)
        )
    return patched_options


def install_thermal_warm_start_patch(
    base_module,
    profiler: Optional[ProfilingReport] = None,
) -> bool:
    thermal_cls = getattr(base_module, "TransientThermal", None)
    if thermal_cls is None or not hasattr(thermal_cls, "set_params"):
        if profiler is not None:
            profiler.meta["thermal_warm_start_patch"] = "missing TransientThermal"
        return False

    original_set_params = getattr(
        thermal_cls,
        "_v04_original_set_params",
        thermal_cls.set_params,
    )
    thermal_cls._v04_original_set_params = original_set_params

    def set_params_with_initial_guess(self, params):
        if params:
            self._v04_thermal_initial_guess = params[0]
        return original_set_params(self, params)

    thermal_cls.set_params = set_params_with_initial_guess
    if profiler is not None:
        profiler.meta["thermal_warm_start_patch"] = (
            "TransientThermal.set_params[0]"
        )
    return True


def install_jax_fem_timing_patch(profiler: Optional[ProfilingReport]) -> None:
    if profiler is None:
        return
    solver_module = sys.modules.get("jax_fem.solver")
    if solver_module is None or not hasattr(solver_module, "_timing_record"):
        return

    original_timing_record = getattr(
        solver_module,
        "_v04_original_timing_record",
        solver_module._timing_record,
    )
    solver_module._v04_original_timing_record = original_timing_record
    stage_by_timing_name = {
        "linear": (STAGE_SOLVER,),
        "linear_kernel": (STAGE_SOLVER,),
        "linear_residual_check": (STAGE_SOLVER,),
        "sparse_conversion": (STAGE_CONVERSION,),
        "bc_initial_guess": (STAGE_BC_INITIAL_GUESS,),
        "residual_vector": (STAGE_RESIDUAL_VECTOR,),
        "residual_flatten": (STAGE_RESIDUAL_FLATTEN,),
        "residual_bc": (STAGE_RESIDUAL_BC,),
        "residual_projection": (STAGE_RESIDUAL_PROJECTION,),
        "local_assembly": (STAGE_LOCAL_ASSEMBLY, STAGE_ASSEMBLY),
        "global_matrix": (STAGE_GLOBAL_MATRIX, STAGE_ASSEMBLY),
    }

    def timing_record(timing, name, dt):
        original_timing_record(timing, name, dt)
        if (
            name == "linear"
            and isinstance(timing, dict)
            and timing.get("_last_linear_internal_breakdown", False)
        ):
            return
        stages = stage_by_timing_name.get(name)
        if stages is not None:
            for stage in stages:
                profiler.add(stage, float(dt))

    solver_module._timing_record = timing_record
    profiler.meta["solver_timing_source"] = "jax_fem.solver._timing_record"

    original_log_timing_table = getattr(
        solver_module,
        "_v04_original_log_timing_table",
        getattr(solver_module, "_log_timing_table", None),
    )
    if original_log_timing_table is not None:
        solver_module._v04_original_log_timing_table = original_log_timing_table

        def log_timing_table(n_iters, parts, wall_s):
            profiler.linear_iterations += int(n_iters)
            profiler.meta["newton_solve_calls"] = (
                int(profiler.meta.get("newton_solve_calls", 0) or 0) + 1
            )
            if int(n_iters) == 0:
                profiler.meta["newton_zero_iter_solves"] = (
                    int(profiler.meta.get("newton_zero_iter_solves", 0) or 0) + 1
                )
            profiler.meta["last_newton_iterations"] = int(n_iters)
            profiler.meta["newton_wall_seconds"] = (
                float(profiler.meta.get("newton_wall_seconds", 0.0) or 0.0)
                + float(wall_s)
            )
            return original_log_timing_table(n_iters, parts, wall_s)

        solver_module._log_timing_table = log_timing_table
        profiler.meta["newton_timing_source"] = "jax_fem.solver._log_timing_table"

    original_counter_record = getattr(
        solver_module,
        "_v04_original_counter_record",
        getattr(solver_module, "_counter_record", None),
    )
    if original_counter_record is None:
        return

    solver_module._v04_original_counter_record = original_counter_record
    meta_by_counter_name = {
        "bcoo_cache_hits": "jax_bcoo_cache_hits",
        "bcoo_cache_misses": "jax_bcoo_cache_misses",
        "jax_spsolve_calls": "jax_spsolve_calls",
    }

    def counter_record(timing, name, count=1):
        original_counter_record(timing, name, count)
        meta_name = meta_by_counter_name.get(name)
        if meta_name is not None:
            profiler.meta[meta_name] = (
                int(profiler.meta.get(meta_name, 0) or 0) + int(count)
            )

    solver_module._counter_record = counter_record


def install_problem_local_assembly_timing_patch(
    profiler: Optional[ProfilingReport],
) -> bool:
    if profiler is None:
        return False
    problem_module = sys.modules.get("jax_fem.problem")
    problem_cls = getattr(problem_module, "Problem", None) if problem_module else None
    if problem_cls is None:
        profiler.meta["problem_timing_patch"] = "missing jax_fem.problem.Problem"
        return False

    original_split = getattr(
        problem_cls,
        "_v04_original_split_and_compute_cell",
        problem_cls.split_and_compute_cell,
    )
    original_face = getattr(
        problem_cls,
        "_v04_original_compute_face",
        problem_cls.compute_face,
    )
    original_scatter = getattr(
        problem_cls,
        "_v04_original_compute_residual_vars_helper",
        problem_cls.compute_residual_vars_helper,
    )
    problem_cls._v04_original_split_and_compute_cell = original_split
    problem_cls._v04_original_compute_face = original_face
    problem_cls._v04_original_compute_residual_vars_helper = original_scatter

    def split_and_compute_cell_timed(
        self,
        cells_sol_flat,
        np_version,
        jac_flag,
        internal_vars,
    ):
        stage = STAGE_CELL_JACOBIAN if jac_flag else STAGE_CELL_RESIDUAL
        with profiler.stage(stage):
            return original_split(
                self,
                cells_sol_flat,
                np_version,
                jac_flag,
                internal_vars,
            )

    def compute_face_timed(
        self,
        cells_sol_flat,
        np_version,
        jac_flag,
        internal_vars_surfaces,
    ):
        stage = STAGE_FACE_JACOBIAN if jac_flag else STAGE_FACE_RESIDUAL
        with profiler.stage(stage):
            return original_face(
                self,
                cells_sol_flat,
                np_version,
                jac_flag,
                internal_vars_surfaces,
            )

    def compute_residual_vars_helper_timed(
        self,
        weak_form_flat,
        weak_form_face_flat,
    ):
        with profiler.stage(STAGE_RESIDUAL_SCATTER):
            return original_scatter(self, weak_form_flat, weak_form_face_flat)

    problem_cls.split_and_compute_cell = split_and_compute_cell_timed
    problem_cls.compute_face = compute_face_timed
    problem_cls.compute_residual_vars_helper = compute_residual_vars_helper_timed
    profiler.meta["problem_timing_patch"] = (
        "jax_fem.problem.Problem local assembly methods"
    )
    return True


def _is_jax_array_like(value: Any) -> bool:
    value_type = type(value)
    module = getattr(value_type, "__module__", "")
    name = getattr(value_type, "__name__", "")
    if "Tracer" in name:
        return False
    if module.startswith("jaxlib"):
        return hasattr(value, "shape") and hasattr(value, "dtype")
    return False


def _dof_to_quad_cache_key(fe: Any, sol: Any) -> tuple[Any, ...]:
    cells = getattr(fe, "cells", None)
    shape_vals = getattr(fe, "shape_vals", None)
    return (
        id(fe),
        id(sol),
        getattr(sol, "shape", None),
        str(getattr(sol, "dtype", None)),
        id(cells),
        getattr(cells, "shape", None),
        id(shape_vals),
        getattr(shape_vals, "shape", None),
    )


def _increment_meta_counter(
    profiler: ProfilingReport,
    name: str,
    count: int = 1,
) -> None:
    profiler.meta[name] = int(profiler.meta.get(name, 0) or 0) + int(count)


def install_finite_element_timing_patch(
    profiler: Optional[ProfilingReport],
    *,
    cache_enabled: bool = True,
    cache_max_entries: int = DOF_TO_QUAD_CACHE_MAX_ENTRIES,
) -> bool:
    if profiler is None:
        return False
    fe_module = sys.modules.get("jax_fem.fe")
    if fe_module is None:
        try:
            fe_module = __import__("jax_fem.fe", fromlist=["FiniteElement"])
        except Exception as exc:  # pragma: no cover - environment dependent
            profiler.meta["finite_element_timing_patch"] = (
                f"missing jax_fem.fe.FiniteElement: {exc}"
            )
            return False

    fe_cls = getattr(fe_module, "FiniteElement", None)
    original = getattr(
        fe_cls,
        "_v04_original_convert_from_dof_to_quad",
        getattr(fe_cls, "convert_from_dof_to_quad", None),
    )
    if fe_cls is None or original is None:
        profiler.meta["finite_element_timing_patch"] = (
            "missing jax_fem.fe.FiniteElement.convert_from_dof_to_quad"
        )
        return False

    fe_cls._v04_original_convert_from_dof_to_quad = original
    cache: OrderedDict[
        tuple[Any, ...],
        tuple[Any, Any, Any],
    ] = OrderedDict()
    cache_enabled = bool(cache_enabled and cache_max_entries > 0)

    profiler.meta["dof_to_quad_cache_enabled"] = cache_enabled
    profiler.meta["dof_to_quad_cache_scope"] = (
        "same_finite_element_jax_array_identity"
    )
    profiler.meta["dof_to_quad_cache_max_entries"] = (
        int(cache_max_entries) if cache_enabled else 0
    )

    def convert_from_dof_to_quad_timed(self, sol):
        if cache_enabled:
            if _is_jax_array_like(sol):
                key = _dof_to_quad_cache_key(self, sol)
                cached = cache.get(key)
                if cached is not None:
                    cached_fe, cached_sol, cached_value = cached
                    if cached_fe is self and cached_sol is sol:
                        cache.move_to_end(key)
                        _increment_meta_counter(
                            profiler,
                            "dof_to_quad_cache_hits",
                        )
                        profiler.meta["dof_to_quad_cache_entries"] = len(cache)
                        return cached_value
                    cache.pop(key, None)
                _increment_meta_counter(
                    profiler,
                    "dof_to_quad_cache_misses",
                )
                with profiler.stage(STAGE_DOF_TO_QUAD):
                    value = original(self, sol)
                cache[key] = (self, sol, value)
                cache.move_to_end(key)
                while len(cache) > cache_max_entries:
                    cache.popitem(last=False)
                profiler.meta["dof_to_quad_cache_entries"] = len(cache)
                return value
            _increment_meta_counter(
                profiler,
                "dof_to_quad_cache_skipped_non_jax",
            )
        with profiler.stage(STAGE_DOF_TO_QUAD):
            return original(self, sol)

    fe_cls.convert_from_dof_to_quad = convert_from_dof_to_quad_timed
    profiler.meta["finite_element_timing_patch"] = (
        "jax_fem.fe.FiniteElement convert_from_dof_to_quad"
    )
    return True


def configure_problem_cell_assembly_num_cuts(
    num_cuts: Optional[int],
    profiler: Optional[ProfilingReport] = None,
    target_batch_size: Optional[int] = DEFAULT_CELL_TARGET_BATCH_SIZE,
) -> bool:
    chunking = _cell_chunking_options(num_cuts, target_batch_size)
    problem_module = sys.modules.get("jax_fem.problem")
    if problem_module is None:
        try:
            import jax_fem.problem as problem_module
        except Exception as exc:
            if profiler is not None:
                profiler.meta["cell_assembly_num_cuts_patch"] = (
                    f"failed to import jax_fem.problem: {exc}"
                )
            return False

    problem_cls = getattr(problem_module, "Problem", None)
    if problem_cls is None:
        if profiler is not None:
            profiler.meta["cell_assembly_num_cuts_patch"] = (
                "missing jax_fem.problem.Problem"
            )
        return False

    problem_cls.cell_assembly_num_cuts = (
        int(num_cuts) if num_cuts is not None else 20
    )
    problem_cls.cell_assembly_target_batch_size = chunking[
        "cell_assembly_target_batch_size"
    ]
    if profiler is not None:
        profiler.meta.update(chunking)
    return True


def _array_cache_identity(array: Any) -> tuple[Any, Any, str]:
    return (
        id(array),
        getattr(array, "shape", None),
        str(getattr(array, "dtype", None)),
    )


def _activation_args_key(
    args: argparse.Namespace,
    *,
    include_layer_thickness: bool,
) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        int(getattr(args, "active_window_below_layers", 0) or 0),
        int(getattr(args, "layers", 0) or 0),
    )
    if include_layer_thickness:
        lt = getattr(args, "layer_thickness", None)
        key += (None if lt is None else float(lt),)
    return key


def _record_activation_cache_event(
    profiler: Optional[ProfilingReport],
    *,
    hit: bool,
    entries: int,
) -> None:
    if profiler is None:
        return
    key = "activation_cache_hits" if hit else "activation_cache_misses"
    profiler.meta[key] = int(profiler.meta.get(key, 0) or 0) + 1
    profiler.meta["activation_cache_entries"] = entries


def _cached_activation_result(
    cache: Dict[tuple[Any, ...], Any],
    profiler: Optional[ProfilingReport],
    key: tuple[Any, ...],
    compute_result,
):
    if key in cache:
        _record_activation_cache_event(profiler, hit=True, entries=len(cache))
        return cache[key]

    result = compute_result()
    cache[key] = result
    _record_activation_cache_event(profiler, hit=False, entries=len(cache))
    return result


def install_activation_cache_patch(
    base_module,
    profiler: Optional[ProfilingReport] = None,
) -> None:
    """Cache pure layer/window activation masks from the v03 Python loop.

    The v03 driver recomputes whole-cell boolean masks on every scan step.
    For layer_on_scan and moving-window modes those masks only depend on the
    current layer, stable mesh/classification arrays, and a few CLI knobs.
    Keeping this as a wrapper patch preserves v03 output compatibility while
    removing repeated per-step full-array comparisons in v04.
    """
    cache: Dict[tuple[Any, ...], Any] = {}
    if profiler is not None:
        profiler.meta["activation_cache_enabled"] = True

    original = getattr(base_module, "compute_layer_on_scan_cells", None)
    if original is not None:

        def cached_layer_on_scan(
            highest_printed_layer,
            physical_layer_id_cell,
            substrate_cell,
            support_cell,
            args,
            _original=original,
        ):
            key = (
                "layer_on_scan",
                int(highest_printed_layer),
                _activation_args_key(args, include_layer_thickness=False),
                _array_cache_identity(physical_layer_id_cell),
                _array_cache_identity(substrate_cell),
                _array_cache_identity(support_cell),
            )
            return _cached_activation_result(
                cache,
                profiler,
                key,
                lambda: _original(
                    highest_printed_layer,
                    physical_layer_id_cell,
                    substrate_cell,
                    support_cell,
                    args,
                ),
            )

        base_module.compute_layer_on_scan_cells = cached_layer_on_scan

    original = getattr(base_module, "compute_layer_on_scan_cells_by_intersection", None)
    if original is not None:

        def cached_layer_on_scan_intersection(
            highest_printed_layer,
            cell_d_min,
            cell_d_max,
            substrate_cell,
            support_cell,
            args,
            _original=original,
        ):
            key = (
                "layer_on_scan_intersection",
                int(highest_printed_layer),
                _activation_args_key(args, include_layer_thickness=True),
                _array_cache_identity(cell_d_min),
                _array_cache_identity(cell_d_max),
                _array_cache_identity(substrate_cell),
                _array_cache_identity(support_cell),
            )
            return _cached_activation_result(
                cache,
                profiler,
                key,
                lambda: _original(
                    highest_printed_layer,
                    cell_d_min,
                    cell_d_max,
                    substrate_cell,
                    support_cell,
                    args,
                ),
            )

        base_module.compute_layer_on_scan_cells_by_intersection = (
            cached_layer_on_scan_intersection
        )

    original = getattr(base_module, "compute_moving_window_cells", None)
    if original is not None:

        def cached_moving_window(
            state,
            physical_layer_id_cell,
            substrate_cell,
            support_cell,
            args,
            _original=original,
        ):
            current_layer = int(state.layer_idx) + 1
            key = (
                "moving_window",
                current_layer,
                _activation_args_key(args, include_layer_thickness=False),
                _array_cache_identity(physical_layer_id_cell),
                _array_cache_identity(substrate_cell),
                _array_cache_identity(support_cell),
            )
            return _cached_activation_result(
                cache,
                profiler,
                key,
                lambda: _original(
                    state,
                    physical_layer_id_cell,
                    substrate_cell,
                    support_cell,
                    args,
                ),
            )

        base_module.compute_moving_window_cells = cached_moving_window

    original = getattr(base_module, "compute_moving_window_cells_by_intersection", None)
    if original is not None:

        def cached_moving_window_intersection(
            state,
            cell_d_min,
            cell_d_max,
            substrate_cell,
            support_cell,
            args,
            _original=original,
        ):
            current_layer = int(state.layer_idx) + 1
            key = (
                "moving_window_intersection",
                current_layer,
                _activation_args_key(args, include_layer_thickness=True),
                _array_cache_identity(cell_d_min),
                _array_cache_identity(cell_d_max),
                _array_cache_identity(substrate_cell),
                _array_cache_identity(support_cell),
            )
            return _cached_activation_result(
                cache,
                profiler,
                key,
                lambda: _original(
                    state,
                    cell_d_min,
                    cell_d_max,
                    substrate_cell,
                    support_cell,
                    args,
                ),
            )

        base_module.compute_moving_window_cells_by_intersection = (
            cached_moving_window_intersection
        )


def _profiler_meta_increment(
    profiler: Optional[ProfilingReport],
    key: str,
    count: int = 1,
) -> None:
    if profiler is None:
        return
    profiler.meta[key] = int(profiler.meta.get(key, 0) or 0) + int(count)


def _record_setup_detail_timing(
    profiler: Optional[ProfilingReport],
    detail: str,
    seconds: float,
) -> None:
    if profiler is None:
        return
    if profiler.meta.get("setup_recorded_before_first_solve"):
        return
    seconds_key = f"setup_detail_{detail}_seconds"
    calls_key = f"setup_detail_{detail}_calls"
    profiler.meta[seconds_key] = (
        float(profiler.meta.get(seconds_key, 0.0) or 0.0) + float(seconds)
    )
    profiler.meta[calls_key] = int(profiler.meta.get(calls_key, 0) or 0) + 1


def _install_setup_detail_wrapper(
    base_module,
    name: str,
    detail: str,
    profiler: Optional[ProfilingReport],
) -> bool:
    original_attr = f"_v04_setup_detail_original_{name}"
    original = getattr(
        base_module,
        original_attr,
        getattr(base_module, name, None),
    )
    if original is None:
        return False
    setattr(base_module, original_attr, original)

    def wrapped_setup_detail(*args, _original=original, _detail=detail, **kwargs):
        if profiler is None or profiler.meta.get("setup_recorded_before_first_solve"):
            return _original(*args, **kwargs)
        t0 = time.perf_counter()
        try:
            return _original(*args, **kwargs)
        finally:
            _record_setup_detail_timing(
                profiler,
                _detail,
                time.perf_counter() - t0,
            )

    setattr(base_module, name, wrapped_setup_detail)
    return True


def install_setup_detail_timing_patch(
    base_module,
    profiler: Optional[ProfilingReport] = None,
) -> bool:
    """Record high-level v03 setup substeps without changing setup accounting.

    `STAGE_SETUP` remains the full time before the first solver call. These
    detail values are stored in `meta` only, so existing stage totals and
    python-overhead accounting stay stable while the roadmap gains a clearer
    target for the next XLA migration slice.
    """
    targets = (
        ("read_tet4_inp", "mesh_read"),
        ("generate_raster_step_states", "path_generation"),
        ("generate_path_file_step_states", "path_generation"),
        ("Mesh", "mesh_construction"),
        ("TransientThermal", "thermal_problem"),
        ("ThermoMechanical", "mechanics_problem"),
    )
    installed = [
        f"{name}:{detail}"
        for name, detail in targets
        if _install_setup_detail_wrapper(base_module, name, detail, profiler)
    ]
    if profiler is not None:
        profiler.meta["setup_detail_timing_patch"] = ", ".join(installed) or "none"
    return bool(installed)


def _step_predicate_cache_hit(
    base_module,
    args,
    global_step,
) -> Optional[Mapping[str, Any]]:
    cache = getattr(base_module, "_v04_step_predicate_cache", None)
    if not cache or cache.get("args_id") != id(args):
        return None
    return cache.get("entries", {}).get(int(global_step))


def _record_step_predicate_event(
    profiler: Optional[ProfilingReport],
    *,
    hit: bool,
) -> None:
    _profiler_meta_increment(
        profiler,
        "step_predicate_cache_hits" if hit else "step_predicate_cache_misses",
    )


def install_step_predicate_cache_patch(
    base_module,
    profiler: Optional[ProfilingReport] = None,
    enabled: bool = True,
) -> bool:
    """Precompute cheap per-step predicates after v03 path generation.

    The v03 loop calls several scalar predicates on every step. This patch is a
    conservative Phase 2 bridge: it leaves the loop intact, but turns repeated
    modulo/string predicate work into table lookups for the generated path.
    """
    if profiler is not None:
        profiler.meta["step_predicate_cache_enabled"] = bool(enabled)
    if not enabled:
        return False

    generators = [
        name
        for name in ("generate_raster_step_states", "generate_path_file_step_states")
        if getattr(base_module, name, None) is not None
    ]
    original_activate = getattr(
        base_module,
        "_v04_original_should_activate_layer_for_state",
        getattr(base_module, "should_activate_layer_for_state", None),
    )
    original_mechanics = getattr(
        base_module,
        "_v04_original_should_run_mechanics",
        getattr(base_module, "should_run_mechanics", None),
    )
    original_save = getattr(
        base_module,
        "_v04_original_should_save_step",
        getattr(base_module, "should_save_step", None),
    )
    if not generators or original_activate is None or original_mechanics is None or original_save is None:
        if profiler is not None:
            profiler.meta["step_predicate_cache_patch"] = (
                "missing step generator or predicate function"
            )
        return False

    base_module._v04_original_should_activate_layer_for_state = original_activate
    base_module._v04_original_should_run_mechanics = original_mechanics
    base_module._v04_original_should_save_step = original_save

    def build_cache(args, states):
        entries: Dict[int, Dict[str, Any]] = {}
        for state in states:
            global_step = int(state.global_step)
            activate = bool(original_activate(state))
            run_mechanics = bool(original_mechanics(global_step, args))
            save_if_mechanics = bool(original_save(global_step, True, False, args))
            save_if_no_mechanics = bool(original_save(global_step, False, False, args))
            try:
                state._v04_should_activate_layer = activate
                state._v04_should_run_mechanics = run_mechanics
            except Exception:
                pass
            entries[global_step] = {
                "activate_layer": activate,
                "run_mechanics": run_mechanics,
                "save_if_mechanics": save_if_mechanics,
                "save_if_no_mechanics": save_if_no_mechanics,
            }
        base_module._v04_step_predicate_cache = {
            "args_id": id(args),
            "entries": entries,
            "last_global_step": max(entries) if entries else None,
        }
        if profiler is not None:
            profiler.meta["step_predicate_cache_entries"] = len(entries)
            _profiler_meta_increment(profiler, "step_predicate_cache_builds")

    for name in generators:
        original_generator = getattr(
            base_module,
            f"_v04_original_{name}",
            getattr(base_module, name),
        )
        setattr(base_module, f"_v04_original_{name}", original_generator)

        def wrapped_generator(*args, _original=original_generator, **kwargs):
            result = _original(*args, **kwargs)
            try:
                step_args = args[0]
                states = result[0]
                build_cache(step_args, states)
            except Exception as exc:
                if profiler is not None:
                    profiler.meta["step_predicate_cache_error"] = repr(exc)
            return result

        setattr(base_module, name, wrapped_generator)

    def should_activate_layer_for_state_cached(state):
        if hasattr(state, "_v04_should_activate_layer"):
            _record_step_predicate_event(profiler, hit=True)
            return bool(state._v04_should_activate_layer)
        _record_step_predicate_event(profiler, hit=False)
        return original_activate(state)

    def should_run_mechanics_cached(global_step, args):
        entry = _step_predicate_cache_hit(base_module, args, global_step)
        if entry is not None:
            _record_step_predicate_event(profiler, hit=True)
            result = bool(entry["run_mechanics"])
            base_module._v04_current_step_context = {
                "args_id": id(args),
                "global_step": int(global_step),
                "run_mechanics": result,
            }
            return result
        _record_step_predicate_event(profiler, hit=False)
        result = bool(original_mechanics(global_step, args))
        base_module._v04_current_step_context = {
            "args_id": id(args),
            "global_step": int(global_step),
            "run_mechanics": result,
        }
        return result

    def should_save_step_cached(global_step, did_mechanics, is_last, args):
        if is_last:
            _record_step_predicate_event(profiler, hit=True)
            return True
        entry = _step_predicate_cache_hit(base_module, args, global_step)
        if entry is not None:
            _record_step_predicate_event(profiler, hit=True)
            return bool(
                entry["save_if_mechanics"]
                if did_mechanics
                else entry["save_if_no_mechanics"]
            )
        _record_step_predicate_event(profiler, hit=False)
        return original_save(global_step, did_mechanics, is_last, args)

    base_module.should_activate_layer_for_state = should_activate_layer_for_state_cached
    base_module.should_run_mechanics = should_run_mechanics_cached
    base_module.should_save_step = should_save_step_cached
    if profiler is not None:
        profiler.meta["step_predicate_cache_patch"] = (
            "should_activate_layer_for_state, should_run_mechanics, "
            "should_save_step"
        )
    return True


class _LazyPhaseCell:
    def __init__(self, phase_quad, compute_fn):
        self._phase_quad = phase_quad
        self._compute_fn = compute_fn
        self._computed = False
        self._value = None

    def compute(self):
        if not self._computed:
            self._value = self._compute_fn(self._phase_quad)
            self._computed = True
        return self._value


def _lazy_output_postprocess_should_compute(base_module, args) -> Optional[bool]:
    context = getattr(base_module, "_v04_current_step_context", None)
    cache = getattr(base_module, "_v04_step_predicate_cache", None)
    if (
        not context
        or not cache
        or context.get("args_id") != id(args)
        or cache.get("args_id") != id(args)
    ):
        return None

    global_step = int(context["global_step"])
    last_global_step = cache.get("last_global_step")
    if last_global_step is None:
        return None
    is_last = global_step == int(last_global_step)
    if is_last:
        return True

    entry = cache.get("entries", {}).get(global_step)
    if entry is None:
        return None

    did_mechanics = bool(
        context.get("run_mechanics")
        or (
            is_last
            and int(getattr(args, "mechanics_every", 0) or 0) > 0
        )
    )
    return bool(
        entry["save_if_mechanics"]
        if did_mechanics
        else entry["save_if_no_mechanics"]
    )


def install_lazy_output_postprocess_patch(
    base_module,
    profiler: Optional[ProfilingReport] = None,
    enabled: bool = True,
) -> bool:
    """Delay output-only material-state reduction until a VTU save step."""
    if profiler is not None:
        profiler.meta["lazy_output_postprocess_enabled"] = bool(enabled)
    if not enabled:
        return False

    phase_original = getattr(
        base_module,
        "_v04_original_phase_cell_from_quad",
        getattr(base_module, "phase_cell_from_quad", None),
    )
    material_original = getattr(
        base_module,
        "_v04_original_material_cell_state",
        getattr(base_module, "material_cell_state", None),
    )
    if phase_original is None or material_original is None:
        if profiler is not None:
            profiler.meta["lazy_output_postprocess_patch"] = (
                "missing phase_cell_from_quad or material_cell_state"
            )
        return False

    base_module._v04_original_phase_cell_from_quad = phase_original
    base_module._v04_original_material_cell_state = material_original

    def phase_cell_from_quad_lazy(phase_quad):
        _profiler_meta_increment(
            profiler,
            "lazy_output_postprocess_phase_deferred",
        )
        return _LazyPhaseCell(phase_quad, phase_original)

    def material_cell_state_lazy(
        active_cell,
        substrate_cell,
        support_cell,
        args,
        cell_temperature=None,
        phase_cell=None,
    ):
        should_compute = _lazy_output_postprocess_should_compute(
            base_module,
            args,
        )
        previous = getattr(base_module, "_v04_last_material_state", None)
        if should_compute is False and previous is not None:
            _profiler_meta_increment(
                profiler,
                "lazy_output_postprocess_skips",
            )
            return previous

        if isinstance(phase_cell, _LazyPhaseCell):
            phase_cell = phase_cell.compute()

        result = material_original(
            active_cell,
            substrate_cell,
            support_cell,
            args,
            cell_temperature,
            phase_cell=phase_cell,
        )
        base_module._v04_last_material_state = result
        _profiler_meta_increment(
            profiler,
            "lazy_output_postprocess_computes",
        )
        if should_compute is False:
            _profiler_meta_increment(
                profiler,
                "lazy_output_postprocess_forced_computes",
            )
        return result

    base_module.phase_cell_from_quad = phase_cell_from_quad_lazy
    base_module.material_cell_state = material_cell_state_lazy
    if profiler is not None:
        profiler.meta["lazy_output_postprocess_patch"] = (
            "phase_cell_from_quad, material_cell_state"
        )
    return True


def install_quad_scalar_fast_path_patch(
    base_module,
    profiler: Optional[ProfilingReport] = None,
) -> bool:
    """Avoid `arr * ones` in v03 `make_quad_scalar` for one-point cells."""
    original = getattr(
        base_module,
        "_v04_original_make_quad_scalar",
        getattr(base_module, "make_quad_scalar", None),
    )
    np_module = getattr(base_module, "np", None)
    if original is None or np_module is None:
        if profiler is not None:
            profiler.meta["quad_scalar_fast_path_patch"] = (
                "missing make_quad_scalar/base np"
            )
        return False

    base_module._v04_original_make_quad_scalar = original

    def make_quad_scalar_fast(cell_values, num_quads):
        try:
            n_quads = int(num_quads)
        except Exception:
            _profiler_meta_increment(
                profiler,
                "quad_scalar_fast_path_fallbacks",
            )
            return original(cell_values, num_quads)

        if n_quads != 1:
            _profiler_meta_increment(
                profiler,
                "quad_scalar_fast_path_fallbacks",
            )
            return original(cell_values, num_quads)

        arr = np_module.asarray(cell_values)[:, None, None]
        dtype_kind = getattr(getattr(arr, "dtype", None), "kind", "")
        _profiler_meta_increment(profiler, "quad_scalar_fast_path_calls")
        if dtype_kind in ("f", "c"):
            return arr.copy()
        return arr * np_module.ones(())

    base_module.make_quad_scalar = make_quad_scalar_fast
    if profiler is not None:
        profiler.meta["quad_scalar_fast_path_patch"] = "installed"
    return True


def _tables_are_empty(
    tables: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> bool:
    if not isinstance(tables, Mapping):
        return False
    return all(tables.get(key) is None for key in keys)


def _mechanics_material_is_unused(args: argparse.Namespace) -> bool:
    try:
        mechanics_every = int(getattr(args, "mechanics_every", 1))
    except Exception:
        return False
    return mechanics_every == 0 and not bool(
        getattr(args, "release_after_cooling", False)
    )


class _ThermalOnlyMechanicsFE:
    def __init__(self, thermal_fe: Any, node_inds_list: Sequence[Any]):
        self._thermal_fe = thermal_fe
        self.node_inds_list = list(node_inds_list)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._thermal_fe, name)

    def convert_from_dof_to_quad(self, sol):
        return self._thermal_fe.convert_from_dof_to_quad(sol)


class _ThermalOnlyMechanicsProblem:
    def __init__(
        self,
        thermal_fe: Any,
        vec: int,
        node_inds_list: Sequence[Any],
    ):
        self.fes = [_ThermalOnlyMechanicsFE(thermal_fe, node_inds_list)]
        points = getattr(thermal_fe, "points", None)
        try:
            num_nodes = len(points)
        except Exception:
            num_nodes = int(getattr(thermal_fe, "num_total_nodes", 0) or 0)
        self.num_total_dofs_all_vars = int(num_nodes) * int(vec)


def _mechanics_dirichlet_node_inds_from_thermal_fe(
    thermal_fe: Any,
    dirichlet_bc_info: Any,
) -> Optional[list[Any]]:
    if not dirichlet_bc_info:
        return []
    try:
        location_fns = dirichlet_bc_info[0]
    except Exception:
        return []

    thermal_node_inds = list(getattr(thermal_fe, "node_inds_list", []) or [])
    if thermal_node_inds:
        bottom_nodes = thermal_node_inds[0]
        return [bottom_nodes for _ in location_fns]
    return None


def install_thermal_only_mechanics_surrogate_patch(
    base_module,
    args: argparse.Namespace,
    profiler: Optional[ProfilingReport] = None,
    enabled: bool = True,
) -> bool:
    """Avoid building a full mechanics Problem for thermal-only runs."""
    if profiler is not None:
        profiler.meta["thermal_only_mechanics_surrogate_enabled"] = bool(enabled)
    if not enabled:
        return False

    thermal_original = getattr(
        base_module,
        "_v04_original_TransientThermal_for_surrogate",
        getattr(base_module, "TransientThermal", None),
    )
    mechanics_original = getattr(
        base_module,
        "_v04_original_ThermoMechanical_for_surrogate",
        getattr(base_module, "ThermoMechanical", None),
    )
    if thermal_original is None or mechanics_original is None:
        if profiler is not None:
            profiler.meta["thermal_only_mechanics_surrogate_patch"] = (
                "missing TransientThermal or ThermoMechanical"
            )
        return False

    base_module._v04_original_TransientThermal_for_surrogate = thermal_original
    base_module._v04_original_ThermoMechanical_for_surrogate = mechanics_original

    def transient_thermal_with_stash(*ctor_args, **ctor_kwargs):
        thermal = thermal_original(*ctor_args, **ctor_kwargs)
        base_module._v04_last_thermal_problem = thermal
        fes = getattr(thermal, "fes", None) or []
        if fes:
            base_module._v04_last_thermal_fe = fes[0]
        return thermal

    def thermomechanical_or_surrogate(*ctor_args, **ctor_kwargs):
        if _mechanics_material_is_unused(args):
            thermal_fe = getattr(base_module, "_v04_last_thermal_fe", None)
            if thermal_fe is not None:
                vec = int(ctor_kwargs.get("vec", 3) or 3)
                node_inds_list = _mechanics_dirichlet_node_inds_from_thermal_fe(
                    thermal_fe,
                    ctor_kwargs.get("dirichlet_bc_info"),
                )
                if node_inds_list is not None:
                    _profiler_meta_increment(
                        profiler,
                        "thermal_only_mechanics_surrogate_hits",
                    )
                    return _ThermalOnlyMechanicsProblem(
                        thermal_fe,
                        vec,
                        node_inds_list,
                    )

        _profiler_meta_increment(
            profiler,
            "thermal_only_mechanics_surrogate_fallbacks",
        )
        return mechanics_original(*ctor_args, **ctor_kwargs)

    base_module.TransientThermal = transient_thermal_with_stash
    base_module.ThermoMechanical = thermomechanical_or_surrogate
    if profiler is not None:
        profiler.meta["thermal_only_mechanics_surrogate_patch"] = (
            "TransientThermal stash, ThermoMechanical surrogate"
        )
    return True


def _unused_mechanics_material_quads(T_quad):
    return (T_quad, T_quad, T_quad, T_quad, T_quad, T_quad)


def _optional_float(args: argparse.Namespace, key: str) -> Optional[float]:
    value = getattr(args, key, None)
    return None if value is None else float(value)


def _float_arg(
    args: argparse.Namespace,
    key: str,
    default: float,
) -> float:
    value = getattr(args, key, default)
    return float(default if value is None else value)


def _thermal_material_key(
    args: argparse.Namespace,
    base_module,
) -> tuple[Any, ...]:
    rho_solid = _optional_float(args, "rho_solid")
    if rho_solid is None:
        rho_solid = _float_arg(args, "rho", 7800.0)
    cp_solid = _optional_float(args, "cp_solid")
    if cp_solid is None:
        cp_solid = _float_arg(args, "cp", 500.0)
    k_solid = _optional_float(args, "conductivity_solid")
    if k_solid is None:
        k_solid = _float_arg(args, "conductivity", 20.0)
    rho_liquid = _optional_float(args, "rho_liquid")
    if rho_liquid is None:
        rho_liquid = rho_solid
    cp_liquid = _optional_float(args, "cp_liquid")
    if cp_liquid is None:
        cp_liquid = cp_solid
    k_liquid = _optional_float(args, "conductivity_liquid")
    if k_liquid is None:
        k_liquid = k_solid

    future_layers_are_void = (
        getattr(args, "layer_activation_mode", "front") == "layer_on_scan"
        and getattr(args, "future_layer_mode", "void") == "void"
    )
    unprinted_as_void = (
        future_layers_are_void
        or getattr(args, "powder_mode", "powder") == "void"
    )
    return (
        rho_solid,
        cp_solid,
        k_solid,
        rho_liquid,
        cp_liquid,
        k_liquid,
        _float_arg(args, "rho_powder", 3900.0),
        _float_arg(args, "cp_powder", 500.0),
        _float_arg(args, "conductivity_powder", 1.0),
        _float_arg(args, "inactive_thermal_factor", 1e-6),
        _float_arg(
            args,
            "inactive_mass_factor",
            _float_arg(args, "inactive_thermal_factor", 1e-6),
        ),
        bool(future_layers_are_void),
        _float_arg(args, "solidus_temperature", 0.0),
        _float_arg(args, "liquidus_temperature", 0.0),
        _float_arg(args, "old_layer_thermal_factor", 1e-6),
        _float_arg(args, "latent_heat", 0.0),
        bool(unprinted_as_void),
        float(base_module.STATE_VOID),
        float(base_module.STATE_POWDER),
        float(base_module.STATE_MUSHY),
        float(base_module.STATE_LIQUID),
    )


def _mechanics_material_key(
    args: argparse.Namespace,
    base_module,
) -> Optional[tuple[Any, ...]]:
    if getattr(args, "mechanics_model", "linear_elastic") != "linear_elastic":
        return None
    powder_solid_E = _optional_float(args, "powder_solid_E")
    return (
        _float_arg(args, "young", 2.0e11),
        _float_arg(args, "alpha", 1.2e-5),
        _float_arg(args, "poisson", 0.3),
        _float_arg(args, "mushy_mechanics_factor", 1e-2),
        _float_arg(args, "liquid_mechanics_factor", 1e-4),
        _float_arg(args, "inactive_mechanics_factor", 1e-9),
        bool(
            getattr(args, "layer_activation_mode", "front")
            == "layer_on_scan"
            and getattr(args, "future_layer_mode", "void") == "void"
        ),
        powder_solid_E is not None,
        0.0 if powder_solid_E is None else powder_solid_E,
        _float_arg(args, "powder_solid_yield", 1.0e6),
        _float_arg(args, "powder_solid_hardening", 1.0e7),
        float(base_module.STATE_POWDER),
        float(base_module.STATE_SOLID),
        float(base_module.STATE_MUSHY),
        float(base_module.STATE_LIQUID),
        float(base_module.STATE_SUBSTRATE),
        float(base_module.STATE_SUPPORT),
    )


def _history_kernel_key(args: argparse.Namespace, base_module) -> tuple[Any, ...]:
    return (
        _float_arg(args, "solidus_temperature", 0.0),
        _float_arg(args, "liquidus_temperature", 0.0),
        _float_arg(args, "stress_relaxation_temperature", 0.0),
        bool(getattr(args, "reset_plastic_on_melt", True)),
        float(base_module.STATE_VOID),
        float(base_module.STATE_POWDER),
        float(base_module.STATE_SOLID),
        float(base_module.STATE_MUSHY),
        float(base_module.STATE_LIQUID),
        float(base_module.STATE_SUBSTRATE),
        float(base_module.STATE_SUPPORT),
    )


def _make_jit_thermal_material_kernel(base_module, key: tuple[Any, ...]):
    jax_module = base_module.jax
    jnp = base_module.np
    (
        rho_solid,
        cp_solid,
        k_solid,
        rho_liquid,
        cp_liquid,
        k_liquid,
        rho_powder,
        cp_powder,
        k_powder,
        inactive_thermal_factor,
        inactive_mass_factor,
        strict_active_domain,
        solidus_temperature,
        liquidus_temperature,
        old_layer_thermal_factor,
        latent_heat,
        unprinted_as_void,
        state_void,
        state_powder,
        state_mushy,
        state_liquid,
    ) = key
    has_phase_interval = liquidus_temperature > solidus_temperature
    has_latent_heat = latent_heat > 0.0 and has_phase_interval

    @jax_module.jit
    def kernel(
        T_old_quad,
        active_quad,
        phase_quad,
        printed_quad,
        cooling_only_quad,
    ):
        cp_solid_quad = cp_solid * jnp.ones_like(T_old_quad)
        k_solid_quad = k_solid * jnp.ones_like(T_old_quad)
        cp_powder_quad = cp_powder * jnp.ones_like(T_old_quad)
        k_powder_quad = k_powder * jnp.ones_like(T_old_quad)
        cp_liquid_quad = cp_liquid * jnp.ones_like(T_old_quad)
        k_liquid_quad = k_liquid * jnp.ones_like(T_old_quad)

        if strict_active_domain:
            rho_void = jnp.zeros_like(T_old_quad)
            cp_void = jnp.zeros_like(T_old_quad)
            k_void = jnp.zeros_like(T_old_quad)
        else:
            rho_void = rho_solid * inactive_mass_factor
            cp_void = cp_solid * jnp.ones_like(T_old_quad)
            k_void = (
                k_solid
                * inactive_thermal_factor
                * jnp.ones_like(T_old_quad)
            )

        if has_phase_interval:
            mushy_frac = jnp.clip(
                (T_old_quad - solidus_temperature)
                / (liquidus_temperature - solidus_temperature),
                0.0,
                1.0,
            )
        else:
            mushy_frac = jnp.zeros_like(T_old_quad)

        rho_mushy = (1.0 - mushy_frac) * rho_solid + mushy_frac * rho_liquid
        cp_mushy = (
            (1.0 - mushy_frac) * cp_solid_quad
            + mushy_frac * cp_liquid_quad
        )
        k_mushy = (
            (1.0 - mushy_frac) * k_solid_quad
            + mushy_frac * k_liquid_quad
        )

        is_void = phase_quad == state_void
        is_powder = phase_quad == state_powder
        is_liquid = phase_quad == state_liquid
        is_mushy = phase_quad == state_mushy

        rho_phase = jnp.where(
            is_void,
            rho_void,
            jnp.where(
                is_powder,
                rho_powder,
                jnp.where(
                    is_liquid,
                    rho_liquid,
                    jnp.where(is_mushy, rho_mushy, rho_solid),
                ),
            ),
        )
        cp_phase = jnp.where(
            is_void,
            cp_void,
            jnp.where(
                is_powder,
                cp_powder_quad,
                jnp.where(
                    is_liquid,
                    cp_liquid_quad,
                    jnp.where(is_mushy, cp_mushy, cp_solid_quad),
                ),
            ),
        )
        k_phase = jnp.where(
            is_void,
            k_void,
            jnp.where(
                is_powder,
                k_powder_quad,
                jnp.where(
                    is_liquid,
                    k_liquid_quad,
                    jnp.where(is_mushy, k_mushy, k_solid_quad),
                ),
            ),
        )

        if unprinted_as_void:
            rho_unprinted = rho_void
            cp_unprinted = cp_void
            k_unprinted = k_void
        else:
            rho_unprinted = rho_powder
            cp_unprinted = cp_powder_quad
            k_unprinted = k_powder_quad

        rho_old = rho_solid * jnp.ones_like(T_old_quad)
        cp_old = cp_solid_quad
        k_old = k_solid_quad * old_layer_thermal_factor

        is_printed = printed_quad > 0.5
        is_window = active_quad > 0.5
        is_cooling_only = cooling_only_quad > 0.5

        rho_quad = jnp.where(
            is_window,
            rho_phase,
            jnp.where(is_cooling_only, rho_old, rho_unprinted),
        )
        cp_quad = jnp.where(
            is_window,
            cp_phase,
            jnp.where(is_cooling_only, cp_old, cp_unprinted),
        )
        conductivity_quad = jnp.where(
            is_window,
            k_phase,
            jnp.where(is_cooling_only, k_old, k_unprinted),
        )

        rho_quad = jnp.where(is_printed | is_window, rho_quad, rho_unprinted)
        cp_quad = jnp.where(is_printed | is_window, cp_quad, cp_unprinted)
        conductivity_quad = jnp.where(
            is_printed | is_window,
            conductivity_quad,
            k_unprinted,
        )

        if has_latent_heat:
            in_mushy = (
                (T_old_quad >= solidus_temperature)
                & (T_old_quad <= liquidus_temperature)
                & is_window
            )
            latent_cp = jnp.where(
                in_mushy,
                latent_heat / (liquidus_temperature - solidus_temperature),
                0.0,
            )
        else:
            latent_cp = jnp.zeros_like(T_old_quad)

        return rho_quad, cp_quad, conductivity_quad, latent_cp

    return kernel


def _make_jit_mechanics_material_kernel(base_module, key: tuple[Any, ...]):
    jax_module = base_module.jax
    jnp = base_module.np
    (
        young,
        alpha,
        poisson,
        mushy_mechanics_factor,
        liquid_mechanics_factor,
        inactive_mechanics_factor,
        strict_active_domain,
        has_powder_solid,
        powder_solid_E,
        powder_solid_yield,
        powder_solid_hardening,
        state_powder,
        state_solid,
        state_mushy,
        state_liquid,
        state_substrate,
        state_support,
    ) = key

    @jax_module.jit
    def kernel(T_quad, active_quad, phase_quad):
        inactive_factor = (
            0.0 if strict_active_domain else inactive_mechanics_factor
        )
        E_quad = young * jnp.ones_like(T_quad)
        alpha_base = alpha * jnp.ones_like(T_quad)
        poisson_quad = poisson * jnp.ones_like(T_quad)
        yield_quad = young * jnp.ones_like(T_quad)
        hardening_quad = jnp.zeros_like(T_quad)

        is_solid_like = (
            (phase_quad == state_solid)
            | (phase_quad == state_substrate)
            | (phase_quad == state_support)
        )
        is_mushy = phase_quad == state_mushy
        is_liquid = phase_quad == state_liquid
        is_powder = phase_quad == state_powder

        active_factor_quad = jnp.where(
            is_solid_like,
            1.0,
            jnp.where(
                is_mushy,
                mushy_mechanics_factor,
                jnp.where(
                    is_liquid,
                    liquid_mechanics_factor,
                    inactive_factor,
                ),
            ),
        )
        active_factor_quad = (
            active_factor_quad * active_quad
            + inactive_factor * (1.0 - active_quad)
        )
        alpha_quad = jnp.where(
            is_solid_like,
            alpha_base,
            jnp.zeros_like(alpha_base),
        )
        if has_powder_solid:
            E_quad = jnp.where(is_powder, powder_solid_E, E_quad)
            yield_quad = jnp.where(
                is_powder,
                powder_solid_yield,
                yield_quad,
            )
            hardening_quad = jnp.where(
                is_powder,
                powder_solid_hardening,
                hardening_quad,
            )
            active_factor_quad = jnp.where(
                is_powder,
                active_quad + inactive_factor * (1.0 - active_quad),
                active_factor_quad,
            )
        return (
            active_factor_quad,
            E_quad,
            alpha_quad,
            poisson_quad,
            yield_quad,
            hardening_quad,
        )

    return kernel


def _make_jit_history_kernel(base_module, key: tuple[Any, ...]):
    jax_module = base_module.jax
    jnp = base_module.np
    (
        solidus_temperature,
        liquidus_temperature,
        stress_relaxation_temperature,
        reset_plastic_on_melt,
        state_void,
        state_powder,
        state_solid,
        state_mushy,
        state_liquid,
        state_substrate,
        state_support,
    ) = key
    has_phase_interval = liquidus_temperature > solidus_temperature
    has_relaxation_reference = stress_relaxation_temperature > 0.0

    @jax_module.jit
    def kernel(T_quad, active_quad, phase_quad, T_ref_quad, eqp_quad):
        active = active_quad > 0.5
        fixture = (phase_quad == state_substrate) | (phase_quad == state_support)
        non_fixture = active & (~fixture)
        phase_new = phase_quad

        newly_active_void = non_fixture & (phase_new == state_void)
        phase_new = jnp.where(newly_active_void, state_powder, phase_new)

        if has_phase_interval:
            hot_liquid = non_fixture & (T_quad >= liquidus_temperature)
            mushy = (
                non_fixture
                & (T_quad >= solidus_temperature)
                & (T_quad < liquidus_temperature)
            )
            cold = non_fixture & (T_quad < solidus_temperature)

            old_was_melted = (
                (phase_quad == state_liquid)
                | (phase_quad == state_mushy)
            )
            became_solid = cold & old_was_melted
            stayed_solid = cold & (phase_quad == state_solid)

            phase_new = jnp.where(hot_liquid, state_liquid, phase_new)
            phase_new = jnp.where(mushy, state_mushy, phase_new)
            phase_new = jnp.where(
                became_solid | stayed_solid,
                state_solid,
                phase_new,
            )

            newly_solidified = became_solid
            entered_melted_state = (
                (hot_liquid | mushy)
                & (
                    (phase_quad == state_solid)
                    | (phase_quad == state_mushy)
                    | (phase_quad == state_liquid)
                )
            )
        else:
            became_solid = non_fixture & (phase_quad != state_solid)
            phase_new = jnp.where(non_fixture, state_solid, phase_new)
            newly_solidified = became_solid
            entered_melted_state = jnp.zeros_like(active)

        if has_relaxation_reference:
            T_ref_value = stress_relaxation_temperature * jnp.ones_like(T_quad)
        else:
            T_ref_value = T_quad
        T_ref_new = jnp.where(newly_solidified, T_ref_value, T_ref_quad)
        if reset_plastic_on_melt:
            eqp_new = jnp.where(
                entered_melted_state,
                jnp.zeros_like(eqp_quad),
                eqp_quad,
            )
        else:
            eqp_new = eqp_quad
        return (
            phase_new,
            T_ref_new,
            eqp_new,
            newly_solidified,
            entered_melted_state,
        )

    return kernel


def install_loop_kernel_jit_patch(
    base_module,
    profiler: Optional[ProfilingReport] = None,
    enabled: bool = True,
    skip_unused_mechanics_material: bool = True,
) -> bool:
    """JIT loop-side material/history kernels while preserving v03 fallbacks."""
    if profiler is not None:
        profiler.meta["loop_kernel_jit_enabled"] = bool(enabled)
        profiler.meta["skip_unused_mechanics_material_enabled"] = bool(
            skip_unused_mechanics_material
        )
    if not enabled:
        return False

    if not hasattr(base_module, "jax") or not hasattr(base_module, "np"):
        if profiler is not None:
            profiler.meta["loop_kernel_jit_patch"] = (
                "missing base_module.jax/base_module.np"
            )
        return False

    thermal_original = getattr(
        base_module,
        "_v04_original_thermal_material_quads",
        getattr(base_module, "thermal_material_quads", None),
    )
    mechanics_original = getattr(
        base_module,
        "_v04_original_mechanics_material_quads",
        getattr(base_module, "mechanics_material_quads", None),
    )
    history_original = getattr(
        base_module,
        "_v04_original_update_phase_reference_and_eqp",
        getattr(base_module, "update_phase_reference_and_eqp", None),
    )
    if thermal_original is None or mechanics_original is None or history_original is None:
        if profiler is not None:
            profiler.meta["loop_kernel_jit_patch"] = "missing v03 loop kernels"
        return False

    base_module._v04_original_thermal_material_quads = thermal_original
    base_module._v04_original_mechanics_material_quads = mechanics_original
    base_module._v04_original_update_phase_reference_and_eqp = history_original

    def get_thermal_kernel(args):
        key = _thermal_material_key(args, base_module)
        if key not in _LOOP_KERNEL_JIT_THERMAL_CACHE:
            _LOOP_KERNEL_JIT_THERMAL_CACHE[key] = _make_jit_thermal_material_kernel(
                base_module,
                key,
            )
            _profiler_meta_increment(
                profiler,
                "loop_kernel_jit_thermal_compiles",
            )
        if profiler is not None:
            profiler.meta["loop_kernel_jit_thermal_cache_entries"] = len(
                _LOOP_KERNEL_JIT_THERMAL_CACHE
            )
        return _LOOP_KERNEL_JIT_THERMAL_CACHE[key]

    def get_mechanics_kernel(args):
        key = _mechanics_material_key(args, base_module)
        if key is None:
            return None
        if key not in _LOOP_KERNEL_JIT_MECHANICS_CACHE:
            _LOOP_KERNEL_JIT_MECHANICS_CACHE[key] = _make_jit_mechanics_material_kernel(
                base_module,
                key,
            )
            _profiler_meta_increment(
                profiler,
                "loop_kernel_jit_mechanics_compiles",
            )
        if profiler is not None:
            profiler.meta["loop_kernel_jit_mechanics_cache_entries"] = len(
                _LOOP_KERNEL_JIT_MECHANICS_CACHE
            )
        return _LOOP_KERNEL_JIT_MECHANICS_CACHE[key]

    def get_history_kernel(args):
        key = _history_kernel_key(args, base_module)
        if key not in _LOOP_KERNEL_JIT_HISTORY_CACHE:
            _LOOP_KERNEL_JIT_HISTORY_CACHE[key] = _make_jit_history_kernel(
                base_module,
                key,
            )
            _profiler_meta_increment(
                profiler,
                "loop_kernel_jit_history_compiles",
            )
        if profiler is not None:
            profiler.meta["loop_kernel_jit_history_cache_entries"] = len(
                _LOOP_KERNEL_JIT_HISTORY_CACHE
            )
        return _LOOP_KERNEL_JIT_HISTORY_CACHE[key]

    def thermal_material_quads_jit(
        T_old_quad,
        active_quad,
        phase_quad,
        args,
        tables,
        printed_quad=None,
        cooling_only_quad=None,
    ):
        if _tables_are_empty(tables, THERMAL_TABLE_KEYS):
            if printed_quad is None:
                printed_quad = active_quad
            if cooling_only_quad is None:
                cooling_only_quad = base_module.np.zeros_like(active_quad)
            _profiler_meta_increment(
                profiler,
                "loop_kernel_jit_thermal_calls",
            )
            return get_thermal_kernel(args)(
                T_old_quad,
                active_quad,
                phase_quad,
                printed_quad,
                cooling_only_quad,
            )
        _profiler_meta_increment(
            profiler,
            "loop_kernel_jit_thermal_fallbacks",
        )
        return thermal_original(
            T_old_quad,
            active_quad,
            phase_quad,
            args,
            tables,
            printed_quad=printed_quad,
            cooling_only_quad=cooling_only_quad,
        )

    def mechanics_material_quads_jit(
        T_quad,
        active_quad,
        phase_quad,
        args,
        tables,
    ):
        if (
            skip_unused_mechanics_material
            and _mechanics_material_is_unused(args)
        ):
            _profiler_meta_increment(
                profiler,
                "loop_kernel_jit_mechanics_disabled_skips",
            )
            return _unused_mechanics_material_quads(T_quad)

        kernel = None
        if _tables_are_empty(tables, MECHANICAL_TABLE_KEYS):
            kernel = get_mechanics_kernel(args)
        if kernel is not None:
            _profiler_meta_increment(
                profiler,
                "loop_kernel_jit_mechanics_calls",
            )
            return kernel(T_quad, active_quad, phase_quad)
        _profiler_meta_increment(
            profiler,
            "loop_kernel_jit_mechanics_fallbacks",
        )
        return mechanics_original(T_quad, active_quad, phase_quad, args, tables)

    def update_phase_reference_and_eqp_jit(
        T_quad,
        active_quad,
        phase_quad,
        T_ref_quad,
        eqp_quad,
        args,
    ):
        _profiler_meta_increment(
            profiler,
            "loop_kernel_jit_history_calls",
        )
        return get_history_kernel(args)(
            T_quad,
            active_quad,
            phase_quad,
            T_ref_quad,
            eqp_quad,
        )

    base_module.thermal_material_quads = thermal_material_quads_jit
    base_module.mechanics_material_quads = mechanics_material_quads_jit
    base_module.update_phase_reference_and_eqp = update_phase_reference_and_eqp_jit
    if profiler is not None:
        profiler.meta["loop_kernel_jit_patch"] = (
            "thermal_material_quads, mechanics_material_quads, "
            "update_phase_reference_and_eqp"
        )
    return True


def install_profiling_patches(
    base_module,
    profiler: Optional[ProfilingReport],
) -> None:
    if profiler is None:
        return

    for name in ("generate_raster_step_states", "generate_path_file_step_states"):
        original = getattr(base_module, name, None)
        if original is None:
            continue

        def wrapped_step_generator(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            try:
                profiler.steps = len(result[0])
            except Exception:
                pass
            return result

        setattr(base_module, name, wrapped_step_generator)

    for name in (
        "save_step",
        "write_path_output",
        "write_calibration_template",
        "write_used_config",
    ):
        original = getattr(base_module, name, None)
        if original is None:
            continue

        def wrapped_io(*args, _original=original, **kwargs):
            with profiler.stage(STAGE_IO):
                return _original(*args, **kwargs)

        setattr(base_module, name, wrapped_io)

    loop_stage_by_function = {
        "compute_active_cell": STAGE_ACTIVATION,
        "compute_layer_on_scan_cells": STAGE_ACTIVATION,
        "compute_layer_on_scan_cells_by_intersection": STAGE_ACTIVATION,
        "compute_moving_window_cells": STAGE_ACTIVATION,
        "compute_moving_window_cells_by_intersection": STAGE_ACTIVATION,
        "make_quad_scalar": STAGE_QUAD_STATE,
        "thermal_material_quads": STAGE_MATERIAL,
        "mechanics_material_quads": STAGE_MATERIAL,
        "update_phase_reference_and_eqp": STAGE_HISTORY,
        "compute_cell_temperature": STAGE_POSTPROCESS,
        "material_cell_state": STAGE_POSTPROCESS,
    }
    for name, stage in loop_stage_by_function.items():
        original = getattr(base_module, name, None)
        if original is None:
            continue

        def wrapped_loop_function(
            *args,
            _original=original,
            _stage=stage,
            **kwargs,
        ):
            with profiler.stage(_stage):
                return _original(*args, **kwargs)

        setattr(base_module, name, wrapped_loop_function)


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = _argv(argv)
    runtime_args = preparse_runtime_args(argv_list)
    apply_runtime_env(runtime_args)

    base = load_base_solver()
    args = parse_args(base, argv_list)
    apply_runtime_env(args)
    replacement = linear_options_from_args(args)

    report = ProfilingReport(label=args.profile_label)
    report.meta["base_solver"] = str(BASE_SOLVER_PATH)
    report.meta["linear_solver"] = (
        next(iter(replacement)) if replacement else "base-config"
    )
    report.meta["linear_solver_label"] = _solver_label(replacement)
    report.meta["linear_solver_options"] = (
        {
            key: value
            if isinstance(value, (dict, list, str, int, float, bool, type(None)))
            else getattr(value, "label", repr(value))
            for key, value in replacement.items()
        }
        if replacement
        else copy.deepcopy(replacement)
    )
    report.meta["full_loop_xla"] = False
    report.meta["thermal_warm_start_enabled"] = bool(
        args.xla_thermal_warm_start
    )
    report.meta["loop_kernel_jit_enabled"] = bool(args.xla_jit_loop_kernels)
    report.meta["residual_only_check_enabled"] = bool(
        args.xla_residual_only_check
    )
    report.meta["residual_only_check_scope"] = (
        "thermal" if args.xla_residual_only_check else "disabled"
    )
    report.meta["mechanics_residual_only_check_enabled"] = bool(
        getattr(args, "mechanics_residual_only_check", False)
    )
    report.meta["step_predicate_cache_enabled"] = bool(
        args.xla_step_predicate_cache
    )
    report.meta["skip_unused_mechanics_material_enabled"] = bool(
        args.xla_skip_unused_mechanics_material
    )
    report.meta["thermal_only_mechanics_surrogate_enabled"] = bool(
        args.xla_thermal_only_mechanics_surrogate
    )
    report.meta["quiet_jax_fem_logs_enabled"] = bool(
        args.xla_quiet_jax_fem_logs
    )
    report.meta["jax_fem_log_level"] = (
        "WARNING" if args.xla_quiet_jax_fem_logs else "preserve"
    )
    report.meta["lazy_output_postprocess_enabled"] = bool(
        args.xla_lazy_output_postprocess
    )
    report.meta["dof_to_quad_cache_enabled"] = bool(args.xla_dof_to_quad_cache)
    report.meta["dof_to_quad_cache_scope"] = (
        "same_finite_element_jax_array_identity"
    )
    report.meta["dof_to_quad_cache_max_entries"] = (
        DOF_TO_QUAD_CACHE_MAX_ENTRIES if args.xla_dof_to_quad_cache else 0
    )
    report.meta.update(
        _cell_chunking_options(
            args.xla_cell_num_cuts,
            args.xla_cell_target_batch_size,
        )
    )

    try:
        configure_jax_fem_logging(args.xla_quiet_jax_fem_logs, report)
        print_acceleration_summary(args, replacement)
        if args.xla_show_devices:
            show_jax_devices()
        if args.xla_dry_run:
            return 0

        configure_problem_cell_assembly_num_cuts(
            args.xla_cell_num_cuts,
            report,
            target_batch_size=args.xla_cell_target_batch_size,
        )
        install_jax_fem_timing_patch(report)
        install_problem_local_assembly_timing_patch(report)
        install_finite_element_timing_patch(
            report,
            cache_enabled=args.xla_dof_to_quad_cache,
        )
        if args.xla_thermal_warm_start:
            install_thermal_warm_start_patch(base, report)
        install_thermal_only_mechanics_surrogate_patch(
            base,
            args,
            report,
            enabled=args.xla_thermal_only_mechanics_surrogate,
        )
        install_setup_detail_timing_patch(base, report)
        install_step_predicate_cache_patch(
            base,
            report,
            enabled=args.xla_step_predicate_cache,
        )
        install_lazy_output_postprocess_patch(
            base,
            report,
            enabled=args.xla_lazy_output_postprocess,
        )
        install_activation_cache_patch(base, report)
        install_quad_scalar_fast_path_patch(base, report)
        install_loop_kernel_jit_patch(
            base,
            report,
            enabled=args.xla_jit_loop_kernels,
            skip_unused_mechanics_material=(
                args.xla_skip_unused_mechanics_material
            ),
        )
        install_profiling_patches(base, report)
        install_solver_patch(
            base,
            replacement,
            fallback_to_spsolve=args.xla_fallback_to_spsolve,
            profiler=report,
            profile_solver_call=False,
            thermal_warm_start=args.xla_thermal_warm_start,
            residual_only_check=args.xla_residual_only_check,
        )
        base.parse_args = lambda: args
        rc = base.main()
        return int(rc or 0)
    finally:
        report.finish()
        print(report.summary(), file=sys.stderr)
        if args.profile_json:
            report.dump(args.profile_json)


if __name__ == "__main__":
    raise SystemExit(main())
