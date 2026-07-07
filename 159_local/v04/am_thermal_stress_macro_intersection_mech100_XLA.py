#!/usr/bin/env python3
"""XLA / GPU linear-solver wrapper for the mech100 macro thermal-stress solver.

This module wraps ``am_thermal_stress_macro_intersection_mech100`` (the CPU
reference implementation) and lets the user swap the *linear* solver used
inside the Newton loop without touching the physics driver:

    spsolve  -- SciPy sparse direct solve (CPU reference / fallback)
    jax      -- JAX BiCGStab/CG on GPU (optional Jacobi preconditioner)
    petsc    -- petsc4py KSP (optionally GPU-aware, ksp/pc configurable)
    amgx     -- NVIDIA AMGX (via pyamgx) with persistent resources
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
import importlib
import json
import os
import sys
import time
from contextlib import contextmanager

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

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
    }
)

#: argparse choice -> canonical linear-solver key
_SOLVER_CHOICE_TO_KEY = {
    "spsolve": "spsolve_solver",
    "jax": "jax_solver",
    "petsc": "petsc_solver",
    "amgx": "amgx_solver",
}


def linear_options_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Build the replacement linear-solver options block from CLI args.

    Returns ``None`` when the user asked to keep the base configuration
    (``--xla-linear-solver keep``), otherwise a single-key dict such as
    ``{"jax_solver": {"precond": False}}`` suitable for
    :func:`rewrite_solver_options`.
    """
    choice = getattr(args, "xla_linear_solver", "keep")
    if choice in (None, "keep"):
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
            inner["config"] = str(cfg)
        # AMGX resources are expensive to create; keep them alive across
        # the whole scan (persistent handle managed by the solver adapter).
        inner["persistent_resources"] = True

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

    _replace_linear_block(rewritten, replacement)
    return rewritten


# ---------------------------------------------------------------------------
# Phase 1: profiling report -- solver / conversion / python / io breakdown
# ---------------------------------------------------------------------------

STAGE_SOLVER = "solver"            # time inside the linear solve itself
STAGE_ASSEMBLY = "assembly"        # residual/Jacobian assembly
STAGE_CONVERSION = "conversion"    # PETSc<->SciPy<->JAX BCOO format shuffling
STAGE_TRANSFER = "transfer"        # host<->device copies outside conversion
STAGE_IO = "io"                    # VTU / checkpoint / log writes
STAGE_PYTHON = "python_overhead"   # main-loop bookkeeping not covered above

ALL_STAGES = (
    STAGE_SOLVER,
    STAGE_ASSEMBLY,
    STAGE_CONVERSION,
    STAGE_TRANSFER,
    STAGE_IO,
    STAGE_PYTHON,
)


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

    def finish(self) -> None:
        self.wall_seconds = time.perf_counter() - self.wall_start
        accounted = sum(self.stage_seconds.get(s, 0.0) for s in ALL_STAGES
                        if s != STAGE_PYTHON)
        # Everything not attributed to an explicit stage is Python overhead:
        # path-step bookkeeping, layer-activation checks, output-step
        # predicates, dict churn, etc. This is exactly the per-step fixed
        # cost that dominates at 826k scan steps.
        self.stage_seconds[STAGE_PYTHON] = max(
            0.0, self.wall_seconds - accounted
        )

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
    if gpu.wall_seconds <= cpu.wall_seconds:
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
        STAGE_SOLVER: (
            "the iterative solve itself is slower than the direct CPU "
            "factorization at this problem size; keep spsolve as default "
            "for this tier and re-evaluate on the representative mesh"
        ),
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


# ---------------------------------------------------------------------------
# Base-solver loading & instrumentation
# ---------------------------------------------------------------------------

BASE_MODULE_NAME = "am_thermal_stress_macro_intersection_mech100"


def load_base_solver(module_name: str = BASE_MODULE_NAME):
    """Import the CPU reference solver lazily.

    Kept out of module import time so that option plumbing and unit tests
    work on machines without the full FEM stack installed.
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"could not import base solver module {module_name!r} "
            f"(expected next to {__file__}); option plumbing still works, "
            f"but running the physics requires the full repo checkout"
        ) from exc


def build_arg_parser(parser: Optional[argparse.ArgumentParser] = None
                     ) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        description="mech100 thermal-stress solver, XLA/GPU linear-solver "
                    "wrapper")
    g = parser.add_argument_group("XLA / GPU linear solver")
    g.add_argument("--xla-linear-solver", default="keep",
                   choices=["keep", *sorted(_SOLVER_CHOICE_TO_KEY)],
                   help="linear solver used inside the Newton loop")
    g.add_argument("--xla-jax-precond", action="store_true",
                   help="enable Jacobi preconditioning for the JAX solver")
    g.add_argument("--xla-jax-method", default=None,
                   choices=[None, "bicgstab", "cg", "gmres"],
                   help="JAX iterative method (default: solver's choice)")
    g.add_argument("--xla-jax-tol", type=float, default=None)
    g.add_argument("--xla-petsc-ksp-type", default="gmres")
    g.add_argument("--xla-petsc-pc-type", default="jacobi")
    g.add_argument("--xla-petsc-gpu", action="store_true",
                   help="use PETSc aijcusparse/cuda Mat/Vec types")
    g.add_argument("--xla-amgx-config", default=None,
                   help="path to an AMGX JSON config")

    b = parser.add_argument_group("profiling")
    b.add_argument("--profile-json", default=None,
                   help="write the ProfilingReport to this JSON file")
    b.add_argument("--profile-label", default="run")
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args, passthrough = parser.parse_known_args(argv)

    base = load_base_solver()
    replacement = linear_options_from_args(args)

    report = ProfilingReport(label=args.profile_label)
    report.meta["linear_solver"] = (
        next(iter(replacement)) if replacement else "base-config"
    )

    # The base module is expected to expose `default_solver_options()` and
    # `run(solver_options=..., profiler=..., argv=...)`. We rewrite only the
    # linear block and hand everything else through untouched, so the
    # physics pipeline and output fields stay byte-compatible.
    base_options = base.default_solver_options()
    solver_options = rewrite_solver_options(base_options, replacement)

    try:
        rc = base.run(solver_options=solver_options, profiler=report,
                      argv=passthrough)
    finally:
        report.finish()
        print(report.summary(), file=sys.stderr)
        if args.profile_json:
            report.dump(args.profile_json)
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
