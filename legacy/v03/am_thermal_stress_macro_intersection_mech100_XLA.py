#!/usr/bin/env python3
"""Experimental acceleration wrapper for the macro intersection AM solver.

This entrypoint keeps the production physics in
``am_thermal_stress_macro_intersection_mech100.py`` and only changes the
runtime/linear-solver selection. It intentionally does not claim that the full
transient loop is inside one XLA graph: the current jax-fem Newton solver still
crosses Python, PETSc, SciPy, and sparse-matrix assembly boundaries.

The practical acceleration path exposed here is:

* keep the original thermal/mechanical models, activation logic, material tables,
  and VTU outputs;
* allow ``jax_solver`` or ``amgx_solver`` to replace the CPU SciPy ``spsolve``
  linear solves;
* keep a conservative SciPy fallback for experimental GPU solvers.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V01_DIR = REPO_ROOT / "159_local" / "v01"
ORIGINAL_PATH = SCRIPT_DIR / "am_thermal_stress_macro_intersection_mech100.py"

for path in (str(V01_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


METHOD_KEYS = frozenset({"newton", "arc_length", "dynamic_relax"})
LINEAR_KEYS = frozenset(
    {"jax_solver", "amgx_solver", "spsolve_solver", "petsc_solver", "custom_solver"}
)


def _argv(argv: Sequence[str] | None) -> list[str]:
    return list(sys.argv[1:] if argv is None else argv)


def preparse_runtime_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    _add_runtime_args(parser)
    args, _ = parser.parse_known_args(_argv(argv))
    return args


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--xla-platform",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Set JAX_PLATFORM_NAME before importing jax-fem. 'auto' preserves the environment.",
    )
    parser.add_argument(
        "--xla-preallocate",
        choices=("preserve", "on", "off"),
        default="preserve",
        help="Control XLA_PYTHON_CLIENT_PREALLOCATE. 'preserve' leaves the environment unchanged.",
    )
    parser.add_argument(
        "--xla-mem-fraction",
        type=float,
        default=None,
        help="Set XLA_PYTHON_CLIENT_MEM_FRACTION before importing JAX.",
    )


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


def load_original_module():
    spec = importlib.util.spec_from_file_location(
        "_macro_intersection_mech100_original", ORIGINAL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load original solver module: {ORIGINAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_acceleration_args(parser: argparse.ArgumentParser) -> None:
    _add_runtime_args(parser)
    parser.add_argument(
        "--xla-linear-solver",
        choices=("jax", "amgx", "petsc", "spsolve", "preserve"),
        default="jax",
        help=(
            "Linear solver override for the original jax-fem solver calls. "
            "'jax' uses jax.scipy.sparse.linalg.bicgstab; 'spsolve' matches the original CPU path; "
            "'preserve' leaves every original solver_options block unchanged."
        ),
    )
    parser.add_argument(
        "--xla-jax-precond",
        dest="xla_jax_precond",
        action="store_true",
        default=True,
        help="Use the existing Jacobi preconditioner in jax_fem.solver.jax_solve.",
    )
    parser.add_argument(
        "--no-xla-jax-precond",
        dest="xla_jax_precond",
        action="store_false",
    )
    parser.add_argument(
        "--xla-amgx-config",
        default=None,
        help="Optional AMGX JSON config path when --xla-linear-solver amgx is selected.",
    )
    parser.add_argument(
        "--xla-petsc-ksp",
        default="bcgsl",
        help="PETSc KSP type when --xla-linear-solver petsc is selected.",
    )
    parser.add_argument(
        "--xla-petsc-pc",
        default="ilu",
        help="PETSc PC type when --xla-linear-solver petsc is selected.",
    )
    parser.add_argument(
        "--xla-fallback-to-spsolve",
        dest="xla_fallback_to_spsolve",
        action="store_true",
        default=True,
        help="Retry a failed experimental linear solve with SciPy spsolve.",
    )
    parser.add_argument(
        "--no-xla-fallback-to-spsolve",
        dest="xla_fallback_to_spsolve",
        action="store_false",
    )
    parser.add_argument(
        "--xla-show-devices",
        action="store_true",
        help="Print jax.devices() and the default backend before running.",
    )
    parser.add_argument(
        "--xla-dry-run",
        action="store_true",
        help="Parse arguments, print the selected accelerator settings, and exit before solving.",
    )


def parse_args(original_module, argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv_list = _argv(argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv_list)
    config = original_module.read_config(config_args.config)

    parser = original_module.build_parser(config)
    add_acceleration_args(parser)
    args = parser.parse_args(argv_list)
    args.config = config_args.config
    return args


def linear_options_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    solver_name = args.xla_linear_solver
    if solver_name == "preserve":
        return None
    if solver_name == "spsolve":
        return {"spsolve_solver": {}}
    if solver_name == "jax":
        return {"jax_solver": {"precond": bool(args.xla_jax_precond)}}
    if solver_name == "amgx":
        amgx_options: dict[str, Any] = {}
        if args.xla_amgx_config:
            amgx_options["cfg_path"] = args.xla_amgx_config
        return {"amgx_solver": amgx_options}
    if solver_name == "petsc":
        return {
            "petsc_solver": {
                "ksp_type": args.xla_petsc_ksp,
                "pc_type": args.xla_petsc_pc,
            }
        }
    raise ValueError(f"Unknown --xla-linear-solver value: {solver_name}")


def rewrite_solver_options(
    solver_options: Mapping[str, Any] | None,
    linear_options: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if linear_options is None:
        return copy.deepcopy(solver_options) if solver_options is not None else None

    options: dict[str, Any] = copy.deepcopy(dict(solver_options or {}))
    method_keys = [key for key in METHOD_KEYS if key in options]
    if method_keys:
        for method in method_keys:
            method_options = dict(options.get(method) or {})
            method_options["linear"] = copy.deepcopy(dict(linear_options))
            options[method] = method_options
        return options

    for key in LINEAR_KEYS:
        options.pop(key, None)
    options.update(copy.deepcopy(dict(linear_options)))
    return options


def _solver_label(linear_options: Mapping[str, Any] | None) -> str:
    if linear_options is None:
        return "preserve original solver_options"
    if "jax_solver" in linear_options:
        return f"jax_solver(precond={linear_options['jax_solver'].get('precond', True)})"
    if "amgx_solver" in linear_options:
        cfg_path = linear_options["amgx_solver"].get("cfg_path")
        return f"amgx_solver(cfg_path={cfg_path or 'built-in'})"
    if "petsc_solver" in linear_options:
        opts = linear_options["petsc_solver"]
        return f"petsc_solver(ksp_type={opts.get('ksp_type')}, pc_type={opts.get('pc_type')})"
    if "spsolve_solver" in linear_options:
        return "spsolve_solver(cpu scipy baseline)"
    return str(linear_options)


def install_solver_patch(
    original_module,
    linear_options: Mapping[str, Any] | None,
    fallback_to_spsolve: bool,
) -> None:
    if linear_options is None:
        return

    original_solver = original_module.solver
    fallback_options = {"spsolve_solver": {}}

    def accelerated_solver(problem, solver_options=None):
        patched_options = rewrite_solver_options(solver_options, linear_options)
        try:
            return original_solver(problem, solver_options=patched_options)
        except Exception as exc:
            if not fallback_to_spsolve or "spsolve_solver" in linear_options:
                raise
            print(
                "WARNING: experimental linear solver failed; retrying this solve with "
                f"SciPy spsolve. Error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            retry_options = rewrite_solver_options(solver_options, fallback_options)
            return original_solver(problem, solver_options=retry_options)

    original_module.solver = accelerated_solver


def print_acceleration_summary(
    args: argparse.Namespace,
    linear_options: Mapping[str, Any] | None,
) -> None:
    print("============================================================")
    print("Experimental XLA/JAX acceleration wrapper")
    print(f"original_solver_module = {ORIGINAL_PATH}")
    print(f"linear_solver_override = {_solver_label(linear_options)}")
    print(f"xla_platform           = {args.xla_platform}")
    print(f"xla_preallocate       = {args.xla_preallocate}")
    print(f"xla_mem_fraction      = {args.xla_mem_fraction}")
    print(f"fallback_to_spsolve   = {args.xla_fallback_to_spsolve}")
    print("full_loop_xla         = disabled; original Python/jax-fem loop is preserved")
    print("============================================================")


def show_jax_devices() -> None:
    import jax

    print("JAX devices:", jax.devices())
    print("JAX default backend:", jax.default_backend())
    print("JAX enable x64:", jax.config.read("jax_enable_x64"))


def main(argv: Sequence[str] | None = None):
    argv_list = _argv(argv)
    runtime_args = preparse_runtime_args(argv_list)
    apply_runtime_env(runtime_args)

    original_module = load_original_module()
    args = parse_args(original_module, argv_list)
    apply_runtime_env(args)
    linear_options = linear_options_from_args(args)

    print_acceleration_summary(args, linear_options)
    if args.xla_show_devices:
        show_jax_devices()
    if args.xla_dry_run:
        return 0

    install_solver_patch(
        original_module,
        linear_options,
        fallback_to_spsolve=args.xla_fallback_to_spsolve,
    )
    original_module.parse_args = lambda: args
    return original_module.main()


if __name__ == "__main__":
    raise SystemExit(main())
