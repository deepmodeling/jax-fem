#!/usr/bin/env python3
"""Run the legacy v02 solver with a local Ti-6Al-4V material pack."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_MATERIAL_DIR = PROJECT_ROOT / "materials" / "Ti-6Al-4V"
DEFAULT_SOLVER = Path(__file__).with_name(
    "am_thermal_stress_upgraded.py"
)

TABLE_ARGS = {
    "k_table_solid": "--k-table-solid",
    "cp_table_solid": "--cp-table-solid",
    "k_table_powder": "--k-table-powder",
    "cp_table_powder": "--cp-table-powder",
    "k_table_liquid": "--k-table-liquid",
    "cp_table_liquid": "--cp-table-liquid",
    "E_table": "--E-table",
    "alpha_table": "--alpha-table",
    "poisson_table": "--poisson-table",
    "yield_table": "--yield-table",
    "hardening_table": "--hardening-table",
}

VALUE_ARGS = {
    "rho_solid": "--rho-solid",
    "rho_powder": "--rho-powder",
    "rho_liquid": "--rho-liquid",
    "cp_solid": "--cp-solid",
    "cp_powder": "--cp-powder",
    "cp_liquid": "--cp-liquid",
    "conductivity_solid": "--conductivity-solid",
    "conductivity_powder": "--conductivity-powder",
    "conductivity_liquid": "--conductivity-liquid",
    "solidus_temperature": "--solidus-temperature",
    "liquidus_temperature": "--liquidus-temperature",
    "latent_heat": "--latent-heat",
    "poisson": "--poisson",
    "mechanics_model": "--mechanics-model",
    "absorptivity": "--absorptivity",
    "emissivity": "--emissivity",
    "powder_mode": "--powder-mode",
    "mushy_mechanics_factor": "--mushy-mechanics-factor",
    "liquid_mechanics_factor": "--liquid-mechanics-factor",
}

REQUIRED_TABLE_KEYS = (
    "k_table_solid",
    "cp_table_solid",
    "E_table",
    "alpha_table",
    "yield_table",
)

PROTECTED_PASSTHROUGH_OPTIONS = {
    *TABLE_ARGS.values(),
    *VALUE_ARGS.values(),
    "--reset-plastic-on-melt",
    "--no-reset-plastic-on-melt",
    "--config",
    "--inp",
    "--output-dir",
    "--build-axis",
    "--base-side",
    "--scan-axis",
    "--scan-rotation-per-layer",
    "--jump-speed",
    "--scan-speed",
    "--laser-power",
    "--dt",
    "--beam-radius",
    "--source-depth",
    "--mechanics-every",
    "--thermal-output-every",
    "--mechanics-output-every",
    "--summary-every",
    "--cooling-steps",
    "--max-cells",
    "--powder-mode",
    "--layer-thickness",
    "--layers",
    "--hatch-spacing",
    "--hatch-lines-per-layer",
    "--auto-scan-steps-from-speed",
    "--scan-steps-per-layer",
}


def read_json_object(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(
            f"Material config must be a JSON object: {path}"
        )
    return data


def resolve_material_path(
    value: str,
    material_dir: Path,
    repo_root: Path,
) -> Path:
    """Resolve both historical project-relative and bundle-local paths."""
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path

    candidates = (
        repo_root / raw_path,
        repo_root.parent / raw_path,
        material_dir / raw_path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (material_dir / raw_path.name).resolve()


def material_config_path(
    material_dir: Path,
    explicit_config: Path | None,
) -> Path:
    if explicit_config is not None:
        expanded = explicit_config.expanduser()
        if not expanded.is_absolute():
            pack_relative = material_dir / expanded
            if pack_relative.is_file():
                return pack_relative.resolve()
        return expanded.resolve()
    return (
        material_dir / "ti64_material_config_initial.json"
    ).resolve()


def load_material_pack(
    material_dir: Path,
    config_path: Path,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict, dict[str, Path]]:
    material_dir = material_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if not material_dir.is_dir():
        raise FileNotFoundError(
            f"Material directory not found: {material_dir}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Material config not found: {config_path}"
        )

    config = read_json_object(config_path)
    resolved_tables = {}
    missing_keys = []
    missing_files = []
    for key in REQUIRED_TABLE_KEYS:
        if not config.get(key):
            missing_keys.append(key)
            continue
        table_path = resolve_material_path(
            str(config[key]),
            material_dir,
            repo_root,
        )
        resolved_tables[key] = table_path
        if not table_path.is_file():
            missing_files.append(str(table_path))
    if missing_keys:
        raise ValueError(
            "Material config is missing required table keys: "
            + ", ".join(missing_keys)
        )
    for key, value in config.items():
        if key in TABLE_ARGS and value:
            table_path = resolve_material_path(
                str(value),
                material_dir,
                repo_root,
            )
            resolved_tables[key] = table_path
            if not table_path.is_file():
                missing_files.append(str(table_path))
    if missing_files:
        raise FileNotFoundError(
            "Material table file(s) not found: "
            + ", ".join(sorted(set(missing_files)))
        )
    return config, resolved_tables


def material_args(
    config: dict,
    resolved_tables: dict[str, Path],
) -> list[str]:
    command = []
    for key, option in TABLE_ARGS.items():
        table_path = resolved_tables.get(key)
        if table_path is not None:
            command.extend([option, str(table_path)])

    for key, option in VALUE_ARGS.items():
        if key in config and config[key] is not None:
            command.extend([option, str(config[key])])

    if "reset_plastic_on_melt" in config:
        command.append(
            "--reset-plastic-on-melt"
            if config["reset_plastic_on_melt"]
            else "--no-reset-plastic-on-melt"
        )
    return command


def default_solver_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--inp",
        str(args.inp),
        "--output-dir",
        str(args.output_dir),
        "--build-axis",
        args.build_axis,
        "--base-side",
        args.base_side,
        "--scan-axis",
        args.scan_axis,
        "--scan-rotation-per-layer",
        str(args.scan_rotation_per_layer),
        "--jump-speed",
        str(args.jump_speed),
        "--scan-speed",
        str(args.scan_speed),
        "--laser-power",
        str(args.laser_power),
        "--dt",
        str(args.dt),
        "--beam-radius",
        str(args.beam_radius),
        "--source-depth",
        str(args.source_depth),
        "--mechanics-every",
        str(args.mechanics_every),
        "--thermal-output-every",
        str(args.thermal_output_every),
        "--mechanics-output-every",
        str(args.mechanics_output_every),
        "--summary-every",
        str(args.summary_every),
        "--cooling-steps",
        str(args.cooling_steps),
    ]
    if args.max_cells is not None:
        command.extend(["--max-cells", str(args.max_cells)])
    if args.powder_mode is not None:
        command.extend(["--powder-mode", args.powder_mode])
    if args.layer_thickness is not None:
        command.extend(
            ["--layer-thickness", str(args.layer_thickness)]
        )
    else:
        command.extend(["--layers", str(args.layers)])
    if args.hatch_spacing is not None:
        command.extend(["--hatch-spacing", str(args.hatch_spacing)])
    else:
        command.extend(
            [
                "--hatch-lines-per-layer",
                str(args.hatch_lines_per_layer),
            ]
        )
    if args.auto_scan_steps_from_speed:
        command.append("--auto-scan-steps-from-speed")
    else:
        command.extend(
            [
                "--scan-steps-per-layer",
                str(args.scan_steps_per_layer),
            ]
        )
    return command


def build_pythonpath(
    repo_root: Path,
    base_env: dict[str, str] | None = None,
) -> str:
    env = base_env if base_env is not None else os.environ
    entries = [
        str(repo_root / "legacy" / "v01"),
        str(repo_root),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def build_command(
    args: argparse.Namespace,
    passthrough: list[str],
) -> tuple[list[str], dict[str, str]]:
    validate_passthrough(passthrough)
    material_dir = args.material_dir.expanduser().resolve()
    config_path = material_config_path(
        material_dir,
        args.material_config,
    )
    config, resolved_tables = load_material_pack(
        material_dir,
        config_path,
        REPO_ROOT,
    )

    solver = args.solver.expanduser().resolve()
    if not solver.is_file():
        raise FileNotFoundError(
            f"v02 solver script not found: {solver}"
        )
    command = [args.python, str(solver)]
    command.extend(material_args(config, resolved_tables))
    command.extend(default_solver_args(args))
    command.extend(passthrough)

    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath(REPO_ROOT, env)
    return command, env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run legacy/v02/am_thermal_stress_upgraded.py with a local "
            "Ti-6Al-4V material pack."
        )
    )
    parser.add_argument(
        "--material-dir",
        type=Path,
        default=DEFAULT_MATERIAL_DIR,
    )
    parser.add_argument(
        "--material-config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--solver",
        type=Path,
        default=DEFAULT_SOLVER,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without running the solver.",
    )

    parser.add_argument(
        "--inp",
        type=Path,
        default=PROJECT_ROOT / "schema" / "0119_c3d4_only.inp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "v02_ti64_material_run",
    )
    parser.add_argument("--max-cells", type=int, default=500)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument(
        "--layer-thickness",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hatch-lines-per-layer",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--hatch-spacing",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--scan-steps-per-layer",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--auto-scan-steps-from-speed",
        action="store_true",
    )

    parser.add_argument(
        "--build-axis",
        choices=("x", "y", "z"),
        default="x",
    )
    parser.add_argument(
        "--base-side",
        choices=("min", "max"),
        default="min",
    )
    parser.add_argument(
        "--scan-axis",
        choices=("auto", "x", "y", "z"),
        default="y",
    )
    parser.add_argument(
        "--scan-rotation-per-layer",
        type=float,
        default=67.0,
    )
    parser.add_argument("--jump-speed", type=float, default=1.0)
    parser.add_argument("--scan-speed", type=float, default=0.5)
    parser.add_argument("--laser-power", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--beam-radius", type=float, default=5e-5)
    parser.add_argument("--source-depth", type=float, default=5e-5)
    parser.add_argument(
        "--powder-mode",
        choices=("powder", "void"),
        default=None,
    )

    parser.add_argument("--mechanics-every", type=int, default=20)
    parser.add_argument(
        "--thermal-output-every",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--mechanics-output-every",
        type=int,
        default=20,
    )
    parser.add_argument("--summary-every", type=int, default=1)
    parser.add_argument("--cooling-steps", type=int, default=20)
    return parser


def validate_passthrough(passthrough: list[str]) -> None:
    for token in passthrough:
        option = token.split("=", 1)[0]
        if option in PROTECTED_PASSTHROUGH_OPTIONS:
            raise ValueError(
                "passthrough arguments cannot override validated option "
                f"{option}"
            )


def parse_cli(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        separator = raw_args.index("--")
        launcher_args = raw_args[:separator]
        passthrough = raw_args[separator + 1 :]
    else:
        launcher_args = raw_args
        passthrough = []
    args = build_parser().parse_args(launcher_args)
    validate_passthrough(passthrough)
    return args, passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_cli(argv)
    command, env = build_command(args, passthrough)

    print("PYTHONPATH=" + env["PYTHONPATH"])
    print(shlex.join(command))
    if args.dry_run:
        return 0
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
