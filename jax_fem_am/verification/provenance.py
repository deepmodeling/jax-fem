"""Create a content-addressed provenance manifest for v06 runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


SCHEMA_VERSION = "jax_fem_am.verification.provenance/2"
COMPLETE_CLAIM_LEVEL = "numerical_smoke_only"
INCOMPLETE_CLAIM_LEVEL = "forensic_manifest_only"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_record(roots: Iterable[Path], repo_root: Path) -> dict[str, Any]:
    """Return a deterministic digest of Python runtime source dependencies."""
    repo_root = Path(repo_root).resolve()
    resolved_roots = [Path(root).resolve() for root in roots]
    missing_roots = [str(root) for root in resolved_roots if not root.is_dir()]
    files = sorted(
        {
            path.resolve()
            for root in resolved_roots
            if root.is_dir()
            for path in root.rglob("*.py")
            if path.is_file()
        },
        key=str,
    )
    digest = hashlib.sha256()
    for path in files:
        try:
            name = path.relative_to(repo_root).as_posix()
        except ValueError:
            name = str(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256(path\\0sha256(content)\\n)",
        "sha256": digest.hexdigest(),
        "python_file_count": len(files),
        "roots": [str(root) for root in resolved_roots],
        "missing_roots": missing_roots,
    }


def _file_record(path: Path, repo_root: Optional[Path] = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if repo_root is not None:
        try:
            record["repo_relative_path"] = str(
                resolved.relative_to(Path(repo_root).resolve())
            )
        except ValueError:
            pass
    return record


def _run_git(repo_root: Path, *arguments: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    head = _run_git(repo_root, "rev-parse", "HEAD")
    if head is None:
        return {"available": False}
    branch = _run_git(repo_root, "branch", "--show-current") or None
    status = _run_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    entries = status.splitlines() if status else []
    return {
        "available": True,
        "head": head,
        "branch": branch,
        "dirty": bool(entries),
        "status_porcelain": entries,
    }


def _package_versions(names: Iterable[str]) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def xrd_report_is_valid(report: dict[str, Any]) -> bool:
    gauges = report.get("gauges")
    return bool(
        report.get("claim_level") == "measurement_operator_prediction_only"
        and isinstance(gauges, list)
        and gauges
        and all(
            isinstance(gauge, dict) and gauge.get("status") == "ok"
            for gauge in gauges
        )
    )


def _record_matches_file(record: Any, path: Path) -> bool:
    if not isinstance(record, dict) or not path.is_file():
        return False
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str):
        return False
    try:
        return sha256_file(path) == expected_hash
    except OSError:
        return False


def audit_artifacts_match(report: dict[str, Any], run_dir: Path) -> bool:
    """Verify that an audit still describes every current transient VTU."""
    if not isinstance(report, dict):
        return False
    run_dir = Path(run_dir)
    current_steps = sorted(run_dir.glob("step_*.vtu"))
    transient = report.get("transient")
    if not current_steps or not isinstance(transient, dict):
        return False
    records = transient.get("steps")
    if not isinstance(records, list) or transient.get("step_count") != len(
        current_steps
    ):
        return False
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        name = record.get("name")
        if not isinstance(name, str) or Path(name).name != name or name in by_name:
            return False
        by_name[name] = record
    if set(by_name) != {path.name for path in current_steps}:
        return False
    if not all(_record_matches_file(by_name[path.name], path) for path in current_steps):
        return False

    latest_source = report.get("latest_constrained", {}).get("source")
    release_source = report.get("release", {}).get("source")
    latest = current_steps[-1]
    release = run_dir / "release.vtu"
    return bool(
        isinstance(latest_source, dict)
        and Path(str(latest_source.get("path", ""))).name == latest.name
        and _record_matches_file(latest_source, latest)
        and isinstance(release_source, dict)
        and Path(str(release_source.get("path", ""))).name == release.name
        and _record_matches_file(release_source, release)
    )


def xrd_inputs_match(
    report: dict[str, Any], run_dir: Path, protocol_path: Path
) -> bool:
    """Verify the XRD report's VTU and protocol content hashes."""
    if not isinstance(report, dict):
        return False
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        return False
    vtu_record = inputs.get("vtu")
    protocol_record = inputs.get("protocol")
    if not isinstance(vtu_record, dict) or not isinstance(protocol_record, dict):
        return False
    vtu_name = Path(str(vtu_record.get("path", ""))).name
    vtu_path = Path(run_dir) / vtu_name
    protocol_path = Path(protocol_path)
    return bool(
        vtu_name.startswith("step_")
        and vtu_name.endswith(".vtu")
        and _record_matches_file(vtu_record, vtu_path)
        and Path(str(protocol_record.get("path", ""))).name
        == protocol_path.name
        and _record_matches_file(protocol_record, protocol_path)
    )


def xrd_gauge_ids_match(report: dict[str, Any], protocol_path: Path) -> bool:
    """Require a one-to-one, ordered gauge mapping to the current protocol."""
    try:
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    report_gauges = report.get("gauges") if isinstance(report, dict) else None
    protocol_gauges = protocol.get("gauges") if isinstance(protocol, dict) else None
    if not isinstance(report_gauges, list) or not isinstance(protocol_gauges, list):
        return False
    report_ids = [gauge.get("id") for gauge in report_gauges if isinstance(gauge, dict)]
    protocol_ids = [
        gauge.get("id") for gauge in protocol_gauges if isinstance(gauge, dict)
    ]
    return bool(
        report_ids
        and len(report_ids) == len(report_gauges)
        and len(protocol_ids) == len(protocol_gauges)
        and all(isinstance(gauge_id, str) and gauge_id for gauge_id in report_ids)
        and len(set(report_ids)) == len(report_ids)
        and report_ids == protocol_ids
    )


def thermal_ledger_is_valid(
    summary: dict[str, Any], ledger_path: Path, audit: dict[str, Any]
) -> bool:
    """Validate ledger completeness and every per-step evidence gate."""
    if not isinstance(summary, dict) or not isinstance(audit, dict):
        return False
    try:
        lines = Path(ledger_path).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    audit_count = audit.get("transient", {}).get("step_count")
    expected = summary.get("expected_step_count")
    recorded = summary.get("recorded_step_count")
    if (
        summary.get("schema_version")
        != "v06.thermal-energy-ledger-summary/1"
        or summary.get("complete") is not True
        or summary.get("all_balance_steps_within_tolerance") is not True
        or summary.get("all_assembly_identities_within_tolerance") is not True
        or summary.get("all_pre_solve_state_overrides_within_tolerance")
        is not True
        or summary.get("all_temperature_invariants_valid") is not True
        or not isinstance(audit_count, int)
        or expected != audit_count
        or recorded != audit_count
        or len(rows) != audit_count
    ):
        return False
    return all(
        isinstance(row, dict)
        and row.get("schema_version") == "v06.thermal-energy-ledger-step/1"
        and row.get("step_index") == index
        and row.get("balance_within_solver_tolerance") is True
        and row.get("assembly_identity_within_tolerance") is True
        and row.get("state_override_within_tolerance") is True
        and row.get("temperature_invariants_valid") is True
        for index, row in enumerate(rows)
    )


def response_gate_is_valid(report: dict[str, Any], run_dir: Path) -> bool:
    """Verify the response gate and its hashes against current run evidence."""
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "v06.response-gate/1"
        or report.get("valid") is not True
    ):
        return False
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        return False
    role_names = {
        "used_config": "used_config.json",
        "run_audit": "v06_run_audit.json",
        "xrd_prediction": "xrd_operator_smoke.json",
        "thermal_ledger": "thermal_energy_ledger.jsonl",
    }
    run_dir = Path(run_dir)
    hashes_match = all(
        isinstance(inputs.get(role), dict)
        and Path(str(inputs[role].get("path", ""))).name == name
        and _record_matches_file(inputs[role], run_dir / name)
        for role, name in role_names.items()
    )
    if not hashes_match:
        return False
    try:
        config = json.loads(
            (run_dir / "used_config.json").read_text(encoding="utf-8")
        )
        laser_power = float(config["laser_power"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(laser_power) or laser_power < 0.0:
        return False
    expected_required = laser_power > 0.0
    if report.get("required") is not expected_required:
        return False
    if expected_required:
        return bool(
            report.get("status") == "passed"
            and report.get("claim_level")
            == "manufactured_nonzero_response_smoke_only"
        )
    return bool(
        report.get("status") == "zero_input_invariant_smoke"
        and report.get("claim_level") == "zero_input_invariant_smoke_only"
    )


def _material_table_records(
    used_config: dict[str, Any], work_root: Path, repo_root: Path
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for key, value in sorted(used_config.items()):
        if "_table" not in key or not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = work_root / path
        if path.is_file():
            records[key] = _file_record(path, repo_root)
        else:
            records[key] = {"path": str(path.resolve()), "missing": True}
    return records


def build_manifest(
    *,
    repo_root: Path,
    run_dir: Path,
    mesh: Path,
    material_config: Path,
    label: str,
    work_root: Optional[Path] = None,
    xrd_protocol: Optional[Path] = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    work_root = (
        Path(work_root).expanduser().resolve()
        if work_root is not None
        else repo_root.parent
    )

    inputs = {
        "mesh": _file_record(mesh, repo_root),
        "material_config": _file_record(material_config, repo_root),
    }
    if xrd_protocol is not None:
        inputs["xrd_protocol"] = _file_record(xrd_protocol, repo_root)
    used_config_path = run_dir / "used_config.json"
    used_config: dict[str, Any] = {}
    used_config_valid = False
    if used_config_path.is_file():
        inputs["used_config"] = _file_record(used_config_path, repo_root)
        try:
            with used_config_path.open("r", encoding="utf-8") as stream:
                parsed_config = json.load(stream)
            if isinstance(parsed_config, dict):
                used_config = parsed_config
                used_config_valid = True
        except (json.JSONDecodeError, OSError):
            used_config = {}
    command_path = run_dir / "solver_command.txt"
    if command_path.is_file():
        inputs["solver_command"] = _file_record(command_path, repo_root)

    source_paths = {
        "v06_driver": repo_root / "159_local/v06/driver.py",
        "v06_j2_kernel": repo_root / "159_local/v06/mechanics/j2.py",
        "v06_mechanical_lifecycle": repo_root
        / "159_local/v06/mechanics/lifecycle.py",
        "am_material_validation": repo_root
        / "jax_fem_am/materials/material_validation.py",
        "v06_runner": repo_root / "159_local/v06/run_smoke.sh",
        "v06_nonzero_runner": repo_root
        / "159_local/v06/run_nonzero_smoke.sh",
        "am_provenance": repo_root / "jax_fem_am/verification/provenance.py",
        "v06_run_audit": repo_root
        / "jax_fem_am/verification/run_audit.py",
        "v06_mesh_quality": repo_root
        / "jax_fem_am/verification/mesh_quality.py",
        "v06_weighted_statistics": repo_root
        / "jax_fem_am/verification/weighted.py",
        "v06_thermal_balance": repo_root
        / "jax_fem_am/verification/thermal_balance.py",
        "v06_thermal_ledger": repo_root
        / "jax_fem_am/verification/thermal_ledger.py",
        "v06_response_gate": repo_root
        / "jax_fem_am/verification/response_gate.py",
        "v06_xrd_geometry": repo_root / "jax_fem_am/verification/xrd.py",
        "v06_xrd_vtu": repo_root / "jax_fem_am/verification/xrd_vtu.py",
        "v04_runtime_wrapper": repo_root
        / "159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py",
        "v03_base_solver": repo_root
        / "159_local/v03/am_thermal_stress_macro_intersection_mech100.py",
        "v01_mesh_reader": repo_root
        / "159_local/v01/inp_initial_guess_smoke.py",
    }
    solver_sources = {
        role: _file_record(path, repo_root)
        for role, path in source_paths.items()
        if path.is_file()
    }
    runtime_source_tree = source_tree_record(
        [
            repo_root / "jax_fem",
            repo_root / "159_local/v01",
            repo_root / "159_local/v03",
            repo_root / "159_local/v04",
            repo_root / "159_local/v06",
            repo_root / "jax_fem_am",
        ],
        repo_root,
    )
    runtime_source_tree_complete = bool(
        runtime_source_tree["python_file_count"] > 0
        and not runtime_source_tree["missing_roots"]
    )

    artifact_paths = {
        "profile": run_dir / "profile.json",
        "v06_run_audit": run_dir / "v06_run_audit.json",
        "release_vtu": run_dir / "release.vtu",
        "path_used": run_dir / "path_used.csv",
        "xrd_operator_smoke": run_dir / "xrd_operator_smoke.json",
        "thermal_energy_ledger": run_dir / "thermal_energy_ledger.jsonl",
        "thermal_energy_ledger_summary": run_dir
        / "thermal_energy_ledger_summary.json",
        "v06_response_gate": run_dir / "v06_response_gate.json",
    }
    artifacts = {
        role: _file_record(path, repo_root)
        for role, path in artifact_paths.items()
        if path.is_file()
    }

    output_inventory = {
        path.name: _file_record(path, repo_root)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "v06_manifest.json"
    }

    audit_data: dict[str, Any] = {}
    audit_path = artifact_paths["v06_run_audit"]
    if audit_path.is_file():
        try:
            with audit_path.open("r", encoding="utf-8") as stream:
                audit_data = json.load(stream)
        except (json.JSONDecodeError, OSError):
            audit_data = {}
    profile_data: dict[str, Any] = {}
    profile_path = artifact_paths["profile"]
    if profile_path.is_file():
        try:
            with profile_path.open("r", encoding="utf-8") as stream:
                profile_data = json.load(stream)
        except (json.JSONDecodeError, OSError):
            profile_data = {}
    xrd_data: dict[str, Any] = {}
    xrd_path = artifact_paths["xrd_operator_smoke"]
    if xrd_path.is_file():
        try:
            with xrd_path.open("r", encoding="utf-8") as stream:
                xrd_data = json.load(stream)
        except (json.JSONDecodeError, OSError):
            xrd_data = {}
    thermal_summary_data: dict[str, Any] = {}
    thermal_summary_path = artifact_paths["thermal_energy_ledger_summary"]
    if thermal_summary_path.is_file():
        try:
            with thermal_summary_path.open("r", encoding="utf-8") as stream:
                thermal_summary_data = json.load(stream)
        except (json.JSONDecodeError, OSError):
            thermal_summary_data = {}
    response_gate_data: dict[str, Any] = {}
    response_gate_path = artifact_paths["v06_response_gate"]
    if response_gate_path.is_file():
        try:
            with response_gate_path.open("r", encoding="utf-8") as stream:
                response_gate_data = json.load(stream)
        except (json.JSONDecodeError, OSError):
            response_gate_data = {}

    required_input_roles = {
        "mesh",
        "material_config",
        "solver_command",
        "used_config",
        "xrd_protocol",
    }
    required_artifact_roles = {
        "profile",
        "v06_run_audit",
        "release_vtu",
        "path_used",
        "xrd_operator_smoke",
        "thermal_energy_ledger",
        "thermal_energy_ledger_summary",
        "v06_response_gate",
    }
    required_source_roles = set(source_paths)
    audit_valid = bool(
        audit_data.get("transient", {}).get("all_steps_valid") is True
        and audit_data.get("release", {}).get("valid") is True
    )
    profile_identifies_v06 = bool(
        profile_data.get("meta", {}).get("v06_constitutive_model")
    )
    xrd_operator_valid = xrd_report_is_valid(xrd_data)
    audit_artifacts_current = audit_artifacts_match(audit_data, run_dir)
    xrd_inputs_current = bool(
        xrd_protocol is not None
        and xrd_inputs_match(xrd_data, run_dir, Path(xrd_protocol))
    )
    xrd_gauges_current = bool(
        xrd_protocol is not None
        and xrd_gauge_ids_match(xrd_data, Path(xrd_protocol))
    )
    thermal_ledger_valid = thermal_ledger_is_valid(
        thermal_summary_data,
        artifact_paths["thermal_energy_ledger"],
        audit_data,
    )
    response_gate_valid = response_gate_is_valid(response_gate_data, run_dir)
    has_step_vtu = any(path.name.startswith("step_") for path in run_dir.glob("*.vtu"))
    missing_inputs = sorted(required_input_roles.difference(inputs))
    missing_artifacts = sorted(required_artifact_roles.difference(artifacts))
    missing_sources = sorted(required_source_roles.difference(solver_sources))
    complete = bool(
        not missing_inputs
        and not missing_artifacts
        and not missing_sources
        and used_config_valid
        and runtime_source_tree_complete
        and has_step_vtu
        and audit_valid
        and audit_artifacts_current
        and profile_identifies_v06
        and xrd_operator_valid
        and xrd_inputs_current
        and xrd_gauges_current
        and thermal_ledger_valid
        and response_gate_valid
    )
    claim_level = COMPLETE_CLAIM_LEVEL if complete else INCOMPLETE_CLAIM_LEVEL

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "claim_level": claim_level,
        "run_status": "complete_valid" if complete else "incomplete_or_invalid",
        "claim_note": (
            "A complete manifest records a content-linked numerical smoke run; "
            "it never establishes agreement with a physical experiment."
        ),
        "completeness": {
            "complete": complete,
            "missing_inputs": missing_inputs,
            "missing_artifacts": missing_artifacts,
            "missing_sources": missing_sources,
            "used_config_valid": used_config_valid,
            "runtime_source_tree_complete": runtime_source_tree_complete,
            "has_step_vtu": has_step_vtu,
            "audit_valid": audit_valid,
            "audit_artifacts_match": audit_artifacts_current,
            "profile_identifies_v06": profile_identifies_v06,
            "xrd_operator_valid": xrd_operator_valid,
            "xrd_inputs_match": xrd_inputs_current,
            "xrd_gauge_ids_match": xrd_gauges_current,
            "thermal_ledger_valid": thermal_ledger_valid,
            "response_gate_valid": response_gate_valid,
        },
        "model_boundary": {
            "time_loop": "v03",
            "performance_wrapper": "v04",
            "constitutive_state_adapter": "v06",
            "v05_runtime_dependency": False,
        },
        "repository": git_snapshot(repo_root),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": _package_versions(
                ("jax", "jaxlib", "numpy", "scipy", "meshio")
            ),
        },
        "inputs": inputs,
        "material_tables": _material_table_records(
            used_config, work_root, repo_root
        ),
        "solver_sources": solver_sources,
        "runtime_source_tree": runtime_source_tree,
        "artifacts": artifacts,
        "output_inventory": output_inventory,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a content-addressed v06 run manifest."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--material-config", type=Path, required=True)
    parser.add_argument("--xrd-protocol", type=Path)
    parser.add_argument("--label", default="v06-run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manifest(
        repo_root=args.repo_root,
        run_dir=args.run_dir,
        mesh=args.mesh,
        material_config=args.material_config,
        label=args.label,
        work_root=args.work_root,
        xrd_protocol=args.xrd_protocol,
    )
    output = args.output or (args.run_dir / "v06_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(output),
        "claim_level": manifest["claim_level"],
        "run_status": manifest["run_status"],
    }))
    if args.require_complete and manifest["run_status"] != "complete_valid":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
