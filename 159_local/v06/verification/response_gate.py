"""Gate zero-input invariants and manufactured nonzero v06 smoke responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _finite_scalar(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return value


def evaluate_response_gate(*, laser_power_w, ambient_k, audit, xrd, ledger_rows):
    """Require every coupled output to respond in a nonzero manufactured run."""
    laser_power = _finite_scalar("laser_power_w", laser_power_w)
    ambient = _finite_scalar("ambient_k", ambient_k)
    if laser_power < 0.0:
        raise ValueError("laser_power_w must be nonnegative")
    if laser_power == 0.0:
        return {
            "schema_version": "v06.response-gate/1",
            "claim_level": "zero_input_invariant_smoke_only",
            "required": False,
            "valid": True,
            "status": "zero_input_invariant_smoke",
            "failed_checks": [],
        }

    try:
        maximum_temperature = _finite_scalar(
            "maximum_temperature",
            audit["transient"]["maximum_temperature"],
        )
        constrained_stress = _finite_scalar(
            "constrained_stress",
            audit["latest_constrained"]["stress"]["quality_filtered_max"],
        )
        release_stress = _finite_scalar(
            "release_stress",
            audit["release"]["stress"]["quality_filtered_max"],
        )
        release_displacement = _finite_scalar(
            "release_displacement",
            audit["release"]["displacement_norm"]["maximum"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("audit is missing manufactured-response fields") from error
    if not isinstance(ledger_rows, list) or not ledger_rows:
        raise ValueError("thermal ledger must contain at least one row")
    deposited = sum(
        _finite_scalar("laser_deposited_j", row.get("laser_deposited_j"))
        for row in ledger_rows
        if isinstance(row, dict)
    )
    gauges = xrd.get("gauges") if isinstance(xrd, dict) else None
    xrd_values = (
        [
            _finite_scalar("predicted_microstrain", gauge.get("predicted_microstrain"))
            for gauge in gauges
            if isinstance(gauge, dict) and gauge.get("status") == "ok"
        ]
        if isinstance(gauges, list)
        else []
    )
    checks = {
        "deposited_laser_energy": deposited > 0.0,
        "temperature_response": maximum_temperature > ambient,
        "constrained_stress": constrained_stress > 0.0,
        "release_stress": release_stress > 0.0,
        "release_displacement": release_displacement > 0.0,
        "xrd_operator_response": bool(xrd_values)
        and any(abs(value) > 0.0 for value in xrd_values),
    }
    failed = [name for name, valid in checks.items() if not valid]
    return {
        "schema_version": "v06.response-gate/1",
        "claim_level": "manufactured_nonzero_response_smoke_only",
        "required": True,
        "valid": not failed,
        "status": "passed" if not failed else "failed",
        "failed_checks": failed,
        "checks": checks,
        "values": {
            "laser_power_w": laser_power,
            "laser_deposited_j": deposited,
            "maximum_temperature_k": maximum_temperature,
            "ambient_k": ambient,
            "constrained_stress_max_pa": constrained_stress,
            "release_stress_max_pa": release_stress,
            "release_displacement_max_m": release_displacement,
            "maximum_absolute_xrd_microstrain": (
                max(abs(value) for value in xrd_values) if xrd_values else None
            ),
        },
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "used_config": args.run_dir / "used_config.json",
        "run_audit": args.run_dir / "v06_run_audit.json",
        "xrd_prediction": args.run_dir / "xrd_operator_smoke.json",
        "thermal_ledger": args.run_dir / "thermal_energy_ledger.jsonl",
    }
    loaded = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
        if name != "thermal_ledger"
    }
    ledger_rows = [
        json.loads(line)
        for line in paths["thermal_ledger"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = loaded["used_config"]
    report = evaluate_response_gate(
        laser_power_w=config["laser_power"],
        ambient_k=config["ambient"],
        audit=loaded["run_audit"],
        xrd=loaded["xrd_prediction"],
        ledger_rows=ledger_rows,
    )
    report["inputs"] = {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "required": report["required"],
        "valid": report["valid"],
        "status": report["status"],
    }))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
