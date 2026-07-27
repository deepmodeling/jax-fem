"""Fail-closed gate for the currently approved Kaess material-config bytes.

The G0-v1 ``external_sha256`` contract covers the top-level JSON only. It does
not promote or validate an unapproved dependency-CSV bundle; G0-v2 must add an
explicit manifest mode before such a bundle can pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


class MaterialIdentityError(ValueError):
    """Raised when runtime material bytes are not the approved G0 input."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, label: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterialIdentityError(
            f"{label} is not a readable JSON object: {path}"
        ) from exc
    if not isinstance(document, dict):
        raise MaterialIdentityError(
            f"{label} must contain a JSON object: {path}"
        )
    return document


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterialIdentityError(message)


def validate_material_identity(
    material_config_path,
    *,
    parity_config_path,
    approval_record_path,
) -> dict:
    """Validate exact runtime material bytes against the approved G0 chain."""
    material_path = Path(material_config_path).resolve()
    parity_path = Path(parity_config_path).resolve()
    approval_path = Path(approval_record_path).resolve()
    _require(
        material_path.is_file(),
        f"material config is not a file: {material_path}",
    )

    parity = _load_object(parity_path, "paper-parity config")
    approval = _load_object(approval_path, "G0 approval record")
    _require(
        parity.get("schema_version") == "kaess.paper-parity-config/1",
        "paper-parity config schema is not supported",
    )
    _require(
        approval.get("schema_version") == "kaess.g0-approval/1",
        "G0 approval schema is not supported",
    )
    _require(
        parity.get("status") == "approved",
        "paper-parity config is not approved",
    )
    _require(
        approval.get("gate_id") == "G0"
        and approval.get("decision") == "approved",
        "G0 approval record is not approved",
    )
    _require(
        parity.get("protocol_id") == approval.get("protocol_id"),
        "paper-parity and approval protocol identities differ",
    )

    expected_approval_hash = (
        parity.get("approval", {}).get("approval_record_sha256")
    )
    actual_approval_hash = _sha256(approval_path)
    _require(
        expected_approval_hash == actual_approval_hash,
        "G0 approval record SHA-256 does not match paper-parity config",
    )

    parity_freeze = dict(parity.get("material_freeze") or {})
    _require(
        parity_freeze.get("status") == "approved",
        "paper-parity material freeze is not approved",
    )
    parity_freeze.pop("status", None)
    if parity_freeze.get("path") is None:
        parity_freeze.pop("path", None)
    approval_freeze = approval.get("material_freeze")
    _require(
        parity_freeze == approval_freeze,
        "paper-parity and G0 approval material freezes differ",
    )
    _require(
        parity_freeze.get("mode") == "external_sha256",
        "unsupported G0 material-freeze mode",
    )
    _require(
        parity_freeze.get("environment_variable")
        == "KAESS_MATERIAL_CONFIG",
        "G0 material environment variable must be KAESS_MATERIAL_CONFIG",
    )

    expected_material_hash = parity_freeze.get("sha256")
    actual_material_hash = _sha256(material_path)
    _require(
        actual_material_hash == expected_material_hash,
        "material config SHA-256 does not match the approved G0 identity",
    )
    return {
        "status": "pass",
        "protocol_id": parity["protocol_id"],
        "material_config": str(material_path),
        "material_sha256": actual_material_hash,
        "environment_variable": parity_freeze["environment_variable"],
        "approval_record": str(approval_path),
        "approval_record_sha256": actual_approval_hash,
        "parity_config": str(parity_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Kaess material config against approved G0 bytes"
    )
    parser.add_argument("--material-config", type=Path, required=True)
    parser.add_argument("--parity-config", type=Path, required=True)
    parser.add_argument("--approval-record", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_material_identity(
            args.material_config,
            parity_config_path=args.parity_config,
            approval_record_path=args.approval_record,
        )
    except MaterialIdentityError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
