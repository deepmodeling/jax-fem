import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = (
    REPO_ROOT / "cases" / "kaess_2023" / "run_kaess_phase1.sh",
    REPO_ROOT / "cases" / "kaess_2023" / "run_kaess_phase2.sh",
)
LAUNCHER_GUARD = (
    REPO_ROOT / "cases" / "kaess_2023" / "launcher_guards.sh"
)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approved_fixture(tmp_path):
    material = tmp_path / "material.json"
    material.write_text('{"material": "approved"}\n', encoding="utf-8")
    freeze = {
        "mode": "external_sha256",
        "environment_variable": "KAESS_MATERIAL_CONFIG",
        "source_manifest_evidence_id": "formal-material-config",
        "sha256": _sha256(material),
    }
    approval = tmp_path / "g0-approval.json"
    _write_json(
        approval,
        {
            "schema_version": "kaess.g0-approval/1",
            "gate_id": "G0",
            "protocol_id": "kaess-test",
            "decision": "approved",
            "material_freeze": freeze,
        },
    )
    parity = tmp_path / "paper-parity-config.yaml"
    _write_json(
        parity,
        {
            "schema_version": "kaess.paper-parity-config/1",
            "protocol_id": "kaess-test",
            "status": "approved",
            "approval": {
                "approval_record_sha256": _sha256(approval),
            },
            "material_freeze": {
                **freeze,
                "path": None,
                "status": "approved",
            },
        },
    )
    return material, parity, approval


def test_material_identity_gate_accepts_only_the_g0_approved_bytes(tmp_path):
    from jax_fem_am.verification.material_identity import (
        validate_material_identity,
    )

    material, parity, approval = _approved_fixture(tmp_path)

    result = validate_material_identity(
        material,
        parity_config_path=parity,
        approval_record_path=approval,
    )

    assert result["status"] == "pass"
    assert result["material_sha256"] == _sha256(material)
    assert result["environment_variable"] == "KAESS_MATERIAL_CONFIG"


def test_material_identity_gate_rejects_changed_material_bytes(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
        validate_material_identity,
    )

    material, parity, approval = _approved_fixture(tmp_path)
    material.write_text('{"material": "changed"}\n', encoding="utf-8")

    with pytest.raises(MaterialIdentityError, match="SHA-256"):
        validate_material_identity(
            material,
            parity_config_path=parity,
            approval_record_path=approval,
        )


def test_material_identity_gate_rejects_stale_approval_binding(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
        validate_material_identity,
    )

    material, parity, approval = _approved_fixture(tmp_path)
    approval.write_text(
        approval.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MaterialIdentityError, match="approval.*SHA-256"):
        validate_material_identity(
            material,
            parity_config_path=parity,
            approval_record_path=approval,
        )


def test_material_identity_gate_rejects_unapproved_material_freeze(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
        validate_material_identity,
    )

    material, parity, approval = _approved_fixture(tmp_path)
    document = json.loads(parity.read_text(encoding="utf-8"))
    document["material_freeze"]["status"] = "pending_review"
    _write_json(parity, document)

    with pytest.raises(MaterialIdentityError, match="freeze is not approved"):
        validate_material_identity(
            material,
            parity_config_path=parity,
            approval_record_path=approval,
        )


def test_formal_launchers_use_the_g0_environment_and_identity_gate():
    guard = LAUNCHER_GUARD.read_text(encoding="utf-8")
    assert "kaess_validate_material_identity()" in guard
    assert 'cd "${repo_root}"' in guard
    assert "-m jax_fem_am.verification.material_identity" in guard
    assert "kaess_celsius_to_kelvin()" in guard

    for launcher_path in LAUNCHERS:
        launcher = launcher_path.read_text(encoding="utf-8")

        assert "KAESS_MATERIAL_CONFIG" in launcher
        assert "kaess_celsius_to_kelvin" in launcher
        assert "kaess_validate_material_identity" in launcher
        assert "inputs/paper-parity-config.yaml" in launcher
        assert "inputs/g0-approval.json" in launcher
        assert 'source "${SCRIPT_DIR}/launcher_guards.sh"' in launcher
        assert 'MATERIAL_CONFIG="$(realpath -e "${MATERIAL_CONFIG}")"' in launcher
        assert '"${KAESS_EXTRA_ARGV[@]}"' in launcher


def test_phase2_does_not_inject_unapproved_powder_hardening():
    launcher = LAUNCHERS[1].read_text(encoding="utf-8")

    assert 'POWDER_SOLID_HARDENING:-0}' in launcher
    assert 'POWDER_SOLID_HARDENING:-1e7}' not in launcher


def _run_extra_args_guard(extra_args, *, allow_layers=False):
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'set -e; source "$1"; '
                'kaess_parse_safe_extra_args "$2" "$3"; '
                'printf "%s\\n" "${KAESS_EXTRA_ARGV[@]}"'
            ),
            "bash",
            str(LAUNCHER_GUARD),
            extra_args,
            "1" if allow_layers else "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_launcher_extra_args_guard_accepts_only_frozen_safe_overrides():
    result = _run_extra_args_guard(
        "--mechanics-residual-only-check "
        "--xla-pardiso-mode phase23 "
        "--summary-every 25"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "--mechanics-residual-only-check",
        "--xla-pardiso-mode",
        "phase23",
        "--summary-every",
        "25",
    ]

    assert _run_extra_args_guard(
        "--layers 3",
        allow_layers=True,
    ).returncode == 0
    assert _run_extra_args_guard("--layers 3").returncode != 0


@pytest.mark.parametrize(
    "extra_args",
    [
        "--summary-every 0",
        "--summary-every=0",
        "--summary-every -1",
        "--thermal-output-every -1",
        "--mechanics-output-every=-1",
        "--xla-pardiso-mode unsafe",
    ],
)
def test_launcher_extra_args_guard_rejects_invalid_safe_option_values(
    extra_args,
):
    result = _run_extra_args_guard(extra_args, allow_layers=True)

    assert result.returncode != 0


@pytest.mark.parametrize(
    "extra_args",
    [
        "--config candidate.json",
        "--config=candidate.json",
        "--powder-solid-hardening 1e7",
        "--phase-history-model legacy_reset",
        "--reset-plastic-on-melt",
        "--k-table-powder hacked.csv",
        '--summary-every "25 --config candidate.json"',
    ],
)
def test_launcher_extra_args_guard_rejects_material_and_phase_overrides(
    extra_args,
):
    result = _run_extra_args_guard(extra_args, allow_layers=True)

    assert result.returncode != 0
    assert "kaess launcher:" in result.stderr


def _run_path_args_guard(path_args):
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'set -e; source "$1"; '
                'kaess_parse_safe_path_args "$2"; '
                'printf "%s\\n" "${KAESS_PATH_ARGV[@]}"'
            ),
            "bash",
            str(LAUNCHER_GUARD),
            path_args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_path_args_guard_accepts_generator_parameters_but_not_output():
    accepted = _run_path_args_guard(
        "--power 250 --speed=0.850 --rotation-deg -67 "
        "--sample-step 5e-5 --layers 2"
    )

    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.splitlines() == [
        "--power",
        "250",
        "--speed=0.850",
        "--rotation-deg",
        "-67",
        "--sample-step",
        "5e-5",
        "--layers",
        "2",
    ]

    for unsafe in (
        "--output /tmp/overwrite",
        "--output=/tmp/overwrite",
        '--power "250 --output /tmp/overwrite"',
        "--unknown-option 1",
    ):
        rejected = _run_path_args_guard(unsafe)
        assert rejected.returncode != 0
        assert "unsupported PATH_ARGS option" in rejected.stderr


def test_celsius_conversion_treats_environment_value_as_data(tmp_path):
    marker = tmp_path / "plate-temp-code-ran"
    payload = (
        "150); __import__('pathlib').Path("
        f"{str(marker)!r}).write_text('unsafe'); print(273.15 + (150"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1"; kaess_celsius_to_kelvin "$2" "$3"',
            "bash",
            str(LAUNCHER_GUARD),
            sys.executable,
            payload,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not marker.exists()


def test_identity_gate_cannot_be_shadowed_by_caller_checkout(tmp_path):
    material, parity, approval = _approved_fixture(tmp_path)
    shadow_checkout = tmp_path / "shadow-checkout"
    shadow_module = (
        shadow_checkout
        / "jax_fem_am"
        / "verification"
        / "material_identity.py"
    )
    shadow_module.parent.mkdir(parents=True)
    (shadow_checkout / "jax_fem_am" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (shadow_module.parent / "__init__.py").write_text("", encoding="utf-8")
    shadow_marker = tmp_path / "shadow-gate-ran"
    shadow_module.write_text(
        "from pathlib import Path\n"
        f"Path({str(shadow_marker)!r}).write_text('unsafe')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'set -e; cd "$1"; source "$2"; '
                'kaess_validate_material_identity "$3" "$4" "$5" "$6" "$7"'
            ),
            "bash",
            str(shadow_checkout),
            str(LAUNCHER_GUARD),
            str(REPO_ROOT),
            sys.executable,
            str(material),
            str(parity),
            str(approval),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not shadow_marker.exists()
