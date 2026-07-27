from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = (
    REPO_ROOT / "cases" / "kaess_2023" / "candidates" / "g0-v2-t018"
)
MANIFEST_PATH = CANDIDATE_ROOT / "material-bundle-manifest.json"
REQUEST_PATH = CANDIDATE_ROOT / "g0-reapproval-request.json"
MATERIAL_CONFIG_PATH = (
    CANDIDATE_ROOT
    / "ss316l_material_config_kaess.g0-v2-candidate.json"
)
PAPER_PARITY_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "inputs" / "paper-parity-config.yaml"
)
CANONICAL_G0_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "inputs" / "g0-approval.json"
)
APPROVAL_PATH = (
    REPO_ROOT
    / "cases"
    / "kaess_2023"
    / "inputs"
    / "g0-v2-material-conditional-approval.json"
)
GITATTRIBUTES_PATH = REPO_ROOT / ".gitattributes"
CONTRACT_ROOT = (
    REPO_ROOT / "specs" / "001-kaess-paper-reproduction" / "contracts"
)

MANIFEST_SCHEMA = "material-bundle-manifest.schema.json"
REQUEST_SCHEMA = "g0-reapproval-request.schema.json"
APPROVAL_SCHEMA = "g0-v2-material-approval.schema.json"

PERMITTED_USE_SCOPES = [
    "g1_cpu_validation",
    "g2_cpu_validation",
    "sensitivity_analysis",
]
PROHIBITED_USE_SCOPES = ["formal_run", "promotion"]
EXCLUDED_PENDING_SEMANTICS = [
    "abaqus_total_mean_thermal_expansion_runtime_implementation",
    "activation_reference_temperature_semantics",
]
CONDITIONS_FOR_FINAL_APPROVAL = [
    "verify_abaqus_total_mean_thermal_expansion",
    "verify_activation_reference_temperature_semantics",
    "pass_flow_curve_solver_realization_sensitivity",
    "issue_superseding_final_g0_v2_approval",
]
APPROVAL_MESSAGE = "按上述范围条件批准 G0-v2"
APPROVED_UTC = "2026-07-27T07:07:13Z"
PAPER_PARITY_SHA256 = (
    "7e777d73f736d72578bcfccb80199e361"
    "63d9b2f60fdb08e9e8b3d2fa56320f7"
)
CANONICAL_G0_SHA256 = (
    "206d7af567f0ee4d9113e8780b67b503"
    "936030538f266114834aa63052390839"
)
MANIFEST_SHA256 = (
    "c7cc552afca8c3ebad26160e41c6aa70"
    "1fc099f3b8abfd6182a6360e7258b061"
)
REQUEST_SHA256 = (
    "b94be55ed173186bea9dcacfd5fbd404"
    "4afef3187dc72fd1aa72e00ba1a211c2"
)
MATERIAL_CONFIG_SHA256 = (
    "899a912609db10490872bfe8d1a738a4"
    "0b5c68e03b37dafac8d4c659b8bca178"
)
APPROVAL_SHA256 = (
    "4c917871ea433b6589ad13ec681c09d40"
    "67d8710d0771b99fadcd1681cbc123b"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validator(schema_name: str) -> Draft202012Validator:
    schema = _load_json(CONTRACT_ROOT / schema_name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def _build_minimal_approved_bundle(
    tmp_path: Path,
    *,
    payload_path: str = "table.csv",
    duplicate_payload_alias: bool = False,
) -> dict[str, Path]:
    repo = tmp_path / "repo"
    candidate = (
        repo / "cases" / "kaess_2023" / "candidates" / "g0-v2-t018"
    )
    inputs = repo / "cases" / "kaess_2023" / "inputs"
    candidate.mkdir(parents=True)
    inputs.mkdir(parents=True)

    material = candidate / "material.json"
    _write_json(
        material,
        {
            "material_name": "fixture",
            "rho_solid": 8000.0,
            "rho_powder": 4000.0,
            "rho_liquid": 8000.0,
            "solidus_temperature": 1643.15,
            "liquidus_temperature": 1673.15,
            "latent_heat": 280000.0,
            "conductivity_powder": 0.15,
            "powder_solid_E": 1.0e10,
            "powder_solid_yield": 1.0e6,
            "powder_solid_hardening": 0.0,
            "absorptivity": 0.5,
            "emissivity": 0.4,
            "convection_h": 20.0,
            "mechanics_model": "j2_plastic",
            "poisson": 0.3,
            "mushy_mechanics_factor": 1.0,
            "liquid_mechanics_factor": 1.0,
            "reset_plastic_on_melt": False,
            "stress_relaxation_temperature": None,
            "phase_history_model": "paper_irreversible",
        },
    )
    raw_payload = Path(payload_path)
    payload = candidate / raw_payload
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("T,value\n293.15,1.0\n", encoding="utf-8")

    files = [
        {
            "path": material.name,
            "role": "material_config",
            "sha256": _sha256(material),
            "provenance": "fixture",
        },
        {
            "path": payload_path,
            "role": "fixture_table",
            "sha256": _sha256(payload),
            "provenance": "fixture",
        },
    ]
    if duplicate_payload_alias:
        files.append(
            {
                **files[-1],
                "path": f"./{payload_path}",
            }
        )

    manifest = candidate / "material-bundle-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "kaess.material-bundle-manifest/1",
            "bundle_id": "fixture-g0-v2",
            "status": "pending_review",
            "promotion_eligible": False,
            "hash_algorithm": "sha256",
            "path_bases": {
                "material_config": "manifest_directory",
                "files": "manifest_directory",
                "source_evidence": {
                    "repository_paths": "repository_root",
                    "superseded_external_config_path": "absolute_path",
                },
            },
            "material_config": {
                "path": material.name,
                "sha256": _sha256(material),
            },
            "files": files,
            "source_evidence": {
                "paper_doi": "10.3390/ma16062321",
                "paper_locator": "fixture",
                "superseded_external_config_path": "/external/material.json",
                "superseded_external_config_sha256": "0" * 64,
            },
            "semantic_fields": {
                "density_kg_per_m3": {
                    "solid": 8000.0,
                    "powder": 4000.0,
                    "liquid": 8000.0,
                },
                "melting_interval_K": {
                    "solidus": 1643.15,
                    "liquidus": 1673.15,
                },
                "latent_heat_J_per_kg": 280000.0,
                "powder_conductivity_W_per_mK": {
                    "at_293_15_K": 0.15,
                    "at_1643_15_K": 0.6,
                },
                "powder_weak_solid": {
                    "youngs_modulus_Pa": 1.0e10,
                    "yield_stress_Pa": 1.0e6,
                    "hardening_modulus_Pa": 0.0,
                },
                "thermal_boundary": {
                    "absorptivity": 0.5,
                    "emissivity": 0.4,
                    "convection_W_per_m2K": 20.0,
                },
                "figure_4_property_reading_error": {
                    "temperature_absolute_K": 2.0,
                    "youngs_modulus_absolute_GPa": 0.5,
                    "conductivity_absolute_W_per_mK": 0.1,
                    "specific_heat_absolute_J_per_kgK": 2.0,
                    "thermal_expansion_absolute_per_K": 1.0e-7,
                },
                "mechanics": {
                    "model": "j2_plastic",
                    "plasticity_input": "fixture_flow_curve",
                    "flow_curve_status": "fixture",
                    "flow_curve_reading_error": {
                        "equivalent_plastic_strain_absolute": 0.0032,
                        "flow_stress_absolute_MPa": 1.0,
                    },
                    "flow_curve_temperature_nodes_K": [293.15, 1643.15],
                    "flow_curve_equivalent_plastic_strain_nodes": [0.0, 0.4],
                    "poisson_ratio": 0.3,
                    "mushy_extra_stiffness_factor": 1.0,
                    "liquid_extra_stiffness_factor": 1.0,
                    "reset_plastic_on_melt": False,
                    "stress_relaxation_temperature_K": None,
                },
            },
            "required_runtime_semantics": {
                "table_path_resolution": (
                    "relative_to_material_config_directory"
                ),
                "table_interpolation": "piecewise_linear",
                "table_endpoint_behavior": "clamp",
                "flow_curve_interpolation": "bilinear",
                "flow_curve_endpoint_behavior": "clamp_on_both_axes",
                "flow_curve_residual_and_tangent": "single_exact_map",
                "powder_specific_heat": "same_as_solid",
                "powder_thermal_expansion": "same_as_solid",
                "thermal_expansion_table_form": "total_mean",
                "liquid_specific_heat": "same_as_solid",
                "liquid_conductivity": "same_as_solid",
                "phase_change_energy": "enthalpy_difference",
                "phase_history": (
                    "irreversible_powder_to_solid_after_first_melt"
                ),
                "first_solidification_reference_temperature": "latch_once",
                "remelt_rewrites_reference_temperature": False,
                "remelt_erases_equivalent_plastic_strain": False,
                "duplicate_mushy_or_liquid_mechanics_scaling": False,
                "validation_status": "fixture",
            },
            "known_registered_assumptions": [],
        },
    )

    canonical = inputs / "g0-approval.json"
    _write_json(
        canonical,
        {
            "schema_version": "kaess.g0-approval/1",
            "gate_id": "G0",
            "protocol_id": "kaess-test",
            "decision": "approved",
        },
    )

    parity = inputs / "paper-parity-config.yaml"
    _write_json(
        parity,
        {
            "schema_version": "kaess.paper-parity-config/1",
            "protocol_id": "kaess-test",
            "status": "approved",
            "approval": {
                "approval_record_sha256": _sha256(canonical),
            },
        },
    )

    request = candidate / "g0-reapproval-request.json"
    _write_json(
        request,
        {
            "schema_version": "kaess.g0-reapproval-request/1",
            "request_id": "fixture-request",
            "gate_id": "G0",
            "status": "pending_review",
            "decision": "pending_review",
            "canonical_approval_reference": {
                "path": "cases/kaess_2023/inputs/g0-approval.json",
                "sha256": _sha256(canonical),
            },
            "candidate_bundle_reference": {
                "path": (
                    "cases/kaess_2023/candidates/g0-v2-t018/"
                    "material-bundle-manifest.json"
                ),
                "sha256": _sha256(manifest),
            },
            "differences": [],
        },
    )

    approval = inputs / "g0-v2-material-conditional-approval.json"
    _write_json(
        approval,
        {
            "schema_version": "kaess.g0-v2-material-approval/1",
            "gate_id": "G0-v2",
            "protocol_id": "kaess-test",
            "decision": "conditionally_approved",
            "approved_by": "project_owner",
            "approved_utc": APPROVED_UTC,
            "approval_source": {
                "kind": "codex_user_message",
                "message": APPROVAL_MESSAGE,
            },
            "relationship_to_g0_v1": (
                "does_not_supersede_formal_g0_v1"
            ),
            "canonical_g0_reference": {
                "path": "cases/kaess_2023/inputs/g0-approval.json",
                "sha256": _sha256(canonical),
                "size_bytes": canonical.stat().st_size,
            },
            "paper_parity_reference": {
                "path": (
                    "cases/kaess_2023/inputs/paper-parity-config.yaml"
                ),
                "sha256": _sha256(parity),
                "size_bytes": parity.stat().st_size,
            },
            "request_reference": {
                "path": (
                    "cases/kaess_2023/candidates/g0-v2-t018/"
                    "g0-reapproval-request.json"
                ),
                "sha256": _sha256(request),
                "size_bytes": request.stat().st_size,
            },
            "bundle_manifest_reference": {
                "path": (
                    "cases/kaess_2023/candidates/g0-v2-t018/"
                    "material-bundle-manifest.json"
                ),
                "sha256": _sha256(manifest),
                "size_bytes": manifest.stat().st_size,
            },
            "material_config_reference": {
                "path": (
                    "cases/kaess_2023/candidates/g0-v2-t018/"
                    "material.json"
                ),
                "sha256": _sha256(material),
                "size_bytes": material.stat().st_size,
            },
            "approval_scope": {
                "authorization_level": "validation_only",
                "claim_level": "public_code_to_code",
                "permitted_use_scopes": PERMITTED_USE_SCOPES,
                "prohibited_use_scopes": PROHIBITED_USE_SCOPES,
                "permitted_jax_platforms": ["cpu"],
                "formal_eligible": False,
                "promotion_eligible": False,
            },
            "accepted_registered_assumptions": [],
            "excluded_pending_semantics": EXCLUDED_PENDING_SEMANTICS,
            "conditions_for_final_approval": CONDITIONS_FOR_FINAL_APPROVAL,
            "verification_evidence": {
                "candidate_commit": "0" * 40,
                "full_regression": "fixture",
                "independent_review": "fixture",
            },
        },
    )
    return {
        "repo": repo,
        "candidate": candidate,
        "material": material,
        "payload": payload,
        "manifest": manifest,
        "canonical": canonical,
        "parity": parity,
        "request": request,
        "approval": approval,
    }


def _refresh_fixture_chain(fixture: dict[str, Path]) -> None:
    manifest = _load_json(fixture["manifest"])
    material_hash = _sha256(fixture["material"])
    manifest["material_config"]["sha256"] = material_hash
    material_record = next(
        record
        for record in manifest["files"]
        if record["role"] == "material_config"
    )
    material_record["sha256"] = material_hash
    _write_json(fixture["manifest"], manifest)

    request = _load_json(fixture["request"])
    request["candidate_bundle_reference"]["sha256"] = _sha256(
        fixture["manifest"]
    )
    _write_json(fixture["request"], request)

    approval = _load_json(fixture["approval"])
    for field, path_key in (
        ("request_reference", "request"),
        ("bundle_manifest_reference", "manifest"),
        ("material_config_reference", "material"),
    ):
        path = fixture[path_key]
        approval[field]["sha256"] = _sha256(path)
        approval[field]["size_bytes"] = path.stat().st_size
    _write_json(fixture["approval"], approval)


def _validate_fixture_bundle(
    fixture: dict[str, Path],
    *,
    material_path: Path | None = None,
    manifest_path: Path | None = None,
    use_scope: str = "g1_cpu_validation",
    jax_platform: str | None = "cpu",
    actual_jax_platform: str = "cpu",
) -> dict:
    from jax_fem_am.verification.material_identity import (
        _validate_material_bundle_structure,
    )

    return _validate_material_bundle_structure(
        material_path or fixture["material"],
        repository_root=fixture["repo"],
        bundle_manifest_path=manifest_path or fixture["manifest"],
        reapproval_request_path=fixture["request"],
        approval_record_path=fixture["approval"],
        use_scope=use_scope,
        jax_platform=jax_platform,
        actual_jax_platform=actual_jax_platform,
        expected_approval_sha256=_sha256(fixture["approval"]),
    )


def test_g0v2_conditional_approval_is_schema_valid_and_hash_bound():
    approval = _load_json(APPROVAL_PATH)
    manifest = _load_json(MANIFEST_PATH)
    request = _load_json(REQUEST_PATH)

    _validator(MANIFEST_SCHEMA).validate(manifest)
    _validator(REQUEST_SCHEMA).validate(request)
    _validator(APPROVAL_SCHEMA).validate(approval)

    assert approval["decision"] == "conditionally_approved"
    assert approval["approved_by"] == "project_owner"
    assert approval["approved_utc"] == APPROVED_UTC
    parsed_utc = datetime.fromisoformat(
        approval["approved_utc"].replace("Z", "+00:00")
    )
    assert parsed_utc.tzinfo == timezone.utc
    assert approval["approval_source"] == {
        "kind": "codex_user_message",
        "message": APPROVAL_MESSAGE,
    }
    assert approval["relationship_to_g0_v1"] == (
        "does_not_supersede_formal_g0_v1"
    )
    assert _sha256(PAPER_PARITY_PATH) == PAPER_PARITY_SHA256
    assert _sha256(CANONICAL_G0_PATH) == CANONICAL_G0_SHA256
    assert _sha256(MANIFEST_PATH) == MANIFEST_SHA256
    assert _sha256(REQUEST_PATH) == REQUEST_SHA256
    assert _sha256(MATERIAL_CONFIG_PATH) == MATERIAL_CONFIG_SHA256
    assert _sha256(APPROVAL_PATH) == APPROVAL_SHA256
    assert (
        approval["paper_parity_reference"]["sha256"]
        == PAPER_PARITY_SHA256
    )
    assert (
        approval["canonical_g0_reference"]["sha256"]
        == CANONICAL_G0_SHA256
    )
    assert (
        approval["bundle_manifest_reference"]["sha256"]
        == MANIFEST_SHA256
    )
    assert approval["request_reference"]["sha256"] == REQUEST_SHA256
    assert (
        approval["material_config_reference"]["sha256"]
        == MATERIAL_CONFIG_SHA256
    )
    assert approval["approval_scope"] == {
        "authorization_level": "validation_only",
        "claim_level": "public_code_to_code",
        "permitted_use_scopes": PERMITTED_USE_SCOPES,
        "prohibited_use_scopes": PROHIBITED_USE_SCOPES,
        "permitted_jax_platforms": ["cpu"],
        "formal_eligible": False,
        "promotion_eligible": False,
    }
    assert (
        approval["excluded_pending_semantics"]
        == EXCLUDED_PENDING_SEMANTICS
    )
    assert (
        approval["conditions_for_final_approval"]
        == CONDITIONS_FOR_FINAL_APPROVAL
    )
    assert manifest["status"] == "pending_review"
    assert manifest["promotion_eligible"] is False
    assert request["status"] == "pending_review"
    assert request["decision"] == "pending_review"


def test_g0v2_approval_schema_rejects_scope_escalation_and_extra_fields():
    approval = _load_json(APPROVAL_PATH)
    validator = _validator(APPROVAL_SCHEMA)

    escalated = copy.deepcopy(approval)
    escalated["approval_scope"]["promotion_eligible"] = True
    assert list(validator.iter_errors(escalated))

    extra = copy.deepcopy(approval)
    extra["unreviewed_field"] = "unsafe"
    assert list(validator.iter_errors(extra))

    wrong_message = copy.deepcopy(approval)
    wrong_message["approval_source"]["message"] = "批准 G0-v2"
    assert list(validator.iter_errors(wrong_message))

    wrong_time = copy.deepcopy(approval)
    wrong_time["approved_utc"] = "2026-07-27T15:07:13+08:00"
    assert list(validator.iter_errors(wrong_time))

    missing_condition = copy.deepcopy(approval)
    missing_condition["conditions_for_final_approval"].pop()
    assert list(validator.iter_errors(missing_condition))

    nested_extra = copy.deepcopy(approval)
    nested_extra["approval_scope"]["unreviewed"] = True
    assert list(validator.iter_errors(nested_extra))


@pytest.mark.parametrize("use_scope", PERMITTED_USE_SCOPES)
def test_g0v2_gate_accepts_only_conditionally_permitted_scopes(use_scope):
    from jax_fem_am.verification.material_identity import (
        validate_material_bundle_identity,
    )

    result = validate_material_bundle_identity(
        MATERIAL_CONFIG_PATH,
        approval_record_path=APPROVAL_PATH,
        use_scope=use_scope,
        jax_platform="cpu",
    )

    assert result["status"] == "pass"
    assert result["authorization_level"] == "validation_only"
    assert result["use_scope"] == use_scope
    assert result["jax_platform"] == "cpu"
    assert result["actual_jax_platform"] == "cpu"
    assert result["formal_eligible"] is False
    assert result["promotion_eligible"] is False
    assert result["bundle_manifest_sha256"] == _sha256(MANIFEST_PATH)


@pytest.mark.parametrize("use_scope", PROHIBITED_USE_SCOPES)
def test_g0v2_gate_rejects_formal_and_promotion_scopes(use_scope):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
        validate_material_bundle_identity,
    )

    with pytest.raises(MaterialIdentityError, match="not permitted"):
        validate_material_bundle_identity(
            MATERIAL_CONFIG_PATH,
            approval_record_path=APPROVAL_PATH,
            use_scope=use_scope,
            jax_platform="cpu",
        )


@pytest.mark.parametrize("jax_platform", ["gpu", "CPU", "", None])
def test_g0v2_gate_rejects_non_cpu_platforms(jax_platform):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
        validate_material_bundle_identity,
    )

    with pytest.raises(MaterialIdentityError, match="platform"):
        validate_material_bundle_identity(
            MATERIAL_CONFIG_PATH,
            approval_record_path=APPROVAL_PATH,
            use_scope="g1_cpu_validation",
            jax_platform=jax_platform,
        )


def test_g0v2_public_gate_rejects_self_signed_repository(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
        validate_material_bundle_identity,
    )

    fixture = _build_minimal_approved_bundle(tmp_path)
    with pytest.raises(MaterialIdentityError, match="trust anchor|contained"):
        validate_material_bundle_identity(
            fixture["material"],
            approval_record_path=fixture["approval"],
            use_scope="g1_cpu_validation",
            jax_platform="cpu",
        )


def test_g0v2_gate_checks_actual_jax_backend(monkeypatch):
    from jax_fem_am.verification import material_identity

    monkeypatch.setattr(
        material_identity,
        "_detect_jax_platform",
        lambda: "gpu",
    )
    with pytest.raises(
        material_identity.MaterialIdentityError,
        match="actual JAX backend",
    ):
        material_identity.validate_material_bundle_identity(
            MATERIAL_CONFIG_PATH,
            approval_record_path=APPROVAL_PATH,
            use_scope="g1_cpu_validation",
            jax_platform="cpu",
        )


@pytest.mark.parametrize("use_scope", ["gpu_qualification", "unknown"])
def test_g0v2_gate_rejects_unknown_scopes(use_scope):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
        validate_material_bundle_identity,
    )

    with pytest.raises(MaterialIdentityError, match="not permitted"):
        validate_material_bundle_identity(
            MATERIAL_CONFIG_PATH,
            approval_record_path=APPROVAL_PATH,
            use_scope=use_scope,
            jax_platform="cpu",
        )


def test_g0v2_gate_rejects_changed_bundle_payload(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(tmp_path)
    fixture["payload"].write_text(
        "T,value\n293.15,999.0\n",
        encoding="utf-8",
    )

    with pytest.raises(MaterialIdentityError, match="SHA-256"):
        _validate_fixture_bundle(fixture)


def test_g0v2_gate_rejects_manifest_path_escape_with_valid_hash_chain(
    tmp_path,
):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(
        tmp_path,
        payload_path="../escaped.csv",
    )

    with pytest.raises(MaterialIdentityError, match="contained|relative"):
        _validate_fixture_bundle(fixture)


def test_g0v2_gate_rejects_normalized_duplicate_manifest_paths(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(
        tmp_path,
        duplicate_payload_alias=True,
    )

    with pytest.raises(
        MaterialIdentityError,
        match="canonical|duplicate",
    ):
        _validate_fixture_bundle(fixture)


@pytest.mark.parametrize(
    ("replacement", "error_pattern"),
    [
        (
            '"gate_id": "G0-v2",\n'
            '  "gate_id": "G0-v2",',
            "duplicate",
        ),
        ('"promotion_eligible": NaN', "finite|JSON"),
        ('"promotion_eligible": Infinity', "finite|JSON"),
        ('"promotion_eligible": 1e999', "finite|JSON"),
    ],
)
def test_g0v2_gate_strict_json_rejects_ambiguous_numbers_and_keys(
    tmp_path,
    replacement,
    error_pattern,
):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(tmp_path)
    approval_text = fixture["approval"].read_text(encoding="utf-8")
    if replacement.startswith('"gate_id"'):
        approval_text = approval_text.replace(
            '"gate_id": "G0-v2",',
            replacement,
            1,
        )
    else:
        approval_text = approval_text.replace(
            '"promotion_eligible": false',
            replacement,
            1,
        )
    fixture["approval"].write_text(approval_text, encoding="utf-8")

    with pytest.raises(MaterialIdentityError, match=error_pattern):
        _validate_fixture_bundle(fixture)


def test_g0v2_gate_rejects_runtime_material_outside_manifest_identity(
    tmp_path,
):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(tmp_path)
    copied_material = fixture["candidate"] / "copied-material.json"
    copied_material.write_bytes(fixture["material"].read_bytes())

    with pytest.raises(MaterialIdentityError, match="material config path"):
        _validate_fixture_bundle(
            fixture,
            material_path=copied_material,
        )


def test_g0v2_gate_rejects_unlisted_material_table_with_rehashed_chain(
    tmp_path,
):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(tmp_path)
    unlisted_table = fixture["candidate"] / "unlisted.csv"
    unlisted_table.write_text("T,value\n293.15,2.0\n", encoding="utf-8")
    material = _load_json(fixture["material"])
    material["k_table_solid"] = unlisted_table.name
    _write_json(fixture["material"], material)
    _refresh_fixture_chain(fixture)

    with pytest.raises(MaterialIdentityError, match="not bound"):
        _validate_fixture_bundle(fixture)


def test_g0v2_gate_rejects_symlinked_bundle_payload(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(fixture["payload"].read_bytes())
    fixture["payload"].unlink()
    try:
        fixture["payload"].symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(MaterialIdentityError, match="symlink"):
        _validate_fixture_bundle(fixture)


def test_g0v2_gate_rejects_symlink_alias_for_provided_manifest(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(tmp_path)
    manifest_alias = fixture["candidate"] / "manifest-alias.json"
    try:
        manifest_alias.symlink_to(fixture["manifest"].name)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(MaterialIdentityError, match="symlink"):
        _validate_fixture_bundle(
            fixture,
            manifest_path=manifest_alias,
        )


def test_g0v2_gate_rejects_unreviewed_nested_approval_field(tmp_path):
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
    )

    fixture = _build_minimal_approved_bundle(tmp_path)
    approval = _load_json(fixture["approval"])
    approval["approval_scope"]["unreviewed"] = True
    _write_json(fixture["approval"], approval)

    with pytest.raises(MaterialIdentityError, match="does not satisfy"):
        _validate_fixture_bundle(fixture)


def test_g0v2_gate_requires_explicit_scope_and_platform():
    from jax_fem_am.verification.material_identity import (
        validate_material_bundle_identity,
    )

    common = {
        "approval_record_path": APPROVAL_PATH,
    }
    with pytest.raises(TypeError, match="use_scope"):
        validate_material_bundle_identity(
            MATERIAL_CONFIG_PATH,
            jax_platform="cpu",
            **common,
        )
    with pytest.raises(TypeError, match="jax_platform"):
        validate_material_bundle_identity(
            MATERIAL_CONFIG_PATH,
            use_scope="g1_cpu_validation",
            **common,
        )


def test_formal_launchers_remain_isolated_from_g0v2_conditional_approval():
    launchers = [
        REPO_ROOT / "cases" / "kaess_2023" / "run_kaess_phase1.sh",
        REPO_ROOT / "cases" / "kaess_2023" / "run_kaess_phase2.sh",
    ]
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "inputs/g0-approval.json" in text
        assert "g0-v2-material-conditional-approval.json" not in text
        assert "g0-v2-t018" not in text


def test_content_addressed_g0v2_inputs_have_checkout_stable_attributes():
    text_paths = [
        APPROVAL_PATH,
        PAPER_PARITY_PATH,
        CANONICAL_G0_PATH,
        MANIFEST_PATH,
        REQUEST_PATH,
        MATERIAL_CONFIG_PATH,
        CANDIDATE_ROOT / "flow_curve_table.pending.csv",
        REPO_ROOT
        / "cases"
        / "kaess_2023"
        / "references"
        / "cases"
        / "kaess_2023_fulltext.txt",
        REPO_ROOT / "cases" / "kaess_2023" / "calibrate_figure4.py",
        REPO_ROOT
        / "cases"
        / "kaess_2023"
        / "references"
        / "figure4-vector-spec.json",
    ]
    assert GITATTRIBUTES_PATH.is_file()
    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            *(str(path.relative_to(REPO_ROOT)) for path in text_paths),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    for path in text_paths:
        relative = str(path.relative_to(REPO_ROOT))
        assert f"{relative}: text: set" in lines
        assert f"{relative}: eol: lf" in lines


def test_canonical_relative_path_rejects_empty_normalized_path():
    from jax_fem_am.verification.material_identity import (
        MaterialIdentityError,
        _canonical_relative_parts,
    )

    with pytest.raises(MaterialIdentityError, match="non-empty|canonical"):
        _canonical_relative_parts(".", "probe path")


def test_g0v2_bundle_gate_is_available_through_the_cli():
    command = [
        sys.executable,
        "-m",
        "jax_fem_am.verification.material_identity",
        "--material-config",
        str(MATERIAL_CONFIG_PATH),
        "--approval-record",
        str(APPROVAL_PATH),
        "--use-scope",
        "g1_cpu_validation",
        "--jax-platform",
        "cpu",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["authorization_level"] == "validation_only"
    assert payload["formal_eligible"] is False
    assert payload["promotion_eligible"] is False

    command[-1] = "gpu"
    rejected = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "platform is not permitted" in rejected.stderr


def test_legacy_g0v1_gate_does_not_require_jsonschema_at_import(tmp_path):
    material = tmp_path / "legacy-material.json"
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
    script = """
import sys

class BlockJsonschema:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jsonschema" or fullname.startswith("jsonschema."):
            raise ModuleNotFoundError("jsonschema intentionally blocked")
        return None

sys.meta_path.insert(0, BlockJsonschema())
from jax_fem_am.verification.material_identity import validate_material_identity
result = validate_material_identity(
    sys.argv[1],
    parity_config_path=sys.argv[2],
    approval_record_path=sys.argv[3],
)
assert result["status"] == "pass"
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(material),
            str(parity),
            str(approval),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
