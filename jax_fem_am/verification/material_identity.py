"""Fail-closed material-identity gates for the Kaess reproduction workflow.

The legacy G0-v1 gate remains the authority for formal ``external_sha256``
launchers. The separately anchored G0-v2 gate authorizes its repository bundle
only for the explicitly approved CPU-validation scopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import posixpath
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

class MaterialIdentityError(ValueError):
    """Raised when runtime material bytes are not the approved G0 input."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON number must be finite, got {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON number must be finite, got {value}")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _load_object(path: Path, label: str) -> dict:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except (OSError, ValueError) as exc:
        raise MaterialIdentityError(
            f"{label} is not a readable strict JSON object: {path} ({exc})"
        ) from exc
    if not isinstance(document, dict):
        raise MaterialIdentityError(
            f"{label} must contain a JSON object: {path}"
        )
    return document


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterialIdentityError(message)


_SOURCE_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_ROOT = (
    _SOURCE_REPOSITORY_ROOT
    / "specs"
    / "001-kaess-paper-reproduction"
    / "contracts"
)
_G0V2_APPROVAL_RELATIVE_PATH = (
    "cases/kaess_2023/inputs/g0-v2-material-conditional-approval.json"
)
_G0V2_APPROVAL_SHA256 = (
    "4c917871ea433b6589ad13ec681c09d40"
    "67d8710d0771b99fadcd1681cbc123b"
)
_G0V2_MANIFEST_RELATIVE_PATH = (
    "cases/kaess_2023/candidates/g0-v2-t018/"
    "material-bundle-manifest.json"
)
_G0V2_REQUEST_RELATIVE_PATH = (
    "cases/kaess_2023/candidates/g0-v2-t018/"
    "g0-reapproval-request.json"
)
_APPROVAL_MESSAGE = "按上述范围条件批准 G0-v2"
_PERMITTED_G0V2_SCOPES = [
    "g1_cpu_validation",
    "g2_cpu_validation",
    "sensitivity_analysis",
]
_PROHIBITED_G0V2_SCOPES = ["formal_run", "promotion"]
_EXCLUDED_G0V2_SEMANTICS = [
    "abaqus_total_mean_thermal_expansion_runtime_implementation",
    "activation_reference_temperature_semantics",
]
_FINAL_G0V2_CONDITIONS = [
    "verify_abaqus_total_mean_thermal_expansion",
    "verify_activation_reference_temperature_semantics",
    "pass_flow_curve_solver_realization_sensitivity",
    "issue_superseding_final_g0_v2_approval",
]
_TABLE_CONFIG_FIELDS = (
    "k_table_solid",
    "cp_table_solid",
    "k_table_powder",
    "cp_table_powder",
    "k_table_liquid",
    "cp_table_liquid",
    "E_table",
    "alpha_table",
    "flow_curve_table",
)
_SOURCE_EVIDENCE_IDENTITIES = (
    ("repository_fulltext", "repository_fulltext_sha256"),
    ("repository_paper_pdf", "repository_paper_pdf_sha256"),
    (
        "repository_figure_4_page_image",
        "repository_figure_4_page_image_sha256",
    ),
    (
        "figure_4_vector_calibration_script",
        "figure_4_vector_calibration_script_sha256",
    ),
    ("figure_4_vector_spec", "figure_4_vector_spec_sha256"),
)


def _validate_schema(document: dict, schema_name: str, label: str) -> None:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as exc:
        raise MaterialIdentityError(
            "conditional G0-v2 validation requires the 'verification' "
            "dependency extra (jsonschema)"
        ) from exc

    schema_path = _CONTRACT_ROOT / schema_name
    schema = _load_object(schema_path, f"{label} schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise MaterialIdentityError(
            f"{label} schema is invalid: {schema_path} ({exc.message})"
        ) from exc
    validator = Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )
    error = next(iter(validator.iter_errors(document)), None)
    if error is not None:
        raise MaterialIdentityError(
            f"{label} does not satisfy {schema_name} at "
            f"{error.json_path}: {error.message}"
        )


def _canonical_relative_parts(raw_path: str, label: str) -> tuple[str, ...]:
    _require(
        isinstance(raw_path, str) and bool(raw_path),
        f"{label} must be a non-empty relative path",
    )
    _require(
        "\\" not in raw_path,
        f"{label} must use canonical POSIX separators",
    )
    relative = PurePosixPath(raw_path)
    _require(
        bool(relative.parts),
        f"{label} must be a non-empty canonical relative path",
    )
    _require(
        not relative.is_absolute(),
        f"{label} must be relative",
    )
    _require(
        ".." not in relative.parts,
        f"{label} must be contained and cannot use '..'",
    )
    _require(
        raw_path == relative.as_posix(),
        f"{label} must be a canonical relative path",
    )
    _require(
        not relative.parts[0].endswith(":"),
        f"{label} must not contain a drive prefix",
    )
    return relative.parts


def _resolve_contained_file(base: Path, raw_path: str, label: str) -> Path:
    try:
        root = base.resolve(strict=True)
    except OSError as exc:
        raise MaterialIdentityError(
            f"{label} base is not readable: {base}"
        ) from exc
    _require(root.is_dir(), f"{label} base is not a directory: {root}")
    parts = _canonical_relative_parts(raw_path, label)
    current = root
    for part in parts:
        current = current / part
        try:
            current.lstat()
        except OSError as exc:
            raise MaterialIdentityError(
                f"{label} does not identify an existing file: {current}"
            ) from exc
        _require(
            not current.is_symlink(),
            f"{label} must not traverse a symlink: {current}",
        )
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise MaterialIdentityError(
            f"{label} is not readable: {current}"
        ) from exc
    _require(
        resolved.is_relative_to(root),
        f"{label} must be contained by {root}",
    )
    _require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    return resolved


def _resolve_provided_file(path, repository_root: Path, label: str) -> Path:
    unresolved = Path(path)
    _require(
        ".." not in unresolved.parts,
        f"{label} must not use '..'",
    )
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    try:
        lexical_path = unresolved.absolute()
        relative = lexical_path.relative_to(repository_root)
    except (OSError, ValueError) as exc:
        raise MaterialIdentityError(
            f"{label} must be lexically contained by repository root"
        ) from exc

    current = repository_root
    for part in relative.parts:
        current = current / part
        try:
            current.lstat()
        except OSError as exc:
            raise MaterialIdentityError(
                f"{label} is not an existing file: {current}"
            ) from exc
        _require(
            not current.is_symlink(),
            f"{label} must not traverse a symlink: {current}",
        )
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise MaterialIdentityError(
            f"{label} is not an existing file: {path}"
        ) from exc
    _require(resolved.is_file(), f"{label} is not a file: {resolved}")
    _require(
        resolved.is_relative_to(repository_root),
        f"{label} must be contained by repository root",
    )
    return resolved


def _verify_reference(
    repository_root: Path,
    reference: dict,
    label: str,
) -> tuple[Path, str]:
    path = _resolve_contained_file(
        repository_root,
        reference.get("path"),
        f"{label} path",
    )
    actual_hash = _sha256(path)
    _require(
        reference.get("sha256") == actual_hash,
        f"{label} SHA-256 does not match approval identity",
    )
    if "size_bytes" in reference:
        _require(
            reference["size_bytes"] == path.stat().st_size,
            f"{label} size does not match approval identity",
        )
    return path, actual_hash


def _expect_config_value(
    config: dict,
    key: str,
    expected: Any,
    semantic_label: str,
) -> None:
    _require(
        config.get(key) == expected,
        f"material config {key} differs from manifest {semantic_label}",
    )


def _validate_manifest_semantics(config: dict, manifest: dict) -> None:
    semantic = manifest["semantic_fields"]
    density = semantic.get("density_kg_per_m3")
    if density is not None:
        _expect_config_value(config, "rho_solid", density["solid"], "density")
        _expect_config_value(config, "rho_powder", density["powder"], "density")
        _expect_config_value(config, "rho_liquid", density["liquid"], "density")

    melting = semantic.get("melting_interval_K")
    if melting is not None:
        _expect_config_value(
            config,
            "solidus_temperature",
            melting["solidus"],
            "melting interval",
        )
        _expect_config_value(
            config,
            "liquidus_temperature",
            melting["liquidus"],
            "melting interval",
        )

    if "latent_heat_J_per_kg" in semantic:
        _expect_config_value(
            config,
            "latent_heat",
            semantic["latent_heat_J_per_kg"],
            "latent heat",
        )

    powder_conductivity = semantic.get("powder_conductivity_W_per_mK")
    if powder_conductivity is not None:
        _expect_config_value(
            config,
            "conductivity_powder",
            powder_conductivity["at_293_15_K"],
            "powder conductivity",
        )

    weak_solid = semantic.get("powder_weak_solid")
    if weak_solid is not None:
        for config_key, semantic_key in (
            ("powder_solid_E", "youngs_modulus_Pa"),
            ("powder_solid_yield", "yield_stress_Pa"),
            ("powder_solid_hardening", "hardening_modulus_Pa"),
        ):
            _expect_config_value(
                config,
                config_key,
                weak_solid[semantic_key],
                "powder weak-solid mechanics",
            )

    boundary = semantic.get("thermal_boundary")
    if boundary is not None:
        for config_key, semantic_key in (
            ("absorptivity", "absorptivity"),
            ("emissivity", "emissivity"),
            ("convection_h", "convection_W_per_m2K"),
        ):
            _expect_config_value(
                config,
                config_key,
                boundary[semantic_key],
                "thermal boundary",
            )

    mechanics = semantic.get("mechanics")
    if mechanics is not None:
        for config_key, semantic_key in (
            ("mechanics_model", "model"),
            ("poisson", "poisson_ratio"),
            ("mushy_mechanics_factor", "mushy_extra_stiffness_factor"),
            ("liquid_mechanics_factor", "liquid_extra_stiffness_factor"),
            ("reset_plastic_on_melt", "reset_plastic_on_melt"),
            (
                "stress_relaxation_temperature",
                "stress_relaxation_temperature_K",
            ),
        ):
            _expect_config_value(
                config,
                config_key,
                mechanics[semantic_key],
                "mechanics",
            )

    runtime_semantics = manifest["required_runtime_semantics"]
    if runtime_semantics.get("phase_history") is not None:
        _require(
            runtime_semantics["phase_history"]
            == "irreversible_powder_to_solid_after_first_melt"
            and config.get("phase_history_model") == "paper_irreversible",
            "material config phase history differs from manifest semantics",
        )


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


def _validate_material_bundle_structure(
    material_config_path,
    *,
    repository_root,
    bundle_manifest_path,
    reapproval_request_path,
    approval_record_path,
    use_scope,
    jax_platform,
    actual_jax_platform,
    expected_approval_sha256,
) -> dict:
    """Validate a bundle structure against an explicit external trust anchor."""

    try:
        repo_root = Path(repository_root).resolve(strict=True)
    except OSError as exc:
        raise MaterialIdentityError(
            f"repository root is not readable: {repository_root}"
        ) from exc
    _require(repo_root.is_dir(), f"repository root is not a directory: {repo_root}")

    approval_path = _resolve_provided_file(
        approval_record_path,
        repo_root,
        "G0-v2 approval record",
    )
    approval_hash = _sha256(approval_path)
    _require(
        approval_hash == expected_approval_sha256,
        "G0-v2 approval record SHA-256 differs from the trust anchor",
    )
    approval = _load_object(approval_path, "G0-v2 approval record")
    _validate_schema(
        approval,
        "g0-v2-material-approval.schema.json",
        "G0-v2 approval record",
    )

    _require(
        approval.get("schema_version")
        == "kaess.g0-v2-material-approval/1",
        "G0-v2 approval schema is not supported",
    )
    _require(
        approval.get("gate_id") == "G0-v2"
        and approval.get("decision") == "conditionally_approved",
        "G0-v2 approval record is not conditionally approved",
    )
    _require(
        approval.get("approval_source")
        == {
            "kind": "codex_user_message",
            "message": _APPROVAL_MESSAGE,
        },
        "G0-v2 approval source does not match the recorded user decision",
    )
    _require(
        approval.get("relationship_to_g0_v1")
        == "does_not_supersede_formal_g0_v1",
        "G0-v2 approval must not supersede formal G0-v1",
    )

    scope = approval["approval_scope"]
    expected_scope = {
        "authorization_level": "validation_only",
        "claim_level": "public_code_to_code",
        "permitted_use_scopes": _PERMITTED_G0V2_SCOPES,
        "prohibited_use_scopes": _PROHIBITED_G0V2_SCOPES,
        "permitted_jax_platforms": ["cpu"],
        "formal_eligible": False,
        "promotion_eligible": False,
    }
    _require(
        scope == expected_scope,
        "G0-v2 approval scope differs from the validation-only contract",
    )
    _require(
        use_scope in scope["permitted_use_scopes"],
        f"G0-v2 use scope is not permitted: {use_scope!r}",
    )
    _require(
        jax_platform in scope["permitted_jax_platforms"],
        f"G0-v2 JAX platform is not permitted: {jax_platform!r}",
    )
    _require(
        actual_jax_platform == jax_platform,
        "declared G0-v2 JAX platform does not match the actual JAX backend: "
        f"declared={jax_platform!r}, actual={actual_jax_platform!r}",
    )
    _require(
        approval["excluded_pending_semantics"]
        == _EXCLUDED_G0V2_SEMANTICS,
        "G0-v2 excluded semantics differ from the conditional decision",
    )
    _require(
        approval["conditions_for_final_approval"]
        == _FINAL_G0V2_CONDITIONS,
        "G0-v2 final-approval conditions differ from the conditional decision",
    )

    parity_path, parity_hash = _verify_reference(
        repo_root,
        approval["paper_parity_reference"],
        "paper-parity config",
    )
    canonical_path, canonical_hash = _verify_reference(
        repo_root,
        approval["canonical_g0_reference"],
        "canonical G0-v1 approval",
    )
    request_path, request_hash = _verify_reference(
        repo_root,
        approval["request_reference"],
        "G0-v2 reapproval request",
    )
    manifest_path, manifest_hash = _verify_reference(
        repo_root,
        approval["bundle_manifest_reference"],
        "G0-v2 bundle manifest",
    )
    approved_material_path, material_hash = _verify_reference(
        repo_root,
        approval["material_config_reference"],
        "G0-v2 material config",
    )

    provided_request = _resolve_provided_file(
        reapproval_request_path,
        repo_root,
        "provided G0-v2 reapproval request",
    )
    provided_manifest = _resolve_provided_file(
        bundle_manifest_path,
        repo_root,
        "provided G0-v2 bundle manifest",
    )
    provided_material = _resolve_provided_file(
        material_config_path,
        repo_root,
        "provided G0-v2 material config",
    )
    _require(
        provided_request == request_path,
        "provided reapproval request path differs from approval identity",
    )
    _require(
        provided_manifest == manifest_path,
        "provided bundle manifest path differs from approval identity",
    )
    _require(
        provided_material == approved_material_path,
        "provided material config path differs from approval identity",
    )

    parity = _load_object(parity_path, "paper-parity config")
    canonical = _load_object(canonical_path, "canonical G0-v1 approval")
    request = _load_object(request_path, "G0-v2 reapproval request")
    manifest = _load_object(manifest_path, "G0-v2 bundle manifest")
    material_config = _load_object(
        approved_material_path,
        "G0-v2 material config",
    )
    _validate_schema(
        request,
        "g0-reapproval-request.schema.json",
        "G0-v2 reapproval request",
    )
    _validate_schema(
        manifest,
        "material-bundle-manifest.schema.json",
        "G0-v2 bundle manifest",
    )

    protocol_id = approval["protocol_id"]
    _require(
        parity.get("schema_version") == "kaess.paper-parity-config/1"
        and parity.get("status") == "approved",
        "paper-parity config is not the approved v1 protocol",
    )
    _require(
        canonical.get("schema_version") == "kaess.g0-approval/1"
        and canonical.get("gate_id") == "G0"
        and canonical.get("decision") == "approved",
        "canonical G0-v1 record is not approved",
    )
    _require(
        parity.get("protocol_id")
        == canonical.get("protocol_id")
        == protocol_id,
        "approval, parity, and canonical G0 protocol identities differ",
    )
    _require(
        parity.get("approval", {}).get("approval_record_sha256")
        == canonical_hash,
        "paper-parity config does not bind the canonical G0-v1 approval",
    )

    _require(
        request.get("gate_id") == "G0"
        and request.get("status") == "pending_review"
        and request.get("decision") == "pending_review",
        "G0-v2 reapproval request must remain pending_review",
    )
    request_canonical_path, request_canonical_hash = _verify_reference(
        repo_root,
        request["canonical_approval_reference"],
        "request canonical G0-v1 approval",
    )
    request_manifest_path, request_manifest_hash = _verify_reference(
        repo_root,
        request["candidate_bundle_reference"],
        "request candidate manifest",
    )
    _require(
        request_canonical_path == canonical_path
        and request_canonical_hash == canonical_hash,
        "request and approval bind different canonical G0-v1 records",
    )
    _require(
        request_manifest_path == manifest_path
        and request_manifest_hash == manifest_hash,
        "request and approval bind different candidate manifests",
    )

    _require(
        manifest.get("schema_version")
        == "kaess.material-bundle-manifest/1",
        "material bundle manifest schema is not supported",
    )
    _require(
        manifest.get("status") == "pending_review"
        and manifest.get("promotion_eligible") is False,
        "conditional approval must not mutate candidate manifest status",
    )
    _require(
        manifest.get("hash_algorithm") == "sha256",
        "material bundle hash algorithm must be sha256",
    )
    _require(
        manifest.get("path_bases")
        == {
            "material_config": "manifest_directory",
            "files": "manifest_directory",
            "source_evidence": {
                "repository_paths": "repository_root",
                "superseded_external_config_path": "absolute_path",
            },
        },
        "material bundle path bases differ from the approved contract",
    )

    bundle_root = manifest_path.parent
    normalized_paths: set[str] = set()
    resolved_paths: set[Path] = set()
    records_by_path: dict[Path, dict] = {}
    material_records: list[dict] = []
    for record in manifest["files"]:
        raw_path = record["path"]
        normalized_path = posixpath.normpath(raw_path)
        _require(
            normalized_path not in normalized_paths,
            f"duplicate normalized manifest path: {raw_path}",
        )
        normalized_paths.add(normalized_path)
        resolved = _resolve_contained_file(
            bundle_root,
            raw_path,
            f"manifest file {raw_path!r}",
        )
        _require(
            resolved not in resolved_paths,
            f"duplicate resolved manifest path: {raw_path}",
        )
        resolved_paths.add(resolved)
        actual_hash = _sha256(resolved)
        _require(
            record["sha256"] == actual_hash,
            f"manifest file SHA-256 does not match: {raw_path}",
        )
        records_by_path[resolved] = record
        if record["role"] == "material_config":
            material_records.append(record)

    _require(
        len(material_records) == 1,
        "manifest must contain exactly one material_config role",
    )
    manifest_material = manifest["material_config"]
    manifest_material_path = _resolve_contained_file(
        bundle_root,
        manifest_material["path"],
        "manifest material_config path",
    )
    _require(
        manifest_material_path == approved_material_path,
        "manifest material config path differs from approval identity",
    )
    _require(
        manifest_material["sha256"] == material_hash,
        "manifest material config SHA-256 differs from approval identity",
    )
    _require(
        material_records[0]["path"] == manifest_material["path"]
        and material_records[0]["sha256"] == manifest_material["sha256"],
        "manifest material_config descriptor differs from its file record",
    )

    source_evidence = manifest["source_evidence"]
    for path_field, hash_field in _SOURCE_EVIDENCE_IDENTITIES:
        has_path = path_field in source_evidence
        has_hash = hash_field in source_evidence
        _require(
            has_path == has_hash,
            f"source evidence must pair {path_field} with {hash_field}",
        )
        if has_path:
            source_path = _resolve_contained_file(
                repo_root,
                source_evidence[path_field],
                f"source evidence {path_field}",
            )
            _require(
                source_evidence[hash_field] == _sha256(source_path),
                f"source evidence SHA-256 does not match: {path_field}",
            )

    for config_field in _TABLE_CONFIG_FIELDS:
        if config_field not in material_config:
            continue
        raw_dependency = material_config[config_field]
        _require(
            isinstance(raw_dependency, str),
            f"material config {config_field} must be a relative path",
        )
        dependency_path = _resolve_contained_file(
            bundle_root,
            raw_dependency,
            f"material config {config_field}",
        )
        _require(
            dependency_path in records_by_path,
            f"material config {config_field} is not bound by the manifest",
        )

    _validate_manifest_semantics(material_config, manifest)
    excluded_assumption = (
        "first_solidification_reference_temperature_is_a_solver_realization_"
        "not_a_reported_paper_field"
    )
    expected_accepted_assumptions = [
        assumption
        for assumption in manifest["known_registered_assumptions"]
        if assumption != excluded_assumption
    ]
    _require(
        approval["accepted_registered_assumptions"]
        == expected_accepted_assumptions,
        "approval assumptions differ from the reviewed manifest assumptions",
    )

    return {
        "status": "pass",
        "protocol_id": protocol_id,
        "authorization_level": scope["authorization_level"],
        "claim_level": scope["claim_level"],
        "use_scope": use_scope,
        "jax_platform": jax_platform,
        "actual_jax_platform": actual_jax_platform,
        "formal_eligible": False,
        "promotion_eligible": False,
        "material_config": str(approved_material_path),
        "material_sha256": material_hash,
        "bundle_manifest": str(manifest_path),
        "bundle_manifest_sha256": manifest_hash,
        "reapproval_request": str(request_path),
        "reapproval_request_sha256": request_hash,
        "approval_record": str(approval_path),
        "approval_record_sha256": approval_hash,
        "paper_parity_config": str(parity_path),
        "paper_parity_sha256": parity_hash,
        "canonical_g0_approval": str(canonical_path),
        "canonical_g0_approval_sha256": canonical_hash,
        "excluded_pending_semantics": list(
            approval["excluded_pending_semantics"]
        ),
        "conditions_for_final_approval": list(
            approval["conditions_for_final_approval"]
        ),
    }


def _detect_jax_platform() -> str:
    try:
        import jax

        return str(jax.default_backend())
    except Exception as exc:
        raise MaterialIdentityError(
            f"cannot determine the actual JAX backend: {exc}"
        ) from exc


def validate_material_bundle_identity(
    material_config_path,
    *,
    approval_record_path,
    use_scope,
    jax_platform,
) -> dict:
    """Authorize the fixed G0-v2 bundle for CPU validation only.

    Unlike the internal structural validator, this public gate is rooted in
    the source checkout and in the SHA-256 of the user's recorded conditional
    approval. Callers cannot substitute their own repository, manifest,
    request, or self-signed approval chain.
    """

    _require(
        jax_platform == "cpu",
        f"G0-v2 JAX platform is not permitted: {jax_platform!r}",
    )
    fixed_approval = _resolve_contained_file(
        _SOURCE_REPOSITORY_ROOT,
        _G0V2_APPROVAL_RELATIVE_PATH,
        "fixed G0-v2 approval record",
    )
    supplied_approval = _resolve_provided_file(
        approval_record_path,
        _SOURCE_REPOSITORY_ROOT,
        "provided G0-v2 approval record",
    )
    _require(
        supplied_approval == fixed_approval,
        "provided G0-v2 approval path differs from the fixed trust anchor",
    )
    return _validate_material_bundle_structure(
        material_config_path,
        repository_root=_SOURCE_REPOSITORY_ROOT,
        bundle_manifest_path=(
            _SOURCE_REPOSITORY_ROOT / _G0V2_MANIFEST_RELATIVE_PATH
        ),
        reapproval_request_path=(
            _SOURCE_REPOSITORY_ROOT / _G0V2_REQUEST_RELATIVE_PATH
        ),
        approval_record_path=fixed_approval,
        use_scope=use_scope,
        jax_platform=jax_platform,
        actual_jax_platform=_detect_jax_platform(),
        expected_approval_sha256=_G0V2_APPROVAL_SHA256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Kaess material bytes against legacy G0-v1 or the "
            "conditional G0-v2 CPU-validation bundle"
        )
    )
    parser.add_argument("--material-config", type=Path, required=True)
    parser.add_argument("--parity-config", type=Path)
    parser.add_argument("--approval-record", type=Path, required=True)
    parser.add_argument("--use-scope")
    parser.add_argument("--jax-platform")
    args = parser.parse_args(argv)
    bundle_options = {
        "--use-scope": args.use_scope,
        "--jax-platform": args.jax_platform,
    }
    bundle_mode = any(value is not None for value in bundle_options.values())
    try:
        if bundle_mode:
            missing = [
                option
                for option, value in bundle_options.items()
                if value is None
            ]
            if missing:
                parser.error(
                    "conditional G0-v2 mode requires "
                    + ", ".join(missing)
                )
            if args.parity_config is not None:
                parser.error(
                    "--parity-config is derived from the G0-v2 approval "
                    "record and must not be supplied in bundle mode"
                )
            result = validate_material_bundle_identity(
                args.material_config,
                approval_record_path=args.approval_record,
                use_scope=args.use_scope,
                jax_platform=args.jax_platform,
            )
        else:
            if args.parity_config is None:
                parser.error(
                    "legacy G0-v1 mode requires --parity-config"
                )
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
