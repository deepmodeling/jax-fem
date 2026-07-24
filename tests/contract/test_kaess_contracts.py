import copy
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
from jsonschema import Draft202012Validator

from jax_fem_am.verification.backend_qualification import (
    ContractValidationError,
    _checkpoint_metric_truth,
    _load_g0_performance_protocol,
    canonical_json_sha256,
    formal_promotion_allowed,
    g0_performance_protocol_sha256,
    inspect_native_checkpoint,
    load_json_strict,
    manifest_acceptance_model_sha256,
    manifest_cpu_hardware_sha256,
    manifest_environment_sha256,
    manifest_input_bundle_sha256,
    ndarray_sha256,
    validate_backend_qualification_bundle,
    validate_paper_comparison_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = (
    REPO_ROOT / "specs" / "001-kaess-paper-reproduction" / "contracts"
)
THRESHOLD_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "inputs" / "threshold-set.json"
)
PARITY_CONFIG_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "inputs" / "paper-parity-config.yaml"
)
APPROVAL_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "inputs" / "g0-approval.json"
)
SOURCE_MANIFEST_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "inputs" / "source-manifest.yaml"
)
PAPER_PATH = (
    REPO_ROOT
    / "cases"
    / "kaess_2023"
    / "references"
    / "cases"
    / "kaess_2023_paper.pdf"
)
FIG8_PATH = (
    REPO_ROOT
    / "cases"
    / "kaess_2023"
    / "references"
    / "digitized"
    / "fig8_sigma_x.csv"
)
FIG9_PATH = (
    REPO_ROOT
    / "cases"
    / "kaess_2023"
    / "references"
    / "digitized"
    / "fig9_bending.csv"
)
H40 = "0" * 40
H64 = "0" * 64
CHECK_IDS = (
    "artifact_rehash",
    "candidate_membership",
    "source_identity",
    "acceptance_model_identity",
    "level_coverage",
    "checkpoint_dtype",
    "checkpoint_shape",
    "comparison_mask_identity",
    "placement_reconciliation",
    "gate_consistency",
    "metric_recalculation",
    "performance_recalculation",
    "performance_protocol_identity",
)
PAPER_COMPARISONS = (
    (
        "figure8_sigma_x_sign_sequence",
        "sigma_x_depth_curve",
        "Figure 8",
        "sign_sequence",
        "unknown",
        None,
    ),
    (
        "figure8_sigma_x_peak_relative_error",
        "sigma_x_depth_curve",
        "Figure 8",
        "relative_error",
        0.0,
        "fraction",
    ),
    (
        "figure8_sigma_x_trough_relative_error",
        "sigma_x_depth_curve",
        "Figure 8",
        "relative_error",
        0.0,
        "fraction",
    ),
    (
        "figure8_sigma_x_zero_crossing_depth_error",
        "sigma_x_zero_crossing",
        "Figure 8",
        "absolute_error",
        0.0,
        "local_element_height",
    ),
    (
        "figure9_bending_curve_nrmse",
        "front_bending_curve",
        "Figure 9",
        "nrmse",
        0.0,
        "fraction",
    ),
    (
        "figure9_max_front_bending_relative_error",
        "max_front_bending",
        "Figure 9",
        "relative_error",
        0.0,
        "fraction",
    ),
    (
        "figure9_max_front_bending_absolute_error",
        "max_front_bending",
        "Figure 9",
        "absolute_error",
        0.0,
        "um",
    ),
    (
        "figure9_release_direction",
        "release_direction",
        "Figure 9",
        "direction",
        "indeterminate",
        None,
    ),
)
BACKEND_NUMERIC_THRESHOLDS = {
    "temperature_field_relative_l2": 0.001,
    "sigma_x_field_relative_l2": 0.01,
    "eqp_field_relative_l2": 0.01,
    "displacement_field_relative_l2": 0.02,
    "peak_temperature_relative_error": 0.001,
    "front_bending_curve_relative_l2": 0.02,
    "max_front_bending_error": 0.5,
    "linear_solve_count_delta_fraction": 0.1,
}
BACKEND_NUMERIC_UNITS = {
    "temperature_field_relative_l2": ("K", "fraction"),
    "sigma_x_field_relative_l2": ("MPa", "fraction"),
    "eqp_field_relative_l2": ("fraction", "fraction"),
    "displacement_field_relative_l2": ("um", "fraction"),
    "peak_temperature_relative_error": ("K", "fraction"),
    "front_bending_curve_relative_l2": ("um", "fraction"),
    "max_front_bending_error": ("um", "um"),
    "linear_solve_count_delta_fraction": ("count", "fraction"),
}


def _schema(name: str) -> dict:
    return json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))


def _validator(name: str) -> Draft202012Validator:
    schema = _schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def _artifact(path: str = "evidence.json") -> dict:
    return {"path": path, "sha256": H64, "size_bytes": 1}


def _real_artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _run_artifact(role: str = "input", path: str = "input.json") -> dict:
    return {"role": role, **_artifact(path)}


def _real_run_artifact(role: str, path: Path) -> dict:
    return {"role": role, **_real_artifact(path)}


def _not_run_stage() -> dict:
    return {
        "status": "not_run",
        "local_assembly_backend": "not_applicable",
        "global_matrix_backend": "not_applicable",
        "linear_solver_backend": "not_applicable",
        "state_residency_backend": "not_applicable",
    }


def _uncertainty() -> dict:
    return {
        "applicability": "not_applicable",
        "method": "not available",
        "value": None,
        "unit": None,
        "evidence": [_artifact()],
    }


def _minimal_run_manifest() -> dict:
    return {
        "schema_version": "kaess.run-manifest/1",
        "run_id": "cpu-1",
        "case_id": "kaess",
        "claim_level": "verification_only",
        "status": "planned",
        "created_utc": "2026-07-24T00:00:00Z",
        "code": {
            "repository": "jax-fem",
            "checkout_path": "/repo",
            "branch": "test",
            "commit": H40,
            "dirty": False,
        },
        "environment": {
            "python": "3.13",
            "jax": "0.10.2",
            "jaxlib": "0.10.2",
            "platform": "linux",
            "hardware": {},
            "variables": {},
        },
        "backend": {
            "mode": "cpu_reference",
            "jax_platform": "cpu",
            "orchestration_backend": "host_python",
            "full_loop_xla": False,
            "precision": "float64",
            "cpu_threads": 1,
            "mkl_threads": 1,
            "omp_threads": 1,
            "thermal": _not_run_stage(),
            "mechanics": _not_run_stage(),
            "release": _not_run_stage(),
            "host_device_transfers": {
                "measured": False,
                "count": None,
                "bytes": None,
            },
            "expected_cpu_operations": [],
            "unexpected_fallbacks": [],
        },
        "command": "true",
        "inputs": [_run_artifact()],
        "phase_gates": {},
        "qoi": [],
        "artifacts": [],
    }


def _minimal_unsupported_qualification() -> dict:
    return {
        "schema_version": "kaess.backend-qualification/2",
        "qualification_id": "q1",
        "execution_mode": "gpu_dominant_experimental",
        "verdict": "unsupported",
        "promotion_eligible": False,
        "reason_code": "no_gpu_device",
        "message": "GPU unavailable",
        "probe_artifacts": [_artifact("probe.json")],
    }


def _minimal_failed_validation() -> dict:
    return {
        "schema_version": "kaess.backend-qualification-validation/1",
        "validation_id": "v1",
        "validator_version": "1",
        "created_utc": "2026-07-24T00:00:00Z",
        "qualification_id": "q1",
        "qualification_artifact": _artifact("qualification.json"),
        "qualification_candidate_run_id": "candidate-1",
        "qualification_cpu_reference_run_id": "cpu-1",
        "qualification_candidate_manifest_artifact": _artifact(
            "candidate.json"
        ),
        "execution_mode": "hybrid_gpu_assembly_cpu_pardiso",
        "identity": {
            "commit": H40,
            "dirty_diff_sha256": None,
            "input_bundle_sha256": H64,
            "acceptance_model_sha256": H64,
            "precision": "float32",
            "cpu_environment_sha256": H64,
            "candidate_environment_sha256": H64,
            "cpu_hardware_sha256": H64,
            "candidate_hardware_sha256": H64,
            "same_cpu_hardware": False,
            "cpu_thread_budget": {
                "cpu_control_threads": 1,
                "candidate_threads": 2,
                "same_budget": False,
            },
            "performance_protocol_sha256": H64,
            "sequential_execution": False,
            "execution_order_artifact": _artifact("order.json"),
            "checkpoint_sha256": H64,
            "checkpoint_dtype": "float32",
            "checkpoint_shape": [1],
            "mask_sha256": H64,
            "candidate_in_qualification": False,
            "mode_matches_manifest": False,
        },
        "checks": {
            key: {"status": "fail", "evidence": [_artifact()]}
            for key in CHECK_IDS
        },
        "recomputed_performance": {
            "measured": False,
            "cpu_median_wall_seconds": None,
            "candidate_median_wall_seconds": None,
            "speedup": None,
            "linear_solve_count_delta_fraction": None,
            "evidence": [_artifact("performance.json")],
        },
        "promotion_eligible": False,
        "verdict": "fail",
    }


def _minimal_blocked_paper_comparison() -> dict:
    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": "kaess.paper-comparison/2",
        "protocol_id": threshold["protocol_id"],
        "case_id": "kaess",
        "run_id": "candidate-1",
        "run_manifest": _artifact("manifest.json"),
        "claim_boundary": {
            "reproduction_level": "public_code_to_code",
            "experimental_validation": False,
        },
        "reference": {
            "doi": "10.3390/ma16062321",
            "source_sha256": H64,
            "digitized_data": [
                {
                    "figure": "Figure 8",
                    **_artifact("fig8.csv"),
                    "uncertainty": _uncertainty(),
                },
                {
                    "figure": "Figure 9",
                    **_artifact("fig9.csv"),
                    "uncertainty": _uncertainty(),
                },
            ],
        },
        "threshold_set": {
            "threshold_set_id": threshold["threshold_set_id"],
            "version": threshold["version"],
            "threshold_set_artifact": _artifact("threshold-set.json"),
            "approved": True,
            "approved_by": threshold["approved_by"],
            "approved_utc": threshold["approved_utc"],
            "approval_artifact": _artifact("g0-approval.json"),
            "metrics": copy.deepcopy(threshold["metrics"]),
        },
        "comparisons": [
            {
                "comparison_id": comparison_id,
                "qoi_id": qoi_id,
                "paper_location": paper_location,
                "metric": metric,
                "value": value,
                "value_unit": value_unit,
                "threshold_metric_id": None,
                "status": "not_comparable",
                "evidence": [_artifact()],
            }
            for (
                comparison_id,
                qoi_id,
                paper_location,
                metric,
                value,
                value_unit,
            ) in PAPER_COMPARISONS
        ],
        "uncertainty": {
            "digitization": _uncertainty(),
            "discretization": _uncertainty(),
            "repeatability": _uncertainty(),
            "input_assumptions": _uncertainty(),
            "combination_method": "reported_separately",
        },
        "semantic_validation": {
            "validator_id": "kaess-paper-comparison-semantic-validator",
            "validator_version": "1",
            "status": "not_run",
            "validated_payload_sha256": H64,
            "run_manifest_sha256": H64,
            "threshold_set_sha256": H64,
            "paper_parity_config_sha256": H64,
            "g0_approval_sha256": H64,
            "artifact": _artifact("semantic-validation.json"),
        },
        "verdict": "blocked",
    }


def _hybrid_stage() -> dict:
    return {
        "status": "used",
        "local_assembly_backend": "gpu",
        "global_matrix_backend": "cpu",
        "linear_solver_backend": "cpu_pardiso",
        "state_residency_backend": "mixed",
    }


def _candidate_hybrid_manifest() -> dict:
    manifest = _minimal_run_manifest()
    manifest.update(
        {
            "run_id": "candidate-1",
            "case_id": "standard-10x30-t150-p250-v850",
            "claim_level": "public_code_to_code",
            "status": "running",
        }
    )
    manifest["environment"]["cuda"] = "13"
    manifest["environment"]["driver"] = "test"
    manifest["environment"]["hardware"] = {
        "cpu": "test-cpu",
        "gpu": "test-gpu",
    }
    manifest["backend"].update(
        {
            "mode": "hybrid_gpu_assembly_cpu_pardiso",
            "jax_platform": "gpu",
            "thermal": _hybrid_stage(),
            "mechanics": _hybrid_stage(),
            "release": _hybrid_stage(),
        }
    )
    manifest["inputs"] = [
        _real_run_artifact("paper_parity_config", PARITY_CONFIG_PATH),
        _real_run_artifact("threshold_set", THRESHOLD_PATH),
        _real_run_artifact("g0_approval", APPROVAL_PATH),
        _real_run_artifact("source_manifest", SOURCE_MANIFEST_PATH),
    ]
    return manifest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _refresh_artifact_identity(payload, target_path: Path) -> None:
    """Refresh every embedded identity that resolves to target_path."""
    target_path = target_path.resolve()
    if isinstance(payload, dict):
        if {"path", "sha256", "size_bytes"} <= set(payload):
            if Path(payload["path"]).resolve() == target_path:
                payload.update(_real_artifact(target_path))
        for value in payload.values():
            _refresh_artifact_identity(value, target_path)
    elif isinstance(payload, list):
        for value in payload:
            _refresh_artifact_identity(value, target_path)


def _rewrite_candidate_and_refresh_bundle(
    qualification_path: Path,
    validation_path: Path,
    candidate_manifest_path: Path,
    candidate_manifest: dict,
) -> None:
    _write_json(candidate_manifest_path, candidate_manifest)
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    _refresh_artifact_identity(qualification, candidate_manifest_path)
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    _refresh_artifact_identity(validation, candidate_manifest_path)
    _refresh_artifact_identity(validation, qualification_path)
    _write_json(validation_path, validation)


def _write_paper_bundle(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = tmp_path / "run-manifest.json"
    manifest = _candidate_hybrid_manifest()
    manifest["status"] = "completed"
    _write_json(manifest_path, manifest)

    semantic_path = tmp_path / "paper-semantic-validation.json"

    report = _minimal_blocked_paper_comparison()
    report["case_id"] = "standard-10x30-t150-p250-v850"
    report["run_manifest"] = _real_artifact(manifest_path)
    report["reference"]["source_sha256"] = hashlib.sha256(
        PAPER_PATH.read_bytes()
    ).hexdigest()
    report["reference"]["digitized_data"] = [
        {
            "figure": "Figure 8",
            **_real_artifact(FIG8_PATH),
            "uncertainty": {
                **_uncertainty(),
                "evidence": [_real_artifact(FIG8_PATH)],
            },
        },
        {
            "figure": "Figure 9",
            **_real_artifact(FIG9_PATH),
            "uncertainty": {
                **_uncertainty(),
                "evidence": [_real_artifact(FIG9_PATH)],
            },
        },
    ]
    report["threshold_set"]["threshold_set_artifact"] = _real_artifact(
        THRESHOLD_PATH
    )
    report["threshold_set"]["approval_artifact"] = _real_artifact(APPROVAL_PATH)
    evidence = _real_artifact(manifest_path)
    for comparison in report["comparisons"]:
        comparison["evidence"] = [evidence]
    for component in report["uncertainty"].values():
        if isinstance(component, dict):
            component["evidence"] = [_real_artifact(FIG8_PATH)]
    report["semantic_validation"].update(
        {
            "status": "pass",
            "run_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "threshold_set_sha256": hashlib.sha256(
                THRESHOLD_PATH.read_bytes()
            ).hexdigest(),
            "paper_parity_config_sha256": hashlib.sha256(
                PARITY_CONFIG_PATH.read_bytes()
            ).hexdigest(),
            "g0_approval_sha256": hashlib.sha256(
                APPROVAL_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    report["verdict"] = "blocked"
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    semantic_payload = {
        "schema_version": "kaess.paper-comparison-semantic-validation/1",
        **{
            key: report["semantic_validation"][key]
            for key in (
                "validator_id",
                "validator_version",
                "status",
                "validated_payload_sha256",
                "run_manifest_sha256",
                "threshold_set_sha256",
                "paper_parity_config_sha256",
                "g0_approval_sha256",
            )
        },
    }
    _write_json(semantic_path, semantic_payload)
    report["semantic_validation"]["artifact"] = _real_artifact(semantic_path)
    report_path = tmp_path / "paper-comparison.json"
    _write_json(report_path, report)
    return report_path, manifest_path


def _write_paper_report_with_refreshed_semantic(
    report_path: Path,
    report: dict,
) -> None:
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    semantic_path = Path(report["semantic_validation"]["artifact"]["path"])
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    for key in semantic:
        if key != "schema_version":
            semantic[key] = report["semantic_validation"][key]
    _write_json(semantic_path, semantic)
    report["semantic_validation"]["artifact"] = _real_artifact(semantic_path)
    _write_json(report_path, report)


def _numeric_metric(metric_id: str, evidence: dict | None = None) -> dict:
    return {
        "metric_id": metric_id,
        "metric_kind": "numeric",
        "cpu_value": 1.0,
        "candidate_value": 1.0,
        "value_unit": BACKEND_NUMERIC_UNITS[metric_id][0],
        "error": 0.0,
        "error_unit": BACKEND_NUMERIC_UNITS[metric_id][1],
        "threshold": BACKEND_NUMERIC_THRESHOLDS[metric_id],
        "comparison_operator": "<=",
        "status": "pass",
        "evidence": evidence or _artifact(),
    }


def _digest_metric(metric_id: str, evidence: dict | None = None) -> dict:
    return {
        "metric_id": metric_id,
        "metric_kind": "digest_match",
        "cpu_sha256": H64,
        "candidate_sha256": H64,
        "match": True,
        "status": "pass",
        "evidence": evidence or _artifact(),
    }


def _categorical_metric(metric_id: str, evidence: dict | None = None) -> dict:
    return {
        "metric_id": metric_id,
        "metric_kind": "categorical_match",
        "cpu_value": "upward",
        "candidate_value": "upward",
        "match": True,
        "status": "pass",
        "evidence": evidence or _artifact(),
    }


def _passed_gate(evidence: dict | None = None) -> dict:
    return {"status": "pass", "evidence": [evidence or _artifact()]}


def _set_identical_checkpoint_metric_truth(
    qualification: dict,
    checkpoint_state: dict[str, np.ndarray],
) -> None:
    active_mask = checkpoint_state["active_mask"]
    numeric_values = {
        "temperature_field_relative_l2": float(
            np.linalg.norm(checkpoint_state["temperature"][active_mask])
        ),
        "sigma_x_field_relative_l2": float(
            np.linalg.norm(checkpoint_state["sigma_x_mpa"][active_mask])
        ),
        "eqp_field_relative_l2": float(
            np.linalg.norm(checkpoint_state["eqp"][active_mask])
        ),
        "displacement_field_relative_l2": float(
            np.linalg.norm(checkpoint_state["displacement_um"][active_mask])
        ),
        "peak_temperature_relative_error": float(
            np.max(checkpoint_state["temperature"][active_mask])
        ),
        "front_bending_curve_relative_l2": float(
            np.linalg.norm(checkpoint_state["front_bending_curve_um"])
        ),
        "max_front_bending_error": float(
            np.max(np.abs(checkpoint_state["front_bending_curve_um"]))
        ),
        "linear_solve_count_delta_fraction": float(
            checkpoint_state["linear_solve_count"].item()
        ),
    }
    for group_name in ("field_metrics", "qoi_metrics", "convergence_metrics"):
        for metric in qualification[group_name]:
            metric_id = metric["metric_id"]
            if metric["metric_kind"] == "numeric":
                metric["cpu_value"] = numeric_values[metric_id]
                metric["candidate_value"] = numeric_values[metric_id]
                metric["error"] = 0.0
            elif metric["metric_kind"] == "digest_match":
                array_name = {
                    "accepted_increment_digest_match": "accepted_increments",
                    "fallback_event_digest_match": "fallback_events",
                }[metric_id]
                digest = ndarray_sha256(checkpoint_state[array_name])
                metric["cpu_sha256"] = digest
                metric["candidate_sha256"] = digest
    digest_arrays = {
        "activation_event_digest_match": "activation_events",
        "active_element_digest_match": "active_mask",
        "phase_state_digest_match": "phase_state",
    }
    for metric in qualification["event_metrics"]:
        digest = ndarray_sha256(checkpoint_state[digest_arrays[metric["metric_id"]]])
        metric["cpu_sha256"] = digest
        metric["candidate_sha256"] = digest


def _passing_qualification(evidence: dict | None = None) -> dict:
    evidence = evidence or _artifact()
    return {
        "schema_version": "kaess.backend-qualification/2",
        "qualification_id": "q1",
        "execution_mode": "hybrid_gpu_assembly_cpu_pardiso",
        "verdict": "pass",
        "promotion_eligible": False,
        "levels": ["kernel"],
        "cpu_reference_run_ids": ["cpu-1"],
        "candidate_run_ids": ["candidate-1"],
        "level_run_pairs": {
            "kernel": {
                "cpu_run_ids": ["cpu-1"],
                "candidate_run_ids": ["candidate-1"],
                "status": "pass",
                "evidence": [evidence],
            }
        },
        "source_identity": {
            "same_identity": True,
            "commit": H40,
            "dirty_diff_sha256": None,
            "input_bundle_sha256": H64,
            "acceptance_model_sha256": H64,
        },
        "comparison_scope": {
            "mask_id": "printed_and_active",
            "mask_artifact": evidence,
            "active_or_printed_only": True,
            "paper_path_ids": [
                "figure7-stress-path-v1",
                "figure7-bending-path-v1",
            ],
            "excluded_domains": ["inactive", "permanent_powder"],
        },
        "checkpoint_precision": "float64",
        "stage_gates": {
            "backend_parity": _passed_gate(evidence),
            "energy_audit": _passed_gate(evidence),
            "convergence_audit": _passed_gate(evidence),
        },
        "field_metrics": [
            _numeric_metric(metric_id, evidence)
            for metric_id in (
                "temperature_field_relative_l2",
                "sigma_x_field_relative_l2",
                "eqp_field_relative_l2",
                "displacement_field_relative_l2",
            )
        ],
        "event_metrics": [
            _digest_metric(metric_id, evidence)
            for metric_id in (
                "activation_event_digest_match",
                "active_element_digest_match",
                "phase_state_digest_match",
            )
        ],
        "qoi_metrics": [
            _numeric_metric("peak_temperature_relative_error", evidence),
            _numeric_metric("front_bending_curve_relative_l2", evidence),
            _numeric_metric("max_front_bending_error", evidence),
            _categorical_metric("release_direction_match", evidence),
        ],
        "convergence_metrics": [
            _numeric_metric("linear_solve_count_delta_fraction", evidence),
            _digest_metric("accepted_increment_digest_match", evidence),
            _digest_metric("fallback_event_digest_match", evidence),
        ],
        "placement_evidence": {
            "placement_verified": True,
            "run_manifest_artifacts": [evidence],
            "profiler_artifacts": [evidence],
            "orchestration_backend": "host_python",
            "full_loop_xla": False,
            "cpu_pardiso_calls": 1,
            "unexpected_fallback_count": 0,
        },
        "performance": {
            "measured": False,
            "cold_wall_seconds": None,
            "steady_wall_seconds": None,
            "speedup": None,
            "cpu_threads": None,
            "linear_solve_count_delta_fraction": None,
            "evidence": [evidence],
        },
        "numerically_qualified": True,
        "performance_qualified": False,
    }


def _passing_nonpromotion_validation(evidence: dict | None = None) -> dict:
    evidence = evidence or _artifact()
    validation = _minimal_failed_validation()
    validation.update(
        {
            "execution_mode": "hybrid_gpu_assembly_cpu_pardiso",
            "promotion_eligible": False,
            "verdict": "pass_not_promotion_eligible",
        }
    )
    validation["identity"].update(
        {
            "precision": "float64",
            "same_cpu_hardware": True,
            "cpu_thread_budget": {
                "cpu_control_threads": 1,
                "candidate_threads": 1,
                "same_budget": True,
            },
            "sequential_execution": True,
            "checkpoint_dtype": "float64",
            "candidate_in_qualification": True,
            "mode_matches_manifest": True,
        }
    )
    validation["identity"]["execution_order_artifact"] = evidence
    validation["checks"] = {
        key: {"status": "pass", "evidence": [evidence]}
        for key in CHECK_IDS
    }
    validation["recomputed_performance"]["evidence"] = [evidence]
    return validation


def _write_backend_bundle(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    cpu_checkpoint = tmp_path / "cpu-checkpoint.npz"
    candidate_checkpoint = tmp_path / "candidate-checkpoint.npz"
    active_mask = np.array([True, True], dtype=np.bool_)
    checkpoint_state = {
        "temperature": np.array([300.0, 301.0], dtype=np.float64),
        "sigma_x_mpa": np.array([10.0, 20.0], dtype=np.float64),
        "eqp": np.array([0.1, 0.2], dtype=np.float64),
        "displacement_um": np.zeros((2, 3), dtype=np.float64),
        "front_bending_curve_um": np.array([0.0, 1.0], dtype=np.float64),
        "activation_events": np.array([1, 2], dtype=np.int64),
        "active_mask": active_mask,
        "phase_state": np.array([1, 1], dtype=np.int64),
        "accepted_increments": np.array([1, 2, 3], dtype=np.int64),
        "fallback_events": np.array([], dtype=np.int64),
        "linear_solve_count": np.array(5, dtype=np.int64),
    }
    for path in (cpu_checkpoint, candidate_checkpoint):
        np.savez(path, **checkpoint_state)
    mask_path = tmp_path / "active-mask.npy"
    np.save(mask_path, active_mask, allow_pickle=False)
    profiler_path = tmp_path / "profiler.json"
    performance_path = tmp_path / "performance.json"
    order_path = tmp_path / "execution-order.json"
    level_path = tmp_path / "kernel-level.json"
    _write_json(
        level_path,
        {
            "schema_version": "kaess.qualification-level/1",
            "level": "kernel",
            "case_id": "standard-10x30-t150-p250-v850",
        },
    )
    _write_json(
        profiler_path,
        {
            "run_id": "candidate-1",
            "execution_mode": "hybrid_gpu_assembly_cpu_pardiso",
            "orchestration_backend": "host_python",
            "full_loop_xla": False,
            "cpu_pardiso_calls": 1,
            "unexpected_fallback_count": 0,
            "stages": {
                "thermal": _hybrid_stage(),
                "mechanics": _hybrid_stage(),
                "release": _hybrid_stage(),
            },
        },
    )
    _write_json(
        performance_path,
        {
            "schema_version": "kaess.performance-evidence/1",
            "measured": False,
            "cpu_run_ids": ["cpu-1"],
            "candidate_run_ids": ["candidate-1"],
            "cpu_threads": 1,
            "cpu_wall_seconds_samples": [],
            "candidate_wall_seconds_samples": [],
            "cpu_linear_solve_count_samples": [],
            "candidate_linear_solve_count_samples": [],
        },
    )
    _write_json(
        order_path,
        {
            "sequential": True,
            "runs": [
                {
                    "run_id": "cpu-1",
                    "started_utc": "2026-07-24T00:00:00Z",
                    "completed_utc": "2026-07-24T00:01:00Z",
                },
                {
                    "run_id": "candidate-1",
                    "started_utc": "2026-07-24T00:02:00Z",
                    "completed_utc": "2026-07-24T00:03:00Z",
                },
            ],
        },
    )

    frozen_inputs = [
        _real_run_artifact("paper_parity_config", PARITY_CONFIG_PATH),
        _real_run_artifact("threshold_set", THRESHOLD_PATH),
        _real_run_artifact("g0_approval", APPROVAL_PATH),
        _real_run_artifact("qualification_level", level_path),
    ]
    cpu_manifest = _minimal_run_manifest()
    cpu_manifest.update(
        {
            "case_id": "standard-10x30-t150-p250-v850",
            "status": "completed",
        }
    )
    cpu_manifest["environment"]["hardware"] = {"cpu": "test-cpu"}
    cpu_manifest["inputs"] = copy.deepcopy(frozen_inputs)
    cpu_manifest["artifacts"] = [
        _real_run_artifact("native_float64_checkpoint", cpu_checkpoint),
        _real_run_artifact("comparison_mask", mask_path),
        _real_run_artifact("profiler", profiler_path),
    ]

    candidate_manifest = _candidate_hybrid_manifest()
    candidate_manifest["status"] = "completed"
    candidate_manifest["inputs"] = copy.deepcopy(frozen_inputs)
    candidate_manifest["artifacts"] = [
        _real_run_artifact(
            "native_float64_checkpoint", candidate_checkpoint
        ),
        _real_run_artifact("comparison_mask", mask_path),
        _real_run_artifact("profiler", profiler_path),
    ]

    cpu_manifest_path = tmp_path / "cpu-manifest.json"
    candidate_manifest_path = tmp_path / "candidate-manifest.json"
    _write_json(cpu_manifest_path, cpu_manifest)
    _write_json(candidate_manifest_path, candidate_manifest)

    profiler_artifact = _real_artifact(profiler_path)
    qualification = _passing_qualification(profiler_artifact)
    _set_identical_checkpoint_metric_truth(qualification, checkpoint_state)
    qualification["performance"]["evidence"] = [
        _real_artifact(performance_path)
    ]
    qualification["source_identity"].update(
        {
            "commit": candidate_manifest["code"]["commit"],
            "dirty_diff_sha256": candidate_manifest["code"].get(
                "dirty_diff_sha256"
            ),
            "input_bundle_sha256": manifest_input_bundle_sha256(
                candidate_manifest
            ),
            "acceptance_model_sha256": manifest_acceptance_model_sha256(
                candidate_manifest
            ),
        }
    )
    qualification["comparison_scope"]["mask_artifact"] = _real_artifact(
        mask_path
    )
    qualification["level_run_pairs"]["kernel"]["evidence"] = [
        _real_artifact(candidate_manifest_path)
    ]
    qualification["placement_evidence"]["run_manifest_artifacts"] = [
        _real_artifact(cpu_manifest_path),
        _real_artifact(candidate_manifest_path),
    ]
    qualification["placement_evidence"]["profiler_artifacts"] = [
        profiler_artifact
    ]
    qualification_path = tmp_path / "qualification.json"
    _write_json(qualification_path, qualification)

    validation = _passing_nonpromotion_validation(profiler_artifact)
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    validation["qualification_candidate_manifest_artifact"] = _real_artifact(
        candidate_manifest_path
    )
    validation["identity"].update(
        {
            "commit": candidate_manifest["code"]["commit"],
            "dirty_diff_sha256": candidate_manifest["code"].get(
                "dirty_diff_sha256"
            ),
            "input_bundle_sha256": manifest_input_bundle_sha256(
                candidate_manifest
            ),
            "acceptance_model_sha256": manifest_acceptance_model_sha256(
                candidate_manifest
            ),
            "cpu_environment_sha256": manifest_environment_sha256(
                cpu_manifest
            ),
            "candidate_environment_sha256": manifest_environment_sha256(
                candidate_manifest
            ),
            "cpu_hardware_sha256": manifest_cpu_hardware_sha256(cpu_manifest),
            "candidate_hardware_sha256": manifest_cpu_hardware_sha256(
                candidate_manifest
            ),
            "performance_protocol_sha256": g0_performance_protocol_sha256(
                PARITY_CONFIG_PATH,
                THRESHOLD_PATH,
                APPROVAL_PATH,
            ),
            "execution_order_artifact": _real_artifact(order_path),
            "checkpoint_sha256": hashlib.sha256(
                candidate_checkpoint.read_bytes()
            ).hexdigest(),
            "checkpoint_dtype": "float64",
            "checkpoint_shape": [2],
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        }
    )
    validation_path = tmp_path / "qualification-validation.json"
    _write_json(validation_path, validation)
    return (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    )


def _enable_measured_performance(
    qualification_path: Path,
    validation_path: Path,
    *,
    reported_speedup: float = 1.25,
) -> None:
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    performance_path = Path(qualification["performance"]["evidence"][0]["path"])
    performance_evidence = json.loads(
        performance_path.read_text(encoding="utf-8")
    )
    performance_evidence.update(
        {
            "measured": True,
            "cpu_wall_seconds_samples": [10.0, 10.0],
            "candidate_wall_seconds_samples": [8.0, 8.0],
            "cpu_linear_solve_count_samples": [5, 5],
            "candidate_linear_solve_count_samples": [5, 5],
        }
    )
    _write_json(performance_path, performance_evidence)
    qualification["performance"]["evidence"] = [
        _real_artifact(performance_path)
    ]
    evidence = qualification["performance"]["evidence"]
    qualification["performance"].update(
        {
            "measured": True,
            "cold_wall_seconds": 10.0,
            "steady_wall_seconds": 8.0,
            "speedup": reported_speedup,
            "cpu_threads": 1,
            "linear_solve_count_delta_fraction": 0.0,
            "cpu_wall_seconds_samples": [10.0, 10.0],
            "candidate_wall_seconds_samples": [8.0, 8.0],
            "cpu_linear_solve_count_samples": [5, 5],
            "candidate_linear_solve_count_samples": [5, 5],
        }
    )
    qualification["performance_qualified"] = True
    qualification["stage_gates"]["performance_gate"] = {
        "status": "pass",
        "evidence": evidence,
    }
    _write_json(qualification_path, qualification)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    validation["recomputed_performance"].update(
        {
            "measured": True,
            "cpu_median_wall_seconds": 10.0,
            "candidate_median_wall_seconds": 8.0,
            "speedup": 1.25,
            "linear_solve_count_delta_fraction": 0.0,
        }
    )
    _write_json(validation_path, validation)


SCHEMA_BUILDERS = {
    "run-manifest.schema.json": _minimal_run_manifest,
    "paper-comparison.schema.json": _minimal_blocked_paper_comparison,
    "backend-qualification.schema.json": _minimal_unsupported_qualification,
    "backend-qualification-validation.schema.json": _minimal_failed_validation,
}


def test_contract_schemas_are_valid_draft_2020_12():
    for schema_name in SCHEMA_BUILDERS:
        schema = _schema(schema_name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("schema_name", SCHEMA_BUILDERS)
def test_contract_minimal_objects_are_valid(schema_name):
    errors = list(
        _validator(schema_name).iter_errors(SCHEMA_BUILDERS[schema_name]())
    )
    assert not errors, "\n".join(error.message for error in errors)


@pytest.mark.parametrize("schema_name", SCHEMA_BUILDERS)
def test_contract_rejects_missing_top_level_required_field(schema_name):
    payload = SCHEMA_BUILDERS[schema_name]()
    payload.pop("schema_version")

    assert not _validator(schema_name).is_valid(payload)


def test_run_manifest_rejects_invalid_datetime_format():
    payload = _minimal_run_manifest()
    payload["created_utc"] = "yesterday"

    assert not _validator("run-manifest.schema.json").is_valid(payload)


def test_run_manifest_rejects_dirty_checkout_without_diff_hash():
    payload = _minimal_run_manifest()
    payload["code"]["dirty"] = True

    assert not _validator("run-manifest.schema.json").is_valid(payload)


def test_run_manifest_rejects_non_float64_precision():
    payload = _minimal_run_manifest()
    payload["backend"]["precision"] = "float32"

    assert not _validator("run-manifest.schema.json").is_valid(payload)


def test_paper_comparison_rejects_empty_uncertainty_payload():
    payload = _minimal_blocked_paper_comparison()
    payload["uncertainty"]["digitization"] = {
        "applicability": "bounded",
        "method": "unknown",
        "value": {"junk": 1},
        "unit": "fraction",
        "evidence": [_artifact()],
    }

    assert not _validator("paper-comparison.schema.json").is_valid(payload)


def test_paper_comparison_requires_figure_8_and_figure_9_evidence():
    payload = _minimal_blocked_paper_comparison()
    payload["reference"]["digitized_data"][1]["figure"] = "Figure 8"

    assert not _validator("paper-comparison.schema.json").is_valid(payload)


def test_paper_comparison_requires_all_17_frozen_threshold_metrics():
    payload = _minimal_blocked_paper_comparison()
    payload["threshold_set"]["metrics"].pop()

    assert not _validator("paper-comparison.schema.json").is_valid(payload)


def test_paper_comparison_requires_null_threshold_for_not_comparable():
    payload = _minimal_blocked_paper_comparison()
    payload["comparisons"][0]["threshold_metric_id"] = (
        "figure9_bending_curve_nrmse"
    )

    assert not _validator("paper-comparison.schema.json").is_valid(payload)


def test_paper_comparison_requires_threshold_for_pass_or_fail():
    payload = _minimal_blocked_paper_comparison()
    payload["comparisons"][4]["status"] = "pass"

    assert not _validator("paper-comparison.schema.json").is_valid(payload)


def test_paper_comparison_rejects_all_fail_partial_verdict():
    payload = _minimal_blocked_paper_comparison()
    payload["verdict"] = "partial"
    payload["semantic_validation"]["status"] = "pass"
    for comparison in payload["comparisons"]:
        comparison["status"] = "fail"
        comparison["threshold_metric_id"] = "figure9_bending_curve_nrmse"

    assert not _validator("paper-comparison.schema.json").is_valid(payload)


def test_unsupported_qualification_cannot_be_promoted():
    payload = _minimal_unsupported_qualification()
    payload["promotion_eligible"] = True

    assert not _validator("backend-qualification.schema.json").is_valid(payload)


def test_validation_promotion_requires_promotion_verdict():
    payload = _minimal_failed_validation()
    payload["promotion_eligible"] = True

    assert not _validator(
        "backend-qualification-validation.schema.json"
    ).is_valid(payload)


def test_passing_validation_rejects_nonsequential_execution():
    payload = _passing_nonpromotion_validation()
    payload["identity"]["sequential_execution"] = False

    assert not _validator(
        "backend-qualification-validation.schema.json"
    ).is_valid(payload)


def test_strict_json_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"run_id": "a", "run_id": "b"}\n', encoding="utf-8")

    with pytest.raises(ContractValidationError, match="duplicate_key"):
        load_json_strict(path)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_loader_rejects_nonfinite_numbers(tmp_path, token):
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value": {token}}}\n', encoding="utf-8")

    with pytest.raises(ContractValidationError, match="nonfinite_number"):
        load_json_strict(path)


def test_paper_semantics_bind_report_to_g0_and_current_run(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)

    validate_paper_comparison_bundle(
        report_path,
        run_manifest_path=manifest_path,
        parity_config_path=PARITY_CONFIG_PATH,
        threshold_set_path=THRESHOLD_PATH,
        approval_record_path=APPROVAL_PATH,
        source_manifest_path=SOURCE_MANIFEST_PATH,
        artifact_root=REPO_ROOT,
    )


def test_paper_semantics_reject_threshold_content_drift(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["threshold_set"]["metrics"][0]["value"] *= 2
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="threshold_metrics"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_wrong_g0_hash(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["semantic_validation"]["g0_approval_sha256"] = H64
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="g0_approval_hash"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_run_without_frozen_threshold_input(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"] = [
        item for item in manifest["inputs"] if item["role"] != "threshold_set"
    ]
    _write_json(manifest_path, manifest)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    updated_manifest_artifact = _real_artifact(manifest_path)
    report["run_manifest"] = updated_manifest_artifact
    for comparison in report["comparisons"]:
        comparison["evidence"] = [updated_manifest_artifact]
    report["semantic_validation"]["run_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="run_threshold_binding"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_wrong_current_run_id(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["run_id"] = "other-run"
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="run_id"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_threshold_misbinding(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["comparisons"][4]["threshold_metric_id"] = (
        "hybrid_temperature_qoi_difference"
    )
    report["comparisons"][4]["status"] = "pass"
    report["comparisons"][4]["value"] = 0.0005
    report["verdict"] = "partial"
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="threshold_binding"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_passing_nonpromotion_qualification_is_schema_valid():
    assert _validator("backend-qualification.schema.json").is_valid(
        _passing_qualification()
    )
    assert _validator(
        "backend-qualification-validation.schema.json"
    ).is_valid(_passing_nonpromotion_validation())


def test_qualification_rejects_untyped_metric():
    payload = _passing_qualification()
    payload["field_metrics"][0].pop("metric_kind")

    assert not _validator("backend-qualification.schema.json").is_valid(payload)


def test_qualification_rejects_kernel_only_promotion():
    payload = _passing_qualification()
    payload["promotion_eligible"] = True

    assert not _validator("backend-qualification.schema.json").is_valid(payload)


def test_qualification_rejects_experimental_promotion():
    payload = _passing_qualification()
    payload["execution_mode"] = "gpu_dominant_experimental"
    payload["promotion_eligible"] = True

    assert not _validator("backend-qualification.schema.json").is_valid(payload)


def test_run_manifest_rejects_hybrid_placement_claimed_as_full_gpu():
    payload = _candidate_hybrid_manifest()
    payload["backend"]["mode"] = "full_gpu"

    assert not _validator("run-manifest.schema.json").is_valid(payload)


def test_formal_promotion_requires_verdict_and_boolean():
    validation = _passing_nonpromotion_validation()
    assert formal_promotion_allowed(validation) is False

    validation["verdict"] = "pass_promotion_eligible"
    assert formal_promotion_allowed(validation) is False

    validation["promotion_eligible"] = True
    assert formal_promotion_allowed(validation) is True


def test_native_checkpoint_rejects_float32_state(tmp_path):
    path = tmp_path / "checkpoint.npz"
    np.savez(
        path,
        temperature=np.array([300.0], dtype=np.float32),
        active_mask=np.array([True]),
    )

    with pytest.raises(ContractValidationError, match="checkpoint_dtype"):
        inspect_native_checkpoint(path)


def test_native_checkpoint_rejects_nonfinite_state(tmp_path):
    path = tmp_path / "checkpoint.npz"
    np.savez(
        path,
        temperature=np.array([np.nan], dtype=np.float64),
        active_mask=np.array([True]),
    )

    with pytest.raises(ContractValidationError, match="checkpoint_finite"):
        inspect_native_checkpoint(path)


def test_backend_semantics_validate_current_candidate_bundle(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)

    validate_backend_qualification_bundle(
        qualification_path,
        validation_path=validation_path,
        candidate_manifest_path=candidate_manifest_path,
        cpu_manifest_paths=[cpu_manifest_path],
        parity_config_path=PARITY_CONFIG_PATH,
        artifact_root=REPO_ROOT,
    )


def test_backend_semantics_reject_candidate_outside_qualification(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    qualification["candidate_run_ids"] = ["different-candidate"]
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="candidate_membership"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_environment_identity_drift(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["identity"]["candidate_environment_sha256"] = H64
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="environment_identity"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_execution_mode_drift(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["execution_mode"] = "full_gpu"
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="execution_mode"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_thread_budget_drift(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["identity"]["cpu_thread_budget"]["candidate_threads"] = 2
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="thread_budget"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_cpu_hardware_identity_drift(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["identity"]["candidate_hardware_sha256"] = H64
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="hardware_identity"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_artifact_hash_drift(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"]["sha256"] = H64
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="artifact_rehash"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_recalculate_metric_status(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    qualification["field_metrics"][0]["error"] = 1.0
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="metric_recalculation"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_conflicting_optional_gate(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    qualification["stage_gates"]["build_complete"] = {
        "status": "fail",
        "evidence": qualification["performance"]["evidence"],
    }
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="gate_consistency"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_mask_identity_drift(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["identity"]["mask_sha256"] = H64
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="comparison_mask_identity"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_recompute_measured_performance(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    _enable_measured_performance(qualification_path, validation_path)

    validate_backend_qualification_bundle(
        qualification_path,
        validation_path=validation_path,
        candidate_manifest_path=candidate_manifest_path,
        cpu_manifest_paths=[cpu_manifest_path],
        parity_config_path=PARITY_CONFIG_PATH,
        artifact_root=REPO_ROOT,
    )


def test_backend_semantics_reject_forged_speedup(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_manifest_path,
        cpu_manifest_path,
    ) = _write_backend_bundle(tmp_path)
    _enable_measured_performance(
        qualification_path,
        validation_path,
        reported_speedup=9.0,
    )

    with pytest.raises(
        ContractValidationError, match="performance_recalculation"
    ):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_manifest_path,
            cpu_manifest_paths=[cpu_manifest_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_numeric_metric_forgery(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["field_metrics"][0]["candidate_value"] = 999.0
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="metric_recalculation"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_numeric_threshold_forgery(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["field_metrics"][0]["threshold"] = 1.0
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="metric_recalculation"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_numeric_unit_forgery(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    max_bending = next(
        metric
        for metric in qualification["qoi_metrics"]
        if metric["metric_id"] == "max_front_bending_error"
    )
    max_bending["value_unit"] = "K"
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="metric_recalculation"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_digest_match_forgery(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["event_metrics"][0]["candidate_sha256"] = "1" * 64
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="metric_recalculation"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_ghost_candidate_ids(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["candidate_run_ids"].append("ghost-candidate")
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="candidate_membership"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_profiler_summary_disagreement(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["placement_evidence"]["cpu_pardiso_calls"] = 2
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(
        ContractValidationError, match="placement_reconciliation"
    ):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_mkl_thread_budget_drift(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["backend"]["mkl_threads"] = 2
    _rewrite_candidate_and_refresh_bundle(
        qualification_path,
        validation_path,
        candidate_path,
        candidate,
    )

    with pytest.raises(ContractValidationError, match="thread_budget"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_secondary_checkpoint_shape_drift(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    checkpoint_record = next(
        item
        for item in candidate["artifacts"]
        if item["role"] == "native_float64_checkpoint"
    )
    checkpoint_path = Path(checkpoint_record["path"])
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        state = {name: checkpoint[name] for name in checkpoint.files}
    state["displacement_um"] = np.zeros((3, 3), dtype=np.float64)
    np.savez(checkpoint_path, **state)
    checkpoint_record.update(_real_artifact(checkpoint_path))
    _rewrite_candidate_and_refresh_bundle(
        qualification_path,
        validation_path,
        candidate_path,
        candidate,
    )

    with pytest.raises(ContractValidationError, match="checkpoint_shape"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_checkpoint_value_metric_drift(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    checkpoint_record = next(
        item
        for item in candidate["artifacts"]
        if item["role"] == "native_float64_checkpoint"
    )
    checkpoint_path = Path(checkpoint_record["path"])
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        state = {name: checkpoint[name] for name in checkpoint.files}
    state["temperature"] = np.array([600.0, -600.0], dtype=np.float64)
    np.savez(checkpoint_path, **state)
    checkpoint_record.update(_real_artifact(checkpoint_path))
    _rewrite_candidate_and_refresh_bundle(
        qualification_path,
        validation_path,
        candidate_path,
        candidate,
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["identity"]["checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="metric_recalculation"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_checkpoint_mask_content_drift(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    checkpoint_record = next(
        item
        for item in candidate["artifacts"]
        if item["role"] == "native_float64_checkpoint"
    )
    checkpoint_path = Path(checkpoint_record["path"])
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        state = {name: checkpoint[name] for name in checkpoint.files}
    state["active_mask"] = np.array([True, False], dtype=np.bool_)
    np.savez(checkpoint_path, **state)
    checkpoint_record.update(_real_artifact(checkpoint_path))
    _rewrite_candidate_and_refresh_bundle(
        qualification_path,
        validation_path,
        candidate_path,
        candidate,
    )

    with pytest.raises(
        ContractValidationError, match="comparison_mask_identity"
    ):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_overlapping_execution_order(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    order_path = Path(
        validation["identity"]["execution_order_artifact"]["path"]
    )
    order = json.loads(order_path.read_text(encoding="utf-8"))
    order["runs"][1]["started_utc"] = "2026-07-24T00:00:30Z"
    _write_json(order_path, order)
    validation["identity"]["execution_order_artifact"] = _real_artifact(
        order_path
    )
    _write_json(validation_path, validation)

    with pytest.raises(
        ContractValidationError, match="performance_protocol_identity"
    ):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_decoy_qualification_artifact(tmp_path):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    profiler_path = Path(
        json.loads(candidate_path.read_text(encoding="utf-8"))["artifacts"][2][
            "path"
        ]
    )
    validation["qualification_artifact"] = _real_artifact(profiler_path)
    _write_json(validation_path, validation)

    with pytest.raises(
        ContractValidationError, match="qualification_artifact_binding"
    ):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_small_value_bound_to_wrong_threshold(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparison = report["comparisons"][4]
    comparison["threshold_metric_id"] = "hybrid_temperature_qoi_difference"
    comparison["value"] = 0.0005
    comparison["status"] = "pass"
    report["verdict"] = "partial"
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="threshold_binding"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_decoy_run_manifest_artifact(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    decoy_path = tmp_path / "decoy.json"
    _write_json(decoy_path, {"not": "a run manifest"})
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["run_manifest"] = _real_artifact(decoy_path)
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="run_manifest_binding"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_running_manifest(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    _write_json(manifest_path, manifest)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _refresh_artifact_identity(report, manifest_path)
    report["semantic_validation"]["run_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="run_status"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_conflicting_semantic_artifact(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    semantic_path = Path(report["semantic_validation"]["artifact"]["path"])
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["status"] = "fail"
    _write_json(semantic_path, semantic)
    report["semantic_validation"]["artifact"] = _real_artifact(semantic_path)
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="semantic_artifact"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_reject_performance_samples_without_raw_evidence(
    tmp_path,
):
    paths = _write_backend_bundle(tmp_path)
    qualification_path, validation_path, candidate_path, cpu_path = paths
    _enable_measured_performance(qualification_path, validation_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["performance"].update(
        {
            "cpu_wall_seconds_samples": [1.0e6, 1.0e6],
            "candidate_wall_seconds_samples": [1.0, 1.0],
            "speedup": 1.0e6,
        }
    )
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(
        ContractValidationError, match="performance_recalculation"
    ):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_reject_comparable_value_without_typed_qoi_evidence(
    tmp_path,
):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparison = report["comparisons"][4]
    comparison.update(
        {
            "threshold_metric_id": "figure9_bending_curve_nrmse",
            "status": "pass",
            "value": 0.05,
        }
    )
    report["verdict"] = "partial"
    report["semantic_validation"]["validated_payload_sha256"] = (
        canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key != "semantic_validation"
            }
        )
    )
    _write_json(report_path, report)

    with pytest.raises(ContractValidationError, match="paper_metric_evidence"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_g0_protocol_rejects_config_metric_drift(tmp_path):
    config = json.loads(PARITY_CONFIG_PATH.read_text(encoding="utf-8"))
    speedup = next(
        metric
        for metric in config["threshold_set"]["metrics"]
        if metric["metric_id"] == "accelerated_wall_speedup"
    )
    speedup["value"] = 999.0
    config_path = tmp_path / "paper-parity-config.json"
    _write_json(config_path, config)

    with pytest.raises(
        ContractValidationError, match="performance_protocol_identity"
    ):
        _load_g0_performance_protocol(
            config_path,
            THRESHOLD_PATH,
            APPROVAL_PATH,
        )


def test_checkpoint_metric_truth_uses_active_domain_mask():
    def checkpoint(temperature: np.ndarray, active_value: float) -> dict:
        return {
            "state": {
                "temperature": temperature,
                "sigma_x_mpa": np.array(
                    [active_value, 1.0e12], dtype=np.float64
                ),
                "eqp": np.array(
                    [active_value, 1.0e12], dtype=np.float64
                ),
                "displacement_um": np.array(
                    [
                        [active_value, 0.0, 0.0],
                        [1.0e12, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                "front_bending_curve_um": np.array(
                    [0.0, 1.0], dtype=np.float64
                ),
                "activation_events": np.array([1, 2], dtype=np.int64),
                "active_mask": np.array([True, False], dtype=np.bool_),
                "phase_state": np.array([1, 0], dtype=np.int64),
                "accepted_increments": np.array([1], dtype=np.int64),
                "fallback_events": np.array([], dtype=np.int64),
                "linear_solve_count": np.array(1, dtype=np.int64),
            }
        }

    truth = _checkpoint_metric_truth(
        checkpoint(np.array([1.0, 1.0e12], dtype=np.float64), 1.0),
        checkpoint(np.array([2.0, 1.0e12], dtype=np.float64), 2.0),
    )

    for metric_id in (
        "temperature_field_relative_l2",
        "sigma_x_field_relative_l2",
        "eqp_field_relative_l2",
        "displacement_field_relative_l2",
    ):
        assert truth[metric_id]["error"] == pytest.approx(1.0)
    assert truth["peak_temperature_relative_error"]["error"] == pytest.approx(
        1.0
    )


def test_backend_semantics_rejects_run_reused_across_levels(tmp_path):
    (
        qualification_path,
        validation_path,
        candidate_path,
        cpu_path,
    ) = _write_backend_bundle(tmp_path)
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    qualification["levels"].append("small_domain")
    qualification["level_run_pairs"]["small_domain"] = {
        "cpu_run_ids": ["cpu-1"],
        "candidate_run_ids": ["candidate-1"],
        "status": "pass",
        "evidence": [_real_artifact(candidate_path)],
    }
    _write_json(qualification_path, qualification)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    _write_json(validation_path, validation)

    with pytest.raises(ContractValidationError, match="level_coverage"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_rejects_qualification_level_artifact_drift(
    tmp_path,
):
    (
        qualification_path,
        validation_path,
        candidate_path,
        cpu_path,
    ) = _write_backend_bundle(tmp_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    wrong_level_path = tmp_path / "wrong-level.json"
    _write_json(
        wrong_level_path,
        {
            "schema_version": "kaess.qualification-level/1",
            "level": "small_domain",
            "case_id": candidate["case_id"],
        },
    )
    level_record = next(
        item
        for item in candidate["inputs"]
        if item["role"] == "qualification_level"
    )
    level_record.update(
        _real_run_artifact("qualification_level", wrong_level_path)
    )
    _rewrite_candidate_and_refresh_bundle(
        qualification_path,
        validation_path,
        candidate_path,
        candidate,
    )

    with pytest.raises(ContractValidationError, match="level_coverage"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "jax-fem-fork"),
        ("checkout_path", "/different/repo"),
        ("branch", "different-branch"),
    ],
)
def test_backend_semantics_rejects_checkout_identity_drift(
    tmp_path,
    field,
    value,
):
    (
        qualification_path,
        validation_path,
        candidate_path,
        cpu_path,
    ) = _write_backend_bundle(tmp_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["code"][field] = value
    _rewrite_candidate_and_refresh_bundle(
        qualification_path,
        validation_path,
        candidate_path,
        candidate,
    )

    with pytest.raises(ContractValidationError, match="source_identity"):
        validate_backend_qualification_bundle(
            qualification_path,
            validation_path=validation_path,
            candidate_manifest_path=candidate_path,
            cpu_manifest_paths=[cpu_path],
            parity_config_path=PARITY_CONFIG_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_rejects_unbound_source_hash(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["reference"]["source_sha256"] = H64
    _write_paper_report_with_refreshed_semantic(report_path, report)

    with pytest.raises(ContractValidationError, match="paper_source_identity"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


def test_paper_semantics_rejects_missing_source_manifest_input(tmp_path):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"] = [
        item
        for item in manifest["inputs"]
        if item["role"] != "source_manifest"
    ]
    _write_json(manifest_path, manifest)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    _refresh_artifact_identity(report, manifest_path)
    report["semantic_validation"]["run_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    _write_paper_report_with_refreshed_semantic(report_path, report)

    with pytest.raises(
        ContractValidationError,
        match="run_source_manifest_binding",
    ):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            artifact_root=REPO_ROOT,
        )


@pytest.mark.parametrize("tamper", ["wrong_pdf_hash", "path_escape"])
def test_paper_semantics_rehashes_authoritative_source_manifest(
    tmp_path,
    tamper,
):
    report_path, manifest_path = _write_paper_bundle(tmp_path)
    source_manifest = json.loads(
        SOURCE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    paper_entry = next(
        item
        for item in source_manifest["evidence"]
        if item["evidence_id"] == "paper-pdf"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if tamper == "wrong_pdf_hash":
        paper_entry["sha256"] = H64
        report["reference"]["source_sha256"] = H64
    else:
        paper_entry["repository_path"] = "../outside-paper.pdf"
    tampered_source_path = tmp_path / f"source-manifest-{tamper}.json"
    _write_json(tampered_source_path, source_manifest)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_record = next(
        item
        for item in manifest["inputs"]
        if item["role"] == "source_manifest"
    )
    source_record.update(
        _real_run_artifact("source_manifest", tampered_source_path)
    )
    _write_json(manifest_path, manifest)
    _refresh_artifact_identity(report, manifest_path)
    report["semantic_validation"]["run_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    _write_paper_report_with_refreshed_semantic(report_path, report)

    with pytest.raises(ContractValidationError, match="paper_source_identity"):
        validate_paper_comparison_bundle(
            report_path,
            run_manifest_path=manifest_path,
            parity_config_path=PARITY_CONFIG_PATH,
            threshold_set_path=THRESHOLD_PATH,
            approval_record_path=APPROVAL_PATH,
            source_manifest_path=tampered_source_path,
            artifact_root=REPO_ROOT,
        )


def test_backend_semantics_allows_level_local_input_and_profiler_evidence(
    tmp_path,
):
    (
        qualification_path,
        validation_path,
        candidate_path,
        cpu_path,
    ) = _write_backend_bundle(tmp_path)
    level_input_path = tmp_path / "small-domain-level.json"
    _write_json(
        level_input_path,
        {
            "schema_version": "kaess.qualification-level/1",
            "level": "small_domain",
            "case_id": "standard-10x30-t150-p250-v850",
        },
    )

    cpu_two = json.loads(cpu_path.read_text(encoding="utf-8"))
    cpu_two["run_id"] = "cpu-2"
    cpu_level = next(
        item
        for item in cpu_two["inputs"]
        if item["role"] == "qualification_level"
    )
    cpu_level.update(_real_run_artifact("qualification_level", level_input_path))
    cpu_two_path = tmp_path / "cpu-manifest-2.json"
    _write_json(cpu_two_path, cpu_two)

    candidate_two = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_two["run_id"] = "candidate-2"
    candidate_level = next(
        item
        for item in candidate_two["inputs"]
        if item["role"] == "qualification_level"
    )
    candidate_level.update(
        _real_run_artifact("qualification_level", level_input_path)
    )
    profiler_two_path = tmp_path / "profiler-2.json"
    _write_json(
        profiler_two_path,
        {
            "run_id": "candidate-2",
            "execution_mode": "hybrid_gpu_assembly_cpu_pardiso",
            "orchestration_backend": "host_python",
            "full_loop_xla": False,
            "cpu_pardiso_calls": 7,
            "unexpected_fallback_count": 0,
            "stages": {
                stage: candidate_two["backend"][stage]
                for stage in ("thermal", "mechanics", "release")
            },
        },
    )
    profiler_record = next(
        item
        for item in candidate_two["artifacts"]
        if item["role"] == "profiler"
    )
    profiler_record.update(_real_artifact(profiler_two_path))
    candidate_two_path = tmp_path / "candidate-manifest-2.json"
    _write_json(candidate_two_path, candidate_two)

    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    qualification["levels"].append("small_domain")
    qualification["cpu_reference_run_ids"].append("cpu-2")
    qualification["candidate_run_ids"].append("candidate-2")
    qualification["level_run_pairs"]["small_domain"] = {
        "cpu_run_ids": ["cpu-2"],
        "candidate_run_ids": ["candidate-2"],
        "status": "pass",
        "evidence": [
            _real_artifact(cpu_two_path),
            _real_artifact(candidate_two_path),
        ],
    }
    qualification["placement_evidence"]["run_manifest_artifacts"].extend(
        [_real_artifact(cpu_two_path), _real_artifact(candidate_two_path)]
    )
    qualification["placement_evidence"]["profiler_artifacts"].append(
        _real_artifact(profiler_two_path)
    )
    performance_path = Path(qualification["performance"]["evidence"][0]["path"])
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["cpu_run_ids"].append("cpu-2")
    performance["candidate_run_ids"].append("candidate-2")
    _write_json(performance_path, performance)
    qualification["performance"]["evidence"] = [
        _real_artifact(performance_path)
    ]
    _write_json(qualification_path, qualification)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["qualification_artifact"] = _real_artifact(qualification_path)
    order_path = Path(
        validation["identity"]["execution_order_artifact"]["path"]
    )
    order = json.loads(order_path.read_text(encoding="utf-8"))
    order["runs"].extend(
        [
            {
                "run_id": "cpu-2",
                "started_utc": "2026-07-24T00:04:00Z",
                "completed_utc": "2026-07-24T00:05:00Z",
            },
            {
                "run_id": "candidate-2",
                "started_utc": "2026-07-24T00:06:00Z",
                "completed_utc": "2026-07-24T00:07:00Z",
            },
        ]
    )
    _write_json(order_path, order)
    validation["identity"]["execution_order_artifact"] = _real_artifact(
        order_path
    )
    _write_json(validation_path, validation)

    validate_backend_qualification_bundle(
        qualification_path,
        validation_path=validation_path,
        candidate_manifest_path=candidate_path,
        candidate_manifest_paths=[candidate_path, candidate_two_path],
        cpu_manifest_paths=[cpu_path, cpu_two_path],
        parity_config_path=PARITY_CONFIG_PATH,
        threshold_set_path=THRESHOLD_PATH,
        approval_record_path=APPROVAL_PATH,
        artifact_root=REPO_ROOT,
    )
