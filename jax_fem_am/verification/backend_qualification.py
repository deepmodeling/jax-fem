"""Strict JSON-contract and cross-file validation for Kaess evidence bundles."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from .thermal_balance import compute_discrete_balance


CONTRACT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "specs"
    / "001-kaess-paper-reproduction"
    / "contracts"
)

PAPER_THRESHOLD_BINDINGS: dict[str, str | None] = {
    "figure8_sigma_x_sign_sequence": None,
    "figure8_sigma_x_peak_relative_error": None,
    "figure8_sigma_x_trough_relative_error": None,
    "figure8_sigma_x_zero_crossing_depth_error": (
        "figure8_zero_crossing_depth_error"
    ),
    "figure9_bending_curve_nrmse": "figure9_bending_curve_nrmse",
    "figure9_max_front_bending_relative_error": None,
    "figure9_max_front_bending_absolute_error": None,
    "figure9_release_direction": None,
}

BACKEND_THRESHOLD_BINDINGS = {
    "temperature_field_relative_l2": "hybrid_temperature_qoi_difference",
    "peak_temperature_relative_error": "hybrid_temperature_qoi_difference",
    "sigma_x_field_relative_l2": "hybrid_stress_qoi_difference",
    "eqp_field_relative_l2": "hybrid_stress_qoi_difference",
    "displacement_field_relative_l2": (
        "hybrid_release_displacement_relative"
    ),
    "front_bending_curve_relative_l2": (
        "hybrid_release_displacement_relative"
    ),
    "max_front_bending_error": "hybrid_release_displacement_absolute",
    "linear_solve_count_delta_fraction": (
        "accelerated_linear_solve_increase"
    ),
}

BACKEND_METRIC_UNITS = {
    "temperature_field_relative_l2": ("K", "fraction"),
    "sigma_x_field_relative_l2": ("MPa", "fraction"),
    "eqp_field_relative_l2": ("fraction", "fraction"),
    "displacement_field_relative_l2": ("um", "fraction"),
    "peak_temperature_relative_error": ("K", "fraction"),
    "front_bending_curve_relative_l2": ("um", "fraction"),
    "max_front_bending_error": ("um", "um"),
    "linear_solve_count_delta_fraction": ("count", "fraction"),
}

CHECKPOINT_FLOAT64_QOI_ARRAYS = {
    "temperature",
    "sigma_x_mpa",
    "eqp",
    "displacement_um",
    "front_bending_curve_um",
}


class ContractValidationError(ValueError):
    """Fail-closed contract error with stable machine-readable issue codes."""

    def __init__(self, issues: Iterable[tuple[str, str]]):
        self.issues = tuple(issues)
        detail = "; ".join(f"{code}: {message}" for code, message in self.issues)
        super().__init__(detail)


def _error(code: str, message: str) -> ContractValidationError:
    return ContractValidationError(((code, message),))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Hash canonical finite JSON without introducing self-referential fields."""
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error("canonical_json", str(exc)) from exc
    return hashlib.sha256(encoded).hexdigest()


def manifest_input_bundle_sha256(manifest: Mapping[str, Any]) -> str:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise _error("input_bundle", "manifest inputs must be a non-empty array")
    ordered = sorted(
        inputs,
        key=lambda item: (
            str(item.get("role", "")) if isinstance(item, Mapping) else "",
            str(item.get("path", "")) if isinstance(item, Mapping) else "",
        ),
    )
    return canonical_json_sha256(ordered)


# Only placement, resource, and solver-family controls may differ.  In
# particular, tolerances, iteration limits, residual/fallback controls,
# surrogate flags, and opaque external solver configs stay in the physics /
# acceptance identity.
_NONPHYSICS_CONFIG_KEYS = {
    "config",
    "inp",
    "output_dir",
    "path_file",
    "path_output",
    "profile_json",
    "profile_label",
    "xla_cell_num_cuts",
    "xla_cell_target_batch_size",
    "xla_dof_to_quad_cache",
    "xla_dry_run",
    "xla_jax_gmres_restart",
    "xla_jax_gmres_solve_method",
    "xla_jax_method",
    "xla_jax_precond",
    "xla_jit_loop_kernels",
    "xla_lazy_output_postprocess",
    "xla_linear_solver",
    "xla_mem_fraction",
    "xla_pardiso_mode",
    "xla_petsc_gpu",
    "xla_petsc_ksp_type",
    "xla_petsc_pc_type",
    "xla_platform",
    "xla_preallocate",
    "xla_quiet_jax_fem_logs",
    "xla_show_devices",
    "xla_skip_unused_mechanics_material",
    "xla_step_predicate_cache",
    "xla_thermal_warm_start",
}


def _physics_config_view(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _physics_config_view(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in _NONPHYSICS_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_physics_config_view(item) for item in value]
    return value


def physics_input_bundle_sha256(
    manifest: Mapping[str, Any],
    artifact_root: Path,
) -> str:
    """Hash frozen physics inputs while normalizing allowed backend controls."""
    inputs = _input_by_role(manifest)
    used_config_record = inputs.get("used_config")
    if used_config_record is None:
        raise _error(
            "runtime_input_identity",
            "manifest is missing the final used_config input",
        )
    used_config_path = _validate_file_identity(
        used_config_record,
        artifact_root,
        code="runtime_input_identity",
    )
    used_config = _require_mapping(
        load_json_strict(used_config_path),
        code="runtime_input_identity",
        description="used config",
    )
    physics_records = [
        {
            "role": role,
            "sha256": record.get("sha256"),
            "size_bytes": record.get("size_bytes"),
        }
        for role, record in sorted(inputs.items())
        if role not in {"solver_command", "used_config"}
    ]
    return canonical_json_sha256(
        {
            "input_records": physics_records,
            "normalized_used_config": _physics_config_view(used_config),
        }
    )


def manifest_acceptance_model_sha256(manifest: Mapping[str, Any]) -> str:
    required_roles = {
        "paper_parity_config",
        "threshold_set",
        "g0_approval",
    }
    inputs = _input_by_role(manifest)
    missing = sorted(required_roles - set(inputs))
    if missing:
        raise _error(
            "acceptance_model",
            f"manifest is missing acceptance inputs: {', '.join(missing)}",
        )
    return canonical_json_sha256(
        [inputs[role] for role in sorted(required_roles)]
    )


def manifest_environment_sha256(manifest: Mapping[str, Any]) -> str:
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise _error("environment_identity", "manifest environment is missing")
    return canonical_json_sha256(environment)


def manifest_cpu_hardware_sha256(manifest: Mapping[str, Any]) -> str:
    environment = manifest.get("environment")
    hardware = (
        environment.get("hardware") if isinstance(environment, Mapping) else None
    )
    if not isinstance(hardware, Mapping) or "cpu" not in hardware:
        raise _error(
            "hardware_identity",
            "manifest hardware must identify the CPU",
        )
    return canonical_json_sha256({"cpu": hardware["cpu"]})


def inspect_native_checkpoint(path: Path) -> dict[str, Any]:
    """Open a native checkpoint and prove its state arrays are float64."""
    path = Path(path).resolve()
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            names = tuple(checkpoint.files)
            if not names:
                raise _error("checkpoint_shape", f"{path}: checkpoint is empty")
            invalid_qoi = {
                name: str(checkpoint[name].dtype)
                for name in CHECKPOINT_FLOAT64_QOI_ARRAYS & set(names)
                if (
                    checkpoint[name].dtype != np.dtype(np.float64)
                    or checkpoint[name].size == 0
                )
            }
            if invalid_qoi:
                raise _error(
                    "checkpoint_dtype",
                    f"{path}: native QoI arrays must be non-empty float64 "
                    f"{invalid_qoi}",
                )
            floating = {
                name: checkpoint[name]
                for name in names
                if np.issubdtype(checkpoint[name].dtype, np.floating)
            }
            if not floating:
                raise _error(
                    "checkpoint_dtype",
                    f"{path}: checkpoint has no floating state arrays",
                )
            wrong = {
                name: str(array.dtype)
                for name, array in floating.items()
                if array.dtype != np.dtype(np.float64)
            }
            if wrong:
                raise _error(
                    "checkpoint_dtype",
                    f"{path}: non-float64 state arrays {wrong}",
                )
            nonfinite = [
                name
                for name, array in floating.items()
                if not np.all(np.isfinite(array))
            ]
            if nonfinite:
                raise _error(
                    "checkpoint_finite",
                    f"{path}: non-finite state arrays {nonfinite}",
                )
            primary = floating.get("temperature")
            if primary is None:
                primary = next(iter(floating.values()))
            active_mask = (
                checkpoint["active_mask"] if "active_mask" in names else None
            )
            if active_mask is not None and active_mask.dtype != np.dtype(np.bool_):
                raise _error(
                    "checkpoint_mask",
                    f"{path}: active_mask must be boolean",
                )
            return {
                "sha256": sha256_file(path),
                "dtype": "float64",
                "shape": list(primary.shape),
                "arrays": names,
                "array_shapes": {
                    name: list(checkpoint[name].shape) for name in names
                },
                "array_dtypes": {
                    name: str(checkpoint[name].dtype) for name in names
                },
                "state": {
                    name: np.array(checkpoint[name], copy=True) for name in names
                },
                "active_mask_shape": (
                    list(active_mask.shape) if active_mask is not None else None
                ),
                "active_mask": (
                    np.array(active_mask, copy=True)
                    if active_mask is not None
                    else None
                ),
            }
    except ContractValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise _error("checkpoint_read", f"{path}: {exc}") from exc


def formal_promotion_allowed(validation: Mapping[str, Any]) -> bool:
    """Require both the semantic verdict and its promotion boolean."""
    return bool(
        validation.get("verdict") == "pass_promotion_eligible"
        and validation.get("promotion_eligible") is True
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error("duplicate_key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise _error("nonfinite_number", f"non-finite JSON number {token!r}")


def _strict_float(token: str) -> float:
    value = float(token)
    if not np.isfinite(value):
        raise _error("nonfinite_number", f"non-finite JSON number {token!r}")
    return value


def load_json_strict(path: Path) -> Any:
    """Load finite RFC JSON and reject duplicate keys at every object level."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _error("json_read", f"{path}: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except ContractValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise _error("json_syntax", f"{path}: {exc}") from exc


def load_jsonl_strict(path: Path) -> list[Any]:
    """Load non-empty finite RFC JSON objects from a JSON Lines artifact."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise _error("json_read", f"{path}: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise _error(
            "energy_audit_evidence",
            f"{path}: JSON Lines evidence must be non-empty without blank rows",
        )
    rows: list[Any] = []
    for index, line in enumerate(lines):
        try:
            rows.append(
                json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_constant,
                    parse_float=_strict_float,
                )
            )
        except ContractValidationError:
            raise
        except json.JSONDecodeError as exc:
            raise _error(
                "json_syntax",
                f"{path}:{index + 1}: {exc}",
            ) from exc
    return rows


def validate_json_contract(payload: Any, schema_name: str) -> None:
    """Validate one payload with the explicitly selected Draft 2020-12 dialect."""
    schema_path = CONTRACT_ROOT / schema_name
    schema = load_json_strict(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise _error("schema_meta", f"{schema_path}: {exc}") from exc
    validator = Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        issues = []
        for error in errors:
            pointer = "/" + "/".join(
                str(part) for part in error.absolute_path
            )
            issues.append(
                (
                    "schema_instance",
                    f"{schema_name}{pointer}: {error.message}",
                )
            )
        raise ContractValidationError(issues)


def _resolve_artifact_path(record: Mapping[str, Any], root: Path) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise _error("artifact_path", "artifact path must be a non-empty string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def _validate_file_identity(
    record: Mapping[str, Any],
    root: Path,
    *,
    code: str = "artifact_rehash",
) -> Path:
    path = _resolve_artifact_path(record, root)
    if not path.is_file():
        raise _error(code, f"artifact does not exist: {path}")
    actual_size = path.stat().st_size
    if record.get("size_bytes") != actual_size:
        raise _error(
            code,
            f"{path}: size {record.get('size_bytes')!r} != {actual_size}",
        )
    actual_hash = sha256_file(path)
    if record.get("sha256") != actual_hash:
        raise _error(
            code,
            f"{path}: sha256 {record.get('sha256')!r} != {actual_hash}",
        )
    return path


def _validate_file_identity_at(
    record: Mapping[str, Any],
    expected_path: Path,
    root: Path,
    *,
    code: str = "artifact_binding",
) -> Path:
    actual_path = _resolve_artifact_path(record, root)
    expected_path = Path(expected_path).resolve()
    if actual_path != expected_path:
        raise _error(
            code,
            f"artifact path {actual_path} != expected {expected_path}",
        )
    return _validate_file_identity(record, root, code=code)


def _iter_file_identities(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "sha256", "size_bytes"} <= set(value):
            yield value
        for nested in value.values():
            yield from _iter_file_identities(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_file_identities(nested)


def _validate_all_artifacts(payload: Any, root: Path) -> None:
    checked: set[tuple[str, str, Any]] = set()
    for record in _iter_file_identities(payload):
        identity = (
            str(record.get("path")),
            str(record.get("sha256")),
            record.get("size_bytes"),
        )
        if identity not in checked:
            _validate_file_identity(record, root)
            checked.add(identity)


def _require_equal(
    observed: Any,
    expected: Any,
    code: str,
    description: str,
) -> None:
    if observed != expected:
        raise _error(
            code,
            f"{description}: observed {observed!r}, expected {expected!r}",
        )


def _require_mapping(
    value: Any,
    *,
    code: str,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(code, f"{description} must be a JSON object")
    return value


def _raw_identity(path: Path) -> tuple[str, int]:
    path = Path(path).resolve()
    return sha256_file(path), path.stat().st_size


def _input_by_role(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in manifest.get("inputs", []):
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        if isinstance(role, str):
            if role in indexed:
                raise _error("run_input_role", f"duplicate input role {role!r}")
            indexed[role] = item
    return indexed


def _validate_run_input_binding(
    inputs: Mapping[str, Mapping[str, Any]],
    role: str,
    path: Path,
    artifact_root: Path,
    code: str,
) -> None:
    record = inputs.get(role)
    if record is None:
        raise _error(code, f"run manifest is missing input role {role!r}")
    _require_equal(
        _resolve_artifact_path(record, artifact_root),
        Path(path).resolve(),
        code,
        f"{role} path",
    )
    expected_hash, expected_size = _raw_identity(path)
    _require_equal(record.get("sha256"), expected_hash, code, f"{role} hash")
    _require_equal(
        record.get("size_bytes"),
        expected_size,
        code,
        f"{role} size",
    )


def g0_performance_protocol_sha256(
    parity_config_path: Path,
    threshold_set_path: Path,
    approval_record_path: Path,
) -> str:
    """Bind performance acceptance to all three immutable G0 artifacts."""
    return canonical_json_sha256(
        {
            "paper_parity_config_sha256": sha256_file(parity_config_path),
            "threshold_set_sha256": sha256_file(threshold_set_path),
            "g0_approval_sha256": sha256_file(approval_record_path),
        }
    )


def _load_g0_performance_protocol(
    parity_config_path: Path,
    threshold_set_path: Path,
    approval_record_path: Path,
) -> dict[str, float | int]:
    config = _require_mapping(
        load_json_strict(parity_config_path),
        code="performance_protocol_identity",
        description="paper parity config",
    )
    threshold = _require_mapping(
        load_json_strict(threshold_set_path),
        code="performance_protocol_identity",
        description="threshold set",
    )
    approval = _require_mapping(
        load_json_strict(approval_record_path),
        code="performance_protocol_identity",
        description="G0 approval record",
    )
    threshold_hash = sha256_file(threshold_set_path)
    approval_hash = sha256_file(approval_record_path)
    _require_equal(
        config.get("status"),
        "approved",
        "performance_protocol_identity",
        "G0 config status",
    )
    _require_equal(
        approval.get("decision"),
        "approved",
        "performance_protocol_identity",
        "G0 approval decision",
    )
    _require_equal(
        threshold.get("status"),
        "approved",
        "performance_protocol_identity",
        "G0 threshold status",
    )
    for observed, expected, description in (
        (
            threshold.get("protocol_id"),
            config.get("protocol_id"),
            "threshold/config protocol",
        ),
        (
            approval.get("protocol_id"),
            config.get("protocol_id"),
            "approval/config protocol",
        ),
    ):
        _require_equal(
            observed,
            expected,
            "performance_protocol_identity",
            description,
        )
    _require_equal(
        config.get("approval", {}).get("approval_record_sha256"),
        approval_hash,
        "performance_protocol_identity",
        "G0 approval hash",
    )
    _require_equal(
        config.get("threshold_set", {}).get("content_sha256"),
        threshold_hash,
        "performance_protocol_identity",
        "G0 threshold hash",
    )
    _require_equal(
        approval.get("threshold_artifact", {}).get("sha256"),
        threshold_hash,
        "performance_protocol_identity",
        "approval threshold hash",
    )
    _require_equal(
        config.get("threshold_set", {}).get("metrics"),
        threshold.get("metrics"),
        "performance_protocol_identity",
        "config/frozen threshold metrics",
    )
    metrics = _metric_index(threshold.get("metrics"))
    speed = metrics.get("accelerated_wall_speedup", {})
    solve = metrics.get("accelerated_linear_solve_increase", {})
    for metric, operator, unit, name in (
        (speed, "greater_than_or_equal", "ratio", "speedup"),
        (solve, "less_than_or_equal", "fraction", "linear solve increase"),
    ):
        _require_equal(
            metric.get("operator"),
            operator,
            "performance_protocol_identity",
            f"{name} operator",
        )
        _require_equal(
            metric.get("unit"),
            unit,
            "performance_protocol_identity",
            f"{name} unit",
        )
        if not isinstance(metric.get("value"), (int, float)):
            raise _error(
                "performance_protocol_identity",
                f"{name} threshold is not numeric",
            )
    sample_count = speed.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise _error(
            "performance_protocol_identity",
            "speedup sample_count is not an integer",
        )
    return {
        "speedup": float(speed["value"]),
        "linear_solve_increase": float(solve["value"]),
        "sample_count": sample_count,
    }


def _metric_index(metrics: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(metrics, list):
        raise _error("threshold_metrics", "threshold metrics must be an array")
    indexed: dict[str, Mapping[str, Any]] = {}
    for metric in metrics:
        if not isinstance(metric, Mapping):
            raise _error("threshold_metrics", "threshold metric must be an object")
        metric_id = metric.get("metric_id")
        if not isinstance(metric_id, str) or not metric_id:
            raise _error("threshold_metrics", "threshold metric id is missing")
        if metric_id in indexed:
            raise _error(
                "threshold_metrics", f"duplicate threshold metric {metric_id!r}"
            )
        indexed[metric_id] = metric
    return indexed


def _threshold_status(metric: Mapping[str, Any], value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error(
            "comparison_value",
            f"{metric.get('metric_id')}: numeric threshold needs numeric value",
        )
    limit = metric.get("value")
    operator = metric.get("operator")
    if not isinstance(limit, (int, float)) or isinstance(limit, bool):
        raise _error(
            "threshold_metrics",
            f"{metric.get('metric_id')}: threshold value is not numeric",
        )
    if operator == "less_than_or_equal":
        passed = value <= limit
    elif operator == "greater_than_or_equal":
        passed = value >= limit
    elif operator == "equal":
        passed = value == limit
    else:
        raise _error(
            "threshold_metrics",
            f"{metric.get('metric_id')}: unsupported operator {operator!r}",
        )
    if (
        metric.get("combined_with")
        and not passed
        and metric.get("combination") == "maximum"
    ):
        raise _error(
            "threshold_combination_unresolved",
            f"{metric.get('metric_id')}: external combined threshold is required",
        )
    return "pass" if passed else "fail"


def _expected_paper_verdict(comparisons: list[Mapping[str, Any]]) -> str:
    statuses = [item.get("status") for item in comparisons]
    if any(status == "fail" for status in statuses):
        return "fail"
    if statuses and all(status == "pass" for status in statuses):
        return "pass"
    if any(status == "pass" for status in statuses):
        return "partial"
    return "blocked"


def validate_paper_comparison_bundle(
    report_path: Path,
    *,
    run_manifest_path: Path,
    parity_config_path: Path,
    threshold_set_path: Path,
    approval_record_path: Path,
    source_manifest_path: Path,
    artifact_root: Path,
) -> None:
    """Validate a paper report against raw artifacts and the approved G0 chain."""
    report_path = Path(report_path).resolve()
    run_manifest_path = Path(run_manifest_path).resolve()
    parity_config_path = Path(parity_config_path).resolve()
    threshold_set_path = Path(threshold_set_path).resolve()
    approval_record_path = Path(approval_record_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()

    report = load_json_strict(report_path)
    manifest = load_json_strict(run_manifest_path)
    config = _require_mapping(
        load_json_strict(parity_config_path),
        code="paper_parity_config",
        description="paper parity config",
    )
    threshold = _require_mapping(
        load_json_strict(threshold_set_path),
        code="threshold_artifact_binding",
        description="threshold set",
    )
    approval = _require_mapping(
        load_json_strict(approval_record_path),
        code="g0_approval_binding",
        description="G0 approval record",
    )
    source_manifest = _require_mapping(
        load_json_strict(source_manifest_path),
        code="paper_source_identity",
        description="source manifest",
    )

    validate_json_contract(report, "paper-comparison.schema.json")
    validate_json_contract(manifest, "run-manifest.schema.json")
    _validate_all_artifacts(report, artifact_root)
    _validate_all_artifacts(manifest, artifact_root)

    config_hash, _ = _raw_identity(parity_config_path)
    threshold_hash, _ = _raw_identity(threshold_set_path)
    approval_hash, _ = _raw_identity(approval_record_path)
    manifest_hash, _ = _raw_identity(run_manifest_path)

    _validate_file_identity_at(
        report.get("run_manifest", {}),
        run_manifest_path,
        artifact_root,
        code="run_manifest_binding",
    )
    _validate_file_identity_at(
        report.get("threshold_set", {}).get("threshold_set_artifact", {}),
        threshold_set_path,
        artifact_root,
        code="threshold_artifact_binding",
    )
    _validate_file_identity_at(
        report.get("threshold_set", {}).get("approval_artifact", {}),
        approval_record_path,
        artifact_root,
        code="g0_approval_binding",
    )

    _require_equal(config.get("status"), "approved", "g0_status", "G0 status")
    _require_equal(
        approval.get("decision"), "approved", "g0_status", "G0 decision"
    )
    _require_equal(
        config.get("approval", {}).get("approval_record_sha256"),
        approval_hash,
        "g0_approval_hash",
        "config approval record",
    )
    _require_equal(
        config.get("threshold_set", {}).get("content_sha256"),
        threshold_hash,
        "threshold_artifact_hash",
        "config threshold artifact",
    )
    _require_equal(
        approval.get("threshold_artifact", {}).get("sha256"),
        threshold_hash,
        "threshold_artifact_hash",
        "approval threshold artifact",
    )
    _require_equal(
        report.get("semantic_validation", {}).get(
            "paper_parity_config_sha256"
        ),
        config_hash,
        "paper_parity_config_hash",
        "paper parity config",
    )
    _require_equal(
        report.get("semantic_validation", {}).get("g0_approval_sha256"),
        approval_hash,
        "g0_approval_hash",
        "report G0 approval",
    )
    _require_equal(
        report.get("semantic_validation", {}).get("threshold_set_sha256"),
        threshold_hash,
        "threshold_artifact_hash",
        "report threshold artifact",
    )
    _require_equal(
        report.get("semantic_validation", {}).get("run_manifest_sha256"),
        manifest_hash,
        "run_manifest_hash",
        "report run manifest",
    )

    report_payload = {
        key: value for key, value in report.items() if key != "semantic_validation"
    }
    _require_equal(
        report.get("semantic_validation", {}).get("validated_payload_sha256"),
        canonical_json_sha256(report_payload),
        "validated_payload_hash",
        "paper comparison payload",
    )

    _require_equal(
        report.get("protocol_id"),
        config.get("protocol_id"),
        "protocol_id",
        "report/config protocol",
    )
    _require_equal(
        approval.get("protocol_id"),
        config.get("protocol_id"),
        "protocol_id",
        "approval/config protocol",
    )
    _require_equal(
        threshold.get("protocol_id"),
        config.get("protocol_id"),
        "protocol_id",
        "threshold/config protocol",
    )
    _require_equal(
        report.get("run_id"),
        manifest.get("run_id"),
        "run_id",
        "report/current manifest run",
    )
    _require_equal(
        report.get("case_id"),
        manifest.get("case_id"),
        "case_id",
        "report/current manifest case",
    )
    _require_equal(
        report.get("case_id"),
        config.get("standard_case_id"),
        "case_id",
        "report/G0 standard case",
    )
    _require_equal(
        report.get("claim_boundary", {}).get("reproduction_level"),
        manifest.get("claim_level"),
        "claim_level",
        "report/current manifest claim",
    )
    if manifest.get("status") not in {"completed", "accepted"}:
        raise _error(
            "run_status",
            "paper comparison requires a completed or accepted current run",
        )
    _require_equal(
        manifest.get("claim_level"),
        config.get("claim_level"),
        "claim_level",
        "manifest/G0 claim",
    )

    report_threshold = report.get("threshold_set", {})
    _require_equal(
        report_threshold.get("threshold_set_id"),
        threshold.get("threshold_set_id"),
        "threshold_identity",
        "threshold set id",
    )
    _require_equal(
        report_threshold.get("version"),
        threshold.get("version"),
        "threshold_identity",
        "threshold set version",
    )
    _require_equal(
        report_threshold.get("approved_by"),
        approval.get("approved_by"),
        "threshold_approver",
        "threshold approver",
    )
    _require_equal(
        report_threshold.get("approved_utc"),
        approval.get("approved_utc"),
        "threshold_approver",
        "threshold approval time",
    )
    _require_equal(
        threshold.get("approved_by"),
        approval.get("approved_by"),
        "threshold_approver",
        "artifact approver",
    )
    _require_equal(
        threshold.get("approved_utc"),
        approval.get("approved_utc"),
        "threshold_approver",
        "artifact approval time",
    )
    _require_equal(
        report_threshold.get("metrics"),
        threshold.get("metrics"),
        "threshold_metrics",
        "report/frozen threshold metrics",
    )
    _require_equal(
        config.get("threshold_set", {}).get("metrics"),
        threshold.get("metrics"),
        "threshold_metrics",
        "config/frozen threshold metrics",
    )
    threshold_index = _metric_index(threshold.get("metrics"))

    inputs = _input_by_role(manifest)
    _validate_run_input_binding(
        inputs,
        "paper_parity_config",
        parity_config_path,
        artifact_root,
        "run_parity_config_binding",
    )
    _validate_run_input_binding(
        inputs,
        "threshold_set",
        threshold_set_path,
        artifact_root,
        "run_threshold_binding",
    )
    _validate_run_input_binding(
        inputs,
        "g0_approval",
        approval_record_path,
        artifact_root,
        "run_g0_binding",
    )
    _validate_run_input_binding(
        inputs,
        "source_manifest",
        source_manifest_path,
        artifact_root,
        "run_source_manifest_binding",
    )

    _require_equal(
        source_manifest.get("schema_version"),
        "kaess.source-manifest/1",
        "paper_source_identity",
        "source manifest schema",
    )
    _require_equal(
        source_manifest.get("hash_algorithm"),
        "sha256",
        "paper_source_identity",
        "source manifest hash algorithm",
    )
    _require_equal(
        source_manifest.get("protocol_id"),
        config.get("protocol_id"),
        "paper_source_identity",
        "source manifest protocol",
    )
    _require_equal(
        source_manifest.get("claim_level"),
        config.get("claim_level"),
        "paper_source_identity",
        "source manifest claim",
    )
    _require_equal(
        source_manifest.get("paper_doi"),
        config.get("paper_doi"),
        "paper_source_identity",
        "source manifest/config DOI",
    )
    evidence = source_manifest.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise _error(
            "paper_source_identity",
            "source manifest evidence must be a non-empty array",
        )
    evidence_ids = [
        item.get("evidence_id")
        for item in evidence
        if isinstance(item, Mapping)
        and isinstance(item.get("evidence_id"), str)
        and item.get("evidence_id")
    ]
    if len(evidence_ids) != len(evidence) or len(set(evidence_ids)) != len(
        evidence_ids
    ):
        raise _error(
            "paper_source_identity",
            "source manifest evidence IDs must be non-empty and unique",
        )
    paper_entries = (
        [
            item
            for item in evidence
            if isinstance(item, Mapping)
            and item.get("evidence_id") == "paper-pdf"
        ]
    )
    if len(paper_entries) != 1:
        raise _error(
            "paper_source_identity",
            "source manifest requires exactly one 'paper-pdf' evidence entry",
        )
    paper_entry = paper_entries[0]
    _require_equal(
        paper_entry.get("source_class"),
        "paper_text",
        "paper_source_identity",
        "paper source class",
    )
    _require_equal(
        paper_entry.get("status"),
        "verified",
        "paper_source_identity",
        "paper source status",
    )
    repository_path = paper_entry.get("repository_path")
    if not isinstance(repository_path, str) or not repository_path:
        raise _error(
            "paper_source_identity",
            "paper source repository_path must be a non-empty string",
        )
    repository_relative = Path(repository_path)
    if repository_relative.is_absolute():
        raise _error(
            "paper_source_identity",
            "paper source repository_path must be repository-relative",
        )
    paper_path = (artifact_root / repository_relative).resolve()
    try:
        paper_path.relative_to(artifact_root)
    except ValueError as exc:
        raise _error(
            "paper_source_identity",
            "paper source path escapes the artifact root",
        ) from exc
    if not paper_path.is_file():
        raise _error(
            "paper_source_identity",
            f"authoritative paper source does not exist: {paper_path}",
        )
    paper_sha256 = sha256_file(paper_path)
    _require_equal(
        paper_entry.get("sha256"),
        paper_sha256,
        "paper_source_identity",
        "source manifest paper hash",
    )
    _require_equal(
        source_manifest.get("paper_doi"),
        report.get("reference", {}).get("doi"),
        "paper_source_identity",
        "paper DOI",
    )
    _require_equal(
        report.get("reference", {}).get("source_sha256"),
        paper_sha256,
        "paper_source_identity",
        "report paper hash",
    )

    comparisons = report.get("comparisons", [])
    for comparison in comparisons:
        comparison_id = comparison.get("comparison_id")
        if comparison_id not in PAPER_THRESHOLD_BINDINGS:
            raise _error(
                "threshold_binding",
                f"unknown paper comparison {comparison_id!r}",
            )
        status = comparison.get("status")
        threshold_id = comparison.get("threshold_metric_id")
        expected_threshold_id = PAPER_THRESHOLD_BINDINGS[comparison_id]
        if status == "not_comparable":
            _require_equal(
                threshold_id,
                None,
                "threshold_binding",
                f"{comparison_id} threshold",
            )
            continue
        if expected_threshold_id is None:
            raise _error(
                "threshold_binding",
                f"{comparison_id}: no independent frozen G0 threshold",
            )
        _require_equal(
            threshold_id,
            expected_threshold_id,
            "threshold_binding",
            f"{comparison_id} frozen threshold",
        )
        metric = threshold_index.get(threshold_id)
        if metric is None:
            raise _error(
                "threshold_binding",
                f"{comparison.get('comparison_id')}: unknown threshold "
                f"{threshold_id!r}",
            )
        _require_equal(
            comparison.get("value_unit"),
            metric.get("unit"),
            "threshold_unit",
            f"{comparison.get('comparison_id')} unit",
        )
        expected_status = _threshold_status(metric, comparison.get("value"))
        _require_equal(
            status,
            expected_status,
            "comparison_status",
            f"{comparison.get('comparison_id')} status",
        )
        raise _error(
            "paper_metric_evidence",
            f"{comparison_id}: comparable paper values require a typed raw "
            "QoI evidence artifact; T032/T035 have not produced one",
        )

    expected_verdict = _expected_paper_verdict(comparisons)
    _require_equal(
        report.get("verdict"),
        expected_verdict,
        "paper_verdict",
        "paper comparison verdict",
    )
    if expected_verdict != "blocked":
        _require_equal(
            report.get("semantic_validation", {}).get("status"),
            "pass",
            "semantic_validation_status",
            "paper semantic validation status",
        )

    semantic_record = report.get("semantic_validation", {}).get("artifact", {})
    semantic_path = _validate_file_identity(
        semantic_record,
        artifact_root,
        code="semantic_artifact",
    )
    semantic_artifact = load_json_strict(semantic_path)
    if not isinstance(semantic_artifact, Mapping):
        raise _error(
            "semantic_artifact",
            "paper semantic-validation artifact must be an object",
        )
    required_semantic_fields = {
        "schema_version",
        "validator_id",
        "validator_version",
        "status",
        "validated_payload_sha256",
        "run_manifest_sha256",
        "threshold_set_sha256",
        "paper_parity_config_sha256",
        "g0_approval_sha256",
    }
    if set(semantic_artifact) != required_semantic_fields:
        raise _error(
            "semantic_artifact",
            "paper semantic-validation artifact fields do not match contract",
        )
    _require_equal(
        semantic_artifact.get("schema_version"),
        "kaess.paper-comparison-semantic-validation/1",
        "semantic_artifact",
        "semantic artifact schema",
    )
    report_semantic = report.get("semantic_validation", {})
    for field in required_semantic_fields - {"schema_version"}:
        _require_equal(
            semantic_artifact.get(field),
            report_semantic.get(field),
            "semantic_artifact",
            f"semantic artifact {field}",
        )


def _artifact_by_role(
    manifest: Mapping[str, Any], role: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    if len(matches) != 1:
        raise _error(
            "artifact_role",
            f"run {manifest.get('run_id')!r} requires exactly one {role!r} "
            f"artifact, found {len(matches)}",
        )
    return matches[0]


def ndarray_sha256(array: np.ndarray) -> str:
    """Hash an ndarray with explicit dtype and shape framing."""
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _relative_l2(cpu: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    cpu_norm = float(np.linalg.norm(cpu))
    candidate_norm = float(np.linalg.norm(candidate))
    denominator = max(cpu_norm, np.finfo(np.float64).tiny)
    error = float(np.linalg.norm(candidate - cpu) / denominator)
    return cpu_norm, candidate_norm, error


def _checkpoint_metric_truth(
    cpu_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    cpu_state = cpu_checkpoint.get("state", {})
    candidate_state = candidate_checkpoint.get("state", {})
    required = {
        "temperature",
        "sigma_x_mpa",
        "eqp",
        "displacement_um",
        "front_bending_curve_um",
        "activation_events",
        "active_mask",
        "phase_state",
        "accepted_increments",
        "fallback_events",
        "linear_solve_count",
    }
    missing = sorted(
        required - set(cpu_state)
        | required - set(candidate_state)
    )
    if missing:
        raise _error(
            "metric_recalculation",
            f"native checkpoints are missing metric arrays: {', '.join(missing)}",
        )

    for role, state in (
        ("CPU", cpu_state),
        ("candidate", candidate_state),
    ):
        for array_name in CHECKPOINT_FLOAT64_QOI_ARRAYS:
            continuous = np.asarray(state[array_name])
            if (
                continuous.dtype != np.dtype(np.float64)
                or continuous.size == 0
                or not np.all(np.isfinite(continuous))
            ):
                raise _error(
                    "checkpoint_dtype",
                    f"{role} {array_name} must be a non-empty finite float64 "
                    "array",
                )

    for array_name in (
        "activation_events",
        "phase_state",
        "accepted_increments",
        "fallback_events",
    ):
        discrete = np.asarray(cpu_state[array_name])
        candidate_discrete = np.asarray(candidate_state[array_name])
        for role, value in (
            ("CPU", discrete),
            ("candidate", candidate_discrete),
        ):
            if (
                value.ndim < 1
                or not np.issubdtype(value.dtype, np.integer)
                or np.issubdtype(value.dtype, np.bool_)
            ):
                raise _error(
                    "checkpoint_discrete_state",
                    f"{role} {array_name} must be a non-scalar integer array",
                )
    solve_counts: dict[str, int] = {}
    for role, value in (
        ("CPU", np.asarray(cpu_state["linear_solve_count"])),
        (
            "candidate",
            np.asarray(candidate_state["linear_solve_count"]),
        ),
    ):
        if (
            value.ndim != 0
            or not np.issubdtype(value.dtype, np.integer)
            or np.issubdtype(value.dtype, np.bool_)
            or int(value.item()) < 0
        ):
            raise _error(
                "checkpoint_discrete_state",
                f"{role} linear_solve_count must be a nonnegative integer scalar",
            )
        solve_counts[role] = int(value.item())

    cpu_mask = np.asarray(cpu_state["active_mask"])
    candidate_mask = np.asarray(candidate_state["active_mask"])
    if (
        cpu_mask.dtype != np.dtype(np.bool_)
        or candidate_mask.dtype != np.dtype(np.bool_)
        or cpu_mask.ndim != 1
        or candidate_mask.ndim != 1
        or not np.array_equal(cpu_mask, candidate_mask)
    ):
        raise _error(
            "comparison_mask_identity",
            "CPU and candidate checkpoints require the same one-dimensional "
            "boolean active mask",
        )
    if not np.any(cpu_mask):
        raise _error(
            "comparison_mask_identity",
            "active-domain metric mask must select at least one entry",
        )

    def active_values(state: Mapping[str, Any], array_name: str) -> np.ndarray:
        array = np.asarray(state[array_name], dtype=np.float64)
        if array.ndim == 0 or array.shape[0] != cpu_mask.size:
            raise _error(
                "metric_recalculation",
                f"{array_name}: leading dimension {array.shape!r} does not "
                f"match active mask size {cpu_mask.size}",
            )
        return array[cpu_mask]

    truth: dict[str, dict[str, Any]] = {}
    for metric_id, array_name in (
        ("temperature_field_relative_l2", "temperature"),
        ("sigma_x_field_relative_l2", "sigma_x_mpa"),
        ("eqp_field_relative_l2", "eqp"),
        ("displacement_field_relative_l2", "displacement_um"),
    ):
        cpu_value, candidate_value, error = _relative_l2(
            active_values(cpu_state, array_name),
            active_values(candidate_state, array_name),
        )
        truth[metric_id] = {
            "cpu_value": cpu_value,
            "candidate_value": candidate_value,
            "error": error,
        }

    cpu_curve = np.asarray(
        cpu_state["front_bending_curve_um"], dtype=np.float64
    ).reshape(-1)
    candidate_curve = np.asarray(
        candidate_state["front_bending_curve_um"], dtype=np.float64
    ).reshape(-1)
    curve_cpu_value, curve_candidate_value, curve_error = _relative_l2(
        cpu_curve,
        candidate_curve,
    )
    truth["front_bending_curve_relative_l2"] = {
        "cpu_value": curve_cpu_value,
        "candidate_value": curve_candidate_value,
        "error": curve_error,
    }

    cpu_temperature = active_values(cpu_state, "temperature")
    candidate_temperature = active_values(candidate_state, "temperature")
    cpu_peak = float(np.max(cpu_temperature))
    candidate_peak = float(np.max(candidate_temperature))
    truth["peak_temperature_relative_error"] = {
        "cpu_value": cpu_peak,
        "candidate_value": candidate_peak,
        "error": abs(candidate_peak - cpu_peak)
        / max(abs(cpu_peak), np.finfo(np.float64).tiny),
    }

    if cpu_curve.size == 0 or candidate_curve.size == 0:
        raise _error(
            "metric_recalculation",
            "front bending curves must not be empty",
        )
    cpu_max = float(np.max(np.abs(cpu_curve)))
    candidate_max = float(np.max(np.abs(candidate_curve)))
    truth["max_front_bending_error"] = {
        "cpu_value": cpu_max,
        "candidate_value": candidate_max,
        "error": abs(candidate_max - cpu_max),
    }

    def release_direction(curve: np.ndarray) -> str:
        endpoint = float(curve[-1])
        if endpoint > 0.0:
            return "upward"
        if endpoint < 0.0:
            return "downward"
        return "neutral"

    truth["release_direction_match"] = {
        "cpu_value": release_direction(cpu_curve),
        "candidate_value": release_direction(candidate_curve),
    }

    for metric_id, array_name in (
        ("activation_event_digest_match", "activation_events"),
        ("active_element_digest_match", "active_mask"),
        ("phase_state_digest_match", "phase_state"),
        ("accepted_increment_digest_match", "accepted_increments"),
        ("fallback_event_digest_match", "fallback_events"),
    ):
        truth[metric_id] = {
            "cpu_sha256": ndarray_sha256(np.asarray(cpu_state[array_name])),
            "candidate_sha256": ndarray_sha256(
                np.asarray(candidate_state[array_name])
            ),
        }

    cpu_solves = solve_counts["CPU"]
    candidate_solves = solve_counts["candidate"]
    solve_error = (
        max(0.0, (candidate_solves - cpu_solves) / abs(cpu_solves))
        if cpu_solves
        else (0.0 if candidate_solves == 0 else float("inf"))
    )
    truth["linear_solve_count_delta_fraction"] = {
        "cpu_value": float(cpu_solves),
        "candidate_value": float(candidate_solves),
        "error": float(solve_error),
    }
    return truth


def _checkpoint_pair_parity_passes(
    truth: Mapping[str, Mapping[str, Any]],
    frozen_thresholds: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return threshold truth; malformed evidence still fails closed."""
    passed = True
    for metric_id, threshold_metric_id in BACKEND_THRESHOLD_BINDINGS.items():
        if metric_id == "linear_solve_count_delta_fraction":
            continue
        metric_truth = truth[metric_id]
        threshold = frozen_thresholds[threshold_metric_id].get("value")
        if not isinstance(threshold, (int, float)) or isinstance(
            threshold, bool
        ):
            raise _error(
                "metric_recalculation",
                f"{threshold_metric_id}: frozen threshold is not numeric",
            )
        if metric_id == "max_front_bending_error":
            relative_threshold = frozen_thresholds.get(
                "hybrid_release_displacement_relative", {}
            ).get("value")
            if not isinstance(relative_threshold, (int, float)) or isinstance(
                relative_threshold, bool
            ):
                raise _error(
                    "metric_recalculation",
                    "max front bending relative threshold is missing",
                )
            threshold = max(
                float(threshold),
                float(relative_threshold)
                * abs(float(metric_truth["cpu_value"])),
            )
        if float(metric_truth["error"]) > float(threshold):
            passed = False
    for metric_id in (
        "activation_event_digest_match",
        "active_element_digest_match",
        "phase_state_digest_match",
    ):
        metric_truth = truth[metric_id]
        if metric_truth["cpu_sha256"] != metric_truth["candidate_sha256"]:
            passed = False
    release = truth["release_direction_match"]
    if release["cpu_value"] != release["candidate_value"]:
        passed = False
    return passed


def _checkpoint_pair_convergence_passes(
    truth: Mapping[str, Mapping[str, Any]],
    frozen_thresholds: Mapping[str, Mapping[str, Any]],
) -> bool:
    solve_threshold = frozen_thresholds.get(
        "accelerated_linear_solve_increase", {}
    ).get("value")
    if (
        not isinstance(solve_threshold, (int, float))
        or isinstance(solve_threshold, bool)
        or not np.isfinite(solve_threshold)
    ):
        raise _error(
            "metric_recalculation",
            "accelerated linear-solve threshold is not finite numeric",
        )
    return bool(
        truth["linear_solve_count_delta_fraction"]["error"]
        <= float(solve_threshold)
        and all(
            truth[metric_id]["cpu_sha256"]
            == truth[metric_id]["candidate_sha256"]
            for metric_id in (
                "accepted_increment_digest_match",
                "fallback_event_digest_match",
            )
        )
    )


def _validate_metric_truth(
    metric: Mapping[str, Any],
    frozen_thresholds: Mapping[str, Mapping[str, Any]],
    recomputed: Mapping[str, Any],
) -> None:
    metric_id = metric.get("metric_id")
    kind = metric.get("metric_kind")
    status = metric.get("status")
    if kind == "numeric":
        cpu_value = metric.get("cpu_value")
        candidate_value = metric.get("candidate_value")
        error = metric.get("error")
        threshold = metric.get("threshold")
        operator = metric.get("comparison_operator")
        numeric_values = (cpu_value, candidate_value, error, threshold)
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in numeric_values
        ):
            raise _error(
                "metric_recalculation",
                f"{metric_id}: numeric metric is not numeric",
            )
        for field, observed in (
            ("cpu_value", cpu_value),
            ("candidate_value", candidate_value),
            ("error", error),
        ):
            expected = recomputed.get(field)
            if not isinstance(expected, (int, float)) or not np.isclose(
                observed,
                expected,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise _error(
                    "metric_recalculation",
                    f"{metric_id}: reported {field} {observed!r} != "
                    f"checkpoint value {expected!r}",
                )
        expected_units = BACKEND_METRIC_UNITS.get(str(metric_id))
        if expected_units is None:
            raise _error(
                "metric_recalculation",
                f"{metric_id}: no unit contract",
            )
        _require_equal(
            metric.get("value_unit"),
            expected_units[0],
            "metric_recalculation",
            f"{metric_id} value unit",
        )
        _require_equal(
            metric.get("error_unit"),
            expected_units[1],
            "metric_recalculation",
            f"{metric_id} error unit",
        )
        threshold_metric_id = BACKEND_THRESHOLD_BINDINGS.get(str(metric_id))
        if threshold_metric_id is None:
            raise _error(
                "metric_recalculation",
                f"{metric_id}: no frozen G0 threshold binding",
            )
        frozen_threshold = frozen_thresholds.get(threshold_metric_id)
        if frozen_threshold is None:
            raise _error(
                "metric_recalculation",
                f"{metric_id}: missing frozen threshold {threshold_metric_id!r}",
            )
        expected_threshold = frozen_threshold.get("value")
        if metric_id == "max_front_bending_error":
            relative_threshold = frozen_thresholds.get(
                "hybrid_release_displacement_relative", {}
            ).get("value")
            if not isinstance(relative_threshold, (int, float)):
                raise _error(
                    "metric_recalculation",
                    "max front bending relative threshold is missing",
                )
            expected_threshold = max(
                float(expected_threshold),
                float(relative_threshold) * abs(float(recomputed["cpu_value"])),
            )
        _require_equal(
            operator,
            "<=",
            "metric_recalculation",
            f"{metric_id} comparison operator",
        )
        if not isinstance(expected_threshold, (int, float)) or not np.isclose(
            threshold,
            expected_threshold,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise _error(
                "metric_recalculation",
                f"{metric_id}: reported threshold {threshold!r} != frozen "
                f"threshold {expected_threshold!r}",
            )
        passed = error <= threshold
        expected_status = "pass" if passed else "fail"
        _require_equal(
            status,
            expected_status,
            "metric_recalculation",
            f"{metric_id} status",
        )
    elif kind == "digest_match":
        _require_equal(
            metric.get("cpu_sha256"),
            recomputed.get("cpu_sha256"),
            "metric_recalculation",
            f"{metric_id} CPU digest",
        )
        _require_equal(
            metric.get("candidate_sha256"),
            recomputed.get("candidate_sha256"),
            "metric_recalculation",
            f"{metric_id} candidate digest",
        )
        recomputed_match = (
            recomputed.get("cpu_sha256")
            == recomputed.get("candidate_sha256")
        )
        _require_equal(
            metric.get("match"),
            recomputed_match,
            "metric_recalculation",
            f"{metric_id} digest match",
        )
        _require_equal(
            status,
            "pass" if recomputed_match else "fail",
            "metric_recalculation",
            f"{metric_id} status",
        )
    elif kind == "categorical_match":
        _require_equal(
            metric.get("cpu_value"),
            recomputed.get("cpu_value"),
            "metric_recalculation",
            f"{metric_id} CPU category",
        )
        _require_equal(
            metric.get("candidate_value"),
            recomputed.get("candidate_value"),
            "metric_recalculation",
            f"{metric_id} candidate category",
        )
        recomputed_match = (
            recomputed.get("cpu_value") == recomputed.get("candidate_value")
        )
        _require_equal(
            metric.get("match"),
            recomputed_match,
            "metric_recalculation",
            f"{metric_id} categorical match",
        )
        _require_equal(
            status,
            "pass" if recomputed_match else "fail",
            "metric_recalculation",
            f"{metric_id} status",
        )
    else:
        raise _error(
            "metric_recalculation",
            f"{metric_id}: unknown metric kind {kind!r}",
        )


def _energy_number(
    payload: Mapping[str, Any],
    key: str,
    *,
    nonnegative: bool = False,
) -> float:
    value = payload.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or (nonnegative and float(value) < 0.0)
    ):
        domain = "finite nonnegative" if nonnegative else "finite"
        raise _error(
            "energy_audit_evidence",
            f"{key} must be a {domain} JSON number",
        )
    return float(value)


def _require_close(
    observed: Any,
    expected: float,
    *,
    description: str,
) -> None:
    if (
        not isinstance(observed, (int, float))
        or isinstance(observed, bool)
        or not np.isfinite(observed)
        or not np.isclose(
            float(observed),
            float(expected),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ):
        raise _error(
            "energy_audit_evidence",
            f"{description}: observed {observed!r}, expected {expected!r}",
        )


def _validate_energy_run_truth(
    *,
    ledger_path: Path,
    summary_path: Path,
    audit_path: Path,
    threshold: float,
    expected_thermal_solve_count: int,
) -> bool:
    if (
        not isinstance(expected_thermal_solve_count, int)
        or isinstance(expected_thermal_solve_count, bool)
        or expected_thermal_solve_count < 1
    ):
        raise _error(
            "energy_audit_evidence",
            "expected thermal solve count must be a positive integer",
        )
    rows = load_jsonl_strict(ledger_path)
    summary = _require_mapping(
        load_json_strict(summary_path),
        code="energy_audit_evidence",
        description="thermal energy summary",
    )
    audit = _require_mapping(
        load_json_strict(audit_path),
        code="energy_audit_evidence",
        description="run audit",
    )
    _require_equal(
        summary.get("schema_version"),
        "v06.thermal-energy-ledger-summary/1",
        "energy_audit_evidence",
        "thermal energy summary schema",
    )
    _require_equal(
        audit.get("schema_version"),
        "v06.run-audit.2",
        "energy_audit_evidence",
        "run audit schema",
    )
    transient = _require_mapping(
        audit.get("transient"),
        code="energy_audit_evidence",
        description="run audit transient section",
    )
    output_step_count = transient.get("step_count")
    if (
        not isinstance(output_step_count, int)
        or isinstance(output_step_count, bool)
        or output_step_count < 0
    ):
        raise _error(
            "energy_audit_evidence",
            "run audit transient step_count must be a nonnegative integer",
        )
    recorded_step_count = summary.get("recorded_step_count")
    expected_step_count = summary.get("expected_step_count")
    if (
        not isinstance(recorded_step_count, int)
        or isinstance(recorded_step_count, bool)
        or not isinstance(expected_step_count, int)
        or isinstance(expected_step_count, bool)
        or recorded_step_count < 1
        or expected_step_count < 1
        or recorded_step_count != len(rows)
        or expected_step_count != expected_thermal_solve_count
        or recorded_step_count != expected_step_count
    ):
        raise _error(
            "energy_audit_evidence",
            "ledger row count does not match its recorded/expected thermal "
            "solve counts",
        )

    balance_passes: list[bool] = []
    assembly_passes: list[bool] = []
    state_override_passes: list[bool] = []
    temperature_passes: list[bool] = []
    relative_errors: list[float] = []
    absolute_errors: list[float] = []
    assembly_errors: list[float] = []
    state_override_values: list[float] = []
    for index, raw_row in enumerate(rows):
        row = _require_mapping(
            raw_row,
            code="energy_audit_evidence",
            description=f"thermal ledger row {index}",
        )
        _require_equal(
            row.get("schema_version"),
            "v06.thermal-energy-ledger-step/1",
            "energy_audit_evidence",
            f"thermal ledger row {index} schema",
        )
        _require_equal(
            row.get("step_index"),
            index,
            "energy_audit_evidence",
            f"thermal ledger row {index} index",
        )
        try:
            balance = compute_discrete_balance(
                storage_j=_energy_number(row, "storage_j"),
                laser_deposited_j=_energy_number(
                    row, "laser_deposited_j", nonnegative=True
                ),
                laser_commanded_j=(
                    None
                    if row.get("laser_commanded_j") is None
                    else _energy_number(
                        row, "laser_commanded_j", nonnegative=True
                    )
                ),
                laser_absorbed_nominal_j=_energy_number(
                    row, "laser_absorbed_nominal_j", nonnegative=True
                ),
                front_loss_j=_energy_number(row, "front_loss_j"),
                old_layer_loss_j=_energy_number(row, "old_layer_loss_j"),
                surface_loss_j=_energy_number(row, "surface_loss_j"),
                dirichlet_exchange_into_domain_j=_energy_number(
                    row, "dirichlet_exchange_into_domain_j"
                ),
                assembly_identity_error_j=_energy_number(
                    row, "assembly_identity_error_j", nonnegative=True
                ),
                free_residual_l1_j=_energy_number(
                    row, "free_residual_l1_j", nonnegative=True
                ),
                free_residual_l2_j=_energy_number(
                    row, "free_residual_l2_j", nonnegative=True
                ),
            )
        except ValueError as exc:
            raise _error(
                "energy_audit_evidence",
                f"thermal ledger row {index}: {exc}",
            ) from exc
        _require_close(
            row.get("balance_error_j"),
            balance.balance_error_j,
            description=f"thermal ledger row {index} balance error",
        )
        _require_close(
            row.get("relative_balance_error"),
            balance.relative_balance_error,
            description=f"thermal ledger row {index} relative balance error",
        )
        balance_scale = (
            abs(balance.storage_j)
            + abs(balance.laser_deposited_j)
            + abs(balance.front_loss_j)
            + abs(balance.old_layer_loss_j)
            + abs(balance.surface_loss_j)
            + abs(balance.dirichlet_exchange_into_domain_j)
        )
        _require_close(
            row.get("balance_scale_j"),
            balance_scale,
            description=f"thermal ledger row {index} balance scale",
        )
        free_node_count = row.get("free_node_count")
        if (
            not isinstance(free_node_count, int)
            or isinstance(free_node_count, bool)
            or free_node_count < 0
        ):
            raise _error(
                "energy_audit_evidence",
                f"thermal ledger row {index} free_node_count must be a "
                "nonnegative integer",
            )
        dt_s = _energy_number(row, "dt_s")
        if dt_s <= 0.0:
            raise _error(
                "energy_audit_evidence",
                f"thermal ledger row {index} dt_s must be positive",
            )
        residual_tolerance_w = _energy_number(
            row,
            "solver_residual_tolerance_w",
            nonnegative=True,
        )
        recorded_absolute_tolerance = _energy_number(
            row,
            "absolute_balance_tolerance_j",
            nonnegative=True,
        )
        absolute_tolerance = (
            np.sqrt(max(free_node_count, 1))
            * dt_s
            * residual_tolerance_w
        )
        _require_close(
            recorded_absolute_tolerance,
            absolute_tolerance,
            description=(
                f"thermal ledger row {index} absolute balance tolerance"
            ),
        )
        relative_tolerance = _energy_number(
            row,
            "relative_balance_tolerance",
            nonnegative=True,
        )
        balance_passed = bool(
            abs(balance.balance_error_j)
            <= absolute_tolerance + relative_tolerance * balance_scale
        )
        _require_equal(
            row.get("balance_within_solver_tolerance"),
            balance_passed,
            "energy_audit_evidence",
            f"thermal ledger row {index} balance status",
        )

        assembly_signed = _energy_number(
            row, "assembly_identity_signed_j"
        )
        _require_close(
            balance.assembly_identity_error_j,
            abs(assembly_signed),
            description=f"thermal ledger row {index} assembly error",
        )
        recorded_assembly_tolerance = _energy_number(
            row,
            "assembly_identity_tolerance_j",
            nonnegative=True,
        )
        explicit_total = (
            balance.storage_j
            - balance.laser_deposited_j
            + balance.front_loss_j
            + balance.old_layer_loss_j
            + balance.surface_loss_j
        )
        residual_total = explicit_total - assembly_signed
        assembly_tolerance = 1.0e-12 + 1.0e-10 * max(
            abs(explicit_total),
            abs(residual_total),
            np.finfo(np.float64).tiny,
        )
        _require_close(
            recorded_assembly_tolerance,
            assembly_tolerance,
            description=(
                f"thermal ledger row {index} assembly tolerance"
            ),
        )
        assembly_passed = abs(assembly_signed) <= assembly_tolerance
        _require_equal(
            row.get("assembly_identity_within_tolerance"),
            assembly_passed,
            "energy_audit_evidence",
            f"thermal ledger row {index} assembly status",
        )

        state_override = row.get("pre_solve_state_override_j")
        if state_override is not None:
            state_override = _energy_number(
                row, "pre_solve_state_override_j"
            )
            state_override_values.append(state_override)
        recorded_state_override_tolerance = _energy_number(
            row,
            "state_override_tolerance_j",
            nonnegative=True,
        )
        state_override_tolerance = max(absolute_tolerance, 1.0e-12)
        _require_close(
            recorded_state_override_tolerance,
            state_override_tolerance,
            description=(
                f"thermal ledger row {index} state-override tolerance"
            ),
        )
        state_override_passed = bool(
            state_override is None
            or abs(state_override) <= state_override_tolerance
        )
        _require_equal(
            row.get("state_override_within_tolerance"),
            state_override_passed,
            "energy_audit_evidence",
            f"thermal ledger row {index} state-override status",
        )

        invariants = _require_mapping(
            row.get("temperature_invariants"),
            code="energy_audit_evidence",
            description=f"thermal ledger row {index} temperature invariants",
        )
        _require_equal(
            invariants.get("claim_level"),
            "physical_temperature_invariant_diagnostic",
            "energy_audit_evidence",
            f"thermal ledger row {index} temperature-invariant claim",
        )
        coefficient_preconditions = invariants.get(
            "coefficient_preconditions_valid"
        )
        temperatures_finite = invariants.get(
            "all_new_temperatures_finite"
        )
        source_free = invariants.get("source_free")
        if any(
            not isinstance(value, bool)
            for value in (
                coefficient_preconditions,
                temperatures_finite,
                source_free,
            )
        ):
            raise _error(
                "energy_audit_evidence",
                f"thermal ledger row {index} invariant preconditions must be "
                "boolean",
            )
        expected_source_free = (
            balance.laser_deposited_j <= np.finfo(np.float64).eps
        )
        _require_equal(
            source_free,
            expected_source_free,
            "energy_audit_evidence",
            f"thermal ledger row {index} source-free status",
        )
        lower_bound = _energy_number(invariants, "lower_bound_k")
        upper_bound = invariants.get("upper_bound_k")
        if source_free:
            upper_bound = _energy_number(invariants, "upper_bound_k")
            if lower_bound > upper_bound:
                raise _error(
                    "energy_audit_evidence",
                    f"thermal ledger row {index} invariant bounds are reversed",
                )
        elif upper_bound is not None:
            raise _error(
                "energy_audit_evidence",
                f"thermal ledger row {index} source-bearing upper bound must "
                "be null",
            )
        violation_counts = (
            invariants.get("lower_violation_count"),
            invariants.get("upper_violation_count"),
        )
        lower_count, upper_count = violation_counts
        if (
            not isinstance(lower_count, int)
            or isinstance(lower_count, bool)
            or lower_count < 0
            or (
                source_free
                and (
                    not isinstance(upper_count, int)
                    or isinstance(upper_count, bool)
                    or upper_count < 0
                )
            )
            or (not source_free and upper_count is not None)
        ):
            raise _error(
                "energy_audit_evidence",
                f"thermal ledger row {index} invariant violation counts are "
                "invalid",
            )
        _energy_number(invariants, "atol_k", nonnegative=True)
        temperature_passed = bool(
            coefficient_preconditions
            and temperatures_finite
            and lower_count == 0
            and upper_count in (None, 0)
        )
        _require_equal(
            invariants.get("valid"),
            temperature_passed,
            "energy_audit_evidence",
            f"thermal ledger row {index} nested temperature status",
        )
        _require_equal(
            row.get("temperature_invariants_valid"),
            temperature_passed,
            "energy_audit_evidence",
            f"thermal ledger row {index} temperature status",
        )

        balance_passes.append(balance_passed)
        assembly_passes.append(assembly_passed)
        state_override_passes.append(state_override_passed)
        temperature_passes.append(temperature_passed)
        relative_errors.append(balance.relative_balance_error)
        absolute_errors.append(abs(balance.balance_error_j))
        assembly_errors.append(balance.assembly_identity_error_j)

    summary_expected = {
        "recorded_step_count": len(rows),
        "expected_step_count": expected_thermal_solve_count,
        "all_balance_steps_within_tolerance": all(balance_passes),
        "all_assembly_identities_within_tolerance": all(assembly_passes),
        "all_pre_solve_state_overrides_within_tolerance": all(
            state_override_passes
        ),
        "all_temperature_invariants_valid": all(temperature_passes),
    }
    for field, expected in summary_expected.items():
        _require_equal(
            summary.get(field),
            expected,
            "energy_audit_evidence",
            f"thermal energy summary {field}",
        )
    solver_completed = summary.get("solver_completed")
    if not isinstance(solver_completed, bool):
        raise _error(
            "energy_audit_evidence",
            "thermal energy summary solver_completed must be boolean",
        )
    complete = bool(solver_completed and all(summary_expected.values()))
    _require_equal(
        summary.get("complete"),
        complete,
        "energy_audit_evidence",
        "thermal energy summary complete flag",
    )
    maximum_relative = max(relative_errors)
    for field, expected in (
        ("maximum_relative_balance_error", maximum_relative),
        ("maximum_absolute_balance_error_j", max(absolute_errors)),
        ("maximum_assembly_identity_error_j", max(assembly_errors)),
        (
            "cumulative_pre_solve_state_override_j",
            sum(state_override_values),
        ),
    ):
        _require_close(
            summary.get(field),
            expected,
            description=f"thermal energy summary {field}",
        )
    return bool(complete and maximum_relative <= threshold)


def _validate_energy_audit_truth(
    qualification: Mapping[str, Any],
    *,
    manifests_by_id: Mapping[str, Mapping[str, Any]],
    frozen_thresholds: Mapping[str, Mapping[str, Any]],
    threshold_set_path: Path,
    artifact_root: Path,
) -> bool:
    energy_gate = qualification.get("stage_gates", {}).get("energy_audit")
    if not isinstance(energy_gate, Mapping):
        raise _error(
            "energy_audit_evidence",
            "evaluated qualification is missing the energy audit gate",
        )
    evidence = energy_gate.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        raise _error(
            "energy_audit_evidence",
            "energy audit gate requires exactly one typed evidence wrapper",
        )
    wrapper_path = _validate_file_identity(
        evidence[0],
        artifact_root,
        code="energy_audit_evidence",
    )
    wrapper = _require_mapping(
        load_json_strict(wrapper_path),
        code="energy_audit_evidence",
        description="energy audit evidence wrapper",
    )
    validate_json_contract(wrapper, "energy-audit-evidence.schema.json")
    _require_equal(
        wrapper.get("qualification_id"),
        qualification.get("qualification_id"),
        "energy_audit_evidence",
        "energy evidence qualification id",
    )
    _require_equal(
        wrapper.get("threshold_set_sha256"),
        sha256_file(threshold_set_path),
        "energy_audit_evidence",
        "energy evidence threshold set",
    )
    threshold_record = frozen_thresholds.get("thermal_energy_closure")
    if not isinstance(threshold_record, Mapping):
        raise _error(
            "energy_audit_evidence",
            "frozen thermal_energy_closure threshold is missing",
        )
    _require_equal(
        threshold_record.get("operator"),
        "less_than_or_equal",
        "energy_audit_evidence",
        "thermal energy threshold operator",
    )
    _require_equal(
        threshold_record.get("unit"),
        "fraction",
        "energy_audit_evidence",
        "thermal energy threshold unit",
    )
    threshold = _energy_number(
        threshold_record,
        "value",
        nonnegative=True,
    )

    run_records = wrapper.get("runs", [])
    run_ids = [
        str(record.get("run_id"))
        for record in run_records
        if isinstance(record, Mapping)
    ]
    if (
        len(run_ids) != len(run_records)
        or len(set(run_ids)) != len(run_ids)
        or set(run_ids) != set(manifests_by_id)
    ):
        raise _error(
            "energy_audit_evidence",
            "energy evidence runs must uniquely and exactly cover qualification "
            "CPU/candidate runs",
        )
    used_paths: set[Path] = set()
    run_passes: list[bool] = []
    role_fields = {
        "ledger": "thermal_energy_ledger",
        "summary": "thermal_energy_ledger_summary",
        "run_audit": "v06_run_audit",
    }
    for run_record in run_records:
        run_id = str(run_record["run_id"])
        manifest = manifests_by_id[run_id]
        resolved: dict[str, Path] = {}
        for field, role in role_fields.items():
            wrapper_record = run_record[field]
            manifest_record = _artifact_by_role(manifest, role)
            manifest_path = _validate_file_identity(
                manifest_record,
                artifact_root,
                code="energy_audit_evidence",
            )
            wrapper_path_for_role = _validate_file_identity_at(
                wrapper_record,
                manifest_path,
                artifact_root,
                code="energy_audit_evidence",
            )
            if wrapper_path_for_role in used_paths:
                raise _error(
                    "energy_audit_evidence",
                    "different qualification runs must not reuse thermal "
                    "energy evidence paths",
                )
            used_paths.add(wrapper_path_for_role)
            resolved[field] = wrapper_path_for_role
        used_config_path = _validate_file_identity(
            _input_by_role(manifest)["used_config"],
            artifact_root,
            code="energy_audit_evidence",
        )
        used_config = _require_mapping(
            load_json_strict(used_config_path),
            code="energy_audit_evidence",
            description=f"{run_id} used config",
        )
        derived = _require_mapping(
            used_config.get("derived"),
            code="energy_audit_evidence",
            description=f"{run_id} used config derived section",
        )
        expected_thermal_solve_count = derived.get("total_steps")
        if (
            not isinstance(expected_thermal_solve_count, int)
            or isinstance(expected_thermal_solve_count, bool)
            or expected_thermal_solve_count < 1
        ):
            raise _error(
                "energy_audit_evidence",
                f"{run_id} used config derived.total_steps must be a "
                "positive integer",
            )
        run_passes.append(
            _validate_energy_run_truth(
                ledger_path=resolved["ledger"],
                summary_path=resolved["summary"],
                audit_path=resolved["run_audit"],
                threshold=threshold,
                expected_thermal_solve_count=expected_thermal_solve_count,
            )
        )
    energy_passed = all(run_passes)
    _require_equal(
        energy_gate.get("status"),
        "pass" if energy_passed else "fail",
        "energy_audit_evidence",
        "energy audit gate status",
    )
    return energy_passed


def _validate_checkpoint_gate_evidence(
    qualification: Mapping[str, Any],
    *,
    evidence_by_run: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
) -> None:
    expected_paths = {
        Path(evidence["checkpoint_path"]).resolve()
        for evidence in evidence_by_run.values()
    }
    for gate_id in ("backend_parity", "convergence_audit"):
        gate = qualification.get("stage_gates", {}).get(gate_id)
        records = gate.get("evidence") if isinstance(gate, Mapping) else None
        if not isinstance(records, list):
            raise _error(
                "gate_evidence_binding",
                f"{gate_id} evidence must be an array",
            )
        observed_paths = {
            _validate_file_identity(
                record,
                artifact_root,
                code="gate_evidence_binding",
            ).resolve()
            for record in records
            if isinstance(record, Mapping)
        }
        if (
            len(observed_paths) != len(records)
            or observed_paths != expected_paths
        ):
            raise _error(
                "gate_evidence_binding",
                f"{gate_id} evidence must exactly cover every native "
                "qualification checkpoint",
            )


def _validate_gate_truth(
    qualification: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    """Reject a passing verdict when any declared gate or level failed."""
    if qualification.get("verdict") == "pass":
        required_gate_ids = {
            "backend_parity",
            "energy_audit",
            "convergence_audit",
        }
        for gate_id, gate in qualification.get("stage_gates", {}).items():
            allowed_statuses = (
                {"pass"} if gate_id in required_gate_ids
                else {"pass", "not_applicable"}
            )
            if not isinstance(gate, Mapping) or gate.get(
                "status"
            ) not in allowed_statuses:
                raise _error(
                    "gate_consistency",
                    f"{gate_id}: passing qualification cannot contain a "
                    "failed gate",
                )
        for level_id, pair in qualification.get("level_run_pairs", {}).items():
            if not isinstance(pair, Mapping) or pair.get("status") != "pass":
                raise _error(
                    "gate_consistency",
                    f"{level_id}: passing qualification requires a passing level",
                )
    if str(validation.get("verdict", "")).startswith("pass"):
        for check_id, check in validation.get("checks", {}).items():
            if not isinstance(check, Mapping) or check.get("status") != "pass":
                raise _error(
                    "gate_consistency",
                    f"{check_id}: passing validation requires a passing check",
                )


def _reject_untyped_stage_promotion(
    qualification: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    if (
        qualification.get("promotion_eligible") is True
        or formal_promotion_allowed(validation)
    ):
        raise _error(
            "stage_gate_evidence",
            "formal promotion is blocked until T028/T031 produce typed, "
            "recomputable build/cooling/release stage evidence",
        )


def _validate_level_membership(
    qualification: Mapping[str, Any],
    current_candidate_run_id: str,
    candidate_manifest_run_ids: set[str],
    cpu_manifest_run_ids: set[str],
) -> None:
    candidates = set(qualification.get("candidate_run_ids", []))
    references = set(qualification.get("cpu_reference_run_ids", []))
    if current_candidate_run_id not in candidates:
        raise _error(
            "candidate_membership",
            f"current run {current_candidate_run_id!r} is not a qualification "
            "candidate",
        )
    if candidate_manifest_run_ids != candidates:
        raise _error(
            "candidate_membership",
            "provided candidate manifests do not exactly match qualification "
            "candidate IDs",
        )
    if cpu_manifest_run_ids != references:
        raise _error(
            "candidate_membership",
            "provided CPU manifests do not exactly match qualification "
            "reference IDs",
        )
    if candidates & references:
        raise _error(
            "candidate_membership",
            "candidate and CPU reference run IDs overlap",
        )
    levels = set(qualification.get("levels", []))
    pairs = qualification.get("level_run_pairs", {})
    if not isinstance(pairs, Mapping) or set(pairs) != levels:
        raise _error(
            "level_coverage",
            "level_run_pairs must match the declared qualification levels",
        )
    for level, pair in pairs.items():
        if not isinstance(pair, Mapping):
            raise _error("level_coverage", f"{level}: level pair is invalid")
        pair_cpu = set(pair.get("cpu_run_ids", []))
        pair_candidates = set(pair.get("candidate_run_ids", []))
        if not pair_cpu <= references or not pair_candidates <= candidates:
            raise _error(
                "candidate_membership",
                f"{level}: level pair references undeclared run IDs",
            )
    paired_cpu = {
        run_id
        for pair in pairs.values()
        for run_id in pair.get("cpu_run_ids", [])
    }
    paired_candidates = {
        run_id
        for pair in pairs.values()
        for run_id in pair.get("candidate_run_ids", [])
    }
    if paired_cpu != references or paired_candidates != candidates:
        raise _error(
            "candidate_membership",
            "union of level pairs must exactly cover declared CPU and candidate "
            "run IDs",
        )
    cpu_occurrences = Counter(
        str(run_id)
        for pair in pairs.values()
        for run_id in pair.get("cpu_run_ids", [])
    )
    candidate_occurrences = Counter(
        str(run_id)
        for pair in pairs.values()
        for run_id in pair.get("candidate_run_ids", [])
    )
    reused_cpu = sorted(
        run_id for run_id in references if cpu_occurrences[run_id] != 1
    )
    reused_candidates = sorted(
        run_id for run_id in candidates if candidate_occurrences[run_id] != 1
    )
    if reused_cpu or reused_candidates:
        raise _error(
            "level_coverage",
            "each supplied run must belong to exactly one qualification level; "
            f"CPU={reused_cpu}, candidate={reused_candidates}",
        )
    if current_candidate_run_id not in paired_candidates:
        raise _error(
            "candidate_membership",
            "current candidate is not present in any qualification level pair",
        )


def _validate_stage_placement(
    mode: str,
    backend: Mapping[str, Any],
    placement: Mapping[str, Any],
) -> None:
    _require_equal(
        placement.get("orchestration_backend"),
        backend.get("orchestration_backend"),
        "placement_reconciliation",
        "orchestration backend",
    )
    _require_equal(
        placement.get("full_loop_xla"),
        backend.get("full_loop_xla"),
        "placement_reconciliation",
        "full-loop XLA flag",
    )
    if placement.get("unexpected_fallback_count") != 0:
        raise _error(
            "placement_reconciliation",
            "qualification contains unexpected backend fallbacks",
        )
    stages = (backend.get("thermal"), backend.get("mechanics"), backend.get("release"))
    used = [stage for stage in stages if stage.get("status") == "used"]
    if not used:
        raise _error(
            "placement_reconciliation",
            "qualification candidate did not execute any solver stage",
        )
    if mode == "hybrid_gpu_assembly_cpu_pardiso":
        if placement.get("cpu_pardiso_calls", 0) < 1:
            raise _error(
                "placement_reconciliation",
                "hybrid mode must record CPU PARDISO calls",
            )
        expected = {
            "local_assembly_backend": "gpu",
            "global_matrix_backend": "cpu",
            "linear_solver_backend": "cpu_pardiso",
        }
    elif mode == "full_gpu":
        if placement.get("cpu_pardiso_calls") != 0:
            raise _error(
                "placement_reconciliation",
                "full-GPU mode cannot record CPU PARDISO calls",
            )
        expected = {
            "local_assembly_backend": "gpu",
            "global_matrix_backend": "gpu",
            "state_residency_backend": "gpu",
        }
        if any(
            stage.get("linear_solver_backend")
            not in {"gpu_jax", "gpu_petsc", "gpu_amgx"}
            for stage in used
        ):
            raise _error(
                "placement_reconciliation",
                "full-GPU mode used a non-GPU linear solver",
            )
    else:
        raise _error(
            "execution_mode",
            f"mode {mode!r} is not eligible for backend qualification",
        )
    for stage in used:
        for key, expected_value in expected.items():
            _require_equal(
                stage.get(key),
                expected_value,
                "placement_reconciliation",
                f"stage {key}",
            )


def _validate_profiler_placement(
    qualification: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    candidate_manifests: tuple[Mapping[str, Any], ...],
    candidate_manifest_paths: tuple[Path, ...],
    cpu_manifest_paths: tuple[Path, ...],
    artifact_root: Path,
) -> None:
    placement = qualification.get("placement_evidence", {})
    manifest_records = placement.get("run_manifest_artifacts", [])
    observed_manifest_paths = {
        _validate_file_identity(record, artifact_root).resolve()
        for record in manifest_records
        if isinstance(record, Mapping)
    }
    expected_manifest_paths = set(candidate_manifest_paths) | set(cpu_manifest_paths)
    if (
        observed_manifest_paths != expected_manifest_paths
        or len(manifest_records) != len(expected_manifest_paths)
    ):
        raise _error(
            "placement_reconciliation",
            "placement run-manifest artifacts do not exactly match supplied runs",
        )
    profiler_records = placement.get("profiler_artifacts", [])
    if not profiler_records:
        raise _error(
            "placement_reconciliation",
            "placement evidence contains no profiler artifact",
        )
    profiler_paths = {
        _validate_file_identity(record, artifact_root).resolve()
        for record in profiler_records
        if isinstance(record, Mapping)
    }
    candidate_profiler_paths = {
        _validate_file_identity(
            _artifact_by_role(manifest, "profiler"),
            artifact_root,
        ).resolve()
        for manifest in candidate_manifests
    }
    if (
        profiler_paths != candidate_profiler_paths
        or len(profiler_records) != len(candidate_profiler_paths)
    ):
        raise _error(
            "placement_reconciliation",
            "qualification profiler artifacts do not exactly match candidate "
            "manifest profiler artifacts",
        )
    manifests_by_run_id = {
        str(manifest.get("run_id")): manifest
        for manifest in candidate_manifests
    }
    seen_profiler_run_ids: set[str] = set()
    for record in profiler_records:
        if not isinstance(record, Mapping):
            raise _error(
                "placement_reconciliation",
                "profiler artifact record must be an object",
            )
        profiler_path = _validate_file_identity(record, artifact_root)
        profiler = load_json_strict(profiler_path)
        if not isinstance(profiler, Mapping):
            raise _error(
                "placement_reconciliation",
                f"{profiler_path}: profiler payload must be an object",
            )
        run_id = str(profiler.get("run_id"))
        manifest = manifests_by_run_id.get(run_id)
        if manifest is None or run_id in seen_profiler_run_ids:
            raise _error(
                "placement_reconciliation",
                f"{profiler_path}: profiler run_id is missing, duplicate or "
                "not a candidate",
            )
        seen_profiler_run_ids.add(run_id)
        backend = manifest.get("backend", {})
        expected = {
            "execution_mode": backend.get("mode"),
            "orchestration_backend": backend.get("orchestration_backend"),
            "full_loop_xla": backend.get("full_loop_xla"),
            "unexpected_fallback_count": len(
                backend.get("unexpected_fallbacks", [])
            ),
            "stages": {
                stage_name: backend.get(stage_name)
                for stage_name in ("thermal", "mechanics", "release")
            },
        }
        for key, expected_value in expected.items():
            _require_equal(
                profiler.get(key),
                expected_value,
                "placement_reconciliation",
                f"profiler {key}",
            )
        calls = profiler.get("cpu_pardiso_calls")
        if backend.get("mode") == "hybrid_gpu_assembly_cpu_pardiso":
            if not isinstance(calls, int) or calls < 1:
                raise _error(
                    "placement_reconciliation",
                    f"{run_id}: hybrid profiler requires CPU PARDISO calls",
                )
        elif backend.get("mode") == "full_gpu" and calls != 0:
            raise _error(
                "placement_reconciliation",
                f"{run_id}: full-GPU profiler records CPU PARDISO calls",
            )
        if run_id == str(candidate.get("run_id")):
            for key in (
                "orchestration_backend",
                "full_loop_xla",
                "cpu_pardiso_calls",
                "unexpected_fallback_count",
            ):
                _require_equal(
                    profiler.get(key),
                    placement.get(key),
                    "placement_reconciliation",
                    f"current profiler {key}",
                )
    if seen_profiler_run_ids != set(manifests_by_run_id):
        raise _error(
            "placement_reconciliation",
            "candidate profiler evidence does not cover every candidate run",
        )


def _load_performance_evidence(
    qualification: Mapping[str, Any],
    *,
    cpu_threads: int,
    artifact_root: Path,
    execution_intervals: Mapping[str, tuple[datetime, datetime]],
    linear_solve_counts: Mapping[str, int],
    manifests_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    performance = qualification.get("performance", {})
    records = performance.get("evidence", [])
    if not isinstance(records, list) or len(records) != 1:
        raise _error(
            "performance_recalculation",
            "performance requires exactly one raw evidence artifact",
        )
    path = _validate_file_identity(
        records[0],
        artifact_root,
        code="performance_recalculation",
    )
    evidence = load_json_strict(path)
    required_fields = {
        "schema_version",
        "measured",
        "cpu_run_ids",
        "candidate_run_ids",
        "cpu_threads",
        "cpu_wall_seconds_samples",
        "candidate_wall_seconds_samples",
        "cpu_linear_solve_count_samples",
        "candidate_linear_solve_count_samples",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_fields:
        raise _error(
            "performance_recalculation",
            "raw performance evidence fields do not match the contract",
        )
    _require_equal(
        evidence.get("schema_version"),
        "kaess.performance-evidence/1",
        "performance_recalculation",
        "performance evidence schema",
    )
    _require_equal(
        evidence.get("cpu_threads"),
        cpu_threads,
        "thread_budget",
        "performance evidence CPU threads",
    )
    _require_equal(
        performance.get("measured"),
        evidence.get("measured"),
        "performance_recalculation",
        "performance measured flag",
    )
    measured = evidence.get("measured") is True
    if measured:
        performance_pair = qualification.get("level_run_pairs", {}).get(
            "performance_pair"
        )
        if not isinstance(performance_pair, Mapping):
            raise _error(
                "performance_protocol_identity",
                "measured performance requires a performance_pair level",
            )
        expected_cpu_run_ids = [
            str(run_id)
            for run_id in performance_pair.get("cpu_run_ids", [])
        ]
        expected_candidate_run_ids = [
            str(run_id)
            for run_id in performance_pair.get("candidate_run_ids", [])
        ]
    else:
        expected_cpu_run_ids = []
        expected_candidate_run_ids = []
    _require_equal(
        evidence.get("cpu_run_ids"),
        expected_cpu_run_ids,
        "performance_recalculation",
        "performance evidence CPU runs",
    )
    _require_equal(
        evidence.get("candidate_run_ids"),
        expected_candidate_run_ids,
        "performance_recalculation",
        "performance evidence candidate runs",
    )

    def expected_wall_seconds(run_ids: list[str]) -> list[float]:
        values: list[float] = []
        starts: list[datetime] = []
        for run_id in run_ids:
            interval = execution_intervals.get(run_id)
            if interval is None:
                raise _error(
                    "performance_protocol_identity",
                    f"performance run {run_id!r} has no execution interval",
                )
            started, completed = interval
            manifest = manifests_by_id.get(run_id)
            if not isinstance(manifest, Mapping):
                raise _error(
                    "performance_protocol_identity",
                    f"performance run {run_id!r} has no run manifest",
                )
            try:
                manifest_completed = datetime.fromisoformat(
                    str(manifest.get("completed_utc")).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise _error(
                    "performance_protocol_identity",
                    f"performance run {run_id!r} has no valid completed_utc",
                ) from exc
            if manifest_completed != completed:
                raise _error(
                    "performance_protocol_identity",
                    f"performance run {run_id!r} completion time is not bound "
                    "to its manifest",
                )
            resource_usage = manifest.get("resource_usage")
            wall_seconds = (
                resource_usage.get("wall_seconds")
                if isinstance(resource_usage, Mapping)
                else None
            )
            duration = float((completed - started).total_seconds())
            if (
                not isinstance(wall_seconds, (int, float))
                or isinstance(wall_seconds, bool)
                or not np.isfinite(wall_seconds)
                or not np.isclose(
                    float(wall_seconds),
                    duration,
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                raise _error(
                    "performance_protocol_identity",
                    f"performance run {run_id!r} wall time is not bound to "
                    "its manifest resource usage",
                )
            starts.append(started)
            values.append(duration)
        if starts != sorted(starts):
            raise _error(
                "performance_protocol_identity",
                "performance run IDs must follow execution order",
            )
        return values

    expected_sequences = {
        "cpu_wall_seconds_samples": expected_wall_seconds(
            expected_cpu_run_ids
        ),
        "candidate_wall_seconds_samples": expected_wall_seconds(
            expected_candidate_run_ids
        ),
        "cpu_linear_solve_count_samples": [
            linear_solve_counts[run_id] for run_id in expected_cpu_run_ids
        ],
        "candidate_linear_solve_count_samples": [
            linear_solve_counts[run_id]
            for run_id in expected_candidate_run_ids
        ],
    }
    for field in (
        "cpu_wall_seconds_samples",
        "candidate_wall_seconds_samples",
        "cpu_linear_solve_count_samples",
        "candidate_linear_solve_count_samples",
    ):
        _require_equal(
            performance.get(field, []),
            evidence.get(field),
            "performance_recalculation",
            f"performance evidence {field}",
        )
        observed = evidence.get(field)
        expected = expected_sequences[field]
        if (
            not isinstance(observed, list)
            or len(observed) != len(expected)
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in observed
            )
            or not np.allclose(
                np.asarray(observed, dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise _error(
                "performance_recalculation",
                f"{field} is not bound to execution/checkpoint evidence",
            )
    return evidence


def _validate_performance_truth(
    qualification: Mapping[str, Any],
    validation: Mapping[str, Any],
    protocol: Mapping[str, float | int],
    raw_performance: Mapping[str, Any],
) -> None:
    performance = qualification.get("performance", {})
    recomputed = validation.get("recomputed_performance", {})
    if raw_performance.get("measured") is not True:
        _require_equal(
            recomputed.get("measured"),
            False,
            "performance_recalculation",
            "unmeasured performance",
        )
        return
    cpu_samples = raw_performance.get("cpu_wall_seconds_samples")
    candidate_samples = raw_performance.get("candidate_wall_seconds_samples")
    cpu_solves = raw_performance.get("cpu_linear_solve_count_samples")
    candidate_solves = raw_performance.get(
        "candidate_linear_solve_count_samples"
    )
    sample_count = int(protocol["sample_count"])
    if not all(
        isinstance(samples, list) and len(samples) >= sample_count
        for samples in (
            cpu_samples,
            candidate_samples,
            cpu_solves,
            candidate_solves,
        )
    ):
        raise _error(
            "performance_protocol_identity",
            "measured performance does not meet the frozen sample count",
        )
    cpu_median = float(statistics.median(cpu_samples))
    candidate_median = float(statistics.median(candidate_samples))
    candidate_cold = float(candidate_samples[0])
    candidate_steady = float(statistics.median(candidate_samples[1:]))
    speedup = cpu_median / candidate_median
    cpu_solve_median = float(statistics.median(cpu_solves))
    candidate_solve_median = float(statistics.median(candidate_solves))
    solve_delta = (
        (candidate_solve_median - cpu_solve_median) / cpu_solve_median
        if cpu_solve_median
        else (0.0 if candidate_solve_median == 0 else float("inf"))
    )
    for observed, expected, name in (
        (
            performance.get("cold_wall_seconds"),
            candidate_cold,
            "qualification cold wall time",
        ),
        (
            performance.get("steady_wall_seconds"),
            candidate_steady,
            "qualification steady wall time",
        ),
        (performance.get("speedup"), speedup, "qualification speedup"),
        (
            performance.get("linear_solve_count_delta_fraction"),
            solve_delta,
            "qualification linear solve delta",
        ),
        (recomputed.get("cpu_median_wall_seconds"), cpu_median, "CPU median"),
        (
            recomputed.get("candidate_median_wall_seconds"),
            candidate_median,
            "candidate median",
        ),
        (recomputed.get("speedup"), speedup, "validation speedup"),
        (
            recomputed.get("linear_solve_count_delta_fraction"),
            solve_delta,
            "validation linear solve delta",
        ),
    ):
        if not isinstance(observed, (int, float)) or not np.isclose(
            observed, expected, rtol=1e-12, atol=1e-12
        ):
            raise _error(
                "performance_recalculation",
                f"{name}: observed {observed!r}, expected {expected!r}",
            )
    performance_passed = (
        speedup >= float(protocol["speedup"])
        and solve_delta <= float(protocol["linear_solve_increase"])
    )
    _require_equal(
        qualification.get("performance_qualified"),
        performance_passed,
        "performance_recalculation",
        "performance qualification flag",
    )
    performance_gate = qualification.get("stage_gates", {}).get(
        "performance_gate"
    )
    if performance_gate is not None:
        _require_equal(
            performance_gate.get("status"),
            "pass" if performance_passed else "fail",
            "performance_recalculation",
            "performance gate status",
        )


def validate_backend_qualification_bundle(
    qualification_path: Path,
    *,
    validation_path: Path,
    candidate_manifest_path: Path,
    cpu_manifest_paths: Iterable[Path],
    parity_config_path: Path,
    artifact_root: Path,
    threshold_set_path: Path | None = None,
    approval_record_path: Path | None = None,
    candidate_manifest_paths: Iterable[Path] | None = None,
) -> None:
    """Cross-check qualification, semantic verdict, runs, checkpoints and placement."""
    qualification_path = Path(qualification_path).resolve()
    validation_path = Path(validation_path).resolve()
    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    cpu_manifest_paths = tuple(Path(path).resolve() for path in cpu_manifest_paths)
    parity_config_path = Path(parity_config_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    threshold_set_path = Path(
        threshold_set_path
        if threshold_set_path is not None
        else parity_config_path.parent / "threshold-set.json"
    ).resolve()
    approval_record_path = Path(
        approval_record_path
        if approval_record_path is not None
        else parity_config_path.parent / "g0-approval.json"
    ).resolve()
    if candidate_manifest_paths is None:
        candidate_manifest_paths = (candidate_manifest_path,)
    else:
        candidate_manifest_paths = tuple(
            Path(path).resolve() for path in candidate_manifest_paths
        )
    if candidate_manifest_path not in candidate_manifest_paths:
        raise _error(
            "candidate_membership",
            "the current candidate manifest must be included in all candidate "
            "manifest paths",
        )
    if len(set(candidate_manifest_paths)) != len(candidate_manifest_paths):
        raise _error(
            "candidate_membership",
            "candidate manifest paths must be unique",
        )

    qualification = load_json_strict(qualification_path)
    validation = load_json_strict(validation_path)
    candidate_manifests = tuple(
        load_json_strict(path) for path in candidate_manifest_paths
    )
    candidate = candidate_manifests[
        candidate_manifest_paths.index(candidate_manifest_path)
    ]
    cpu_manifests = tuple(load_json_strict(path) for path in cpu_manifest_paths)

    validate_json_contract(
        qualification, "backend-qualification.schema.json"
    )
    validate_json_contract(
        validation, "backend-qualification-validation.schema.json"
    )
    for manifest in (*candidate_manifests, *cpu_manifests):
        validate_json_contract(manifest, "run-manifest.schema.json")
    for payload in (
        qualification,
        validation,
        *candidate_manifests,
        *cpu_manifests,
    ):
        _validate_all_artifacts(payload, artifact_root)
    _validate_file_identity_at(
        validation.get("qualification_artifact", {}),
        qualification_path,
        artifact_root,
        code="qualification_artifact_binding",
    )
    _validate_file_identity_at(
        validation.get("qualification_candidate_manifest_artifact", {}),
        candidate_manifest_path,
        artifact_root,
        code="candidate_manifest_binding",
    )

    _require_equal(
        validation.get("qualification_id"),
        qualification.get("qualification_id"),
        "qualification_id",
        "validation/qualification id",
    )
    _require_equal(
        validation.get("qualification_candidate_run_id"),
        candidate.get("run_id"),
        "candidate_membership",
        "validation/current candidate run",
    )
    _require_equal(
        validation.get("execution_mode"),
        qualification.get("execution_mode"),
        "execution_mode",
        "validation/qualification mode",
    )
    _require_equal(
        qualification.get("execution_mode"),
        candidate.get("backend", {}).get("mode"),
        "execution_mode",
        "qualification/manifest mode",
    )
    if any(
        manifest.get("backend", {}).get("mode")
        != qualification.get("execution_mode")
        for manifest in candidate_manifests
    ):
        raise _error(
            "execution_mode",
            "all candidate manifests must use the qualified execution mode",
        )
    if any(
        manifest.get("backend", {}).get("mode") != "cpu_reference"
        for manifest in cpu_manifests
    ):
        raise _error(
            "execution_mode",
            "all CPU reference manifests must use cpu_reference mode",
        )
    candidate_run_ids = {
        str(manifest.get("run_id")) for manifest in candidate_manifests
    }
    cpu_run_ids = {str(manifest.get("run_id")) for manifest in cpu_manifests}
    if len(candidate_run_ids) != len(candidate_manifests):
        raise _error(
            "candidate_membership",
            "candidate manifests must have unique run IDs",
        )
    if len(cpu_run_ids) != len(cpu_manifests):
        raise _error(
            "candidate_membership",
            "CPU manifests must have unique run IDs",
        )
    _validate_level_membership(
        qualification,
        str(candidate.get("run_id")),
        candidate_run_ids,
        cpu_run_ids,
    )
    cpu_manifests_by_id = {
        str(manifest.get("run_id")): manifest for manifest in cpu_manifests
    }
    current_cpu_run_id = str(
        validation.get("qualification_cpu_reference_run_id")
    )
    if current_cpu_run_id not in cpu_manifests_by_id:
        raise _error(
            "candidate_membership",
            "validation CPU reference is not a supplied qualification run",
        )
    current_candidate_run_id = str(candidate.get("run_id"))
    paired_together = any(
        current_candidate_run_id in pair.get("candidate_run_ids", [])
        and current_cpu_run_id in pair.get("cpu_run_ids", [])
        for pair in qualification.get("level_run_pairs", {}).values()
        if isinstance(pair, Mapping)
    )
    if not paired_together:
        raise _error(
            "candidate_membership",
            "validation CPU and candidate runs do not share a level pair",
        )
    cpu_manifest = cpu_manifests_by_id[current_cpu_run_id]

    all_manifests = (*candidate_manifests, *cpu_manifests)
    required_runtime_input_roles = {
        "mesh",
        "material_config",
        "scan_path",
        "solver_command",
        "used_config",
        "xrd_protocol",
    }
    for manifest in all_manifests:
        if manifest.get("status") not in {"completed", "accepted"}:
            raise _error(
                "run_status",
                f"qualification run {manifest.get('run_id')!r} is not complete",
            )
        inputs = _input_by_role(manifest)
        missing_runtime_roles = sorted(
            required_runtime_input_roles - set(inputs)
        )
        if missing_runtime_roles:
            raise _error(
                "runtime_input_identity",
                f"run {manifest.get('run_id')!r} is missing runtime inputs: "
                f"{', '.join(missing_runtime_roles)}",
            )
        for role, path in (
            ("paper_parity_config", parity_config_path),
            ("threshold_set", threshold_set_path),
            ("g0_approval", approval_record_path),
        ):
            _validate_run_input_binding(
                inputs,
                role,
                path,
                artifact_root,
                "acceptance_model_identity",
            )
        used_config_path = _validate_file_identity(
            inputs["used_config"],
            artifact_root,
            code="runtime_input_identity",
        )
        used_config = _require_mapping(
            load_json_strict(used_config_path),
            code="runtime_input_identity",
            description=f"run {manifest.get('run_id')!r} used config",
        )
        scan_path = _validate_file_identity(
            inputs["scan_path"],
            artifact_root,
            code="runtime_input_identity",
        )
        mesh_path = _validate_file_identity(
            inputs["mesh"],
            artifact_root,
            code="runtime_input_identity",
        )
        material_path = _validate_file_identity(
            inputs["material_config"],
            artifact_root,
            code="runtime_input_identity",
        )
        configured_material_path = used_config.get("config")
        if (
            not isinstance(configured_material_path, str)
            or not Path(configured_material_path).is_absolute()
        ):
            raise _error(
                "runtime_input_identity",
                f"run {manifest.get('run_id')!r} used config must record an "
                "absolute config path",
            )
        _require_equal(
            Path(configured_material_path).resolve(),
            material_path,
            "runtime_input_identity",
            f"run {manifest.get('run_id')!r} material config path",
        )
        configured_mesh_path = used_config.get("inp")
        if (
            not isinstance(configured_mesh_path, str)
            or not Path(configured_mesh_path).is_absolute()
        ):
            raise _error(
                "runtime_input_identity",
                f"run {manifest.get('run_id')!r} used config must record an "
                "absolute inp path",
            )
        _require_equal(
            Path(configured_mesh_path).resolve(),
            mesh_path,
            "runtime_input_identity",
            f"run {manifest.get('run_id')!r} mesh path",
        )
        configured_scan_path = used_config.get("path_file")
        if (
            not isinstance(configured_scan_path, str)
            or not Path(configured_scan_path).is_absolute()
        ):
            raise _error(
                "runtime_input_identity",
                f"run {manifest.get('run_id')!r} used config must record an "
                "absolute path_file",
            )
        _require_equal(
            Path(configured_scan_path).resolve(),
            scan_path,
            "runtime_input_identity",
            f"run {manifest.get('run_id')!r} scan path",
        )
        command_path = _validate_file_identity(
            inputs["solver_command"],
            artifact_root,
            code="runtime_input_identity",
        )
        try:
            recorded_command = command_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise _error(
                "runtime_input_identity",
                f"{command_path}: cannot read solver command: {exc}",
            ) from exc
        _require_equal(
            recorded_command,
            manifest.get("command"),
            "runtime_input_identity",
            f"run {manifest.get('run_id')!r} solver command",
        )
    manifests_by_id = {
        str(manifest.get("run_id")): manifest for manifest in all_manifests
    }
    level_threshold_truth: dict[str, bool] = {}
    level_parity_truth: dict[str, bool] = {}
    level_convergence_truth: dict[str, bool] = {}
    for level_id, pair in qualification.get("level_run_pairs", {}).items():
        level_case_ids: set[str] = set()
        for run_id in (
            *pair.get("cpu_run_ids", []),
            *pair.get("candidate_run_ids", []),
        ):
            manifest = manifests_by_id[str(run_id)]
            inputs = _input_by_role(manifest)
            level_record = inputs.get("qualification_level")
            if level_record is None:
                raise _error(
                    "level_coverage",
                    f"{run_id}: missing 'qualification_level' input",
                )
            level_path = _validate_file_identity(
                level_record,
                artifact_root,
                code="level_coverage",
            )
            level_payload = load_json_strict(level_path)
            expected_fields = {"schema_version", "level", "case_id"}
            if (
                not isinstance(level_payload, Mapping)
                or set(level_payload) != expected_fields
            ):
                raise _error(
                    "level_coverage",
                    f"{run_id}: qualification-level artifact fields do not "
                    "match the contract",
                )
            _require_equal(
                level_payload.get("schema_version"),
                "kaess.qualification-level/1",
                "level_coverage",
                f"{run_id} qualification-level schema",
            )
            _require_equal(
                level_payload.get("level"),
                level_id,
                "level_coverage",
                f"{run_id} qualification level",
            )
            _require_equal(
                level_payload.get("case_id"),
                manifest.get("case_id"),
                "level_coverage",
                f"{run_id} qualification case",
            )
            level_case_ids.add(str(level_payload.get("case_id")))
        if len(level_case_ids) != 1:
            raise _error(
                "level_coverage",
                f"{level_id}: CPU/candidate case IDs differ",
            )
    commits = {manifest.get("code", {}).get("commit") for manifest in all_manifests}
    checkout_identities = {
        (
            manifest.get("code", {}).get("repository"),
            manifest.get("code", {}).get("checkout_path"),
            manifest.get("code", {}).get("branch"),
        )
        for manifest in all_manifests
    }
    dirty_states = {
        (
            manifest.get("code", {}).get("dirty"),
            manifest.get("code", {}).get("dirty_diff_sha256"),
        )
        for manifest in all_manifests
    }
    for level_id, pair in qualification.get("level_run_pairs", {}).items():
        level_input_hashes = {
            physics_input_bundle_sha256(
                manifests_by_id[str(run_id)],
                artifact_root,
            )
            for run_id in (
                *pair.get("cpu_run_ids", []),
                *pair.get("candidate_run_ids", []),
            )
        }
        if len(level_input_hashes) != 1:
            raise _error(
                "source_identity",
                f"{level_id}: CPU/candidate input bundles differ",
            )
    acceptance_hashes = {
        manifest_acceptance_model_sha256(manifest)
        for manifest in all_manifests
    }
    if (
        len(checkout_identities) != 1
        or len(commits) != 1
        or len(dirty_states) != 1
    ):
        raise _error(
            "source_identity",
            "CPU and candidate repository, checkout path, branch, commit or "
            "dirty diff are not identical",
        )
    if len(acceptance_hashes) != 1:
        raise _error(
            "source_identity",
            "CPU and candidate acceptance models differ",
        )
    source = qualification.get("source_identity", {})
    identity = validation.get("identity", {})
    _require_equal(
        source.get("same_identity"),
        True,
        "source_identity",
        "source identity flag",
    )
    commit = next(iter(commits))
    dirty_diff = next(iter(dirty_states))[1]
    current_input_hashes = {
        physics_input_bundle_sha256(candidate, artifact_root),
        physics_input_bundle_sha256(cpu_manifest, artifact_root),
    }
    if len(current_input_hashes) != 1:
        raise _error(
            "source_identity",
            "current CPU/candidate input bundles differ",
        )
    input_hash = next(iter(current_input_hashes))
    acceptance_hash = next(iter(acceptance_hashes))
    for observed, expected, code, description in (
        (source.get("commit"), commit, "source_identity", "source commit"),
        (
            source.get("dirty_diff_sha256"),
            dirty_diff,
            "source_identity",
            "source dirty diff",
        ),
        (
            source.get("input_bundle_sha256"),
            input_hash,
            "source_identity",
            "source input bundle",
        ),
        (
            source.get("acceptance_model_sha256"),
            acceptance_hash,
            "acceptance_model_identity",
            "source acceptance model",
        ),
        (identity.get("commit"), commit, "source_identity", "validation commit"),
        (
            identity.get("dirty_diff_sha256"),
            dirty_diff,
            "source_identity",
            "validation dirty diff",
        ),
        (
            identity.get("input_bundle_sha256"),
            input_hash,
            "source_identity",
            "validation input bundle",
        ),
        (
            identity.get("acceptance_model_sha256"),
            acceptance_hash,
            "acceptance_model_identity",
            "validation acceptance model",
        ),
    ):
        _require_equal(observed, expected, code, description)

    if not cpu_manifests:
        raise _error(
            "environment_identity",
            "at least one CPU reference manifest is required",
        )
    candidate_environment_hashes = {
        manifest_environment_sha256(manifest)
        for manifest in candidate_manifests
    }
    cpu_environment_hashes = {
        manifest_environment_sha256(manifest) for manifest in cpu_manifests
    }
    if len(candidate_environment_hashes) != 1 or len(cpu_environment_hashes) != 1:
        raise _error(
            "environment_identity",
            "repeat manifests do not use stable CPU/candidate environments",
        )
    _require_equal(
        identity.get("candidate_environment_sha256"),
        next(iter(candidate_environment_hashes)),
        "environment_identity",
        "candidate environment",
    )
    _require_equal(
        identity.get("cpu_environment_sha256"),
        next(iter(cpu_environment_hashes)),
        "environment_identity",
        "CPU environment",
    )
    cpu_hardware_hashes = {
        manifest_cpu_hardware_sha256(manifest) for manifest in cpu_manifests
    }
    candidate_hardware_hashes = {
        manifest_cpu_hardware_sha256(manifest)
        for manifest in candidate_manifests
    }
    if len(cpu_hardware_hashes) != 1 or len(candidate_hardware_hashes) != 1:
        raise _error(
            "hardware_identity",
            "repeat manifests do not use stable CPU hardware",
        )
    cpu_hardware = next(iter(cpu_hardware_hashes))
    candidate_hardware = next(iter(candidate_hardware_hashes))
    _require_equal(
        identity.get("cpu_hardware_sha256"),
        cpu_hardware,
        "hardware_identity",
        "CPU hardware",
    )
    _require_equal(
        identity.get("candidate_hardware_sha256"),
        candidate_hardware,
        "hardware_identity",
        "candidate CPU hardware",
    )
    _require_equal(
        cpu_hardware,
        candidate_hardware,
        "hardware_identity",
        "CPU/candidate hardware",
    )
    _require_equal(
        identity.get("same_cpu_hardware"),
        True,
        "hardware_identity",
        "same CPU hardware flag",
    )

    def thread_tuple(manifest: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        backend = manifest.get("backend", {})
        return (
            backend.get("cpu_threads"),
            backend.get("mkl_threads"),
            backend.get("omp_threads"),
        )

    thread_budgets = {thread_tuple(manifest) for manifest in all_manifests}
    if len(thread_budgets) != 1:
        raise _error(
            "thread_budget",
            "CPU, MKL and OMP thread budgets must all match",
        )
    cpu_threads, _, _ = thread_tuple(cpu_manifest)
    candidate_threads, _, _ = thread_tuple(candidate)
    budget = identity.get("cpu_thread_budget", {})
    _require_equal(
        budget.get("cpu_control_threads"),
        cpu_threads,
        "thread_budget",
        "CPU control threads",
    )
    _require_equal(
        budget.get("candidate_threads"),
        candidate_threads,
        "thread_budget",
        "candidate threads",
    )
    _require_equal(
        budget.get("same_budget"),
        cpu_threads == candidate_threads,
        "thread_budget",
        "thread budget equality",
    )
    if qualification.get("performance", {}).get("measured") is True:
        _require_equal(
            qualification.get("performance", {}).get("cpu_threads"),
            cpu_threads,
            "thread_budget",
            "performance CPU thread count",
        )
    performance_protocol = _load_g0_performance_protocol(
        parity_config_path,
        threshold_set_path,
        approval_record_path,
    )
    frozen_thresholds = _metric_index(
        load_json_strict(threshold_set_path).get("metrics")
    )
    _require_equal(
        identity.get("performance_protocol_sha256"),
        g0_performance_protocol_sha256(
            parity_config_path,
            threshold_set_path,
            approval_record_path,
        ),
        "performance_protocol_identity",
        "performance protocol",
    )
    _require_equal(
        identity.get("sequential_execution"),
        True,
        "performance_protocol_identity",
        "sequential execution",
    )
    order_path = _validate_file_identity(
        identity.get("execution_order_artifact", {}),
        artifact_root,
        code="performance_protocol_identity",
    )
    order = load_json_strict(order_path)
    _require_equal(
        order.get("sequential"),
        True,
        "performance_protocol_identity",
        "execution-order artifact",
    )
    order_runs = order.get("runs")
    expected_order_run_ids = candidate_run_ids | cpu_run_ids
    observed_order_run_ids = {
        item.get("run_id")
        for item in order_runs
        if isinstance(item, Mapping)
    } if isinstance(order_runs, list) else set()
    if (
        not isinstance(order_runs, list)
        or observed_order_run_ids != expected_order_run_ids
        or len(order_runs) != len(expected_order_run_ids)
    ):
        raise _error(
            "performance_protocol_identity",
            "execution-order artifact must cover exactly the supplied runs",
        )
    intervals: list[tuple[datetime, datetime, str]] = []
    for item in order_runs:
        if not isinstance(item, Mapping):
            raise _error(
                "performance_protocol_identity",
                "execution-order run record must be an object",
            )
        try:
            started = datetime.fromisoformat(
                str(item.get("started_utc")).replace("Z", "+00:00")
            )
            completed = datetime.fromisoformat(
                str(item.get("completed_utc")).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise _error(
                "performance_protocol_identity",
                f"invalid execution-order timestamp: {exc}",
            ) from exc
        if started.tzinfo is None or completed.tzinfo is None or completed <= started:
            raise _error(
                "performance_protocol_identity",
                f"{item.get('run_id')}: execution interval is invalid",
            )
        intervals.append((started, completed, str(item.get("run_id"))))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise _error(
                "performance_protocol_identity",
                f"runs {previous[2]!r} and {current[2]!r} overlap",
            )
    execution_intervals = {
        run_id: (started, completed)
        for started, completed, run_id in intervals
    }

    evidence_by_run: dict[str, dict[str, Any]] = {}
    for manifest in all_manifests:
        run_id = str(manifest.get("run_id"))
        checkpoint_record = _artifact_by_role(
            manifest, "native_float64_checkpoint"
        )
        checkpoint_path = _validate_file_identity(
            checkpoint_record, artifact_root
        )
        checkpoint = inspect_native_checkpoint(checkpoint_path)
        mask_record = _artifact_by_role(manifest, "comparison_mask")
        mask_path = _validate_file_identity(mask_record, artifact_root)
        try:
            mask = np.load(mask_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise _error(
                "comparison_mask_identity",
                f"{mask_path}: cannot read comparison mask: {exc}",
            ) from exc
        if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(np.bool_):
            raise _error(
                "comparison_mask_identity",
                f"{mask_path}: comparison mask must be a boolean ndarray",
            )
        embedded_mask = checkpoint.get("active_mask")
        if embedded_mask is None or not np.array_equal(embedded_mask, mask):
            raise _error(
                "comparison_mask_identity",
                f"{run_id}: checkpoint active_mask differs from external mask",
            )
        evidence_by_run[run_id] = {
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
            "mask": mask,
            "mask_path": mask_path,
            "mask_sha256": sha256_file(mask_path),
        }
    checkpoint_paths = {
        evidence["checkpoint_path"] for evidence in evidence_by_run.values()
    }
    if len(checkpoint_paths) != len(evidence_by_run):
        raise _error(
            "checkpoint_identity",
            "each qualification run must have a distinct native checkpoint "
            "artifact path",
        )

    for level_id, pair in qualification.get("level_run_pairs", {}).items():
        cpu_shapes = {
            canonical_json_sha256(
                evidence_by_run[str(run_id)]["checkpoint"]["array_shapes"]
            )
            for run_id in pair.get("cpu_run_ids", [])
        }
        candidate_shapes = {
            canonical_json_sha256(
                evidence_by_run[str(run_id)]["checkpoint"]["array_shapes"]
            )
            for run_id in pair.get("candidate_run_ids", [])
        }
        if len(cpu_shapes) != 1 or candidate_shapes != cpu_shapes:
            raise _error(
                "checkpoint_shape",
                f"{level_id}: CPU/candidate checkpoint array shapes differ",
            )
        level_mask_hashes = {
            evidence_by_run[str(run_id)]["mask_sha256"]
            for run_id in (
                *pair.get("cpu_run_ids", []),
                *pair.get("candidate_run_ids", []),
            )
        }
        if len(level_mask_hashes) != 1:
            raise _error(
                "comparison_mask_identity",
                f"{level_id}: CPU/candidate comparison masks differ",
            )
        parity_passed = True
        convergence_passed = True
        for cpu_run_id in pair.get("cpu_run_ids", []):
            for candidate_run_id in pair.get("candidate_run_ids", []):
                pair_truth = _checkpoint_metric_truth(
                    evidence_by_run[str(cpu_run_id)]["checkpoint"],
                    evidence_by_run[str(candidate_run_id)]["checkpoint"],
                )
                parity_passed = (
                    _checkpoint_pair_parity_passes(
                        pair_truth,
                        frozen_thresholds,
                    )
                    and parity_passed
                )
                convergence_passed = (
                    _checkpoint_pair_convergence_passes(
                        pair_truth,
                        frozen_thresholds,
                    )
                    and convergence_passed
                )
        level_passed = parity_passed and convergence_passed
        level_threshold_truth[str(level_id)] = level_passed
        level_parity_truth[str(level_id)] = parity_passed
        level_convergence_truth[str(level_id)] = convergence_passed
        _require_equal(
            pair.get("status"),
            "pass" if level_passed else "fail",
            "metric_recalculation",
            f"{level_id} level status",
        )

    candidate_evidence = evidence_by_run[str(candidate.get("run_id"))]
    cpu_evidence = evidence_by_run[current_cpu_run_id]
    candidate_checkpoint = candidate_evidence["checkpoint"]
    checkpoint_metric_truth = _checkpoint_metric_truth(
        cpu_evidence["checkpoint"],
        candidate_checkpoint,
    )
    _require_equal(
        identity.get("checkpoint_sha256"),
        candidate_checkpoint["sha256"],
        "checkpoint_hash",
        "candidate checkpoint",
    )
    _require_equal(
        identity.get("checkpoint_dtype"),
        candidate_checkpoint["dtype"],
        "checkpoint_dtype",
        "candidate checkpoint dtype",
    )
    _require_equal(
        identity.get("checkpoint_shape"),
        candidate_checkpoint["shape"],
        "checkpoint_shape",
        "candidate checkpoint shape",
    )

    candidate_mask_path = candidate_evidence["mask_path"]
    candidate_mask_hash = candidate_evidence["mask_sha256"]
    _require_equal(
        identity.get("mask_sha256"),
        candidate_mask_hash,
        "comparison_mask_identity",
        "validation mask",
    )
    _validate_file_identity_at(
        qualification.get("comparison_scope", {}).get("mask_artifact", {}),
        candidate_mask_path,
        artifact_root,
        code="comparison_mask_identity",
    )
    _require_equal(
        qualification.get("comparison_scope", {})
        .get("mask_artifact", {})
        .get("sha256"),
        candidate_mask_hash,
        "comparison_mask_identity",
        "qualification mask",
    )

    _require_equal(
        identity.get("precision"),
        candidate.get("backend", {}).get("precision"),
        "checkpoint_dtype",
        "candidate precision",
    )
    _require_equal(
        identity.get("candidate_in_qualification"),
        True,
        "candidate_membership",
        "candidate membership flag",
    )
    _require_equal(
        identity.get("mode_matches_manifest"),
        True,
        "execution_mode",
        "mode reconciliation flag",
    )
    _validate_stage_placement(
        str(qualification.get("execution_mode")),
        candidate.get("backend", {}),
        qualification.get("placement_evidence", {}),
    )
    _validate_profiler_placement(
        qualification,
        candidate,
        candidate_manifests=candidate_manifests,
        candidate_manifest_paths=candidate_manifest_paths,
        cpu_manifest_paths=cpu_manifest_paths,
        artifact_root=artifact_root,
    )
    raw_performance = _load_performance_evidence(
        qualification,
        cpu_threads=int(cpu_threads),
        artifact_root=artifact_root,
        execution_intervals=execution_intervals,
        linear_solve_counts={
            run_id: int(
                np.asarray(
                    evidence["checkpoint"]["state"]["linear_solve_count"]
                ).item()
            )
            for run_id, evidence in evidence_by_run.items()
        },
        manifests_by_id=manifests_by_id,
    )
    _validate_checkpoint_gate_evidence(
        qualification,
        evidence_by_run=evidence_by_run,
        artifact_root=artifact_root,
    )
    energy_passed = _validate_energy_audit_truth(
        qualification,
        manifests_by_id=manifests_by_id,
        frozen_thresholds=frozen_thresholds,
        threshold_set_path=threshold_set_path,
        artifact_root=artifact_root,
    )
    levels_passed = all(level_threshold_truth.values())
    parity_passed = all(level_parity_truth.values())
    stage_gates = qualification.get("stage_gates", {})
    _require_equal(
        stage_gates.get("backend_parity", {}).get("status"),
        "pass" if parity_passed else "fail",
        "gate_consistency",
        "backend parity gate status",
    )
    convergence_passed = all(level_convergence_truth.values())
    _require_equal(
        stage_gates.get("convergence_audit", {}).get("status"),
        "pass" if convergence_passed else "fail",
        "gate_consistency",
        "convergence audit gate status",
    )
    _require_equal(
        qualification.get("numerically_qualified"),
        bool(levels_passed and energy_passed and convergence_passed),
        "metric_recalculation",
        "qualification numerical status",
    )
    _validate_gate_truth(qualification, validation)
    for key in (
        "field_metrics",
        "event_metrics",
        "qoi_metrics",
        "convergence_metrics",
    ):
        for metric in qualification.get(key, []):
            metric_id = str(metric.get("metric_id"))
            _validate_metric_truth(
                metric,
                frozen_thresholds,
                checkpoint_metric_truth.get(metric_id, {}),
            )
    _validate_performance_truth(
        qualification,
        validation,
        performance_protocol,
        raw_performance,
    )

    qualification_promotes = qualification.get("promotion_eligible") is True
    validation_promotes = formal_promotion_allowed(validation)
    _reject_untyped_stage_promotion(qualification, validation)
    if qualification_promotes != validation_promotes:
        raise _error(
            "promotion_two_condition",
            "qualification and semantic validation disagree on promotion",
        )
    if validation_promotes and qualification.get("verdict") != "pass":
        raise _error(
            "promotion_two_condition",
            "semantic promotion requires a passing qualification",
        )
