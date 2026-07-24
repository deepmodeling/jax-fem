import copy
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "inputs" / "paper-parity-config.yaml"
)
REQUIRED_QOI_IDS = {
    "sigma_x_depth_curve",
    "sigma_x_zero_crossing",
    "front_bending_curve",
    "max_front_bending",
    "release_direction",
}
REQUIRED_STOP_RULES = {
    "require_g0_before_solver_changes",
    "require_g2_before_cpu_reference",
    "require_g3_before_accelerated_qualification",
    "require_g4_before_scale_bridge",
    "require_build_and_cooling_before_release",
}


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _validate_config(config: dict) -> None:
    assert config["schema_version"] == "kaess.paper-parity-config/1"
    assert config["protocol_id"] == "kaess-2023-public-v1"
    assert config["claim_level"] == "public_code_to_code"
    assert config["status"] in {"review_required", "approved"}

    material_freeze = config["material_freeze"]
    if material_freeze["mode"] == "external_sha256":
        assert material_freeze["path"] is None
        assert material_freeze["environment_variable"] == "KAESS_MATERIAL_CONFIG"
        assert (
            material_freeze["source_manifest_evidence_id"]
            == "formal-material-config"
        )

    approval = config["approval"]
    if config["status"] == "approved":
        assert approval["approved_by"]
        assert approval["approved_utc"]
        assert approval["approval_record_sha256"]

    qoi = config["quantities_of_interest"]
    assert REQUIRED_QOI_IDS <= set(qoi)
    for qoi_id, definition in qoi.items():
        assert definition["stage"] in {
            "build_complete",
            "cooling_complete",
            "release_complete",
        }
        assert definition["unit"]
        assert definition["path_id"]
        assert definition["interpolation"]
        assert definition["checkpoint_dtype"] == "float64"

    thresholds = config["threshold_set"]
    assert thresholds["threshold_set_id"]
    assert thresholds["version"]
    metric_ids = [metric["metric_id"] for metric in thresholds["metrics"]]
    assert len(metric_ids) == len(set(metric_ids))
    assert metric_ids

    calibration = set(config["calibration_case_ids"])
    held_out = set(config["held_out_case_ids"])
    assert held_out
    assert calibration.isdisjoint(held_out)
    assert REQUIRED_STOP_RULES <= set(config["stop_rules"])


def test_parity_config_freezes_qoi_thresholds_and_stop_rules():
    _validate_config(_load_config())


def test_parity_config_rejects_calibration_held_out_overlap():
    config = copy.deepcopy(_load_config())
    overlap = config["held_out_case_ids"][0]
    config["calibration_case_ids"].append(overlap)

    with pytest.raises(AssertionError):
        _validate_config(config)


def test_approved_parity_config_requires_auditable_approval():
    config = copy.deepcopy(_load_config())
    config["status"] = "approved"
    config["approval"] = {
        "approved_by": None,
        "approved_utc": None,
        "approval_record_sha256": None,
    }

    with pytest.raises(AssertionError):
        _validate_config(config)
