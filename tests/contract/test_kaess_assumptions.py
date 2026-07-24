import copy
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / "cases" / "kaess_2023" / "inputs"
ASSUMPTIONS_PATH = INPUT_ROOT / "assumptions.yaml"
DEVIATIONS_PATH = INPUT_ROOT / "deviations.yaml"
AUTHOR_REQUESTS_PATH = INPUT_ROOT / "author-input-requests.md"
QOI_IDS = {
    "peak_temperature",
    "melt_volume",
    "thermal_energy_closure",
    "sigma_x_depth_curve",
    "sigma_x_zero_crossing",
    "front_bending_curve",
    "max_front_bending",
    "release_direction",
}


def _load_json_yaml(path: Path) -> dict:
    # The project uses the JSON-compatible YAML 1.2 subset to avoid adding a
    # parser dependency solely for protocol metadata.
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_assumptions(document: dict, request_text: str) -> None:
    assert document["schema_version"] == "kaess.assumptions/1"
    assert document["protocol_id"] == "kaess-2023-public-v1"
    records = document["assumptions"]
    assert records

    assumption_ids = [record["assumption_id"] for record in records]
    assert len(assumption_ids) == len(set(assumption_ids))
    for record in records:
        assert record["source_class"] in {"assumption", "inferred"}
        assert record["impact"] in {"low", "medium", "high", "critical"}
        assert record["decision"] in {
            "open",
            "accepted",
            "rejected",
            "replaced_by_author_data",
        }
        assert record["rationale"].strip()
        assert isinstance(record["range"], dict)
        assert set(record["affected_qoi_ids"]) <= QOI_IDS

        if record["impact"] == "critical" and record["decision"] == "open":
            has_author_request = bool(record.get("author_request_id"))
            has_sensitivity = bool(record.get("sensitivity_required"))
            assert has_author_request or has_sensitivity
            if has_author_request:
                assert record["author_request_id"] in request_text


def _validate_deviations(document: dict) -> None:
    assert document["schema_version"] == "kaess.deviations/1"
    records = document["deviations"]
    assert records
    deviation_ids = [record["deviation_id"] for record in records]
    assert len(deviation_ids) == len(set(deviation_ids))
    assert {
        "P0-BC",
        "P0-HS",
        "P0-ACT",
        "P0-SURF",
        "P0-COOL",
        "P0-MAT",
        "P0-J2",
        "P0-HIST",
        "P0-REL",
    } <= set(deviation_ids)
    for record in records:
        assert record["severity"] in {"P0", "P1", "P2"}
        assert record["resolution"] in {
            "open",
            "fixed",
            "accepted_assumption",
            "cannot_resolve",
        }
        assert set(record["affected_qoi_ids"]) <= QOI_IDS
        assert record["evidence_paths"]


def test_assumptions_register_classifies_critical_unknowns():
    request_text = AUTHOR_REQUESTS_PATH.read_text(encoding="utf-8")
    _validate_assumptions(_load_json_yaml(ASSUMPTIONS_PATH), request_text)


def test_assumptions_register_rejects_unmapped_critical_unknown():
    document = copy.deepcopy(_load_json_yaml(ASSUMPTIONS_PATH))
    record = document["assumptions"][0]
    record["impact"] = "critical"
    record["decision"] = "open"
    record["sensitivity_required"] = False
    record.pop("author_request_id", None)

    with pytest.raises(AssertionError):
        _validate_assumptions(document, AUTHOR_REQUESTS_PATH.read_text("utf-8"))


def test_deviations_register_covers_every_p0_physics_gap():
    _validate_deviations(_load_json_yaml(DEVIATIONS_PATH))
