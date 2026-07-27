import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "inputs" / "source-manifest.yaml"
)
ALLOWED_SOURCE_CLASSES = {
    "paper_text",
    "paper_table",
    "figure_digitized",
    "abaqus_semantics",
    "author_artifact",
    "inferred",
    "assumption",
    "project_decision",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_manifest() -> dict:
    # JSON is a valid YAML 1.2 subset. Keeping this file in that subset avoids
    # adding a YAML runtime dependency solely for the reproduction protocol.
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _validate_manifest(manifest: dict) -> None:
    assert manifest["schema_version"] == "kaess.source-manifest/1"
    assert manifest["claim_level"] == "public_code_to_code"
    records = manifest["evidence"]
    assert records

    evidence_ids = [record["evidence_id"] for record in records]
    assert len(evidence_ids) == len(set(evidence_ids))

    for record in records:
        assert record["source_class"] in ALLOWED_SOURCE_CLASSES
        assert record["impact"] in {"low", "medium", "high", "critical"}
        assert record["status"] in {"verified", "inferred", "assumed", "missing"}
        assert record["source_locator"].strip()
        assert SHA256_PATTERN.fullmatch(record["sha256"])

        repository_path = record.get("repository_path")
        if repository_path is not None:
            path = REPO_ROOT / repository_path
            assert path.is_file(), repository_path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        external_path = record.get("external_path")
        if external_path is not None and Path(external_path).is_file():
            assert (
                hashlib.sha256(Path(external_path).read_bytes()).hexdigest()
                == record["sha256"]
            )

    required_ids = {
        "paper-pdf",
        "paper-fulltext",
        "benchmark-metadata",
        "figure-8-digitized-curve",
        "figure-9-digitized-bending",
        "reference-hex-mesh",
        "formal-material-config",
        "scan-path-reconstruction",
        "paper-parity-config-approved",
        "g0-approval-record",
        "approved-threshold-set",
    }
    assert required_ids <= set(evidence_ids)


def test_source_manifest_is_complete_and_content_addressed():
    _validate_manifest(_load_manifest())


def test_source_manifest_rejects_missing_hash():
    manifest = copy.deepcopy(_load_manifest())
    manifest["evidence"][0].pop("sha256")

    with pytest.raises((AssertionError, KeyError)):
        _validate_manifest(manifest)


def test_source_manifest_rejects_unknown_source_class():
    manifest = copy.deepcopy(_load_manifest())
    manifest["evidence"][0]["source_class"] = "internet_search"

    with pytest.raises(AssertionError):
        _validate_manifest(manifest)


def test_digitized_figure_evidence_binds_pdf_paths_axes_and_reading_bounds():
    manifest = _load_manifest()
    evidence = {
        record["evidence_id"]: record
        for record in manifest["evidence"]
    }
    paper_sha256 = evidence["paper-pdf"]["sha256"]
    assert "figure-8-digitized-anchors" not in evidence

    figure8 = evidence["figure-8-digitized-curve"]
    assert "anchor" not in figure8["quantity"].lower()
    assert "anchor-only" not in figure8["source_locator"].lower()
    assert figure8["repository_path"].endswith("digitized/fig8_sigma_x.csv")
    fig8_digitization = figure8["digitization"]
    assert fig8_digitization == {
        "paper_pdf_evidence_id": "paper-pdf",
        "method_id": "pdf_vector_axis_calibrated_piecewise_linear_v1",
        "paper_page": 10,
        "paper_figure": "Figure 8b",
        "pdf_drawing_indices": [53],
        "output_row_count": 41,
        "axis_calibration": {
            "x": {
                "pdf_points": [388.855377, 484.732971],
                "values": [0.0, -0.4],
                "quantity": "cantilever_z",
                "unit": "mm",
            },
            "y": {
                "pdf_points": [114.762787, 211.186386],
                "values": [1500.0, -500.0],
                "quantity": "sigma_x",
                "unit": "MPa",
            },
        },
        "sampling": {
            "start_mm": 0.0,
            "stop_mm": -0.4,
            "step_mm": -0.01,
            "interpolation": "piecewise_linear",
            "endpoint_policy": "first_segment_linear_extension",
            "maximum_extension_mm": 0.000228,
        },
        "uncertainty_kind": "absolute_symmetric_reading_bound",
        "reading_bounds": {
            "cantilever_z_mm": 0.02,
            "sigma_x_mpa": 50.0,
        },
    }

    figure9 = evidence["figure-9-digitized-bending"]
    assert figure9["repository_path"].endswith("digitized/fig9_bending.csv")
    fig9_digitization = figure9["digitization"]
    assert fig9_digitization["paper_pdf_evidence_id"] == "paper-pdf"
    assert fig9_digitization["method_id"] == "pdf_vector_axis_calibrated_v1"
    assert fig9_digitization["paper_page"] == 11
    assert fig9_digitization["paper_figures"] == ["Figure 9a", "Figure 9b"]
    assert fig9_digitization["pdf_drawing_indices"] == [19, 44, 51]
    assert fig9_digitization["output_row_count"] == 20
    assert fig9_digitization["uncertainty_kind"] == (
        "absolute_symmetric_reading_bound"
    )
    assert fig9_digitization["reading_bounds"] == {
        "max_front_bending_um": 0.3
    }
    assert fig9_digitization["axis_calibrations"] == [
        {
            "panel": "Figure 9a",
            "x_pdf_points": [217.847992, 332.687988],
            "x_values": [0.0, 1000.0],
            "x_quantity": "build_plate_temperature",
            "x_unit": "degC",
            "y_pdf_points": [118.213028, 233.833038],
            "y_values_um": [16.0, 0.0],
        },
        {
            "panel": "Figure 9b",
            "x_pdf_points": [404.388, 518.807983],
            "x_values": [0.0, 400.0],
            "x_quantity": "laser_power",
            "x_unit": "W",
            "y_pdf_points": [117.67305, 232.933044],
            "y_values_um": [16.0, 0.0],
        },
    ]

    for csv_name, evidence_id in [
        ("fig8_sigma_x.csv", "figure-8-digitized-curve"),
        ("fig9_bending.csv", "figure-9-digitized-bending"),
    ]:
        csv_path = (
            REPO_ROOT
            / "cases"
            / "kaess_2023"
            / "references"
            / "digitized"
            / csv_name
        )
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == evidence[evidence_id]["digitization"]["output_row_count"]
        assert {row["digitized_evidence_id"] for row in rows} == {evidence_id}
        assert {row["source_pdf_evidence_id"] for row in rows} == {
            "paper-pdf"
        }
        assert {row["source_pdf_sha256"] for row in rows} == {
            paper_sha256
        }
