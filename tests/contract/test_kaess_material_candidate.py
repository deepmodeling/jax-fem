from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jax_fem_am.config.loaders import read_config
from jax_fem_am.config.schema import build_parser
from jax_fem_am.materials.material_validation import validate_material_inputs
from jax_fem_am.materials.tables import load_property_tables


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = (
    REPO_ROOT / "cases" / "kaess_2023" / "candidates" / "g0-v2-t018"
)
CONFIG_PATH = (
    CANDIDATE_ROOT
    / "ss316l_material_config_kaess.g0-v2-candidate.json"
)
MANIFEST_PATH = CANDIDATE_ROOT / "material-bundle-manifest.json"
REQUEST_PATH = CANDIDATE_ROOT / "g0-reapproval-request.json"
CALIBRATION_PATH = (
    CANDIDATE_ROOT / "flow_curve_table.pending.vector-calibration.json"
)
PROPERTY_CALIBRATION_PATH = (
    CANDIDATE_ROOT / "figure4_material_properties.vector-calibration.json"
)
CALIBRATION_SCRIPT_PATH = (
    REPO_ROOT / "cases" / "kaess_2023" / "calibrate_figure4.py"
)
CALIBRATION_SPEC_PATH = (
    REPO_ROOT
    / "cases"
    / "kaess_2023"
    / "references"
    / "figure4-vector-spec.json"
)


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_contained_file(base: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    assert not relative.is_absolute()
    assert ".." not in relative.parts
    root = base.resolve(strict=True)
    unresolved = root / relative
    resolved = unresolved.resolve(strict=True)
    assert resolved.is_relative_to(root)
    assert resolved.is_file()
    return resolved


def _load_calibration_module():
    spec = importlib.util.spec_from_file_location(
        "kaess_calibrate_figure4",
        CALIBRATION_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pending_material_candidate_has_a_complete_unapproved_hash_chain():
    config = _load_json(CONFIG_PATH)
    manifest = _load_json(MANIFEST_PATH)
    request = _load_json(REQUEST_PATH)

    assert config["_candidate_status"] == "pending_review"
    assert manifest["status"] == "pending_review"
    assert manifest["promotion_eligible"] is False
    assert request["status"] == "pending_review"
    assert request["decision"] == "pending_review"

    assert manifest["material_config"] == {
        "path": CONFIG_PATH.name,
        "sha256": _sha256(CONFIG_PATH),
    }
    assert _resolve_contained_file(
        CANDIDATE_ROOT,
        manifest["material_config"]["path"],
    ) == CONFIG_PATH.resolve()
    records = manifest["files"]
    assert manifest["path_bases"] == {
        "files": "manifest_directory",
        "material_config": "manifest_directory",
        "source_evidence": {
            "repository_paths": "repository_root",
            "superseded_external_config_path": "absolute_path",
        },
    }
    assert len({record["path"] for record in records}) == len(records)
    resolved_records = []
    for record in records:
        candidate_file = _resolve_contained_file(
            CANDIDATE_ROOT,
            record["path"],
        )
        resolved_records.append(candidate_file)
        assert record["sha256"] == _sha256(candidate_file)
    assert len(set(resolved_records)) == len(resolved_records)

    assert (
        request["candidate_bundle_reference"]["sha256"]
        == _sha256(MANIFEST_PATH)
    )
    canonical = request["canonical_approval_reference"]
    canonical_path = _resolve_contained_file(REPO_ROOT, canonical["path"])
    assert canonical["sha256"] == _sha256(canonical_path)

    source_evidence = manifest["source_evidence"]
    for path_field, hash_field in (
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
    ):
        source_path = _resolve_contained_file(
            REPO_ROOT,
            source_evidence[path_field],
        )
        assert source_evidence[hash_field] == _sha256(source_path)
    assert Path(
        source_evidence["superseded_external_config_path"]
    ).is_absolute()

    assert config["flow_curve_table"] == "flow_curve_table.pending.csv"
    assert "yield_table" not in config
    assert "hardening_table" not in config


def test_pending_candidate_loads_the_same_flow_curve_from_any_working_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config = read_config(str(CONFIG_PATH))
    args = build_parser(config).parse_args(
        ["--config", str(CONFIG_PATH)]
    )
    tables = load_property_tables(args)

    assert validate_material_inputs(args, tables) is True
    assert tables["yield"] is None
    assert tables["hardening"] is None
    curve = tables["flow_curve"]
    assert curve.path == CANDIDATE_ROOT / "flow_curve_table.pending.csv"
    np.testing.assert_allclose(
        curve.temperatures,
        [
            293.15,
            673.15,
            1073.15,
            1273.15,
            1643.15,
            1673.15,
            1873.15,
        ],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(curve.plastic_strains, [0.0, 0.4])
    np.testing.assert_allclose(
        curve.stresses[:5] / 1.0e6,
        [
            [500.0, 600.0],
            [400.0, 500.0],
            [300.0, 400.0],
            [150.0, 200.0],
            [20.0, 30.0],
        ],
    )
    np.testing.assert_allclose(
        curve.stresses[5:] / 1.0e6,
        [[1.0, 1.0], [1.0, 1.0]],
    )


def test_pending_flow_curve_metadata_preserves_the_claim_boundary():
    metadata_path = (
        CANDIDATE_ROOT / "flow_curve_table.pending.metadata.json"
    )
    metadata = _load_json(metadata_path)
    manifest = _load_json(MANIFEST_PATH)

    image = REPO_ROOT / metadata["source"]["image_path"]
    assert metadata["source"]["image_sha256"] == _sha256(image)
    assert metadata["status"] == "pending_review"
    assert metadata["promotion_eligible"] is False
    assert metadata["reading_error"]["formal_promotion_allowed"] is False
    assert "not the authors' original Abaqus plastic table" in (
        metadata["claim_boundary"]
    )
    assert len(metadata["digitized_curve_endpoints"]) == 5
    assert len(metadata["solver_realization_nodes"]) == 2

    metadata_record = next(
        record
        for record in manifest["files"]
        if record["path"] == metadata_path.name
    )
    assert metadata_record["sha256"] == _sha256(metadata_path)


def test_flow_curve_vector_calibration_quantifies_the_reading_error():
    calibration = _load_json(CALIBRATION_PATH)
    metadata_path = (
        CANDIDATE_ROOT / "flow_curve_table.pending.metadata.json"
    )
    metadata = _load_json(metadata_path)
    manifest = _load_json(MANIFEST_PATH)

    pdf_path = REPO_ROOT / calibration["source"]["pdf_path"]
    assert calibration["source"]["pdf_sha256"] == _sha256(pdf_path)
    assert calibration["status"] == "independently_calibrated"
    assert calibration["method"]["source_geometry"] == "PDF_vector_objects"
    assert calibration["axis_fit"]["stress_max_residual_MPa"] <= 0.6
    assert calibration["curve_fit"]["max_rounding_residual_MPa"] <= 0.6
    assert calibration["curve_fit"]["max_endpoint_strain_offset"] <= 0.0032

    frozen = calibration["frozen_reading_error"]
    assert frozen == {
        "equivalent_plastic_strain_absolute": 0.0032,
        "flow_stress_absolute_MPa": 1.0,
    }
    assert metadata["reading_error"] == {
        "status": "quantified_from_pdf_vector",
        "formal_promotion_allowed": False,
        **frozen,
        "calibration_path": CALIBRATION_PATH.name,
        "calibration_sha256": _sha256(CALIBRATION_PATH),
    }
    curve_1370 = next(
        curve
        for curve in metadata["digitized_curve_endpoints"]
        if curve["temperature_C"] == 1370.0
    )
    assert [
        point["flow_stress_MPa"] for point in curve_1370["points"]
    ] == [20.0, 30.0]
    assert curve_1370["source_origin"] == "paper_author_assumption"

    with (CANDIDATE_ROOT / "flow_curve_table.pending.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        flow_rows = list(csv.DictReader(stream))
    assert {
        row["source"]
        for row in flow_rows
        if float(row["temperature_K"]) == 1643.15
    } == {"paper_author_assumption"}

    calibration_record = next(
        record
        for record in manifest["files"]
        if record["path"] == CALIBRATION_PATH.name
    )
    assert calibration_record["sha256"] == _sha256(CALIBRATION_PATH)


def test_vector_nodes_are_the_single_source_for_property_table_values():
    module = _load_calibration_module()
    source_spec = _load_json(CALIBRATION_SPEC_PATH)
    assert all(
        "table_frozen" not in panel
        for panel in source_spec["panels"].values()
    )

    mutated_spec = copy.deepcopy(source_spec)
    mutated_spec["panels"]["figure_4a"]["vector_frozen"]["value"][1] = 184.25
    pdf_path = REPO_ROOT / mutated_spec["source"]["pdf_path"]
    calibration = module.calibrate_figure4(
        pdf_path,
        mutated_spec,
        spec_sha256="mutation-test",
    )

    assert calibration["frozen_tables"]["E_table.csv"]["value"][1] == (
        184.25e9
    )


@pytest.mark.parametrize(
    "field",
    ["temperature_C", "value"],
)
def test_property_calibration_rejects_mismatched_vector_lengths(field):
    module = _load_calibration_module()
    source_spec = _load_json(CALIBRATION_SPEC_PATH)
    mutated_spec = copy.deepcopy(source_spec)
    mutated_spec["panels"]["figure_4a"]["vector_frozen"][field].pop()

    with pytest.raises(ValueError, match="vector-node lengths"):
        module.calibrate_figure4(
            REPO_ROOT / mutated_spec["source"]["pdf_path"],
            mutated_spec,
            spec_sha256="mutation-test",
        )


def test_flow_calibration_rejects_unbound_extra_endpoint_values():
    module = _load_calibration_module()
    source_spec = _load_json(CALIBRATION_SPEC_PATH)
    mutated_spec = copy.deepcopy(source_spec)
    mutated_spec["flow_panel"]["curves"][-1][
        "frozen_flow_stress_MPa"
    ].append(999.0)

    with pytest.raises(ValueError, match="exactly two"):
        module.calibrate_figure4_flow_curve(
            REPO_ROOT / mutated_spec["source"]["pdf_path"],
            mutated_spec,
            spec_sha256="mutation-test",
        )


@pytest.mark.parametrize(
    ("panel_id", "mutation"),
    [
        (
            "figure_4a",
            {"rule": "arbitrary_value", "source_class": "solver_extension"},
        ),
        (
            "figure_4a",
            {"rule": "endpoint_extension", "source_class": ""},
        ),
        (
            "figure_4e",
            {"rule": "linear_extrapolation", "source_class": "unknown"},
        ),
    ],
)
def test_property_calibration_rejects_unverified_additional_nodes(
    panel_id,
    mutation,
):
    module = _load_calibration_module()
    source_spec = _load_json(CALIBRATION_SPEC_PATH)
    mutated_spec = copy.deepcopy(source_spec)
    node = mutated_spec["panels"][panel_id]["table_mapping"][
        "additional_nodes"
    ][0]
    node.update(mutation)

    with pytest.raises(ValueError, match="additional node|source class"):
        module.calibrate_figure4(
            REPO_ROOT / mutated_spec["source"]["pdf_path"],
            mutated_spec,
            spec_sha256="mutation-test",
        )


def test_property_additional_nodes_are_derived_not_arbitrary_values():
    spec = _load_json(CALIBRATION_SPEC_PATH)
    calibration = _load_json(PROPERTY_CALIBRATION_PATH)
    for panel in spec["panels"].values():
        for node in panel["table_mapping"]["additional_nodes"]:
            assert "value_SI" not in node

    youngs_modulus = calibration["frozen_tables"]["E_table.csv"]
    assert youngs_modulus["value"][-1] == youngs_modulus["value"][-2]
    alpha = calibration["frozen_tables"]["alpha_table.csv"]
    expected_room_temperature_alpha = (
        alpha["value"][1]
        + (20.0 - 100.0)
        * (alpha["value"][2] - alpha["value"][1])
        / (200.0 - 100.0)
    )
    assert alpha["value"][0] == pytest.approx(
        expected_room_temperature_alpha,
        rel=0.0,
        abs=1.0e-15,
    )


def test_manifest_path_resolver_rejects_traversal_and_symlink_escape(
    tmp_path,
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    inside = bundle / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    escape = bundle / "escape.txt"
    escape.symlink_to(outside)

    assert _resolve_contained_file(bundle, "inside.txt") == inside.resolve()
    with pytest.raises(AssertionError):
        _resolve_contained_file(bundle, "../outside.txt")
    with pytest.raises(AssertionError):
        _resolve_contained_file(bundle, str(outside))
    with pytest.raises(AssertionError):
        _resolve_contained_file(bundle, "escape.txt")


@pytest.mark.parametrize(
    ("relative_path", "old", "new"),
    [
        (
            "flow_curve_table.pending.csv",
            "1643.15,0.4,30000000,paper_author_assumption",
            "1643.15,0.4,20000000,paper_author_assumption",
        ),
        (
            "flow_curve_table.pending.metadata.json",
            '"temperature_K": 1673.15',
            '"temperature_K": 1663.15',
        ),
    ],
)
def test_calibration_command_rejects_runtime_flow_bundle_drift(
    tmp_path,
    relative_path,
    old,
    new,
):
    candidate_copy = tmp_path / "candidate"
    shutil.copytree(CANDIDATE_ROOT, candidate_copy)
    target = candidate_copy / relative_path
    original = target.read_text(encoding="utf-8")
    assert old in original
    target.write_text(original.replace(old, new, 1), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATION_SCRIPT_PATH),
            "--pdf",
            str(REPO_ROOT / _load_json(CALIBRATION_SPEC_PATH)["source"]["pdf_path"]),
            "--spec",
            str(CALIBRATION_SPEC_PATH),
            "--check-output",
            str(PROPERTY_CALIBRATION_PATH),
            "--check-flow-output",
            str(CALIBRATION_PATH),
            "--bundle-dir",
            str(candidate_copy),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "flow" in result.stderr.lower()


def test_figure4_property_tables_match_the_independent_vector_calibration():
    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATION_SCRIPT_PATH),
            "--pdf",
            str(
                REPO_ROOT
                / "cases"
                / "kaess_2023"
                / "references"
                / "cases"
                / "kaess_2023_paper.pdf"
            ),
            "--spec",
            str(CALIBRATION_SPEC_PATH),
            "--check-output",
            str(PROPERTY_CALIBRATION_PATH),
            "--check-flow-output",
            str(CALIBRATION_PATH),
            "--bundle-dir",
            str(CANDIDATE_ROOT),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    calibration = _load_json(PROPERTY_CALIBRATION_PATH)
    manifest = _load_json(MANIFEST_PATH)

    pdf_path = REPO_ROOT / calibration["source"]["pdf_path"]
    assert calibration["source"]["pdf_sha256"] == _sha256(pdf_path)
    assert calibration["status"] == "independently_calibrated"
    assert calibration["method"]["source_geometry"] == "PDF_vector_objects"

    expected_errors = {
        "E_table.csv": ("youngs_modulus_absolute_GPa", 0.5),
        "k_solid_table.csv": (
            "conductivity_absolute_W_per_mK",
            0.1,
        ),
        "cp_solid_table.csv": (
            "specific_heat_absolute_J_per_kgK",
            2.0,
        ),
        "alpha_table.csv": (
            "thermal_expansion_absolute_per_K",
            1.0e-7,
        ),
    }
    for table_name, (error_name, error_limit) in expected_errors.items():
        frozen = calibration["frozen_tables"][table_name]
        with (CANDIDATE_ROOT / table_name).open(
            newline="",
            encoding="utf-8",
        ) as stream:
            rows = list(csv.DictReader(stream))
        np.testing.assert_allclose(
            [float(row["T"]) for row in rows],
            frozen["temperature_K"],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            [float(row["value"]) for row in rows],
            frozen["value"],
            rtol=0.0,
            atol=1.0e-12,
        )
        assert [row["source"] for row in rows] == frozen["source_class"]
        assert (
            calibration["frozen_reading_error"][error_name]
            <= error_limit
        )

        table_record = next(
            record
            for record in manifest["files"]
            if record["path"] == table_name
        )
        assert table_record["sha256"] == _sha256(
            CANDIDATE_ROOT / table_name
        )

    cp = calibration["frozen_tables"]["cp_solid_table.csv"]
    assert cp["temperature_K"][-2:] == [1643.15, 1723.15]
    assert cp["value"][-2:] == [670.0, 750.0]

    calibration_record = next(
        record
        for record in manifest["files"]
        if record["path"] == PROPERTY_CALIBRATION_PATH.name
    )
    assert calibration_record["sha256"] == _sha256(
        PROPERTY_CALIBRATION_PATH
    )
