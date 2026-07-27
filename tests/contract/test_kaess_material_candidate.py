from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

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


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    records = manifest["files"]
    assert len({record["path"] for record in records}) == len(records)
    for record in records:
        candidate_file = CANDIDATE_ROOT / record["path"]
        assert candidate_file.is_file()
        assert record["sha256"] == _sha256(candidate_file)

    assert (
        request["candidate_bundle_reference"]["sha256"]
        == _sha256(MANIFEST_PATH)
    )
    canonical = request["canonical_approval_reference"]
    canonical_path = REPO_ROOT / canonical["path"]
    assert canonical["sha256"] == _sha256(canonical_path)

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
            [20.0, 20.0],
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
