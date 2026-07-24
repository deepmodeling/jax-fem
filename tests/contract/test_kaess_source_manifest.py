import copy
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
        "reference-hex-mesh",
        "formal-material-config",
        "scan-path-reconstruction",
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
