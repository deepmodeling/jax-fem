"""Load versioned, source-backed experimental case metadata."""

from __future__ import annotations

import json
from pathlib import Path


CASE_DIR = (
    Path(__file__).resolve().parents[2] / "cases" / "kaess_2023" / "references" / "cases"
)


def load_case(case_id):
    path = CASE_DIR / f"{case_id}.json"
    if not path.is_file():
        raise KeyError(f"unknown validation case: {case_id}")
    case = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "case_id",
        "role",
        "material",
        "source",
        "measurement",
        "reported_anchors",
        "data_availability",
    }
    missing = sorted(required.difference(case))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    if case["case_id"] != case_id:
        raise ValueError(f"{path} declares case_id={case['case_id']!r}")
    return case

