"""Evidence-gated comparison entry points for experimental cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .cases import CASE_DIR, load_case
from .metrics import evaluate_anchors


class EvidenceLevelError(RuntimeError):
    """Raised when the available data cannot support the requested claim."""


def _microstrain_payload(payload):
    if not isinstance(payload, dict) or payload.get("units") != "microstrain":
        raise ValueError(
            "anchor predictions must declare units='microstrain'; convert "
            "dimensionless elastic strain explicitly before comparison"
        )
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be an object keyed by anchor id")
    return predictions


def screen_anchor_predictions(case, payload):
    """Evaluate textual paper anchors without issuing a field-accuracy claim."""
    predictions = _microstrain_payload(payload)
    known = {anchor["id"] for anchor in case["reported_anchors"]}
    unknown = sorted(set(predictions).difference(known))
    if unknown:
        raise ValueError(f"unknown anchor prediction ids: {unknown}")
    anchors = evaluate_anchors(case["reported_anchors"], predictions)
    evaluated = sum(
        result["status"] == "evaluated" for result in anchors.values()
    )
    return {
        "schema_version": "v06.anchor-screening/1",
        "case_id": case["case_id"],
        "specimen_id": case.get("specimen_id"),
        "evidence_level": "manual_unverified_screening",
        "units": "microstrain",
        "comparison_quantity": case["measurement"]["comparison_quantity"],
        "data_availability": case["data_availability"]["status"],
        "evaluated_anchor_count": int(evaluated),
        "total_anchor_count": len(case["reported_anchors"]),
        "anchors": anchors,
        "claim_limit": (
            "Predictions supplied to this endpoint are not yet bound to a "
            "solver artifact. Textual extrema/ranges support only manual sign "
            "and magnitude screening, never pointwise accuracy metrics."
        ),
    }


def pointwise_field_comparison(
    case, *, observed, predicted, uncertainty, units, fitted_parameters=0
):
    """Refuse formal metrics until the raw-data contract is implemented."""
    raise EvidenceLevelError(
        "pointwise comparison is disabled until a versioned dataset contract "
        "binds source hashes, gauge ids, coordinates, units, state, and "
        "uncertainty to the solver prediction artifact"
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a screening-only experimental anchor report."
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    case = load_case(args.case)
    with args.predictions.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    report = screen_anchor_predictions(case, payload)
    case_path = CASE_DIR / f"{args.case}.json"
    report["inputs"] = {
        "case_metadata": {
            "path": str(case_path.resolve()),
            "sha256": _sha256(case_path),
        },
        "predictions": {
            "path": str(args.predictions.resolve()),
            "sha256": _sha256(args.predictions),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "evidence_level": report["evidence_level"],
        "evaluated_anchor_count": report["evaluated_anchor_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
