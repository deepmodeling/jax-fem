from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    REPO_ROOT
    / "specs"
    / "001-kaess-paper-reproduction"
    / "evidence"
    / "t021-p0-regression.json"
)


def test_t021_evidence_is_complete_and_does_not_promote_g1_g2():
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert evidence["schema_version"] == (
        "kaess.p0-regression-evidence/1"
    )
    assert evidence["task_id"] == "T021"
    assert evidence["status"] == "pass"
    assert evidence["code"]["commit"] == (
        "8f1603f4ee69a15b0049ad04724aba654db7e740"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", evidence["code"]["commit"])
    assert evidence["code"]["source_tree_clean_before_run"] is True

    result = evidence["result"]
    assert result["exit_code"] == 0
    assert result["passed"] == 588
    assert result["failed"] == 0
    assert result["errors"] == 0
    assert result["unexpected_failures"] == 0
    assert result["skipped"] == 2
    assert result["subtests_passed"] == 16
    assert len(evidence["expected_skips"]) == result["skipped"]
    assert all(
        item["classification"] == "conditional_non_failure"
        for item in evidence["expected_skips"]
    )
    quickstart = evidence["quickstart_physics_solver_result"]
    assert quickstart["exit_code"] == 0
    assert quickstart["passed"] == 99
    assert quickstart["skipped"] == 2
    assert quickstart["subtests_passed"] == 10

    closures = evidence["red_green_closures"]
    closure_by_red = {item["red_task"]: item for item in closures}
    assert {
        red_task: item["implementation_task"]
        for red_task, item in closure_by_red.items()
    } == {
        f"T{task_id:03d}": f"T{task_id + 7:03d}"
        for task_id in range(7, 14)
    }
    assert closure_by_red["T011"]["status"] == (
        "green_pending_material_approval"
    )
    assert all(
        item["status"] == "green"
        for red_task, item in closure_by_red.items()
        if red_task != "T011"
    )
    for item in closures:
        evidence_reference = item.get("evidence")
        if evidence_reference is not None:
            assert (REPO_ROOT / evidence_reference).is_file()

    boundary = evidence["claim_boundary"]
    assert boundary["p0_code_regression"] == "passed"
    assert boundary["g1_g2_checkpoint"] == "not_yet_approved"
    assert {
        "T018 material candidate review and G0 reapproval",
        "anchor sensitivity threshold",
        "material-source and Figure 4(b) reading-error approval",
        "remaining unchecked PAR items",
    }.issubset(boundary["pending"])
