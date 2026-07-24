import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "cases" / "kaess_2023" / "references"
DIGITIZED_ROOT = REFERENCE_ROOT / "digitized"
METADATA_PATH = REFERENCE_ROOT / "cases" / "kaess_2023.json"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (DIGITIZED_ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_figure_8_reference_freezes_the_published_150c_profile_anchors():
    rows = _read_csv("fig8_sigma_x.csv")

    assert {row["source_class"] for row in rows} == {"figure_digitized"}
    assert {row["source_figure"] for row in rows} == {"Figure 8b"}
    assert {row["series_id"] for row in rows} == {"plate150_layer30"}
    assert {row["point_role"] for row in rows} == {
        "surface",
        "peak",
        "zero_crossing",
        "trough",
    }

    by_role = {row["point_role"]: row for row in rows}
    assert float(by_role["surface"]["depth_mm"]) == 0.0
    assert float(by_role["surface"]["sigma_x_mpa"]) == 250.0
    assert float(by_role["peak"]["depth_mm"]) == -0.15
    assert float(by_role["peak"]["sigma_x_mpa"]) == 650.0
    assert float(by_role["zero_crossing"]["depth_mm"]) == -0.21
    assert float(by_role["zero_crossing"]["sigma_x_mpa"]) == 0.0
    assert float(by_role["trough"]["depth_mm"]) == -0.30
    assert float(by_role["trough"]["sigma_x_mpa"]) == -450.0
    assert all(float(row["reading_error_depth_mm"]) > 0.0 for row in rows)
    assert all(float(row["reading_error_sigma_mpa"]) > 0.0 for row in rows)


def test_figure_9_reference_freezes_temperature_and_power_series():
    rows = _read_csv("fig9_bending.csv")
    assert {row["source_class"] for row in rows} == {"figure_digitized"}
    assert {row["source_figure"] for row in rows} == {"Figure 9a", "Figure 9b"}

    temperature_rows = {
        float(row["x_value"]): float(row["max_front_bending_um"])
        for row in rows
        if row["series_id"] == "plate_temperature_fixed_p250_v850"
    }
    assert temperature_rows == {
        20.0: 14.6,
        50.0: 14.5,
        150.0: 14.0,
        300.0: 12.7,
        450.0: 11.9,
        600.0: 10.5,
        750.0: 8.7,
        900.0: 6.1,
    }

    fixed_speed = {
        float(row["x_value"]): float(row["max_front_bending_um"])
        for row in rows
        if row["series_id"] == "power_fixed_v850"
    }
    constant_led = {
        float(row["x_value"]): float(row["max_front_bending_um"])
        for row in rows
        if row["series_id"] == "power_constant_led_0p29_j_per_mm"
    }
    assert fixed_speed == {100.0: 15.5, 350.0: 13.0}
    assert constant_led == {100.0: 12.0, 350.0: 14.3}
    assert all(float(row["reading_error_um"]) == 0.3 for row in rows)


def test_benchmark_metadata_matches_the_frozen_digitized_files():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    verification_status = metadata["source"]["verification_status"].lower()
    assert "figure 8/9 digitized" in verification_status
    assert "not yet digitized" not in verification_status
    assert metadata["quantitative_anchors_status"].startswith("FROZEN")
