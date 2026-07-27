import csv
import json
import math
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "cases" / "kaess_2023" / "references"
DIGITIZED_ROOT = REFERENCE_ROOT / "digitized"
METADATA_PATH = REFERENCE_ROOT / "cases" / "kaess_2023.json"
PAPER_SHA256 = "8bf5e7c2f744a55b9c4e40bbe5f17a5cb6e1649d668132edc961ecb5c3d5e941"
FIG8_METHOD_ID = "pdf_vector_axis_calibrated_piecewise_linear_v1"
FIG9_METHOD_ID = "pdf_vector_axis_calibrated_v1"
UNCERTAINTY_KIND = "absolute_symmetric_reading_bound"

FIG8_COLUMNS = [
    "series_id",
    "point_index",
    "build_plate_temp_c",
    "layer_thickness_um",
    "laser_power_w",
    "scan_speed_mm_s",
    "cantilever_z_mm",
    "sigma_x_mpa",
    "point_role",
    "source_pdf_x_pt",
    "source_pdf_y_pt",
    "reading_bound_z_mm",
    "reading_bound_sigma_x_mpa",
    "uncertainty_kind",
    "digitization_method_id",
    "digitized_evidence_id",
    "source_pdf_evidence_id",
    "source_pdf_sha256",
    "source_page",
    "source_figure",
    "source_class",
]

FIG9_COLUMNS = [
    "series_id",
    "point_index",
    "build_plate_temp_c",
    "layer_thickness_um",
    "laser_power_w",
    "scan_speed_mm_s",
    "line_energy_density_j_per_mm",
    "max_front_bending_um",
    "bending_direction",
    "source_pdf_x_pt",
    "source_pdf_y_pt",
    "reading_bound_bending_um",
    "uncertainty_kind",
    "digitization_method_id",
    "digitized_evidence_id",
    "source_pdf_evidence_id",
    "source_pdf_sha256",
    "source_page",
    "source_figure",
    "source_class",
]


def _read_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (DIGITIZED_ROOT / name).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def _assert_finite(rows: list[dict[str, str]], columns: list[str]) -> None:
    for row in rows:
        for column in columns:
            assert math.isfinite(float(row[column])), (column, row)


def test_figure_8_reference_contains_the_complete_vector_derived_profile():
    columns, rows = _read_csv("fig8_sigma_x.csv")

    assert columns == FIG8_COLUMNS
    assert len(rows) == 41
    assert [int(row["point_index"]) for row in rows] == list(range(41))
    assert len({(row["series_id"], row["point_index"]) for row in rows}) == 41
    assert [float(row["cantilever_z_mm"]) for row in rows] == pytest.approx(
        [-index / 100.0 for index in range(41)],
        abs=1.0e-12,
    )
    assert all(
        float(rows[index]["source_pdf_x_pt"])
        < float(rows[index + 1]["source_pdf_x_pt"])
        for index in range(40)
    )
    _assert_finite(
        rows,
        [
            "cantilever_z_mm",
            "sigma_x_mpa",
            "source_pdf_x_pt",
            "source_pdf_y_pt",
            "reading_bound_z_mm",
            "reading_bound_sigma_x_mpa",
        ],
    )

    assert {row["series_id"] for row in rows} == {"plate150_layer30_p250_v850"}
    assert {float(row["build_plate_temp_c"]) for row in rows} == {150.0}
    assert {float(row["layer_thickness_um"]) for row in rows} == {30.0}
    assert {float(row["laser_power_w"]) for row in rows} == {250.0}
    assert {float(row["scan_speed_mm_s"]) for row in rows} == {850.0}
    assert {row["source_class"] for row in rows} == {"figure_digitized"}
    assert {row["source_figure"] for row in rows} == {"Figure 8b"}
    assert {int(row["source_page"]) for row in rows} == {10}
    assert {row["digitized_evidence_id"] for row in rows} == {
        "figure-8-digitized-curve"
    }
    assert {row["source_pdf_evidence_id"] for row in rows} == {"paper-pdf"}
    assert {row["source_pdf_sha256"] for row in rows} == {PAPER_SHA256}
    assert {row["digitization_method_id"] for row in rows} == {FIG8_METHOD_ID}
    assert {row["uncertainty_kind"] for row in rows} == {UNCERTAINTY_KIND}
    assert {float(row["reading_bound_z_mm"]) for row in rows} == {0.02}
    assert {float(row["reading_bound_sigma_x_mpa"]) for row in rows} == {50.0}
    for row in rows:
        pdf_x = float(row["source_pdf_x_pt"])
        pdf_y = float(row["source_pdf_y_pt"])
        calibrated_z = (
            (pdf_x - 388.855377)
            / (484.732971 - 388.855377)
            * -0.4
        )
        calibrated_sigma = 1500.0 + (
            (pdf_y - 114.762787)
            / (211.186386 - 114.762787)
            * -2000.0
        )
        assert float(row["cantilever_z_mm"]) == pytest.approx(
            calibrated_z,
            abs=1.0e-8,
        )
        assert float(row["sigma_x_mpa"]) == pytest.approx(
            calibrated_sigma,
            abs=2.0e-5,
        )

    by_role = {row["point_role"]: row for row in rows if row["point_role"] != "trace"}
    assert set(by_role) == {
        "surface",
        "peak",
        "first_zero_positive_bracket",
        "first_zero_negative_bracket",
        "trough",
        "second_zero_negative_bracket",
        "second_zero_positive_bracket",
    }
    assert float(by_role["surface"]["cantilever_z_mm"]) == 0.0
    assert float(by_role["peak"]["cantilever_z_mm"]) == -0.15
    assert float(by_role["first_zero_positive_bracket"]["cantilever_z_mm"]) == -0.22
    assert float(by_role["first_zero_positive_bracket"]["sigma_x_mpa"]) > 0.0
    assert float(by_role["first_zero_negative_bracket"]["cantilever_z_mm"]) == -0.23
    assert float(by_role["first_zero_negative_bracket"]["sigma_x_mpa"]) < 0.0
    assert float(by_role["trough"]["cantilever_z_mm"]) == -0.30
    assert float(by_role["second_zero_negative_bracket"]["cantilever_z_mm"]) == -0.39
    assert float(by_role["second_zero_negative_bracket"]["sigma_x_mpa"]) < 0.0
    assert float(by_role["second_zero_positive_bracket"]["cantilever_z_mm"]) == -0.40
    assert float(by_role["second_zero_positive_bracket"]["sigma_x_mpa"]) > 0.0

    sigma = [float(row["sigma_x_mpa"]) for row in rows]
    assert rows[sigma.index(max(sigma))]["point_role"] == "peak"
    assert rows[sigma.index(min(sigma))]["point_role"] == "trough"

    positive = by_role["first_zero_positive_bracket"]
    negative = by_role["first_zero_negative_bracket"]
    z_positive = float(positive["cantilever_z_mm"])
    z_negative = float(negative["cantilever_z_mm"])
    sigma_positive = float(positive["sigma_x_mpa"])
    sigma_negative = float(negative["sigma_x_mpa"])
    zero_crossing = z_positive + (
        -sigma_positive
        * (z_negative - z_positive)
        / (sigma_negative - sigma_positive)
    )

    # Independent visual anchors from the original manual read remain inside
    # the declared symmetric reading bounds of the vector-derived curve.
    assert abs(float(by_role["surface"]["sigma_x_mpa"]) - 250.0) <= 50.0
    assert abs(float(by_role["peak"]["sigma_x_mpa"]) - 650.0) <= 50.0
    assert abs(float(by_role["trough"]["sigma_x_mpa"]) + 450.0) <= 50.0
    assert abs(zero_crossing + 0.21) <= 0.02

    second_negative = by_role["second_zero_negative_bracket"]
    second_positive = by_role["second_zero_positive_bracket"]
    second_zero_crossing = float(second_negative["cantilever_z_mm"]) + (
        -float(second_negative["sigma_x_mpa"])
        * (
            float(second_positive["cantilever_z_mm"])
            - float(second_negative["cantilever_z_mm"])
        )
        / (
            float(second_positive["sigma_x_mpa"])
            - float(second_negative["sigma_x_mpa"])
        )
    )
    assert second_zero_crossing == pytest.approx(-0.3964694106, abs=1.0e-6)


def test_figure_9_reference_contains_all_published_parameter_points():
    columns, rows = _read_csv("fig9_bending.csv")

    assert columns == FIG9_COLUMNS
    assert len(rows) == 20
    assert len({(row["series_id"], row["point_index"]) for row in rows}) == 20
    _assert_finite(
        rows,
        [
            "build_plate_temp_c",
            "layer_thickness_um",
            "laser_power_w",
            "scan_speed_mm_s",
            "line_energy_density_j_per_mm",
            "max_front_bending_um",
            "source_pdf_x_pt",
            "source_pdf_y_pt",
            "reading_bound_bending_um",
        ],
    )

    grouped = {
        series_id: [row for row in rows if row["series_id"] == series_id]
        for series_id in {
            "plate_temperature_fixed_p250_v850",
            "power_fixed_v850",
            "power_constant_led_0p29_j_per_mm",
        }
    }
    assert {series_id: len(series) for series_id, series in grouped.items()} == {
        "plate_temperature_fixed_p250_v850": 8,
        "power_fixed_v850": 6,
        "power_constant_led_0p29_j_per_mm": 6,
    }
    for series in grouped.values():
        assert [int(row["point_index"]) for row in series] == list(range(len(series)))

    temperature_rows = {
        float(row["build_plate_temp_c"]): float(row["max_front_bending_um"])
        for row in grouped["plate_temperature_fixed_p250_v850"]
    }
    assert temperature_rows == pytest.approx(
        {
            20.0: 14.571875,
            50.0: 14.463935,
            150.0: 13.982358,
            300.0: 13.127142,
            450.0: 12.047743,
            600.0: 10.528284,
            750.0: 8.668398,
            900.0: 6.127661,
        },
        abs=1.0e-6,
    )

    fixed_speed = {
        float(row["laser_power_w"]): float(row["max_front_bending_um"])
        for row in grouped["power_fixed_v850"]
    }
    constant_led = {
        float(row["laser_power_w"]): float(row["max_front_bending_um"])
        for row in grouped["power_constant_led_0p29_j_per_mm"]
    }
    assert fixed_speed == pytest.approx(
        {
            100.0: 15.491937,
            150.0: 15.017185,
            200.0: 14.534103,
            250.0: 13.976060,
            300.0: 13.526294,
            350.0: 12.984910,
        },
        abs=1.0e-6,
    )
    assert constant_led == pytest.approx(
        {
            100.0: 12.027072,
            150.0: 12.751690,
            200.0: 13.576261,
            250.0: 13.976054,
            300.0: 14.209265,
            350.0: 14.359189,
        },
        abs=1.0e-6,
    )

    constant_led_rows = grouped["power_constant_led_0p29_j_per_mm"]
    temperature_series = grouped["plate_temperature_fixed_p250_v850"]
    fixed_speed_rows = grouped["power_fixed_v850"]
    assert {float(row["laser_power_w"]) for row in temperature_series} == {250.0}
    assert {float(row["scan_speed_mm_s"]) for row in temperature_series} == {
        850.0
    }
    assert {float(row["build_plate_temp_c"]) for row in fixed_speed_rows} == {
        150.0
    }
    assert {float(row["scan_speed_mm_s"]) for row in fixed_speed_rows} == {850.0}
    assert {float(row["build_plate_temp_c"]) for row in constant_led_rows} == {
        150.0
    }
    assert [float(row["scan_speed_mm_s"]) for row in constant_led_rows] == [
        340.0,
        510.0,
        680.0,
        850.0,
        1020.0,
        1190.0,
    ]
    for row in rows:
        assert float(row["line_energy_density_j_per_mm"]) == pytest.approx(
            float(row["laser_power_w"]) / float(row["scan_speed_mm_s"]),
            abs=1.0e-9,
        )
    assert {
        round(float(row["line_energy_density_j_per_mm"]), 9)
        for row in constant_led_rows
    } == {0.294117647}

    assert {float(row["layer_thickness_um"]) for row in rows} == {30.0}
    assert {row["bending_direction"] for row in rows} == {"upward"}
    assert {float(row["reading_bound_bending_um"]) for row in rows} == {0.3}
    assert {row["uncertainty_kind"] for row in rows} == {UNCERTAINTY_KIND}
    assert {row["digitization_method_id"] for row in rows} == {FIG9_METHOD_ID}
    assert {row["digitized_evidence_id"] for row in rows} == {
        "figure-9-digitized-bending"
    }
    assert {row["source_pdf_evidence_id"] for row in rows} == {"paper-pdf"}
    assert {row["source_pdf_sha256"] for row in rows} == {PAPER_SHA256}
    assert {int(row["source_page"]) for row in rows} == {11}
    assert {row["source_figure"] for row in rows} == {"Figure 9a", "Figure 9b"}
    assert {row["source_class"] for row in rows} == {"figure_digitized"}
    for row in rows:
        if row["source_figure"] == "Figure 9a":
            x_left, x_right, x_max = 217.847992, 332.687988, 1000.0
            y_top, y_bottom = 118.213028, 233.833038
            physical_x = float(row["build_plate_temp_c"])
        else:
            x_left, x_right, x_max = 404.388, 518.807983, 400.0
            y_top, y_bottom = 117.67305, 232.933044
            physical_x = float(row["laser_power_w"])

        calibrated_x = (
            (float(row["source_pdf_x_pt"]) - x_left)
            / (x_right - x_left)
            * x_max
        )
        calibrated_bending_um = 16.0 + (
            (float(row["source_pdf_y_pt"]) - y_top)
            / (y_bottom - y_top)
            * -16.0
        )
        assert physical_x == pytest.approx(calibrated_x, abs=1.0)
        assert float(row["max_front_bending_um"]) == pytest.approx(
            calibrated_bending_um,
            abs=2.0e-5,
        )


def test_benchmark_metadata_declares_complete_digitization_without_claim_promotion():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    digitization = metadata["digitization"]
    assert digitization["status"] == "complete"
    assert digitization["figure8"]["file"] == "digitized/fig8_sigma_x.csv"
    assert digitization["figure8"]["sample_count"] == 41
    assert digitization["figure8"]["method_id"] == FIG8_METHOD_ID
    assert digitization["figure9"]["file"] == "digitized/fig9_bending.csv"
    assert digitization["figure9"]["sample_count"] == 20
    assert digitization["figure9"]["method_id"] == FIG9_METHOD_ID
    assert digitization["uncertainty_kind"] == UNCERTAINTY_KIND
    assert "not standard uncertainty" in digitization["uncertainty_claim_boundary"]
    assert digitization["source_discrepancies"] == [
        "Figure 9a contains a 50 C marker although Table 3 omits 50 C; "
        "the plotted marker is retained as figure evidence and is not "
        "attributed to Table 3."
    ]

    verification_status = metadata["source"]["verification_status"].lower()
    assert "complete" in verification_status
    assert "anchor" not in verification_status
    assert "not yet digitized" not in verification_status
    assert metadata["quantitative_anchors_status"].startswith("COMPLETE")
    assert "experimental validation" in metadata["claim_boundary"]


def test_benchmark_metadata_matches_the_complete_digitized_values_and_ladders():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    _, fig8_rows = _read_csv("fig8_sigma_x.csv")
    _, fig9_rows = _read_csv("fig9_bending.csv")
    frozen = metadata["digitized_reference_results"]

    fig9_series = {
        series_id: [row for row in fig9_rows if row["series_id"] == series_id]
        for series_id in {
            "plate_temperature_fixed_p250_v850",
            "power_fixed_v850",
            "power_constant_led_0p29_j_per_mm",
        }
    }
    assert {
        str(int(float(row["build_plate_temp_c"]))): float(
            row["max_front_bending_um"]
        )
        for row in fig9_series["plate_temperature_fixed_p250_v850"]
    } == {
        key: value
        for key, value in frozen[
            "fig9a_max_front_bending_um_vs_plate_c"
        ].items()
        if key != "reading_bound_bending_um"
    }
    assert {
        str(int(float(row["laser_power_w"]))): float(
            row["max_front_bending_um"]
        )
        for row in fig9_series["power_fixed_v850"]
    } == {
        key: value
        for key, value in frozen[
            "fig9b_max_front_bending_um_vs_power_w_fixed_speed"
        ].items()
        if key != "reading_bound_bending_um"
    }
    assert {
        str(int(float(row["laser_power_w"]))): float(
            row["max_front_bending_um"]
        )
        for row in fig9_series["power_constant_led_0p29_j_per_mm"]
    } == {
        key: value
        for key, value in frozen[
            "fig9b_max_front_bending_um_vs_power_w_constant_led"
        ].items()
        if key != "reading_bound_bending_um"
    }

    process_ladders = metadata["process"]["parameter_ladders"]
    figure_ladders = metadata["process"]["figure_marker_ladders"]
    assert process_ladders["build_plate_temp_c"] == [
        20.0,
        150.0,
        300.0,
        450.0,
        600.0,
        750.0,
        900.0,
    ]
    assert set(figure_ladders["figure9a_build_plate_temp_c"]) == {
        float(row["build_plate_temp_c"]) for row in fig9_rows
    }
    assert set(process_ladders["laser_power_w"]) >= {
        float(row["laser_power_w"]) for row in fig9_rows
    }
    assert set(process_ladders["laser_speed_mm_s"]) >= {
        float(row["scan_speed_mm_s"])
        for row in fig9_series["power_constant_led_0p29_j_per_mm"]
    }

    sigma = [float(row["sigma_x_mpa"]) for row in fig8_rows]
    fig8_summary = frozen["fig8b_sigma_x_profile_150c"]
    assert fig8_summary["surface_mpa"] == sigma[0]
    assert fig8_summary["peak_mpa"] == max(sigma)
    assert fig8_summary["min_mpa"] == min(sigma)
    assert fig8_summary["reading_bound_z_mm"] == 0.02
    assert fig8_summary["reading_bound_sigma_x_mpa"] == 50.0
