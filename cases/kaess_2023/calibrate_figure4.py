#!/usr/bin/env python3
"""Recompute the Kaess Figure 4 material-property calibration from PDF vectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pymupdf


_PLOT_VALUE_TO_SI_SCALE = {
    "GPa": 1.0e9,
    "W_per_mK": 1.0,
    "J_per_kgK": 1.0,
    "microstrain_per_K": 1.0e-6,
}
_VECTOR_PROPERTY_SOURCE_CLASSES = {
    "figure_node",
    "figure_node_author_assumption",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def _fit_axis(coordinates: list[float], values: list[float]) -> dict[str, Any]:
    if len(coordinates) != len(values) or len(coordinates) < 2:
        raise ValueError("axis fit requires matching coordinate/value vectors")
    x_mean = sum(coordinates) / len(coordinates)
    y_mean = sum(values) / len(values)
    denominator = sum((x - x_mean) ** 2 for x in coordinates)
    if denominator == 0.0:
        raise ValueError("axis coordinates are degenerate")
    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(coordinates, values)
    ) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [
        slope * coordinate + intercept - value
        for coordinate, value in zip(coordinates, values)
    ]
    return {
        "slope": _round(slope, 12),
        "intercept": _round(intercept, 12),
        "max_residual": _round(max(abs(value) for value in residuals), 12),
    }


def _drawing_rect(drawing: dict[str, Any]) -> list[float]:
    return [_round(value, 6) for value in drawing["rect"]]


def _assert_close_vector(
    actual: list[float],
    expected: list[float],
    tolerance: float,
    label: str,
) -> None:
    if len(actual) != len(expected):
        raise ValueError(f"{label} length drifted: {actual} != {expected}")
    for index, (observed, frozen) in enumerate(zip(actual, expected)):
        if abs(observed - frozen) > tolerance:
            raise ValueError(
                f"{label}[{index}] drifted: {observed} != {frozen} "
                f"(tolerance {tolerance})"
            )


def _axis_coordinate(
    drawings: list[dict[str, Any]],
    drawing_index: int,
    orientation: str,
) -> float:
    drawing = drawings[drawing_index]
    lines = [item for item in drawing["items"] if item[0] == "l"]
    if not lines:
        raise ValueError(f"drawing {drawing_index} contains no line segment")
    first = lines[0]
    p0, p1 = first[1], first[2]
    if orientation == "vertical":
        if abs(float(p0.x) - float(p1.x)) > 1.0e-3:
            raise ValueError(f"drawing {drawing_index} is not vertical")
        return 0.5 * (float(p0.x) + float(p1.x))
    if orientation == "horizontal":
        if abs(float(p0.y) - float(p1.y)) > 1.0e-3:
            raise ValueError(f"drawing {drawing_index} is not horizontal")
        return 0.5 * (float(p0.y) + float(p1.y))
    raise ValueError(f"unsupported axis orientation: {orientation}")


def _curve_vertices(
    drawings: list[dict[str, Any]],
    curve: dict[str, Any],
) -> list[tuple[float, float]]:
    drawing_index = int(curve["drawing_index"])
    drawing = drawings[drawing_index]
    if len(drawing["items"]) != int(curve["expected_line_items"]):
        raise ValueError(
            f"curve drawing {drawing_index} item count drifted: "
            f"{len(drawing['items'])}"
        )
    _assert_close_vector(
        [float(value) for value in drawing["rect"]],
        [float(value) for value in curve["expected_rect"]],
        float(curve["signature_tolerance_points"]),
        f"curve drawing {drawing_index} rectangle",
    )
    lines = [item for item in drawing["items"] if item[0] == "l"]
    if len(lines) != len(drawing["items"]):
        raise ValueError(f"curve drawing {drawing_index} is not line-only")
    points = [(float(lines[0][1].x), float(lines[0][1].y))]
    previous = lines[0][1]
    for item in lines:
        start, end = item[1], item[2]
        if (
            abs(float(start.x) - float(previous.x)) > 1.0e-3
            or abs(float(start.y) - float(previous.y)) > 1.0e-3
        ):
            raise ValueError(f"curve drawing {drawing_index} is disconnected")
        points.append((float(end.x), float(end.y)))
        previous = end
    return points


def _read_frozen_table(
    path: Path,
) -> tuple[list[float], list[float], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"property table is empty: {path}")
    return (
        [float(row["T"]) for row in rows],
        [float(row["value"]) for row in rows],
        [row["source"] for row in rows],
    )


def _derive_frozen_table(
    panel_id: str,
    panel_spec: dict[str, Any],
    frozen_temperature_C: list[float],
    frozen_plot_value: list[float],
) -> dict[str, Any]:
    """Derive a runtime SI table from the one frozen PDF-vector node set."""

    if "table_frozen" in panel_spec:
        raise ValueError(
            f"{panel_id} contains deprecated independent table_frozen data"
        )
    mapping = panel_spec["table_mapping"]
    plot_unit = panel_spec["plot_unit"]
    try:
        unit_scale = _PLOT_VALUE_TO_SI_SCALE[plot_unit]
    except KeyError as exc:
        raise ValueError(f"unsupported plot unit for {panel_id}: {plot_unit}") from exc
    declared_scale = float(mapping["plot_value_to_SI_scale"])
    if declared_scale != unit_scale:
        raise ValueError(
            f"{panel_id} unit scale drifted: {declared_scale} != {unit_scale}"
        )

    vector_sources = list(mapping["vector_source_class"])
    if len(vector_sources) != len(frozen_temperature_C):
        raise ValueError(
            f"{panel_id} vector source-class count does not match vector nodes"
        )
    if any(
        source not in _VECTOR_PROPERTY_SOURCE_CLASSES
        for source in vector_sources
    ):
        raise ValueError(f"{panel_id} has an unknown vector source class")
    rows = [
        {
            "temperature_C": float(temperature),
            "value_SI": _round(float(value) * unit_scale, 15),
            "source_class": source,
        }
        for temperature, value, source in zip(
            frozen_temperature_C,
            frozen_plot_value,
            vector_sources,
        )
    ]
    derived_additional_nodes = []
    for node in mapping.get("additional_nodes", []):
        target_temperature = float(node["temperature_C"])
        if not math.isfinite(target_temperature):
            raise ValueError(
                f"{panel_id} additional node temperature is not finite"
            )
        rule = node.get("rule")
        source_class = node.get("source_class")
        if rule == "endpoint_extension":
            expected_keys = {
                "temperature_C",
                "rule",
                "from_vector_index",
                "source_class",
            }
            if set(node) != expected_keys or source_class != "solver_extension":
                raise ValueError(
                    f"{panel_id} endpoint-extension additional node is invalid"
                )
            source_index = node["from_vector_index"]
            if isinstance(source_index, bool) or not isinstance(
                source_index, int
            ):
                raise ValueError(
                    f"{panel_id} additional node vector index is invalid"
                )
            if not (
                -len(frozen_temperature_C)
                <= source_index
                < len(frozen_temperature_C)
            ):
                raise ValueError(
                    f"{panel_id} additional node vector index is out of range"
                )
            normalized_index = source_index % len(frozen_temperature_C)
            if normalized_index not in {0, len(frozen_temperature_C) - 1}:
                raise ValueError(
                    f"{panel_id} endpoint extension must inherit an endpoint"
                )
            source_temperature = frozen_temperature_C[normalized_index]
            extends_lower = (
                normalized_index == 0
                and target_temperature < source_temperature
            )
            extends_upper = (
                normalized_index == len(frozen_temperature_C) - 1
                and target_temperature > source_temperature
            )
            if not (extends_lower or extends_upper):
                raise ValueError(
                    f"{panel_id} endpoint extension is not outside the vector range"
                )
            plot_value = frozen_plot_value[normalized_index]
            derivation = {
                "rule": rule,
                "from_vector_index": source_index,
                "from_temperature_C": source_temperature,
            }
        elif rule == "linear_extrapolation":
            expected_keys = {
                "temperature_C",
                "rule",
                "from_vector_indices",
                "source_class",
            }
            if (
                set(node) != expected_keys
                or source_class != "linear_extrapolation"
            ):
                raise ValueError(
                    f"{panel_id} linear-extrapolation additional node is invalid"
                )
            indices = node["from_vector_indices"]
            if (
                not isinstance(indices, list)
                or len(indices) != 2
                or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in indices
                )
            ):
                raise ValueError(
                    f"{panel_id} extrapolation vector indices are invalid"
                )
            first_index, second_index = indices
            try:
                first_temperature = frozen_temperature_C[first_index]
                second_temperature = frozen_temperature_C[second_index]
                first_value = frozen_plot_value[first_index]
                second_value = frozen_plot_value[second_index]
            except IndexError as exc:
                raise ValueError(
                    f"{panel_id} extrapolation vector index is out of range"
                ) from exc
            if first_temperature == second_temperature:
                raise ValueError(
                    f"{panel_id} extrapolation temperatures are degenerate"
                )
            if min(first_temperature, second_temperature) <= target_temperature <= max(
                first_temperature, second_temperature
            ):
                raise ValueError(
                    f"{panel_id} additional node is interpolation, not extrapolation"
                )
            plot_value = first_value + (
                (target_temperature - first_temperature)
                * (second_value - first_value)
                / (second_temperature - first_temperature)
            )
            derivation = {
                "rule": rule,
                "from_vector_indices": indices,
                "from_temperature_C": [
                    first_temperature,
                    second_temperature,
                ],
            }
        else:
            raise ValueError(f"{panel_id} additional node rule is unknown")

        value_SI = _round(float(plot_value) * unit_scale, 15)
        if not math.isfinite(value_SI):
            raise ValueError(f"{panel_id} additional node value is not finite")
        rows.append(
            {
                "temperature_C": target_temperature,
                "value_SI": value_SI,
                "source_class": source_class,
            }
        )
        derived_additional_nodes.append(
            {
                "temperature_C": target_temperature,
                "value_SI": value_SI,
                "source_class": source_class,
                **derivation,
            }
        )
    rows.sort(key=lambda row: row["temperature_C"])
    temperatures = [row["temperature_C"] for row in rows]
    if len(set(temperatures)) != len(temperatures):
        raise ValueError(f"{panel_id} table contains duplicate temperature nodes")

    return {
        "temperature_K": [
            _round(temperature + 273.15, 12) for temperature in temperatures
        ],
        "value": [row["value_SI"] for row in rows],
        "source_class": [row["source_class"] for row in rows],
        "derivation": {
            "vector_node_count": len(frozen_temperature_C),
            "plot_unit": plot_unit,
            "plot_value_to_SI_scale": unit_scale,
            "additional_nodes": derived_additional_nodes,
        },
    }


def calibrate_figure4(
    pdf_path: Path,
    spec: dict[str, Any],
    *,
    spec_sha256: str,
) -> dict[str, Any]:
    """Return a deterministic calibration derived from the original PDF."""

    expected_pdf_sha = spec["source"]["pdf_sha256"]
    actual_pdf_sha = _sha256(pdf_path)
    if actual_pdf_sha != expected_pdf_sha:
        raise ValueError(
            f"PDF SHA-256 mismatch: {actual_pdf_sha} != {expected_pdf_sha}"
        )

    page_index = int(spec["source"]["page_index"])
    with pymupdf.open(pdf_path) as document:
        drawings = document[page_index].get_drawings()

    panels: dict[str, Any] = {}
    frozen_tables: dict[str, Any] = {}
    max_temperature_error = 0.0
    max_errors: dict[str, float] = {}

    for panel_id, panel_spec in spec["panels"].items():
        x_spec = panel_spec["x_axis"]
        y_spec = panel_spec["y_axis"]
        x_coordinates = [
            _axis_coordinate(drawings, index, "vertical")
            for index in x_spec["drawing_indices"]
        ]
        y_coordinates = [
            _axis_coordinate(drawings, index, "horizontal")
            for index in y_spec["drawing_indices"]
        ]
        _assert_close_vector(
            x_coordinates,
            x_spec["expected_coordinates"],
            float(x_spec["signature_tolerance_points"]),
            f"{panel_id} x-axis",
        )
        _assert_close_vector(
            y_coordinates,
            y_spec["expected_coordinates"],
            float(y_spec["signature_tolerance_points"]),
            f"{panel_id} y-axis",
        )
        x_fit = _fit_axis(x_coordinates, x_spec["values"])
        y_fit = _fit_axis(y_coordinates, y_spec["values"])

        vertices = _curve_vertices(drawings, panel_spec["curve"])
        mapped_temperature = [
            x_fit["slope"] * x + x_fit["intercept"] for x, _ in vertices
        ]
        mapped_value = [
            y_fit["slope"] * y + y_fit["intercept"] for _, y in vertices
        ]
        vector_frozen = panel_spec["vector_frozen"]
        frozen_temperature = [
            float(value) for value in vector_frozen["temperature_C"]
        ]
        frozen_value = [float(value) for value in vector_frozen["value"]]
        if not (
            len(vertices) == len(frozen_temperature) == len(frozen_value)
        ):
            raise ValueError(
                f"{panel_id} vector-node lengths drifted: "
                f"PDF={len(vertices)}, temperature={len(frozen_temperature)}, "
                f"value={len(frozen_value)}"
            )
        temperature_rounding = max(
            abs(actual - frozen)
            for actual, frozen in zip(mapped_temperature, frozen_temperature)
        )
        value_rounding = max(
            abs(actual - frozen)
            for actual, frozen in zip(mapped_value, frozen_value)
        )
        temperature_error = x_fit["max_residual"] + temperature_rounding
        value_error = y_fit["max_residual"] + value_rounding
        max_temperature_error = max(max_temperature_error, temperature_error)
        error_name = panel_spec["reading_error_name"]
        max_errors[error_name] = max(
            max_errors.get(error_name, 0.0),
            value_error * float(panel_spec["plot_to_si_error_scale"]),
        )

        panels[panel_id] = {
            "table": panel_spec["table"],
            "curve_drawing_index": panel_spec["curve"]["drawing_index"],
            "curve_drawing_rect": _drawing_rect(
                drawings[panel_spec["curve"]["drawing_index"]]
            ),
            "axis_fit": {
                "temperature_C": {
                    **x_fit,
                    "drawing_indices": x_spec["drawing_indices"],
                    "coordinates": [_round(value, 6) for value in x_coordinates],
                    "values": x_spec["values"],
                },
                panel_spec["plot_unit"]: {
                    **y_fit,
                    "drawing_indices": y_spec["drawing_indices"],
                    "coordinates": [_round(value, 6) for value in y_coordinates],
                    "values": y_spec["values"],
                },
            },
            "mapped_vector_nodes": [
                {
                    "pdf_x": _round(x, 6),
                    "pdf_y": _round(y, 6),
                    "temperature_C": _round(temperature, 6),
                    "value": _round(value, 9),
                }
                for (x, y), temperature, value in zip(
                    vertices,
                    mapped_temperature,
                    mapped_value,
                )
            ],
            "frozen_vector_nodes": {
                "temperature_C": frozen_temperature,
                "value": frozen_value,
            },
            "max_temperature_rounding_residual_C": _round(
                temperature_rounding, 9
            ),
            "max_value_rounding_residual": _round(value_rounding, 9),
            "conservative_temperature_reading_error_C": _round(
                temperature_error, 9
            ),
            "conservative_value_reading_error": _round(value_error, 9),
        }

        frozen_tables[panel_spec["table"]] = _derive_frozen_table(
            panel_id,
            panel_spec,
            frozen_temperature,
            frozen_value,
        )

    limits = {
        key: float(value)
        for key, value in spec["frozen_reading_error"].items()
    }
    if max_temperature_error > limits["temperature_absolute_K"]:
        raise ValueError(
            "temperature reading error exceeds frozen limit: "
            f"{max_temperature_error} > {limits['temperature_absolute_K']}"
        )
    for error_name, measured in max_errors.items():
        if measured > limits[error_name]:
            raise ValueError(
                f"{error_name} exceeds frozen limit: "
                f"{measured} > {limits[error_name]}"
            )

    script_path = Path(__file__).resolve()
    return {
        "schema_version": "kaess.figure4-material-vector-calibration/1",
        "status": "independently_calibrated",
        "promotion_eligible": False,
        "source": {
            "paper_doi": spec["source"]["paper_doi"],
            "pdf_path": spec["source"]["pdf_path"],
            "pdf_sha256": actual_pdf_sha,
            "page_index": page_index,
            "printed_page": spec["source"]["printed_page"],
        },
        "method": {
            "source_geometry": "PDF_vector_objects",
            "extractor": "PyMuPDF",
            "extractor_version": pymupdf.__version__,
            "script_path": spec["method"]["script_path"],
            "script_sha256": _sha256(script_path),
            "spec_path": spec["method"]["spec_path"],
            "spec_sha256": spec_sha256,
            "axis_fit": "ordinary_least_squares_on_frozen_grid_lines",
            "curve_mapping": "connected_line_vertices_through_fitted_axes",
            "error_combination": "axis_fit_max_plus_rounding_max",
        },
        "panels": panels,
        "frozen_tables": frozen_tables,
        "measured_conservative_error": {
            "temperature_absolute_K": _round(max_temperature_error, 9),
            **{
                key: _round(value, 12)
                for key, value in sorted(max_errors.items())
            },
        },
        "frozen_reading_error": limits,
        "claim_boundary": (
            "The calibration reproduces and rounds the plotted Kaess Figure 4 "
            "vector geometry. It does not recover the authors' unpublished "
            "Abaqus input deck, and explicitly labelled extrapolation or solver "
            "extension nodes remain model realizations."
        ),
    }


def calibrate_figure4_flow_curve(
    pdf_path: Path,
    spec: dict[str, Any],
    *,
    spec_sha256: str,
) -> dict[str, Any]:
    """Recompute the five Figure 4(b) vector segments and reading bounds."""

    if _sha256(pdf_path) != spec["source"]["pdf_sha256"]:
        raise ValueError("flow-curve calibration PDF SHA-256 mismatch")
    with pymupdf.open(pdf_path) as document:
        drawings = document[int(spec["source"]["page_index"])].get_drawings()

    panel = spec["flow_panel"]
    x_spec = panel["x_axis"]
    y_spec = panel["y_axis"]
    x_coordinates = [
        _axis_coordinate(drawings, index, "vertical")
        for index in x_spec["drawing_indices"]
    ]
    y_coordinates = [
        _axis_coordinate(drawings, index, "horizontal")
        for index in y_spec["drawing_indices"]
    ]
    _assert_close_vector(
        x_coordinates,
        x_spec["expected_coordinates"],
        float(x_spec["signature_tolerance_points"]),
        "Figure 4(b) strain axis",
    )
    _assert_close_vector(
        y_coordinates,
        y_spec["expected_coordinates"],
        float(y_spec["signature_tolerance_points"]),
        "Figure 4(b) stress axis",
    )
    strain_fit = _fit_axis(x_coordinates, x_spec["values"])
    stress_fit = _fit_axis(y_coordinates, y_spec["values_MPa"])

    endpoints = []
    maximum_stress_rounding = 0.0
    maximum_strain_offset = 0.0
    for curve in panel["curves"]:
        vertices = _curve_vertices(drawings, curve)
        if len(vertices) != 2:
            raise ValueError(
                f"flow curve {curve['temperature_C']} C is not one segment"
            )
        mapped_strain = [
            strain_fit["slope"] * x + strain_fit["intercept"]
            for x, _ in vertices
        ]
        mapped_stress = [
            stress_fit["slope"] * y + stress_fit["intercept"]
            for _, y in vertices
        ]
        frozen_strain = [
            float(value) for value in curve["frozen_equivalent_plastic_strain"]
        ]
        frozen_stress = [
            float(value) for value in curve["frozen_flow_stress_MPa"]
        ]
        if len(frozen_strain) != 2 or len(frozen_stress) != 2:
            raise ValueError(
                f"flow curve {curve['temperature_C']} C must have exactly two "
                "frozen strain and stress values"
            )
        maximum_strain_offset = max(
            maximum_strain_offset,
            max(
                abs(mapped - frozen)
                for mapped, frozen in zip(mapped_strain, frozen_strain)
            ),
        )
        maximum_stress_rounding = max(
            maximum_stress_rounding,
            max(
                abs(mapped - frozen)
                for mapped, frozen in zip(mapped_stress, frozen_stress)
            ),
        )
        endpoints.append(
            {
                "temperature_C": float(curve["temperature_C"]),
                "source_origin": curve["source_origin"],
                "drawing_index": int(curve["drawing_index"]),
                "pdf": {
                    "x0": _round(vertices[0][0], 6),
                    "x1": _round(vertices[1][0], 6),
                    "top": _round(min(vertices[0][1], vertices[1][1]), 6),
                    "bottom": _round(max(vertices[0][1], vertices[1][1]), 6),
                },
                "mapped": {
                    "equivalent_plastic_strain": [
                        _round(value, 12) for value in mapped_strain
                    ],
                    "flow_stress_MPa": [
                        _round(value, 12) for value in mapped_stress
                    ],
                },
                "frozen": {
                    "equivalent_plastic_strain": frozen_strain,
                    "flow_stress_MPa": frozen_stress,
                },
            }
        )

    frozen_error = {
        key: float(value)
        for key, value in panel["frozen_reading_error"].items()
    }
    if (
        stress_fit["max_residual"] + maximum_stress_rounding
        > frozen_error["flow_stress_absolute_MPa"]
    ):
        raise ValueError("flow-stress reading error exceeds frozen bound")
    if (
        strain_fit["max_residual"] + maximum_strain_offset
        > frozen_error["equivalent_plastic_strain_absolute"]
    ):
        raise ValueError("flow-strain reading error exceeds frozen bound")

    return {
        "_comment": (
            "Independent calibration of Figure 4(b) from the original PDF "
            "vector objects. This quantifies digitization error only; it does "
            "not recover the authors' Abaqus table or approve the candidate."
        ),
        "schema_version": "kaess.flow-curve-vector-calibration/1",
        "status": "independently_calibrated",
        "promotion_eligible": False,
        "source": {
            "paper_doi": spec["source"]["paper_doi"],
            "pdf_path": spec["source"]["pdf_path"],
            "pdf_sha256": spec["source"]["pdf_sha256"],
            "page_number_one_based": spec["source"]["printed_page"],
            "figure": "4(b)",
        },
        "method": {
            "source_geometry": "PDF_vector_objects",
            "extractor": "PyMuPDF",
            "extractor_version": pymupdf.__version__,
            "script_path": spec["method"]["script_path"],
            "script_sha256": _sha256(Path(__file__).resolve()),
            "spec_path": spec["method"]["spec_path"],
            "spec_sha256": spec_sha256,
            "axis_model": "least_squares_affine_fit_to_vector_grid_lines",
            "curve_model": "five_connected_vector_line_segments",
            "rounding_model": "frozen_integer_MPa_and_endpoint_strain",
        },
        "axis_fit": {
            "strain_grid": {
                "drawing_indices": x_spec["drawing_indices"],
                "pdf_x": [_round(value, 6) for value in x_coordinates],
                "values": x_spec["values"],
                "value_per_pdf_point": strain_fit["slope"],
                "intercept": strain_fit["intercept"],
                "max_residual": strain_fit["max_residual"],
            },
            "stress_grid": {
                "drawing_indices": y_spec["drawing_indices"],
                "pdf_y": [_round(value, 6) for value in y_coordinates],
                "values_MPa": y_spec["values_MPa"],
                "MPa_per_pdf_point": stress_fit["slope"],
                "intercept_MPa": stress_fit["intercept"],
            },
            "stress_max_residual_MPa": stress_fit["max_residual"],
        },
        "curve_fit": {
            "vector_endpoints": endpoints,
            "max_rounding_residual_MPa": _round(
                maximum_stress_rounding, 12
            ),
            "max_endpoint_strain_offset": _round(
                maximum_strain_offset, 12
            ),
        },
        "solver_realization_nodes": panel["solver_realization_nodes"],
        "frozen_reading_error": frozen_error,
        "claim_boundary": (
            "The frozen error covers extraction and rounding of the plotted "
            "vector geometry only. It excludes model-form uncertainty, the "
            "paper authors' assumed high-temperature curves, and the "
            "unpublished 1673.15 K and 1873.15 K solver-realization nodes."
        ),
    }


def verify_frozen_tables(
    calibration: dict[str, Any],
    bundle_dir: Path,
) -> None:
    for table_name, frozen in calibration["frozen_tables"].items():
        actual_temperature, actual_value, actual_source = _read_frozen_table(
            bundle_dir / table_name
        )
        if actual_temperature != frozen["temperature_K"]:
            raise ValueError(
                f"{table_name} temperature nodes do not match calibration"
            )
        if actual_value != frozen["value"]:
            raise ValueError(f"{table_name} values do not match calibration")
        if actual_source != frozen["source_class"]:
            raise ValueError(
                f"{table_name} source classes do not match calibration"
            )


def _read_flow_table(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"flow table is empty: {path}")
    return [
        {
            "temperature_K": float(row["temperature_K"]),
            "equivalent_plastic_strain": float(
                row["equivalent_plastic_strain"]
            ),
            "flow_stress_Pa": float(row["flow_stress_Pa"]),
            "source": row["source"],
        }
        for row in rows
    ]


def verify_flow_bundle(
    calibration: dict[str, Any],
    bundle_dir: Path,
) -> None:
    """Bind PDF-vector endpoints and explicit solver nodes to runtime files."""

    metadata = _load_json(
        bundle_dir / "flow_curve_table.pending.metadata.json"
    )
    metadata_endpoints = metadata["digitized_curve_endpoints"]
    vector_endpoints = calibration["curve_fit"]["vector_endpoints"]
    if len(metadata_endpoints) != len(vector_endpoints):
        raise ValueError("flow metadata endpoint count does not match calibration")

    expected_rows: list[dict[str, Any]] = []
    for vector, recorded in zip(vector_endpoints, metadata_endpoints):
        temperature_C = float(vector["temperature_C"])
        if float(recorded["temperature_C"]) != temperature_C:
            raise ValueError("flow metadata temperature does not match calibration")
        if float(recorded["temperature_K"]) != _round(
            temperature_C + 273.15, 12
        ):
            raise ValueError("flow metadata Kelvin conversion drifted")
        if recorded["source_origin"] != vector["source_origin"]:
            raise ValueError("flow metadata source origin does not match calibration")
        points = recorded["points"]
        frozen = vector["frozen"]
        if len(points) != len(frozen["equivalent_plastic_strain"]):
            raise ValueError("flow metadata point count does not match calibration")
        for index, point in enumerate(points):
            strain = float(frozen["equivalent_plastic_strain"][index])
            stress_MPa = float(frozen["flow_stress_MPa"][index])
            if float(point["equivalent_plastic_strain"]) != strain:
                raise ValueError("flow metadata plastic strain drifted")
            if float(point["flow_stress_MPa"]) != stress_MPa:
                raise ValueError("flow metadata stress drifted")
            expected_rows.append(
                {
                    "temperature_K": _round(temperature_C + 273.15, 12),
                    "equivalent_plastic_strain": strain,
                    "flow_stress_Pa": stress_MPa * 1.0e6,
                    "source": vector["source_origin"],
                }
            )

    expected_solver_nodes = calibration["solver_realization_nodes"]
    recorded_solver_nodes = metadata["solver_realization_nodes"]
    if recorded_solver_nodes != expected_solver_nodes:
        raise ValueError(
            "flow metadata solver-realization nodes do not match calibration"
        )
    for node in expected_solver_nodes:
        strains = node["equivalent_plastic_strain"]
        stresses = node["flow_stress_MPa"]
        if len(strains) != len(stresses):
            raise ValueError("flow solver-realization node shape is invalid")
        for strain, stress_MPa in zip(strains, stresses):
            expected_rows.append(
                {
                    "temperature_K": float(node["temperature_K"]),
                    "equivalent_plastic_strain": float(strain),
                    "flow_stress_Pa": float(stress_MPa) * 1.0e6,
                    "source": node["source_origin"],
                }
            )

    actual_rows = _read_flow_table(
        bundle_dir / "flow_curve_table.pending.csv"
    )
    if actual_rows != expected_rows:
        raise ValueError(
            "flow runtime CSV does not match vector calibration and "
            "solver-realization metadata"
        )


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key is not allowed: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--check-output", type=Path)
    flow_output = parser.add_mutually_exclusive_group(required=True)
    flow_output.add_argument("--flow-output", type=Path)
    flow_output.add_argument("--check-flow-output", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    args = parser.parse_args(argv)

    spec = _load_json(args.spec)
    calibration = calibrate_figure4(
        args.pdf,
        spec,
        spec_sha256=_sha256(args.spec),
    )
    flow_calibration = calibrate_figure4_flow_curve(
        args.pdf,
        spec,
        spec_sha256=_sha256(args.spec),
    )
    if args.bundle_dir is not None:
        verify_frozen_tables(calibration, args.bundle_dir)
        verify_flow_bundle(flow_calibration, args.bundle_dir)

    if args.output is not None:
        _write_json(args.output, calibration)
    else:
        frozen = _load_json(args.check_output)
        if calibration != frozen:
            raise ValueError(
                f"recomputed calibration differs from {args.check_output}"
            )
    if args.flow_output is not None:
        _write_json(args.flow_output, flow_calibration)
    else:
        frozen_flow = _load_json(args.check_flow_output)
        if flow_calibration != frozen_flow:
            raise ValueError(
                "recomputed flow calibration differs from "
                f"{args.check_flow_output}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
