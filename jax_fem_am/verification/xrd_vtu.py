"""Apply geometry-aware XRD gauge operators to v06 VTU elastic strain."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from jax_fem_am.verification.mesh_quality import audit_tet_mesh

from .xrd import compute_gauge_weights, predict_gauge_microstrain


_COMPONENTS = {
    "xx": (0, 0),
    "yy": (1, 1),
    "zz": (2, 2),
    "xy": (0, 1),
    "yz": (1, 2),
    "xz": (0, 2),
}
_FIELD_PATTERN = re.compile(
    r"^elastic_strain_quad(?P<quad>\d*)_(?P<component>xx|yy|zz|xy|yz|xz)$"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tetra_block(mesh):
    block_indices = [
        index for index, block in enumerate(mesh.cells) if block.type == "tetra"
    ]
    if not block_indices:
        raise ValueError("VTU must contain a tetra cell block")
    if len(block_indices) != 1:
        raise ValueError("VTU must contain exactly one tetra cell block")
    block_index = block_indices[0]
    cell_data = {
        name: np.asarray(arrays[block_index])
        for name, arrays in mesh.cell_data.items()
    }
    return block_index, np.asarray(mesh.cells[block_index].data), cell_data


def _elastic_strain_quads(cell_data, num_cells):
    fields = {}
    for name, values in cell_data.items():
        match = _FIELD_PATTERN.fullmatch(name)
        if match is None:
            continue
        quad_text = match.group("quad")
        quad = int(quad_text) if quad_text else 0
        component = match.group("component")
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) != num_cells:
            raise ValueError(f"{name} must contain one value per tetra cell")
        fields[(quad, component)] = values
    if not fields:
        raise ValueError("VTU has no elastic_strain_quad tensor fields")
    quad_ids = sorted({quad for quad, _component in fields})
    if quad_ids != list(range(len(quad_ids))):
        raise ValueError("elastic strain quadrature indices must be contiguous")
    result = np.zeros((num_cells, len(quad_ids), 3, 3), dtype=np.float64)
    for quad in quad_ids:
        missing = sorted(set(_COMPONENTS).difference(
            component for field_quad, component in fields if field_quad == quad
        ))
        if missing:
            raise ValueError(f"elastic strain quad {quad} is missing components: {missing}")
        for component, (row, col) in _COMPONENTS.items():
            values = fields[(quad, component)]
            result[:, quad, row, col] = values
            result[:, quad, col, row] = values
    return result


def _registration(protocol):
    registration = protocol.get("mesh_to_specimen")
    if not isinstance(registration, dict):
        raise ValueError("protocol must define mesh_to_specimen registration")
    scale = float(registration.get("scale_m_per_mesh_unit", np.nan))
    rotation = np.asarray(registration.get("rotation"), dtype=np.float64)
    translation = np.asarray(registration.get("translation_m"), dtype=np.float64)
    rms = registration.get("registration_rms_m")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("registration scale must be positive and finite")
    if rotation.shape != (3, 3) or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1.0e-10
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10):
        raise ValueError("registration rotation must be right-handed orthonormal")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("registration translation must be a finite 3-vector")
    if rms is None or not np.isfinite(float(rms)) or float(rms) < 0.0:
        raise ValueError("registration_rms_m must be a nonnegative finite value")
    return scale, rotation, translation, float(rms)


def predict_vtu_gauges(vtu_path, protocol, *, quality_threshold=0.05):
    """Predict XRD elastic microstrain for registered gauge definitions."""
    import meshio

    if protocol.get("schema_version") != "v06.xrd-gauges/1":
        raise ValueError("unsupported XRD gauge protocol schema")
    gauges = protocol.get("gauges")
    if not isinstance(gauges, list) or not gauges:
        raise ValueError("protocol must contain at least one gauge")
    quality_threshold = float(quality_threshold)
    if not np.isfinite(quality_threshold) or not 0.0 <= quality_threshold <= 1.0:
        raise ValueError("quality_threshold must be finite and lie in [0, 1]")
    required_state = protocol.get("required_state")
    if required_state != "attached_to_build_plate_before_EDM":
        raise ValueError("unsupported required_state for XRD comparison")

    gauge_sizes = []
    for gauge in gauges:
        if not isinstance(gauge, dict):
            raise ValueError("each gauge must be an object")
        if gauge.get("geometry_model") != "rectangular_box":
            raise ValueError(
                "v06 XRD currently supports only rectangular_box gauge geometry"
            )
        size = np.asarray(gauge.get("size_m"), dtype=np.float64)
        if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ValueError("gauge size_m must be a positive finite 3-vector")
        gauge_sizes.append(size)

    scale, rotation, translation, registration_rms = _registration(protocol)
    maximum_rms_fraction = float(
        protocol.get("maximum_registration_rms_fraction_of_min_gauge", np.nan)
    )
    if not np.isfinite(maximum_rms_fraction) or maximum_rms_fraction < 0.0:
        raise ValueError(
            "maximum_registration_rms_fraction_of_min_gauge must be nonnegative"
        )
    registration_rms_limit = maximum_rms_fraction * min(
        float(np.min(size)) for size in gauge_sizes
    )
    if registration_rms > registration_rms_limit:
        raise ValueError(
            "registration RMS exceeds the protocol limit relative to gauge size"
        )

    measurement_temperature = float(
        protocol.get("measurement_temperature_k", np.nan)
    )
    temperature_tolerance = float(
        protocol.get("temperature_tolerance_k", np.nan)
    )
    if not np.isfinite(measurement_temperature) or measurement_temperature <= 0.0:
        raise ValueError("measurement_temperature_k must be positive and finite")
    if not np.isfinite(temperature_tolerance) or temperature_tolerance < 0.0:
        raise ValueError("temperature_tolerance_k must be nonnegative and finite")

    mesh = meshio.read(vtu_path)
    _block_index, cells, cell_data = _tetra_block(mesh)
    points_mesh = np.asarray(mesh.points, dtype=np.float64)
    temperature = np.asarray(mesh.point_data.get("T", []), dtype=np.float64).reshape(-1)
    if len(temperature) != len(points_mesh) or not np.all(np.isfinite(temperature)):
        raise ValueError("VTU must contain one finite nodal temperature per point")
    if np.any(np.abs(temperature - measurement_temperature) > temperature_tolerance):
        raise ValueError("VTU is outside the protocol measurement temperature band")
    points_specimen = (scale * points_mesh) @ rotation.T + translation
    elastic_quad_mesh = _elastic_strain_quads(cell_data, len(cells))
    elastic_cell_mesh = elastic_quad_mesh.mean(axis=1)
    elastic_cell_specimen = np.einsum(
        "ij,njk,lk->nil", rotation, elastic_cell_mesh, rotation
    )

    mode = np.asarray(cell_data.get("mode_id", []), dtype=np.float64).reshape(-1)
    if len(mode) != len(cells) or not np.all(np.isfinite(mode)):
        raise ValueError("attached XRD protocol requires finite mode_id cell data")
    if not np.all(np.isclose(mode, 5.0, rtol=0.0, atol=1.0e-12)):
        raise ValueError(
            "attached XRD comparison requires a final cooling mode_id=5 VTU"
        )

    printed = np.asarray(cell_data.get("printed", []), dtype=np.float64).reshape(-1)
    mechanics = np.asarray(
        cell_data.get("mechanics_valid", []), dtype=np.float64
    ).reshape(-1)
    if len(printed) != len(cells) or len(mechanics) != len(cells):
        raise ValueError("VTU must contain printed and mechanics_valid cell data")
    if not (
        np.all(np.isfinite(printed))
        and np.all(np.isfinite(mechanics))
        and np.all(np.isin(printed, (0.0, 1.0)))
        and np.all(np.isin(mechanics, (0.0, 1.0)))
    ):
        raise ValueError("printed and mechanics_valid must be finite binary flags")
    quality = audit_tet_mesh(
        points_specimen,
        cells,
        quality_threshold=quality_threshold,
    )
    if quality.inverted_count:
        raise ValueError("VTU contains inverted tetrahedra")
    if quality.degenerate_count:
        raise ValueError("VTU contains degenerate tetrahedra")
    valid_cells = (
        (printed > 0.5)
        & (mechanics > 0.5)
        & np.all(np.isfinite(elastic_cell_specimen), axis=(1, 2))
        & (quality.mean_ratio >= quality_threshold)
    )

    minimum_fill = float(protocol.get("minimum_material_fill_fraction", 0.95))
    reports = []
    seen_ids = set()
    for gauge in gauges:
        gauge_id = gauge.get("id")
        if not isinstance(gauge_id, str) or not gauge_id or gauge_id in seen_ids:
            raise ValueError("gauge ids must be unique nonempty strings")
        seen_ids.add(gauge_id)
        weights = compute_gauge_weights(
            points_specimen,
            cells,
            center=np.asarray(gauge["center_m"], dtype=np.float64),
            size=np.asarray(gauge["size_m"], dtype=np.float64),
            rotation_gauge_to_specimen=np.asarray(
                gauge.get("rotation_gauge_to_specimen", np.eye(3)),
                dtype=np.float64,
            ),
        )
        prediction = predict_gauge_microstrain(
            elastic_cell_specimen,
            weights,
            direction=np.asarray(gauge["direction_specimen"], dtype=np.float64),
            valid_mask=valid_cells,
            minimum_material_fill=minimum_fill,
        )
        reports.append({"id": gauge_id, **prediction})

    return {
        "schema_version": "v06.xrd-predictions/1",
        "claim_level": "measurement_operator_prediction_only",
        "required_state": required_state,
        "input_quantity": "elastic_strain_tensor",
        "input_unit": "1",
        "output_unit": "microstrain",
        "quadrature_projection": "equal_weight_P0",
        "registration_rms_m": registration_rms,
        "registration_rms_limit_m": registration_rms_limit,
        "measurement_temperature_k": measurement_temperature,
        "temperature_tolerance_k": temperature_tolerance,
        "geometry_model": "rectangular_box",
        "quality_threshold": quality_threshold,
        "gauges": reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply registered XRD gauges to a constrained v06 VTU."
    )
    parser.add_argument("--vtu", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--quality-threshold", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.protocol.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    report = predict_vtu_gauges(
        args.vtu,
        protocol,
        quality_threshold=args.quality_threshold,
    )
    report["inputs"] = {
        "vtu": {"path": str(args.vtu.resolve()), "sha256": _sha256(args.vtu)},
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": _sha256(args.protocol),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "claim_level": report["claim_level"],
        "gauge_count": len(report["gauges"]),
        "valid_gauge_count": sum(
            gauge["status"] == "ok" for gauge in report["gauges"]
        ),
    }))
    return 0 if all(gauge["status"] == "ok" for gauge in report["gauges"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
