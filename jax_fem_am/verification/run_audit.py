"""Audit VTU results without treating mesh-singular peak stress as a QoI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .mesh_quality import audit_solid_mesh
from .weighted import weighted_mean, weighted_quantile


def _finite_extrema(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return None, None
    return float(finite.min()), float(finite.max())


def audit_solution_fields(
    *,
    points,
    cells,
    temperature,
    displacement,
    vm_quad,
    eqp,
    printed,
    mechanics_valid,
    ambient,
    quality_threshold=0.05,
    source_free_upper_bound=None,
    temperature_atol_k=1.0e-3,
    excluded_cells=None,
):
    ambient = float(ambient)
    if not np.isfinite(ambient):
        raise ValueError("ambient must be finite")
    quality_threshold = float(quality_threshold)
    if not np.isfinite(quality_threshold) or not 0.0 <= quality_threshold <= 1.0:
        raise ValueError("quality_threshold must be finite and lie in [0, 1]")
    temperature_atol_k = float(temperature_atol_k)
    if not np.isfinite(temperature_atol_k) or temperature_atol_k < 0.0:
        raise ValueError("temperature_atol_k must be nonnegative and finite")
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    num_points = len(points)
    num_cells = len(cells)
    temperature = np.asarray(temperature, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    vm_quad = np.asarray(vm_quad, dtype=np.float64)
    eqp = np.asarray(eqp, dtype=np.float64)
    printed_raw = np.asarray(printed, dtype=np.float64)
    mechanics_valid_raw = np.asarray(mechanics_valid, dtype=np.float64)
    excluded_raw = (
        np.zeros(num_cells, dtype=np.float64)
        if excluded_cells is None
        else np.asarray(excluded_cells, dtype=np.float64)
    )
    if temperature.size != num_points:
        raise ValueError("temperature must contain one scalar per mesh point")
    if displacement.shape != (num_points, 3):
        raise ValueError("displacement must have shape (num_points, 3)")
    if vm_quad.ndim != 2 or vm_quad.shape[0] != num_cells or vm_quad.shape[1] < 1:
        raise ValueError("vm_quad must have shape (num_cells, num_quads)")
    for name, values in (
        ("eqp", eqp),
        ("printed", printed_raw),
        ("mechanics_valid", mechanics_valid_raw),
        ("excluded_cells", excluded_raw),
    ):
        if values.size != num_cells:
            raise ValueError(f"{name} must contain one scalar per mesh cell")
    if not (
        np.all(np.isfinite(printed_raw))
        and np.all(np.isfinite(mechanics_valid_raw))
        and np.all(np.isfinite(excluded_raw))
        and np.all(np.isin(printed_raw, (0.0, 1.0)))
        and np.all(np.isin(mechanics_valid_raw, (0.0, 1.0)))
        and np.all(np.isin(excluded_raw, (0.0, 1.0)))
    ):
        raise ValueError("state flags must be finite binary values")
    temperature = temperature.reshape(-1)
    eqp = eqp.reshape(-1)
    printed = printed_raw.reshape(-1) > 0.5
    mechanics_valid = mechanics_valid_raw.reshape(-1)
    excluded = excluded_raw.reshape(-1) > 0.5
    audited = ~excluded
    quality = audit_solid_mesh(
        points,
        cells,
        quality_threshold=quality_threshold,
    )
    vm_cell = np.max(vm_quad, axis=1)
    cell_mask = printed & audited & np.isfinite(vm_cell)
    global_ids = np.flatnonzero(cell_mask)
    if len(global_ids):
        global_cell = int(global_ids[np.argmax(vm_cell[global_ids])])
        global_max = float(vm_cell[global_cell])
        global_quality = float(quality.mean_ratio[global_cell])
    else:
        global_cell = None
        global_max = None
        global_quality = None

    accepted = (
        cell_mask
        & (quality.mean_ratio >= quality_threshold)
        & (mechanics_valid > 0.5)
    )
    accepted_ids = np.flatnonzero(accepted)
    reported = printed & audited
    quality_rejected = reported & (quality.mean_ratio < quality_threshold)
    quality_rejected_count = int(np.count_nonzero(quality_rejected))
    printed_volume = float(np.sum(quality.volume[reported]))
    quality_rejected_volume_fraction = (
        float(np.sum(quality.volume[quality_rejected]) / printed_volume)
        if printed_volume > 0.0
        else 0.0
    )
    stress_summary = {
        "quality_threshold": quality_threshold,
        "diagnostic_global_max": global_max,
        "diagnostic_global_max_cell": global_cell,
        "diagnostic_global_max_cell_quality": global_quality,
        "quality_filtered_cell_count": int(len(accepted_ids)),
        "quality_filtered_max": None,
        "quality_filtered_volume_weighted_mean": None,
        "quality_filtered_volume_weighted_p95": None,
        "quality_filtered_volume_weighted_p99": None,
        "quality_rejected_cell_count": quality_rejected_count,
        "quality_rejected_volume_fraction": quality_rejected_volume_fraction,
    }
    if len(accepted_ids):
        values = vm_quad[accepted_ids].reshape(-1)
        cell_volumes = quality.volume[accepted_ids]
        volumes = np.repeat(
            cell_volumes / vm_quad.shape[1], vm_quad.shape[1]
        )
        stress_summary.update(
            {
                "quality_filtered_max": float(values.max()),
                "quality_filtered_volume_weighted_mean": weighted_mean(
                    values, volumes
                ),
                "quality_filtered_volume_weighted_p95": weighted_quantile(
                    values, volumes, 0.95
                ),
                "quality_filtered_volume_weighted_p99": weighted_quantile(
                    values, volumes, 0.99
                ),
            }
        )

    temperature_min, temperature_max = _finite_extrema(temperature)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    displacement_min, displacement_max = _finite_extrema(displacement_norm)
    eqp_min, eqp_max = _finite_extrema(eqp[audited])
    finite_temperature = bool(np.all(np.isfinite(temperature)))
    finite_displacement = bool(np.all(np.isfinite(displacement)))
    finite_stress = bool(np.all(np.isfinite(vm_quad[audited])))
    finite_eqp = bool(np.all(np.isfinite(eqp[audited])))
    nonnegative_stress = bool(np.all(vm_quad[audited] >= 0.0))
    nonnegative_eqp = bool(np.all(eqp[audited] >= 0.0))
    below_absolute_zero_count = int(
        np.count_nonzero(np.isfinite(temperature) & (temperature < 0.0))
    )
    below_ambient_count = int(
        np.count_nonzero(
            np.isfinite(temperature)
            & (temperature < ambient - temperature_atol_k)
        )
    )
    if source_free_upper_bound is not None:
        source_free_upper_bound = float(source_free_upper_bound)
        if not np.isfinite(source_free_upper_bound):
            raise ValueError("source_free_upper_bound must be finite")
        above_upper_bound_count = int(
            np.count_nonzero(
                np.isfinite(temperature)
                & (
                    temperature
                    > source_free_upper_bound + temperature_atol_k
                )
            )
        )
    else:
        above_upper_bound_count = None
    mechanics_fraction = (
        float(np.mean(mechanics_valid[reported] > 0.5))
        if np.any(reported)
        else 0.0
    )
    valid_mesh = quality.inverted_count == 0 and quality.degenerate_count == 0
    has_accepted_stress = len(accepted_ids) > 0
    return {
        "valid": bool(
            valid_mesh
            and finite_temperature
            and finite_displacement
            and finite_stress
            and finite_eqp
            and nonnegative_stress
            and nonnegative_eqp
            and below_absolute_zero_count == 0
            and below_ambient_count == 0
            and above_upper_bound_count in (None, 0)
            and mechanics_fraction == 1.0
            and has_accepted_stress
            and quality_rejected_count == 0
        ),
        "units": {
            "temperature": "K",
            "displacement": "m",
            "von_mises_stress": "Pa",
            "equivalent_plastic_strain": "1",
            "mesh_volume": "m^3 when mesh coordinates are metres",
        },
        "mesh": {
            "cell_type": "tetra" if cells.shape[1] == 4 else "hexahedron",
            "num_cells": int(len(cells)),
            "audited_cell_count": int(np.count_nonzero(audited)),
            "excluded_cell_count": int(np.count_nonzero(excluded)),
            "minimum_quality": float(quality.mean_ratio.min()),
            "inverted_count": quality.inverted_count,
            "degenerate_count": quality.degenerate_count,
        },
        "temperature": {
            "all_finite": finite_temperature,
            "minimum": temperature_min,
            "maximum": temperature_max,
            "below_absolute_zero_count": below_absolute_zero_count,
            "below_ambient_count": below_ambient_count,
            "source_free_upper_bound": source_free_upper_bound,
            "above_upper_bound_count": above_upper_bound_count,
            "absolute_tolerance_k": temperature_atol_k,
        },
        "displacement_norm": {
            "all_finite": finite_displacement,
            "minimum": displacement_min,
            "maximum": displacement_max,
        },
        "plastic_strain": {
            "all_finite": finite_eqp,
            "all_nonnegative": nonnegative_eqp,
            "minimum": eqp_min,
            "maximum": eqp_max,
        },
        "mechanics_valid_fraction": mechanics_fraction,
        "stress": {
            **stress_summary,
            "all_finite": finite_stress,
            "all_nonnegative": nonnegative_stress,
        },
    }


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quad_fields(cell_data):
    names = [
        name
        for name in cell_data
        if name == "vm_quad" or re.fullmatch(r"vm_quad\d+", name)
    ]
    if not names:
        raise ValueError("VTU does not contain quadrature von Mises fields")
    names.sort(
        key=lambda name: 0
        if name == "vm_quad"
        else int(name.removeprefix("vm_quad"))
    )
    return np.column_stack([np.asarray(cell_data[name]) for name in names])


def audit_vtu(
    path,
    *,
    ambient,
    quality_threshold,
    source_free_upper_bound=None,
    temperature_atol_k=1.0e-3,
):
    import meshio

    mesh = meshio.read(path)
    solid_blocks = [
        (index, block.type)
        for index, block in enumerate(mesh.cells)
        if block.type in {"tetra", "hexahedron"}
    ]
    if len(solid_blocks) != 1:
        raise ValueError(
            f"{path} must contain exactly one TET4 or HEX8 cell block"
        )
    block_index, _cell_type = solid_blocks[0]
    cell_data = {
        name: arrays[block_index] for name, arrays in mesh.cell_data.items()
    }
    result = audit_solution_fields(
        points=mesh.points,
        cells=mesh.cells[block_index].data,
        temperature=mesh.point_data["T"],
        displacement=mesh.point_data["u"],
        vm_quad=_quad_fields(cell_data),
        eqp=cell_data["eq_plastic_strain"],
        printed=cell_data["printed"],
        mechanics_valid=cell_data["mechanics_valid"],
        ambient=ambient,
        quality_threshold=quality_threshold,
        source_free_upper_bound=source_free_upper_bound,
        temperature_atol_k=temperature_atol_k,
        excluded_cells=cell_data.get(
            "release_removed",
            np.zeros(len(mesh.cells[block_index].data), dtype=np.float64),
        ),
    )
    result["source"] = {
        "path": str(Path(path).resolve()),
        "sha256": _hash(Path(path)),
    }
    return result


def audit_run(
    run_dir,
    *,
    ambient=300.0,
    quality_threshold=0.05,
    source_free_upper_bound=None,
    temperature_atol_k=1.0e-3,
):
    run_dir = Path(run_dir)
    steps = sorted(run_dir.glob("step_*.vtu"))
    release = run_dir / "release.vtu"
    if not steps or not release.is_file():
        raise ValueError("run must contain step_*.vtu and release.vtu")
    step_audits = [
        audit_vtu(
            path,
            ambient=ambient,
            quality_threshold=quality_threshold,
            source_free_upper_bound=source_free_upper_bound,
            temperature_atol_k=temperature_atol_k,
        )
        for path in steps
    ]
    invalid_steps = [
        Path(result["source"]["path"]).name
        for result in step_audits
        if not result["valid"]
    ]
    finite_minima = [
        result["temperature"]["minimum"]
        for result in step_audits
        if result["temperature"]["minimum"] is not None
    ]
    finite_maxima = [
        result["temperature"]["maximum"]
        for result in step_audits
        if result["temperature"]["maximum"] is not None
    ]
    transient = {
        "step_count": len(step_audits),
        "all_steps_valid": not invalid_steps,
        "invalid_step_count": len(invalid_steps),
        "invalid_steps": invalid_steps,
        "minimum_temperature": min(finite_minima) if finite_minima else None,
        "maximum_temperature": max(finite_maxima) if finite_maxima else None,
        "below_absolute_zero_count": sum(
            result["temperature"]["below_absolute_zero_count"]
            for result in step_audits
        ),
        "below_ambient_count": sum(
            result["temperature"]["below_ambient_count"]
            for result in step_audits
        ),
        "above_source_free_upper_bound_count": (
            sum(
                result["temperature"]["above_upper_bound_count"] or 0
                for result in step_audits
            )
            if source_free_upper_bound is not None
            else None
        ),
        "steps": [
            {
                "name": Path(result["source"]["path"]).name,
                "sha256": result["source"]["sha256"],
                "valid": result["valid"],
                "temperature_minimum": result["temperature"]["minimum"],
                "temperature_maximum": result["temperature"]["maximum"],
                "below_ambient_count": result["temperature"][
                    "below_ambient_count"
                ],
                "above_upper_bound_count": result["temperature"][
                    "above_upper_bound_count"
                ],
                "mechanics_valid_fraction": result[
                    "mechanics_valid_fraction"
                ],
            }
            for result in step_audits
        ],
    }
    return {
        "schema_version": "v06.run-audit.2",
        "run_dir": str(run_dir.resolve()),
        "ambient": float(ambient),
        "source_free_upper_bound": source_free_upper_bound,
        "temperature_atol_k": float(temperature_atol_k),
        "units": step_audits[-1]["units"],
        "transient": transient,
        "latest_constrained": step_audits[-1],
        "release": audit_vtu(
            release,
            ambient=ambient,
            quality_threshold=quality_threshold,
            source_free_upper_bound=source_free_upper_bound,
            temperature_atol_k=temperature_atol_k,
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ambient", type=float, default=300.0)
    parser.add_argument("--quality-threshold", type=float, default=0.05)
    parser.add_argument("--source-free-upper-bound", type=float)
    parser.add_argument("--temperature-atol-k", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_run(
        args.run_dir,
        ambient=args.ambient,
        quality_threshold=args.quality_threshold,
        source_free_upper_bound=args.source_free_upper_bound,
        temperature_atol_k=args.temperature_atol_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "constrained_valid": report["transient"]["all_steps_valid"],
        "release_valid": report["release"]["valid"],
        "release_quality_filtered_vm_max": report["release"]["stress"][
            "quality_filtered_max"
        ],
    }, sort_keys=True))
    return 0 if (
        report["transient"]["all_steps_valid"] and report["release"]["valid"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
