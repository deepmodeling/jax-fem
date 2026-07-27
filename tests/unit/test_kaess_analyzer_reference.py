import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as onp
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = REPO_ROOT / "cases" / "kaess_2023" / "analyze_kaess.py"
FIG9_PATH = (
    REPO_ROOT
    / "cases"
    / "kaess_2023"
    / "references"
    / "digitized"
    / "fig9_bending.csv"
)


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("kaess_analyzer", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyzer_uses_the_frozen_figure9a_csv_instead_of_hardcoded_values():
    with FIG9_PATH.open(newline="", encoding="utf-8") as stream:
        expected = {
            int(float(row["build_plate_temp_c"])): float(
                row["max_front_bending_um"]
            )
            for row in csv.DictReader(stream)
            if row["series_id"] == "plate_temperature_fixed_p250_v850"
        }

    analyzer = _load_analyzer()

    assert analyzer.FIG9A_REFERENCE_UM == expected
    assert analyzer.FIG9A_REFERENCE_PATH == FIG9_PATH


def _tetra_points_about(centroid, half_width=5.0e-6):
    offsets = half_width * onp.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
    )
    return onp.asarray(centroid) + offsets


def _fake_mesh(points, cell_type, cells, cell_data):
    block = SimpleNamespace(
        type=cell_type,
        data=onp.asarray(cells, dtype=onp.int64),
    )
    return SimpleNamespace(
        points=onp.asarray(points, dtype=float),
        cells=[block],
        cells_dict={cell_type: block.data},
        cell_data={
            name: [onp.asarray(values, dtype=float)]
            for name, values in cell_data.items()
        },
    )


def test_stress_depth_profile_averages_only_xx_over_quadrature(monkeypatch):
    analyzer = _load_analyzer()
    top = _tetra_points_about([0.475e-3, 0.25e-3, 0.55e-3])
    lower = _tetra_points_about([0.475e-3, 0.25e-3, 0.35e-3])
    points = onp.concatenate([top, lower], axis=0)
    cells = onp.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    cell_data = {
        "stress_quad0_xx": [600.0, 300.0],
        "stress_quad1_xx": [400.0, 100.0],
    }
    for quad_idx in (0, 1):
        for component_idx, component in enumerate(
            ("yy", "zz", "xy", "yz", "xz"),
            start=1,
        ):
            # Distinct large sentinels: none may leak into sigma_xx.
            cell_data[
                f"stress_quad{quad_idx}_{component}"
            ] = onp.full(2, component_idx * (quad_idx + 1) * 1.0e9)
    mesh = _fake_mesh(points, "tetra", cells, cell_data)
    monkeypatch.setattr(
        analyzer,
        "meshio",
        SimpleNamespace(read=lambda _path: mesh),
    )

    profile = analyzer.stress_depth_profile("unused.vtu")

    assert profile is not None
    onp.testing.assert_allclose(
        profile["z_m"],
        [0.55e-3, 0.35e-3],
    )
    onp.testing.assert_allclose(
        profile["sigma_xx_pa"],
        [500.0, 200.0],
    )


def test_stress_depth_profile_supports_hex8_and_single_quad_field(monkeypatch):
    analyzer = _load_analyzer()
    center = onp.array([0.475e-3, 0.25e-3, 0.45e-3])
    half = onp.array([10.0e-6, 10.0e-6, 10.0e-6])
    points = center + half * onp.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    mesh = _fake_mesh(
        points,
        "hexahedron",
        [[0, 1, 2, 3, 4, 5, 6, 7]],
        {
            "stress_quad_xx": [123.0e6],
            "stress_quad_yy": [999.0e6],
        },
    )
    monkeypatch.setattr(
        analyzer,
        "meshio",
        SimpleNamespace(read=lambda _path: mesh),
    )

    profile = analyzer.stress_depth_profile("unused.vtu")

    assert profile is not None
    onp.testing.assert_allclose(profile["z_m"], [0.45e-3])
    onp.testing.assert_allclose(profile["sigma_xx_pa"], [123.0e6])


@pytest.mark.parametrize(
    "cell_data",
    [
        pytest.param(
            {
                "stress_quad_xx": [1.0],
                "stress_quad0_xx": [2.0],
            },
            id="mixed-unindexed-and-numbered",
        ),
        pytest.param(
            {
                "stress_quad0_xx": [1.0],
                "stress_quad2_xx": [2.0],
            },
            id="non-contiguous-quad-indices",
        ),
        pytest.param(
            {
                "stress_quad_xx": [[1.0, 2.0, 3.0]],
            },
            id="non-scalar-cell-field",
        ),
    ],
)
def test_pooled_quad_component_rejects_ambiguous_or_invalid_fields(
    cell_data,
):
    analyzer = _load_analyzer()
    points = _tetra_points_about([0.475e-3, 0.25e-3, 0.45e-3])
    mesh = _fake_mesh(
        points,
        "tetra",
        [[0, 1, 2, 3]],
        cell_data,
    )

    with pytest.raises(ValueError):
        analyzer.pooled_quad_field(mesh, "stress", component="xx")
