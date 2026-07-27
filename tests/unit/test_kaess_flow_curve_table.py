from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_fem_am.config.schema import build_parser
from jax_fem_am.materials.tables import load_property_tables
import jax_fem_am.materials.tables as material_tables


def _write_curve(path, rows):
    path.write_text(
        "temperature_K,equivalent_plastic_strain,flow_stress_Pa,source\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _table_args(config_path, curve_path):
    return SimpleNamespace(
        config=str(config_path),
        k_table_solid=None,
        cp_table_solid=None,
        k_table_powder=None,
        cp_table_powder=None,
        k_table_liquid=None,
        cp_table_liquid=None,
        E_table=None,
        alpha_table=None,
        poisson_table=None,
        yield_table=None,
        hardening_table=None,
        flow_curve_table=curve_path,
    )


def test_flow_curve_table_builds_a_sorted_complete_grid(tmp_path):
    curve_path = tmp_path / "flow.csv"
    _write_curve(
        curve_path,
        [
            "800,0.10,3.4e8,synthetic",
            "300,0.10,6.0e8,synthetic",
            "800,0.00,3.0e8,synthetic",
            "300,0.00,5.0e8,synthetic",
        ],
    )

    table = material_tables.FlowCurveTable(curve_path)

    np.testing.assert_array_equal(table.temperatures, [300.0, 800.0])
    np.testing.assert_array_equal(table.plastic_strains, [0.0, 0.1])
    np.testing.assert_array_equal(
        table.stresses,
        [[5.0e8, 6.0e8], [3.0e8, 3.4e8]],
    )


@pytest.mark.parametrize(
    "rows,match",
    [
        (
            [
                "300,0.00,5.0e8,a",
                "300,0.00,5.1e8,b",
                "800,0.00,3.0e8,c",
                "800,0.10,3.4e8,d",
            ],
            "duplicate",
        ),
        (
            [
                "300,0.00,5.0e8,a",
                "300,0.10,6.0e8,b",
                "800,0.00,3.0e8,c",
            ],
            "complete rectangular grid",
        ),
    ],
)
def test_flow_curve_table_rejects_duplicate_or_missing_nodes(
    tmp_path,
    rows,
    match,
):
    curve_path = tmp_path / "invalid.csv"
    _write_curve(curve_path, rows)

    with pytest.raises(ValueError, match=match):
        material_tables.FlowCurveTable(curve_path)


def test_loader_resolves_flow_curve_relative_to_material_config(tmp_path):
    config_path = tmp_path / "material.json"
    config_path.write_text("{}", encoding="utf-8")
    curve_path = tmp_path / "flow.csv"
    _write_curve(
        curve_path,
        [
            "300,0.00,5.0e8,a",
            "300,0.10,6.0e8,b",
            "800,0.00,3.0e8,c",
            "800,0.10,3.4e8,d",
        ],
    )

    tables = load_property_tables(
        _table_args(config_path, "flow.csv")
    )

    assert tables["flow_curve"].path == curve_path


def test_cli_exposes_flow_curve_table_with_safe_legacy_default():
    assert build_parser().parse_args([]).flow_curve_table is None
    assert (
        build_parser({"flow_curve_table": "flow.csv"})
        .parse_args([])
        .flow_curve_table
        == "flow.csv"
    )
