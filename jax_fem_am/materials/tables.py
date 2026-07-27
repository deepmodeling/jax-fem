"""Temperature-dependent material property tables (T,value CSV).

Origin: legacy/v03/am_thermal_stress_macro_intersection_mech100.py
(PropertyTable, load_property_tables, eval_property). Moved verbatim in the
2026-07-22 restructure.
"""
import csv
import math
from pathlib import Path

import jax.numpy as np
import numpy as onp


class PropertyTable:
    def __init__(self, path):
        self.path = path
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "T" not in reader.fieldnames or "value" not in reader.fieldnames:
                raise ValueError(f"Property table must contain T,value columns: {path}")
            for row in reader:
                rows.append((float(row["T"]), float(row["value"])))
        if len(rows) < 2:
            raise ValueError(f"Property table needs at least two rows: {path}")
        rows.sort(key=lambda item: item[0])
        self.T = np.asarray([r[0] for r in rows])
        self.values = np.asarray([r[1] for r in rows])

    def eval(self, T):
        return np.interp(T, self.T, self.values)


class FlowCurveTable:
    """Rectangular flow-stress grid indexed by temperature and plastic strain."""

    _required_columns = {
        "temperature_K",
        "equivalent_plastic_strain",
        "flow_stress_Pa",
        "source",
    }

    def __init__(self, path):
        self.path = Path(path)
        nodes = {}
        with self.path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = self._required_columns.difference(
                reader.fieldnames or ()
            )
            if missing:
                raise ValueError(
                    "Flow curve must contain "
                    "temperature_K,equivalent_plastic_strain,"
                    f"flow_stress_Pa,source columns: {self.path}"
                )
            for row in reader:
                if not (row.get("source") or "").strip():
                    raise ValueError(
                        "Flow curve nodes require a non-empty source "
                        f"(row {reader.line_num}): {self.path}"
                    )
                temperature = float(row["temperature_K"])
                plastic_strain = float(
                    row["equivalent_plastic_strain"]
                )
                stress = float(row["flow_stress_Pa"])
                if not all(
                    math.isfinite(value)
                    for value in (temperature, plastic_strain, stress)
                ):
                    raise ValueError(
                        f"Flow curve values must be finite: {self.path}"
                    )
                key = (temperature, plastic_strain)
                if key in nodes:
                    raise ValueError(
                        f"Flow curve contains duplicate node {key}: "
                        f"{self.path}"
                    )
                nodes[key] = stress

        temperatures = sorted({key[0] for key in nodes})
        plastic_strains = sorted({key[1] for key in nodes})
        if len(temperatures) < 2 or len(plastic_strains) < 2:
            raise ValueError(
                "Flow curve needs at least two temperatures and two "
                f"plastic-strain points: {self.path}"
            )
        expected = len(temperatures) * len(plastic_strains)
        if len(nodes) != expected:
            raise ValueError(
                "Flow curve must form a complete rectangular grid: "
                f"{self.path}"
            )

        self.temperatures = onp.asarray(temperatures, dtype=onp.float64)
        self.plastic_strains = onp.asarray(
            plastic_strains,
            dtype=onp.float64,
        )
        self.stresses = onp.asarray(
            [
                [nodes[(temperature, strain)] for strain in plastic_strains]
                for temperature in temperatures
            ],
            dtype=onp.float64,
        )


def load_property_tables(args):
    config_path = getattr(args, "config", None)

    def resolve(path):
        if not path:
            return None
        path = Path(path)
        if path.is_absolute() or config_path is None:
            return path
        config_relative = (
            Path(config_path).expanduser().resolve().parent / path
        )
        if config_relative.is_file():
            return config_relative
        # Backward compatibility for older configs whose paths were written
        # relative to the project launch directory rather than the config.
        return path

    tables = {}
    for key, path in [
        ("k_solid", args.k_table_solid),
        ("cp_solid", args.cp_table_solid),
        ("k_powder", args.k_table_powder),
        ("cp_powder", args.cp_table_powder),
        ("k_liquid", args.k_table_liquid),
        ("cp_liquid", args.cp_table_liquid),
        ("E", args.E_table),
        ("alpha", args.alpha_table),
        ("poisson", args.poisson_table),
        ("yield", args.yield_table),
        ("hardening", args.hardening_table),
    ]:
        resolved = resolve(path)
        tables[key] = PropertyTable(resolved) if resolved else None
    flow_curve_path = resolve(
        getattr(args, "flow_curve_table", None)
    )
    tables["flow_curve"] = (
        FlowCurveTable(flow_curve_path) if flow_curve_path else None
    )
    return tables


def eval_property(T_quad, table, default):
    if table is None:
        return default * np.ones_like(T_quad)
    return table.eval(T_quad)
