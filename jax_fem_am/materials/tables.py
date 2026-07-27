"""Temperature-dependent material property tables (T,value CSV).

Origin: legacy/v03/am_thermal_stress_macro_intersection_mech100.py
(PropertyTable, load_property_tables, eval_property). Moved verbatim in the
2026-07-22 restructure.
"""
import csv
from pathlib import Path

import jax.numpy as np


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
    return tables


def eval_property(T_quad, table, default):
    if table is None:
        return default * np.ones_like(T_quad)
    return table.eval(T_quad)
