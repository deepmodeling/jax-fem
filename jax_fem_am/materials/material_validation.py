"""Host-side material-domain checks before v06 enters JAX kernels."""

from __future__ import annotations

import numpy as np


def _table_values(name, table):
    temperatures = np.asarray(table.T, dtype=np.float64).reshape(-1)
    values = np.asarray(table.values, dtype=np.float64).reshape(-1)
    if (
        len(temperatures) < 2
        or values.shape != temperatures.shape
        or not np.all(np.isfinite(temperatures))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name} material table must contain finite T/value pairs")
    if not np.all(np.diff(temperatures) > 0.0):
        raise ValueError(f"{name} material table temperatures must be strictly increasing")
    return values


def _property_values(name, table, fallback):
    if table is not None:
        return _table_values(name, table)
    values = np.asarray([fallback], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} fallback must be finite")
    return values


def _fallback(args, primary, fallback=None):
    value = getattr(args, primary, None)
    if value is None and fallback is not None:
        value = getattr(args, fallback, None)
    return value


def _require_positive(name, values):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} values must be finite and positive")


def _validate_flow_curve(table):
    temperatures = np.asarray(
        table.temperatures,
        dtype=np.float64,
    ).reshape(-1)
    plastic_strains = np.asarray(
        table.plastic_strains,
        dtype=np.float64,
    ).reshape(-1)
    stresses = np.asarray(table.stresses, dtype=np.float64)
    if (
        len(temperatures) < 2
        or len(plastic_strains) < 2
        or stresses.shape != (len(temperatures), len(plastic_strains))
        or not np.all(np.isfinite(temperatures))
        or not np.all(np.isfinite(plastic_strains))
        or not np.all(np.isfinite(stresses))
    ):
        raise ValueError(
            "flow curve must be a finite rectangular grid with at least "
            "two temperatures and two plastic-strain points"
        )
    if not np.all(np.diff(temperatures) > 0.0):
        raise ValueError(
            "flow curve temperatures must be strictly increasing"
        )
    if not np.all(np.diff(plastic_strains) > 0.0):
        raise ValueError(
            "flow curve plastic strains must be strictly increasing"
        )
    if not np.isclose(
        plastic_strains[0],
        0.0,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError("flow curve plastic strain must start at zero")
    _require_positive("flow curve stress", stresses)
    if np.any(np.diff(stresses, axis=1) < 0.0):
        raise ValueError(
            "flow curve stress must be nondecreasing with plastic strain"
        )


def validate_material_inputs(args, tables):
    """Validate every used table and scalar fallback before tracing/JIT.

    This rejects domains that would make isotropic elasticity singular or
    produce nonphysical negative thermal capacity, conductivity, density,
    yield stress, or hardening.
    """
    required_table_keys = {
        "E",
        "alpha",
        "poisson",
        "yield",
        "hardening",
        "flow_curve",
        "k_solid",
        "cp_solid",
        "k_powder",
        "cp_powder",
        "k_liquid",
        "cp_liquid",
    }
    missing = sorted(required_table_keys.difference(tables))
    if missing:
        raise ValueError(f"material table mapping is missing keys: {missing}")

    young = _property_values("E", tables["E"], getattr(args, "young"))
    _require_positive("E", young)

    alpha = _property_values(
        "alpha", tables["alpha"], getattr(args, "alpha")
    )
    if not np.all(np.isfinite(alpha)):
        raise ValueError("alpha values must be finite")

    poisson = _property_values(
        "poisson", tables["poisson"], getattr(args, "poisson")
    )
    if np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
        raise ValueError("poisson values must satisfy -1 < poisson < 0.5")

    flow_curve = tables["flow_curve"]
    if flow_curve is not None:
        if tables["yield"] is not None or tables["hardening"] is not None:
            raise ValueError(
                "flow curve with yield/hardening tables is ambiguous"
            )
        if getattr(args, "mechanics_model", None) != "j2_plastic":
            raise ValueError(
                "flow curve requires mechanics_model=j2_plastic"
            )
        _validate_flow_curve(flow_curve)
    else:
        if (
            getattr(args, "mechanics_model", None) == "j2_plastic"
            and tables["yield"] is None
        ):
            raise ValueError(
                "j2_plastic requires a yield material table or flow curve"
            )
        yield_values = _property_values(
            "yield", tables["yield"], getattr(args, "young")
        )
        _require_positive("yield", yield_values)
        hardening = _property_values(
            "hardening",
            tables["hardening"],
            0.0,
        )
        if np.any(hardening < 0.0):
            raise ValueError(
                "hardening values must be finite and nonnegative"
            )

    for table_name, label, primary, fallback in (
        ("k_solid", "conductivity_solid", "conductivity_solid", "conductivity"),
        ("cp_solid", "cp_solid", "cp_solid", "cp"),
        ("k_powder", "conductivity_powder", "conductivity_powder", "conductivity"),
        ("cp_powder", "cp_powder", "cp_powder", "cp"),
        ("k_liquid", "conductivity_liquid", "conductivity_liquid", "conductivity"),
        ("cp_liquid", "cp_liquid", "cp_liquid", "cp"),
    ):
        values = _property_values(
            table_name,
            tables[table_name],
            _fallback(args, primary, fallback),
        )
        _require_positive(label, values)

    for name, fallback in (
        ("rho", None),
        ("rho_solid", "rho"),
        ("rho_liquid", "rho"),
        ("rho_powder", "rho"),
    ):
        _require_positive(name, [_fallback(args, name, fallback)])

    saturation = getattr(args, "yield_saturation_stress", None)
    if saturation is not None and (
        np.isnan(float(saturation)) or float(saturation) <= 0.0
    ):
        raise ValueError("yield_saturation_stress must be positive or omitted")
    return True
