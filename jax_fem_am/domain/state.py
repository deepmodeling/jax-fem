"""Per-step laser/scan state and activation-time nodal temperature reset.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as np
import numpy as onp


@dataclass
class StepState:
    global_step: int
    mode: str
    layer_idx: int
    hatch_idx: int
    scan_idx: int
    laser_center: onp.ndarray
    laser_power: float
    laser_switch: float
    dt: float
    scan_frac: float
    hatch_frac: float
    front_coord: float
    layer_frac: float
    ambient_temperature: Optional[float] = None
    bottom_temperature: Optional[float] = None


def reset_new_cell_nodal_temperature(T_old, cells, newly_printed_cell, previous_active_cell, value):
    """Reset nodal temperatures of freshly activated cells to the powder value.

    Freshly spread powder enters the build at the preheat/ambient temperature,
    but inactive (void) DOFs drift freely because their thermal mass is scaled
    by inactive_thermal_factor. Only nodes NOT shared with previously active
    material are reset, so the interface to already-printed layers keeps its
    conducted temperature history.
    """
    cells_arr = onp.asarray(cells)
    new_nodes = onp.unique(cells_arr[newly_printed_cell].reshape(-1))
    if new_nodes.size == 0:
        return T_old
    if previous_active_cell.any():
        old_nodes = onp.unique(cells_arr[previous_active_cell].reshape(-1))
    else:
        old_nodes = onp.empty(0, dtype=new_nodes.dtype)
    reset_nodes = onp.setdiff1d(new_nodes, old_nodes, assume_unique=True)
    if reset_nodes.size == 0:
        return T_old
    return np.asarray(T_old).at[reset_nodes, :].set(value)
