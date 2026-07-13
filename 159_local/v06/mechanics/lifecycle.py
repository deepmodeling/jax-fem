"""Pure update rules for mechanical birth, remelt and relaxation state."""

from __future__ import annotations

import jax.numpy as jnp


def effective_thermal_increment(dT, relaxation_managed):
    """Clip positive increments above a stress-relaxation reference.

    For managed material, ``T_ref`` is the relaxation temperature. Temperatures
    above it produce zero, not positive, elastic thermal strain; cooling below
    it retains the negative increment that drives shrinkage. Fixture material
    is left unchanged by setting ``relaxation_managed=False``.
    """
    dT = jnp.asarray(dT)
    managed = jnp.asarray(relaxation_managed, dtype=bool)
    return jnp.where(managed, jnp.minimum(dT, 0.0), dT)


def update_stress_free_reference(
    old_reference,
    total_strain,
    thermal_strain,
    plastic_strain,
    event_mask,
):
    """Capture the current configuration as stress free at lifecycle events.

    The decomposition is ``eps_e = eps - eps_ref - eps_th - eps_p``. Therefore
    ``eps_ref = eps - eps_th - eps_p`` makes the event configuration exactly
    stress free, including a delayed or below-relaxation-temperature event.
    """
    old_reference = jnp.asarray(old_reference)
    total_strain = jnp.asarray(total_strain)
    thermal_strain = jnp.asarray(thermal_strain)
    plastic_strain = jnp.asarray(plastic_strain)
    symmetric_total = 0.5 * (
        total_strain + jnp.swapaxes(total_strain, -1, -2)
    )
    symmetric_plastic = 0.5 * (
        plastic_strain + jnp.swapaxes(plastic_strain, -1, -2)
    )
    symmetric_thermal = 0.5 * (
        thermal_strain + jnp.swapaxes(thermal_strain, -1, -2)
    )
    target = symmetric_total - symmetric_thermal - symmetric_plastic
    mask = jnp.asarray(event_mask, dtype=bool)
    if mask.ndim == old_reference.ndim - 1 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    while mask.ndim < old_reference.ndim:
        mask = mask[..., None]
    return jnp.where(mask, target, old_reference)
