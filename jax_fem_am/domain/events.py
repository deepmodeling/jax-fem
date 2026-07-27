"""Phase lifecycle events for legacy-reset and paper-irreversible histories.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import jax.numpy as np

from jax_fem_am.materials.phases import (
    STATE_LIQUID,
    STATE_MUSHY,
    STATE_POWDER,
    STATE_SOLID,
    STATE_SUBSTRATE,
    STATE_SUPPORT,
    STATE_VOID,
)


def update_phase_reference_and_eqp(T_quad, active_quad, phase_quad, T_ref_quad, eqp_quad, args):
    """Update quadrature-point material phase and stress-free reference temperature.

    ``legacy_reset`` preserves the historical reversible melt/reset behavior.
    ``paper_irreversible`` implements the Kaess et al. (2023), Section 2.5,
    field variable: powder starts at zero and switches permanently to solid
    after first melting.  In that mode a solid point may report a later melt
    event for diagnostics, but it cannot return to a liquid/mushy history
    state, rewrite its first solidification reference, or erase plastic
    history.

    Source: https://doi.org/10.3390/ma16062321
    """
    history_model = getattr(
        args,
        "phase_history_model",
        "legacy_reset",
    )
    if history_model not in ("legacy_reset", "paper_irreversible"):
        raise ValueError(
            "phase_history_model must be 'legacy_reset' or "
            "'paper_irreversible'"
        )
    paper_irreversible = history_model == "paper_irreversible"
    active = active_quad > 0.5
    fixture = (phase_quad == STATE_SUBSTRATE) | (phase_quad == STATE_SUPPORT)
    non_fixture = active & (~fixture)
    phase_new = phase_quad

    newly_active_void = non_fixture & (phase_new == STATE_VOID)
    phase_new = np.where(newly_active_void, STATE_POWDER, phase_new)

    if args.liquidus_temperature > args.solidus_temperature:
        Ts = float(args.solidus_temperature)
        Tl = float(args.liquidus_temperature)
        hot_liquid = non_fixture & (T_quad >= Tl)
        mushy = non_fixture & (T_quad >= Ts) & (T_quad < Tl)
        cold = non_fixture & (T_quad < Ts)

        old_was_melted = (
            (phase_quad == STATE_LIQUID)
            | (phase_quad == STATE_MUSHY)
        )
        old_was_solid = phase_quad == STATE_SOLID
        became_solid = cold & old_was_melted
        if paper_irreversible:
            powder_or_transient = (
                (phase_quad == STATE_POWDER)
                | (phase_quad == STATE_VOID)
                | old_was_melted
            )
            phase_new = np.where(
                hot_liquid & powder_or_transient,
                STATE_LIQUID,
                phase_new,
            )
            phase_new = np.where(
                mushy & powder_or_transient,
                STATE_MUSHY,
                phase_new,
            )
            phase_new = np.where(
                became_solid | old_was_solid,
                STATE_SOLID,
                phase_new,
            )
            entered_melted_state = (
                (hot_liquid | mushy)
                & (old_was_solid | old_was_melted)
            )
        else:
            stayed_solid = cold & old_was_solid
            phase_new = np.where(hot_liquid, STATE_LIQUID, phase_new)
            phase_new = np.where(mushy, STATE_MUSHY, phase_new)
            phase_new = np.where(
                became_solid | stayed_solid,
                STATE_SOLID,
                phase_new,
            )
            entered_melted_state = (
                (hot_liquid | mushy)
                & (old_was_solid | old_was_melted)
            )
        newly_solidified = became_solid
    else:
        # Compatibility / macro consolidation-on-activation mode when no
        # phase-change interval is provided: activated material solidifies
        # directly. This is the standard part-scale AM approach when the mesh
        # cannot resolve the melt pool (beam radius < element size) or the
        # lumped macro path does not carry melt-level energy density.
        became_solid = non_fixture & (phase_quad != STATE_SOLID)
        phase_new = np.where(non_fixture, STATE_SOLID, phase_new)
        newly_solidified = became_solid
        entered_melted_state = np.zeros_like(active, dtype=bool)

    if paper_irreversible:
        T_ref_value = T_quad
        eqp_new = eqp_quad
    else:
        relax_temperature = getattr(
            args,
            "stress_relaxation_temperature",
            None,
        )
        if relax_temperature is not None and relax_temperature > 0.0:
            T_ref_value = relax_temperature * np.ones_like(T_quad)
        else:
            T_ref_value = T_quad
        if getattr(args, "reset_plastic_on_melt", False):
            eqp_new = np.where(
                entered_melted_state,
                np.zeros_like(eqp_quad),
                eqp_quad,
            )
        else:
            eqp_new = eqp_quad
    T_ref_new = np.where(newly_solidified, T_ref_value, T_ref_quad)
    return phase_new, T_ref_new, eqp_new, newly_solidified, entered_melted_state
