"""Phase lifecycle events: solidification reference and plastic-state reset.

Extracted verbatim from 159_local/v03/am_thermal_stress_macro_intersection_mech100.py.
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

    The important modeling change is that T_ref is written when material
    solidifies from liquid/mushy state, not when a layer is merely activated.
    Re-melting moves the point back to a stress-free liquid/mushy state; when it
    solidifies again the reference temperature is overwritten.
    """
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

        old_was_melted = (phase_quad == STATE_LIQUID) | (phase_quad == STATE_MUSHY)
        became_solid = cold & old_was_melted
        stayed_solid = cold & (phase_quad == STATE_SOLID)

        phase_new = np.where(hot_liquid, STATE_LIQUID, phase_new)
        phase_new = np.where(mushy, STATE_MUSHY, phase_new)
        phase_new = np.where(became_solid | stayed_solid, STATE_SOLID, phase_new)

        newly_solidified = became_solid
        entered_melted_state = (hot_liquid | mushy) & ((phase_quad == STATE_SOLID) | (phase_quad == STATE_MUSHY) | (phase_quad == STATE_LIQUID))
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

    relax_T = getattr(args, "stress_relaxation_temperature", None)
    if relax_T is not None and relax_T > 0.0:
        # Stress-free reference is the relaxation temperature: above it the
        # material is assumed to carry no stress (macro calibration knob,
        # Ti64 typically ~1073-1173 K). Residual stress then builds from the
        # constrained shrinkage between relax_T and the local temperature.
        T_ref_value = relax_T * np.ones_like(T_quad)
    else:
        T_ref_value = T_quad
    T_ref_new = np.where(newly_solidified, T_ref_value, T_ref_quad)
    if args.reset_plastic_on_melt:
        eqp_new = np.where(entered_melted_state, np.zeros_like(eqp_quad), eqp_quad)
    else:
        eqp_new = eqp_quad
    return phase_new, T_ref_new, eqp_new, newly_solidified, entered_melted_state
