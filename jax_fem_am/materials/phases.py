"""Material phase/state codes and phase-dependent property maps.

Origin: legacy/v03/am_thermal_stress_macro_intersection_mech100.py
(MODE_TO_ID, STATE_* codes, make_quad_scalar, initial_phase_cell,
phase_cell_from_quad, material_cell_state, thermal_material_quads,
clamp_mechanics_temperature, mechanics_material_quads). Moved verbatim in
the 2026-07-22 restructure.
"""
import jax.numpy as np
import numpy as onp

from jax_fem_am.materials.tables import eval_property


MODE_TO_ID = {
    "scan": 1,
    "hatch_dwell": 2,
    "layer_dwell": 3,
    "recoat": 4,
    "cooling": 5,
    "path": 6,
    "release": 7,
    "jump": 8,
}

# Material/phase codes saved to ParaView and used by the thermal/mechanical
# property maps. Keep these values stable for post-processing scripts.
STATE_VOID = 0.0
STATE_POWDER = 1.0
STATE_SOLID = 2.0
STATE_MUSHY = 3.0
STATE_LIQUID = 4.0
STATE_SUBSTRATE = 5.0
STATE_SUPPORT = 6.0


def make_quad_scalar(cell_values, num_quads):
    arr = np.asarray(cell_values)[:, None, None]
    return arr * np.ones((len(cell_values), num_quads, 1))


def initial_phase_cell(active_cell, substrate_cell, support_cell, args):
    if getattr(args, "layer_activation_mode", "front") == "layer_on_scan" and getattr(args, "future_layer_mode", "void") == "void":
        # Future layers have not been recoated yet, so they are void/inactive.
        # They will become powder when their layer is activated at laser scan start.
        inactive_code = STATE_VOID
    else:
        inactive_code = STATE_POWDER if args.powder_mode == "powder" else STATE_VOID
    phase = inactive_code * onp.ones(len(active_cell), dtype=onp.float64)
    # Active printed cells still start as powder; they only become solid after a
    # melt/solidification event. Fixture regions are treated as solid supports.
    phase[active_cell & (~substrate_cell) & (~support_cell)] = STATE_POWDER
    phase[substrate_cell] = STATE_SUBSTRATE
    phase[support_cell] = STATE_SUPPORT
    return phase


def phase_cell_from_quad(phase_quad):
    return onp.max(onp.asarray(phase_quad)[:, :, 0], axis=1)


def material_cell_state(active_cell, substrate_cell, support_cell, args, cell_temperature=None, phase_cell=None):
    # State encoding for ParaView and CSV post-processing:
    #   0 void/inactive weak material
    #   1 powder
    #   2 solid
    #   3 mushy zone
    #   4 liquid
    #   5 substrate
    #   6 support
    if phase_cell is not None:
        state = onp.asarray(phase_cell, dtype=onp.float64).copy()
    else:
        inactive_code = STATE_POWDER if args.powder_mode == "powder" else STATE_VOID
        state = inactive_code * onp.ones(len(active_cell), dtype=onp.float64)
        state[active_cell] = STATE_SOLID
        if cell_temperature is not None and args.liquidus_temperature > args.solidus_temperature:
            active_non_fixture = active_cell & (~substrate_cell) & (~support_cell)
            mushy = active_non_fixture & (cell_temperature >= args.solidus_temperature) & (cell_temperature < args.liquidus_temperature)
            liquid = active_non_fixture & (cell_temperature >= args.liquidus_temperature)
            state[mushy] = STATE_MUSHY
            state[liquid] = STATE_LIQUID
    state[substrate_cell] = STATE_SUBSTRATE
    state[support_cell] = STATE_SUPPORT
    return state


def thermal_material_quads(T_old_quad, active_quad, phase_quad, args, tables, printed_quad=None, cooling_only_quad=None):
    rho_solid = args.rho_solid if args.rho_solid is not None else args.rho
    cp_solid = args.cp_solid if args.cp_solid is not None else args.cp
    k_solid = args.conductivity_solid if args.conductivity_solid is not None else args.conductivity
    rho_liquid = args.rho_liquid if args.rho_liquid is not None else rho_solid
    cp_liquid = args.cp_liquid if args.cp_liquid is not None else cp_solid
    k_liquid = args.conductivity_liquid if args.conductivity_liquid is not None else k_solid

    cp_solid_quad = eval_property(T_old_quad, tables["cp_solid"], cp_solid)
    k_solid_quad = eval_property(T_old_quad, tables["k_solid"], k_solid)
    cp_powder_quad = eval_property(T_old_quad, tables["cp_powder"], args.cp_powder)
    k_powder_quad = eval_property(T_old_quad, tables["k_powder"], args.conductivity_powder)
    cp_liquid_quad = eval_property(T_old_quad, tables["cp_liquid"], cp_liquid)
    k_liquid_quad = eval_property(T_old_quad, tables["k_liquid"], k_liquid)

    if printed_quad is None:
        printed_quad = active_quad
    if cooling_only_quad is None:
        cooling_only_quad = np.zeros_like(active_quad)

    inactive_mass_factor = getattr(args, "inactive_mass_factor", None)
    if inactive_mass_factor is None:
        inactive_mass_factor = args.inactive_thermal_factor
    rho_void = rho_solid * inactive_mass_factor
    cp_void = cp_solid * np.ones_like(T_old_quad)
    k_void = k_solid * args.inactive_thermal_factor * np.ones_like(T_old_quad)

    if args.liquidus_temperature > args.solidus_temperature:
        mushy_frac = np.clip(
            (T_old_quad - args.solidus_temperature) / (args.liquidus_temperature - args.solidus_temperature),
            0.0,
            1.0,
        )
    else:
        mushy_frac = np.zeros_like(T_old_quad)

    rho_mushy = (1.0 - mushy_frac) * rho_solid + mushy_frac * rho_liquid
    cp_mushy = (1.0 - mushy_frac) * cp_solid_quad + mushy_frac * cp_liquid_quad
    k_mushy = (1.0 - mushy_frac) * k_solid_quad + mushy_frac * k_liquid_quad

    is_void = phase_quad == STATE_VOID
    is_powder = phase_quad == STATE_POWDER
    is_liquid = phase_quad == STATE_LIQUID
    is_mushy = phase_quad == STATE_MUSHY

    # Phase-based properties for cells that belong to the current thermal window.
    rho_phase = np.where(
        is_void,
        rho_void,
        np.where(is_powder, args.rho_powder, np.where(is_liquid, rho_liquid, np.where(is_mushy, rho_mushy, rho_solid))),
    )
    cp_phase = np.where(
        is_void,
        cp_void,
        np.where(is_powder, cp_powder_quad, np.where(is_liquid, cp_liquid_quad, np.where(is_mushy, cp_mushy, cp_solid_quad))),
    )
    k_phase = np.where(
        is_void,
        k_void,
        np.where(is_powder, k_powder_quad, np.where(is_liquid, k_liquid_quad, np.where(is_mushy, k_mushy, k_solid_quad))),
    )

    # Unprinted region: either powder bed or near-void material.
    # In layer_on_scan mode with future_layer_mode=void, future layers are not
    # physically spread yet, so they should not act as a powder heat sink before
    # their scan begins.
    future_layers_are_void = (
        getattr(args, "layer_activation_mode", "front") == "layer_on_scan"
        and getattr(args, "future_layer_mode", "void") == "void"
    )
    if future_layers_are_void or args.powder_mode == "void":
        rho_unprinted = rho_void
        cp_unprinted = cp_void
        k_unprinted = k_void
    else:
        rho_unprinted = args.rho_powder
        cp_unprinted = cp_powder_quad
        k_unprinted = k_powder_quad

    # Printed layers below the moving thermal window: keep heat capacity/history,
    # but strongly reduce conductivity so they no longer dominate local heat loss.
    rho_old = rho_solid * np.ones_like(T_old_quad)
    cp_old = cp_solid_quad
    k_old = k_solid_quad * args.old_layer_thermal_factor

    is_printed = printed_quad > 0.5
    is_window = active_quad > 0.5
    is_cooling_only = cooling_only_quad > 0.5

    rho_quad = np.where(is_window, rho_phase, np.where(is_cooling_only, rho_old, rho_unprinted))
    cp_quad = np.where(is_window, cp_phase, np.where(is_cooling_only, cp_old, cp_unprinted))
    conductivity_quad = np.where(is_window, k_phase, np.where(is_cooling_only, k_old, k_unprinted))

    # Safety: cells that have not been printed are always treated as unprinted,
    # even if a phase history entry is accidentally non-void.
    rho_quad = np.where(is_printed | is_window, rho_quad, rho_unprinted)
    cp_quad = np.where(is_printed | is_window, cp_quad, cp_unprinted)
    conductivity_quad = np.where(is_printed | is_window, conductivity_quad, k_unprinted)

    if args.latent_heat > 0.0 and args.liquidus_temperature > args.solidus_temperature:
        in_mushy = (T_old_quad >= args.solidus_temperature) & (T_old_quad <= args.liquidus_temperature) & is_window
        latent_cp = np.where(in_mushy, args.latent_heat / (args.liquidus_temperature - args.solidus_temperature), 0.0)
    else:
        latent_cp = np.zeros_like(T_old_quad)

    return rho_quad, cp_quad, conductivity_quad, latent_cp


def clamp_mechanics_temperature(T_quad, floor):
    """Floor the mechanics-side temperature field; floor=None keeps it untouched.

    Only the mechanics chain (dT/thermal strain and material-table lookups) sees
    the clamped field; the thermal solution and phase bookkeeping stay unclamped
    so the guard cannot mask undershoot in thermal diagnostics/audits.
    """
    if floor is None:
        return T_quad
    return np.maximum(T_quad, floor)


def mechanics_material_quads(T_quad, active_quad, phase_quad, args, tables):
    E_quad = eval_property(T_quad, tables["E"], args.young)
    alpha_base = eval_property(T_quad, tables["alpha"], args.alpha)
    poisson_quad = eval_property(T_quad, tables["poisson"], args.poisson)
    if args.mechanics_model == "j2_plastic":
        if tables["yield"] is None:
            raise ValueError("--mechanics-model j2_plastic requires --yield-table")
        yield_quad = eval_property(T_quad, tables["yield"], args.young)
        hardening_quad = eval_property(T_quad, tables["hardening"], 0.0)
    else:
        yield_quad = args.young * np.ones_like(T_quad)
        hardening_quad = np.zeros_like(T_quad)

    is_solid_like = (phase_quad == STATE_SOLID) | (phase_quad == STATE_SUBSTRATE) | (phase_quad == STATE_SUPPORT)
    is_mushy = phase_quad == STATE_MUSHY
    is_liquid = phase_quad == STATE_LIQUID

    active_factor_quad = np.where(
        is_solid_like,
        1.0,
        np.where(is_mushy, args.mushy_mechanics_factor, np.where(is_liquid, args.liquid_mechanics_factor, args.inactive_mechanics_factor)),
    )
    active_factor_quad = active_factor_quad * active_quad + args.inactive_mechanics_factor * (1.0 - active_quad)
    alpha_quad = np.where(is_solid_like, alpha_base, np.zeros_like(alpha_base))
    if getattr(args, "powder_solid_E", None) is not None:
        # Weak-solid powder (Kaess 2023): constant E/sigma_y, no hardening,
        # full active factor where mechanically active - the weakness lives in
        # the material values, not the ersatz factor. alpha stays zero above.
        is_powder = phase_quad == STATE_POWDER
        E_quad = np.where(is_powder, args.powder_solid_E, E_quad)
        yield_quad = np.where(is_powder, args.powder_solid_yield, yield_quad)
        hardening_quad = np.where(
            is_powder, getattr(args, "powder_solid_hardening", 1.0e7), hardening_quad)
        active_factor_quad = np.where(
            is_powder,
            active_quad + args.inactive_mechanics_factor * (1.0 - active_quad),
            active_factor_quad,
        )
    return active_factor_quad, E_quad, alpha_quad, poisson_quad, yield_quad, hardening_quad
