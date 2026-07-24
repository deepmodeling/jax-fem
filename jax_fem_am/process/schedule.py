"""Step scheduling predicates (mechanics cadence, output cadence).

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import math


def should_run_mechanics(global_step, args):
    return args.mechanics_every > 0 and global_step % args.mechanics_every == 0


def should_save_step(global_step, did_mechanics, is_last, args):
    if is_last:
        return True
    if did_mechanics and args.mechanics_output_every > 0 and global_step % args.mechanics_output_every == 0:
        return True
    if args.thermal_output_every > 0 and global_step % args.thermal_output_every == 0:
        return True
    return False


def apply_stage_temperature_schedule(
    step_states,
    *,
    process_ambient,
    process_bottom_temperature,
    final_cooldown_temperature=None,
):
    """Freeze per-step ambient and bottom temperatures on ``StepState``.

    Scan, dwell, and recoat states retain their process temperatures.  When a
    final target is supplied, the cooling states follow the same linear
    ``k / N`` ramp for the bottom Dirichlet value and the convection/radiation
    environment, reaching the target on the final cooling increment.
    """

    process_ambient = float(process_ambient)
    process_bottom_temperature = float(process_bottom_temperature)
    if not math.isfinite(process_ambient) or not math.isfinite(
        process_bottom_temperature
    ):
        raise ValueError("stage temperatures must be finite")
    if final_cooldown_temperature is not None:
        final_cooldown_temperature = float(final_cooldown_temperature)
        if not math.isfinite(final_cooldown_temperature):
            raise ValueError("final cooldown temperature must be finite")

    cooling_count = sum(
        1 for state in step_states if state.mode == "cooling"
    )
    cooling_index = 0
    for state in step_states:
        ambient = process_ambient
        bottom_temperature = process_bottom_temperature
        if (
            state.mode == "cooling"
            and final_cooldown_temperature is not None
        ):
            if cooling_count < 1:
                raise ValueError(
                    "a final cooldown schedule requires cooling states"
                )
            cooling_index += 1
            fraction = cooling_index / float(cooling_count)
            ambient = process_ambient + fraction * (
                final_cooldown_temperature - process_ambient
            )
            bottom_temperature = process_bottom_temperature + fraction * (
                final_cooldown_temperature - process_bottom_temperature
            )
        state.ambient_temperature = float(ambient)
        state.bottom_temperature = float(bottom_temperature)
    return step_states
