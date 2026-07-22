"""Step scheduling predicates (mechanics cadence, output cadence).

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""


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
