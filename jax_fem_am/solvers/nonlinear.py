"""Newton solver option helpers for the mechanics solves.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
run_mechanics and run_mechanics_with_cutback intentionally stay in the v03
driver module: v06/driver.py patches base_module.run_mechanics, and the
cutback's bare-name call must keep resolving through the v03 module globals
to hit that patch.
"""


def mechanics_newton_overrides_from_args(args):
    """Newton overrides for the mechanics solves, built from CLI options.

    The legacy defaults (tol=1e-9, rel_tol=1e-11) are far tighter than
    engineering stress accuracy requires and make yield-surface (j2) states
    crawl; line search stabilizes Newton when many quadrature points sit on
    the yield-surface kink.
    """
    overrides = {}
    if args.mechanics_tol is not None:
        overrides["tol"] = float(args.mechanics_tol)
    if args.mechanics_rel_tol is not None:
        overrides["rel_tol"] = float(args.mechanics_rel_tol)
    if args.mechanics_max_iter is not None:
        overrides["max_iter"] = int(args.mechanics_max_iter)
    if args.mechanics_line_search:
        overrides["line_search_flag"] = True
    return overrides
