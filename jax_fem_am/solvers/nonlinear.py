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
    if getattr(args, "mechanics_residual_only_check", False):
        overrides["residual_only_check"] = True
    if getattr(args, "mechanics_acceptance", "legacy") == "abaqus":
        # Abaqus/Standard-style dual criteria (usb 7.2.3): max-norm force
        # residual vs the increment's out-of-balance force scale (0.5%),
        # displacement-correction check (1%), and the linear-convergence
        # fallback (2e-2 after iteration 9). Rationale and provenance:
        # experiments/solver/ABAQUS_SOLVER_NOTES.md (P0).
        overrides["acceptance"] = {
            "force_frac": float(args.mechanics_acceptance_force_frac),
            "disp_frac": float(args.mechanics_acceptance_disp_frac),
            "fallback_frac": float(args.mechanics_acceptance_fallback_frac),
            "fallback_after": int(args.mechanics_acceptance_fallback_after),
        }
    return overrides
