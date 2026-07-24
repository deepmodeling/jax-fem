"""CLI argument schema (argparse parser) for the AM thermal-stress driver.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import argparse

from jax_fem_am.config.loaders import cfg, parse_scalar, read_config


def build_parser(config=None):
    config = config or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--inp", default=cfg(config, "inp", "/home/user/work/159/schema/0119_c3d4_only.inp"))
    parser.add_argument("--max-cells", type=int, default=cfg(config, "max_cells", 0))
    parser.add_argument("--mesh-length-scale", type=float, default=cfg(config, "mesh_length_scale", 1.0))
    parser.add_argument("--path-length-scale", type=float, default=cfg(config, "path_length_scale", None), help="Scale applied to x/y/z coordinates in --path-file. Defaults to --mesh-length-scale.")

    parser.add_argument("--layers", type=int, default=cfg(config, "layers", 50))
    parser.add_argument("--steps", type=int, default=cfg(config, "steps", None), help="Backward-compatible alias for --layers.")
    parser.add_argument("--layer-thickness", type=float, default=cfg(config, "layer_thickness", None), help="Physical layer thickness in scaled mesh units. If provided, it overrides --layers.")
    parser.add_argument("--max-print-layers", type=int, default=cfg(config, "max_print_layers", None), help="Limit printed layers after physical layer-thickness expansion. Useful for debugging full-size builds with realistic layer thickness.")
    parser.add_argument("--active-window-below-layers", type=int, default=cfg(config, "active_window_below_layers", 0), help="Moving thermal active window below current layer. 0 keeps cumulative activation. If 10, current layer L uses layers max(1,L-10)..L as the thermal window.")
    parser.add_argument("--old-layer-thermal-factor", type=float, default=cfg(config, "old_layer_thermal_factor", 1e-6), help="Conductivity multiplier for printed layers that fall below the moving thermal window.")
    parser.add_argument("--old-layer-cooling-h", type=float, default=cfg(config, "old_layer_cooling_h", 0.0), help="Optional volumetric cooling coefficient W/(m^3*K) for printed layers below the moving thermal window. 0 disables this sink.")
    parser.add_argument("--scan-steps-per-layer", type=int, default=cfg(config, "scan_steps_per_layer", 20))
    parser.add_argument("--hatch-lines-per-layer", type=int, default=cfg(config, "hatch_lines_per_layer", 1))
    parser.add_argument("--hatch-spacing", type=float, default=cfg(config, "hatch_spacing", None), help="Physical hatch spacing in the build plane. If provided, it overrides --hatch-lines-per-layer.")
    parser.add_argument("--dt", type=float, default=cfg(config, "dt", 1e-4))

    parser.add_argument("--ambient", type=float, default=cfg(config, "ambient", 300.0))
    parser.add_argument("--preheat-temperature", type=float, default=cfg(config, "preheat_temperature", None))
    parser.add_argument("--bottom-temperature", type=float, default=cfg(config, "bottom_temperature", None), help="Fixed bottom temperature. Defaults to preheat temperature when provided, otherwise ambient.")
    parser.add_argument("--rho", type=float, default=cfg(config, "rho", 7800.0))
    parser.add_argument("--cp", type=float, default=cfg(config, "cp", 500.0))
    parser.add_argument("--conductivity", type=float, default=cfg(config, "conductivity", 20.0))
    parser.add_argument("--rho-solid", type=float, default=cfg(config, "rho_solid", None))
    parser.add_argument("--cp-solid", type=float, default=cfg(config, "cp_solid", None))
    parser.add_argument("--conductivity-solid", type=float, default=cfg(config, "conductivity_solid", None))
    parser.add_argument("--rho-powder", type=float, default=cfg(config, "rho_powder", 3900.0))
    parser.add_argument("--cp-powder", type=float, default=cfg(config, "cp_powder", 500.0))
    parser.add_argument("--conductivity-powder", type=float, default=cfg(config, "conductivity_powder", 1.0))
    parser.add_argument("--rho-liquid", type=float, default=cfg(config, "rho_liquid", None), help="Liquid density. Defaults to solid density when omitted.")
    parser.add_argument("--cp-liquid", type=float, default=cfg(config, "cp_liquid", None), help="Liquid heat capacity. Defaults to solid heat capacity when omitted.")
    parser.add_argument("--conductivity-liquid", type=float, default=cfg(config, "conductivity_liquid", None), help="Liquid thermal conductivity. Defaults to solid conductivity when omitted.")
    parser.add_argument("--powder-mode", choices=("powder", "void"), default=cfg(config, "powder_mode", "powder"))
    parser.add_argument("--layer-activation-mode", choices=("front", "layer_on_scan"), default=cfg(config, "layer_activation_mode", "layer_on_scan"), help="Layer activation model. 'front' keeps the old front-coordinate activation; 'layer_on_scan' activates the whole current layer when laser scanning starts, mimicking recoating/powder spreading.")
    parser.add_argument("--future-layer-mode", choices=("void", "powder"), default=cfg(config, "future_layer_mode", "void"), help="Material treatment for future, not-yet-spread layers when --layer-activation-mode layer_on_scan is used. Use 'void' to make future layers inactive before spreading.")
    parser.add_argument("--layer-activation-geometry", choices=("centroid", "intersection"), default=cfg(config, "layer_activation_geometry", "intersection"), help="How cells are assigned to printed layers. 'centroid' uses cell centroid layer id; 'intersection' activates a cell if its vertex interval intersects the layer band. Use intersection for coarse tetra meshes and macro-layer runs.")
    parser.add_argument("--inactive-thermal-factor", type=float, default=cfg(config, "inactive_thermal_factor", 1e-6))
    parser.add_argument("--inactive-mass-factor", type=float, default=cfg(config, "inactive_mass_factor", None),
                        help="Density scaling for inactive/void cells. Legacy behavior (None) reuses "
                             "--inactive-thermal-factor, which scales k and rho by the SAME factor and therefore leaves "
                             "the void diffusivity k/(rho*cp) at solid-like values: temperature 'ghost-diffuses' 5-10 mm "
                             "into un-spread layers above the laser. Set 1.0 to keep full thermal mass in void (only k "
                             "is reduced), cutting void diffusivity by the k factor and anchoring the otherwise "
                             "near-singular void equations.")
    parser.add_argument("--inactive-mechanics-factor", type=float, default=cfg(config, "inactive_mechanics_factor", 1e-9))

    parser.add_argument("--k-table-solid", default=cfg(config, "k_table_solid", None))
    parser.add_argument("--cp-table-solid", default=cfg(config, "cp_table_solid", None))
    parser.add_argument("--k-table-powder", default=cfg(config, "k_table_powder", None))
    parser.add_argument("--cp-table-powder", default=cfg(config, "cp_table_powder", None))
    parser.add_argument("--k-table-liquid", default=cfg(config, "k_table_liquid", None))
    parser.add_argument("--cp-table-liquid", default=cfg(config, "cp_table_liquid", None))
    parser.add_argument("--solidus-temperature", type=float, default=cfg(config, "solidus_temperature", 0.0))
    parser.add_argument("--liquidus-temperature", type=float, default=cfg(config, "liquidus_temperature", 0.0))
    parser.add_argument("--latent-heat", type=float, default=cfg(config, "latent_heat", 0.0))

    parser.add_argument("--convection-h", type=float, default=cfg(config, "convection_h", 10.0))
    parser.add_argument("--emissivity", type=float, default=cfg(config, "emissivity", 0.0))
    parser.add_argument("--stefan-boltzmann", type=float, default=cfg(config, "stefan_boltzmann", 5.670374419e-8))
    parser.add_argument("--bottom-thermal-bc", choices=("fixed", "convection"), default=cfg(config, "bottom_thermal_bc", "fixed"))
    parser.add_argument("--surface-selection", choices=("box", "exterior"), default=cfg(config, "surface_selection", "box"),
                        help="How convection/radiation faces are selected. 'box' keeps the legacy bounding-box plane selectors "
                             "(only faces exactly on the box planes; zero faces for curved parts). 'exterior' applies the "
                             "surface losses to every mesh-exterior face above the base plane, matching a real part surrounded "
                             "by gas/powder.")
    parser.add_argument("--boundary-tol", type=float, default=cfg(config, "boundary_tol", None),
                        help="Absolute tolerance (mesh length units) for base/exposed/wall plane node selection. Defaults to "
                             "the legacy 1e-8 * bounding-box span, which can miss nodes on real CAD meshes whose base face "
                             "has sub-mm jitter.")
    parser.add_argument("--surface-active-mask", dest="surface_active_mask", action="store_true",
                        default=cfg(config, "surface_active_mask", None),
                        help="Restrict convection/radiation to faces owned by printed material. Void cells are not physical "
                             "surfaces and their near-singular equations produce unphysical temperatures under surface flux. "
                             "Defaults to on for --surface-selection exterior, off for legacy box mode.")
    parser.add_argument("--no-surface-active-mask", dest="surface_active_mask", action="store_false")
    parser.add_argument("--stress-relaxation-temperature", type=float,
                        default=cfg(config, "stress_relaxation_temperature", None),
                        help="Stress-free reference temperature written when material solidifies (macro calibration knob; "
                             "Ti64 typically 1073-1173 K). Without it, T_ref is the local temperature at solidification, "
                             "which in consolidation-on-activation mode equals the powder entry temperature and inverts the "
                             "residual stress sign.")
    parser.add_argument("--quadrature-order", type=int, default=cfg(config, "quadrature_order", None),
                        help="Quadrature order for both thermal and mechanical problems (they must match: phase/material "
                             "state arrays are shared per quadrature point). TET4 defaults to the legacy single-point rule, "
                             "whose rank-1 mass matrix produces large spurious temperature oscillations (observed +-1900K) "
                             "in low-conductivity powder at small Fourier numbers. Use 2 (4-point rule) for a full-rank "
                             "transient term.")
    parser.add_argument("--front-surface-loss-h", type=float, default=cfg(config, "front_surface_loss_h", 0.0), help="Optional volumetric approximation of convection on the moving build front. 0 disables it.")
    parser.add_argument("--front-surface-loss-thickness", type=float, default=cfg(config, "front_surface_loss_thickness", 0.0), help="Thickness for moving-front loss approximation. Defaults to source_depth when front loss is enabled.")
    parser.add_argument("--front-surface-loss-radiation", dest="front_surface_loss_radiation", action="store_true", default=cfg(config, "front_surface_loss_radiation", False))
    parser.add_argument("--no-front-surface-loss-radiation", dest="front_surface_loss_radiation", action="store_false")

    parser.add_argument("--young", type=float, default=cfg(config, "young", 2.0e11))
    parser.add_argument("--poisson", type=float, default=cfg(config, "poisson", 0.3))
    parser.add_argument("--alpha", type=float, default=cfg(config, "alpha", 1.2e-5))
    parser.add_argument("--E-table", dest="E_table", default=cfg(config, "E_table", None))
    parser.add_argument("--alpha-table", default=cfg(config, "alpha_table", None))
    parser.add_argument("--poisson-table", default=cfg(config, "poisson_table", None))
    parser.add_argument("--yield-table", default=cfg(config, "yield_table", None))
    parser.add_argument("--hardening-table", default=cfg(config, "hardening_table", None))
    parser.add_argument("--mechanics-model", choices=("linear_elastic", "j2_plastic"), default=cfg(config, "mechanics_model", "linear_elastic"))
    parser.add_argument("--yield-saturation-stress", type=float, default=cfg(config, "yield_saturation_stress", None),
                        help="Cap on the hardened yield stress (Pa), ~UTS (Ti64: ~1.15e9). Linear isotropic hardening "
                             "extrapolated past its ~10%% strain validity produced ~2 GPa fictitious von Mises at the "
                             "bottom-clamp region; the cap saturates hardening there. None keeps unbounded legacy hardening.")
    parser.add_argument("--bottom-mechanics-bc", choices=("fixed", "elastic", "paper_minimal"),
                        default=cfg(config, "bottom_mechanics_bc", "fixed"),
                        help="'fixed' rigidly clamps the base nodes (legacy; models an infinitely stiff build plate and "
                             "concentrates fictitious stress at the clamp edge). 'elastic' replaces the clamp with a "
                             "Winkler elastic foundation on the base faces. 'paper_minimal' restrains every bottom node "
                             "only in the build direction and adds three deterministic in-plane scalar restraints to "
                             "remove rigid motion while permitting thermal contraction (Kaess 2023 Section 2.3).")
    parser.add_argument("--paper-minimal-anchor-corner",
                        choices=("min_min", "max_min", "max_max", "min_max"),
                        default=cfg(config, "paper_minimal_anchor_corner", "min_min"),
                        help="Bottom-plane corner used by --bottom-mechanics-bc paper_minimal. The paper does not "
                             "publish exact in-plane anchor nodes; the four deterministic variants support the "
                             "required anchor-sensitivity study.")
    parser.add_argument("--bottom-foundation-stiffness", type=float, default=cfg(config, "bottom_foundation_stiffness", 1.0e12),
                        help="Foundation modulus k_s (Pa/m) for --bottom-mechanics-bc elastic. Calibration knob for the "
                             "build-plate compliance; 1e12 approximates a ~25 mm steel plate, larger values approach the "
                             "rigid clamp.")
    parser.add_argument("--powder-mechanics-bc", choices=("none", "elastic"), default=cfg(config, "powder_mechanics_bc", "none"),
                        help="'none' keeps free lateral surfaces (legacy: the surrounding powder bed constrains nothing). "
                             "'elastic' adds horizontal-only Winkler springs on the exterior side faces of printed "
                             "material, modeling lateral support from the unmelted powder bed; springs follow the "
                             "printed state per step and are absent from the release solve (de-powdering). "
                             "Requires --surface-selection exterior.")
    parser.add_argument("--powder-foundation-stiffness", type=float, default=cfg(config, "powder_foundation_stiffness", 1.0e9),
                        help="Foundation modulus k_p (Pa/m) for --powder-mechanics-bc elastic. Calibration knob: "
                             "k_p ~ E_powder / L_embed with loose Ti64 powder E ~ 1-100 MPa and mm-scale embedment "
                             "gives ~1e8-1e11; default 1e9 is a soft support ~3 orders below the build plate.")
    parser.add_argument("--mushy-mechanics-factor", type=float, default=cfg(config, "mushy_mechanics_factor", 1e-2), help="Stress/stiffness scaling for mushy-zone material.")
    parser.add_argument("--liquid-mechanics-factor", type=float, default=cfg(config, "liquid_mechanics_factor", 1e-4), help="Stress/stiffness scaling for liquid material.")
    parser.add_argument("--reset-plastic-on-melt", dest="reset_plastic_on_melt", action="store_true", default=cfg(config, "reset_plastic_on_melt", True))
    parser.add_argument("--no-reset-plastic-on-melt", dest="reset_plastic_on_melt", action="store_false")

    parser.add_argument("--laser-power", type=float, default=cfg(config, "laser_power", 1.0))
    parser.add_argument("--absorptivity", type=float, default=cfg(config, "absorptivity", 0.35))
    parser.add_argument(
        "--source-model",
        choices=("legacy", "paper_hemispherical"),
        default=cfg(config, "source_model", "legacy"),
        help=(
            "Volumetric laser source: legacy uses an in-plane Gaussian with "
            "exponential depth decay; paper_hemispherical uses Kaess (2023) "
            "Equation (1)."
        ),
    )
    parser.add_argument("--beam-radius", type=float, default=cfg(config, "beam_radius", 1.0e-4))
    parser.add_argument("--source-depth", type=float, default=cfg(config, "source_depth", 6.0e-5))

    parser.add_argument("--build-axis", choices=("x", "y", "z"), default=cfg(config, "build_axis", "x"))
    parser.add_argument("--base-side", choices=("min", "max"), default=cfg(config, "base_side", "min"))
    parser.add_argument("--scan-axis", choices=("auto", "x", "y", "z"), default=cfg(config, "scan_axis", "auto"))
    parser.add_argument("--scan-pattern", choices=("raster",), default=cfg(config, "scan_pattern", "raster"))
    parser.add_argument("--scan-rotation-per-layer", type=float, default=cfg(config, "scan_rotation_per_layer", 0.0), help="Rotation angle in degrees applied inside the build plane for each new layer.")
    parser.add_argument("--scan-start", type=float, default=cfg(config, "scan_start", None))
    parser.add_argument("--scan-end", type=float, default=cfg(config, "scan_end", None))
    parser.add_argument("--scan-start-frac", type=float, default=cfg(config, "scan_start_frac", 0.05))
    parser.add_argument("--scan-end-frac", type=float, default=cfg(config, "scan_end_frac", 0.95))
    parser.add_argument("--hatch-start", type=float, default=cfg(config, "hatch_start", None))
    parser.add_argument("--hatch-end", type=float, default=cfg(config, "hatch_end", None))
    parser.add_argument("--hatch-start-frac", type=float, default=cfg(config, "hatch_start_frac", 0.05))
    parser.add_argument("--hatch-end-frac", type=float, default=cfg(config, "hatch_end_frac", 0.95))
    parser.add_argument("--hatch-fixed", type=float, default=cfg(config, "hatch_fixed", None))
    parser.add_argument("--serpentine", dest="serpentine", action="store_true", default=cfg(config, "serpentine", True))
    parser.add_argument("--no-serpentine", dest="serpentine", action="store_false")
    parser.add_argument("--scan-speed", type=float, default=cfg(config, "scan_speed", 0.0))
    parser.add_argument("--auto-scan-steps-from-speed", dest="auto_scan_steps_from_speed", action="store_true", default=cfg(config, "auto_scan_steps_from_speed", False))
    parser.add_argument("--no-auto-scan-steps-from-speed", dest="auto_scan_steps_from_speed", action="store_false")
    parser.add_argument("--dwell-steps-between-layers", type=int, default=cfg(config, "dwell_steps_between_layers", 0))
    parser.add_argument("--dwell-steps-between-hatches", type=int, default=cfg(config, "dwell_steps_between_hatches", 0))
    parser.add_argument("--jump-speed", type=float, default=cfg(config, "jump_speed", 0.0), help="Laser-off travel speed between hatch lines. 0 disables generated jump states.")
    parser.add_argument("--recoat-time", type=float, default=cfg(config, "recoat_time", 0.0))
    parser.add_argument("--recoat-steps", type=int, default=cfg(config, "recoat_steps", 10),
                        help="Number of implicit time steps used to span each recoat interval (dt = recoat_time / recoat_steps). "
                             "Applies to both the raster generator and layer transitions in --path-file runs.")
    parser.add_argument("--path-file", default=cfg(config, "path_file", None))
    parser.add_argument("--path-output", default=cfg(config, "path_output", "path_used.csv"), help="CSV file name/path for the actual sampled path. Set empty string to disable.")

    parser.add_argument("--substrate-thickness", type=float, default=cfg(config, "substrate_thickness", 0.0))
    parser.add_argument("--support-thickness", type=float, default=cfg(config, "support_thickness", 0.0))
    parser.add_argument("--cooling-steps", type=int, default=cfg(config, "cooling_steps", 0))
    parser.add_argument("--cooling-dt", type=float, default=cfg(config, "cooling_dt", None),
                        help="Time step for the trailing cooling states. Defaults to --dt; a larger value lets the part "
                             "actually cool to ambient before release instead of only spanning cooling_steps * dt seconds.")
    parser.add_argument("--reset-activation-temperature", dest="reset_activation_temperature", action="store_true",
                        default=cfg(config, "reset_activation_temperature", False),
                        help="Reset nodal temperatures of newly activated cells (nodes not shared with previously active "
                             "material) to the preheat/ambient temperature, modeling freshly spread powder.")
    parser.add_argument("--no-reset-activation-temperature", dest="reset_activation_temperature", action="store_false")
    parser.add_argument("--activation-reset-temperature", type=float,
                        default=cfg(config, "activation_reset_temperature", None),
                        help="Temperature assigned to newly activated nodes when --reset-activation-temperature is on. "
                             "Defaults to the preheat/ambient temperature. In consolidation-on-activation (route B) runs, "
                             "set it to the stress relaxation temperature: the fresh layer enters hot and stress-free and "
                             "builds stress gradually while cooling, instead of receiving an instantaneous GPa-level "
                             "thermal load step that stalls the mechanics Newton solve.")
    parser.add_argument("--release-after-cooling", dest="release_after_cooling", action="store_true", default=cfg(config, "release_after_cooling", False))
    parser.add_argument("--no-release-after-cooling", dest="release_after_cooling", action="store_false")
    parser.add_argument("--release-anchor-mode",
                        choices=("rigid_body", "box", "paper_minimal_root"),
                        default=cfg(config, "release_anchor_mode", "rigid_body"),
                        help="rigid_body (default): free-free release with 3-point rigid-body anchors (full removal "
                             "from the plate). box: clamp all nodes inside --release-anchor-box (u=0), modeling a "
                             "partial EDM/saw cut that leaves a root attachment. paper_minimal_root: restrain the "
                             "retained release-root bottom only in the build direction plus three in-plane rigid "
                             "DOFs (formal Kaess 2023 cantilever semantics; requires --release-cell-set).")
    parser.add_argument("--release-anchor-box", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        default=cfg(config, "release_anchor_box", None),
                        help="Axis-aligned box (mesh coordinates) whose nodes stay clamped during a box-mode release.")
    parser.add_argument("--release-cut-box", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        default=cfg(config, "release_cut_box", None),
                        help="Cells whose centroid lies inside this box (mesh coordinates) are deactivated in the "
                             "release solve (stiffness AND locked-in stress scaled by the inactive factor) - the "
                             "equivalent of deleting sawed-off support elements (Kaess 2023 Fig 7 semantics).")
    parser.add_argument("--release-cell-set",
                        default=cfg(config, "release_cell_set", None),
                        help="Content-addressed JSON artifact containing the exact zero-based solver cells "
                             "removed for release. Mutually exclusive with --release-cut-box; formal "
                             "paper runs must use this option instead of an unverified geometric box.")
    parser.add_argument("--final-cooldown-temperature", type=float,
                        default=cfg(config, "final_cooldown_temperature", None),
                        help="Ramp the fixed bottom temperature linearly to this value (K) across the final cooling "
                             "steps, modeling a build-plate cooldown to room temperature before release "
                             "(Kaess 2023 style). Requires --bottom-thermal-bc fixed; default keeps the bottom "
                             "temperature constant.")

    parser.add_argument("--mechanics-every", type=int, default=cfg(config, "mechanics_every", 1))
    parser.add_argument("--mechanics-tol", type=float, default=cfg(config, "mechanics_tol", None),
                        help="Absolute Newton residual tolerance for mechanics solves (default keeps legacy 1e-9).")
    parser.add_argument("--mechanics-rel-tol", type=float, default=cfg(config, "mechanics_rel_tol", None),
                        help="Relative Newton residual tolerance for mechanics solves (default keeps legacy 1e-11; "
                             "1e-6 is plenty for engineering stress accuracy and dramatically faster for j2 states).")
    parser.add_argument("--mechanics-max-iter", type=int, default=cfg(config, "mechanics_max_iter", None),
                        help="Newton iteration cap for mechanics solves (solver default 100).")
    parser.add_argument("--mechanics-line-search", dest="mechanics_line_search", action="store_true",
                        default=cfg(config, "mechanics_line_search", False),
                        help="Enable Newton line search for mechanics solves; stabilizes j2 yield-surface states.")
    parser.add_argument("--no-mechanics-line-search", dest="mechanics_line_search", action="store_false")
    parser.add_argument("--mechanics-residual-only-check",
                        dest="mechanics_residual_only_check", action="store_true",
                        default=cfg(config, "mechanics_residual_only_check", False),
                        help="After each mechanics Newton correction, check the residual before rebuilding "
                             "the tangent. If the acceptance criteria pass, the final unused jacfwd tangent "
                             "assembly is skipped. Opt-in to preserve legacy mechanics behavior by default.")
    parser.add_argument("--no-mechanics-residual-only-check",
                        dest="mechanics_residual_only_check", action="store_false")
    parser.add_argument("--mechanics-acceptance", choices=("legacy", "abaqus"),
                        default=cfg(config, "mechanics_acceptance", "legacy"),
                        help="Newton acceptance criteria for mechanics solves. 'legacy' = "
                             "relative/absolute residual test (bitwise-preserving default). 'abaqus' = "
                             "hybrid strict-residual OR Abaqus/Standard-style dual criteria: "
                             "configured tol/rel_tol remain a conservative acceptance exit; "
                             "otherwise use max-norm force residual vs the "
                             "increment's out-of-balance force scale, displacement-correction check, "
                             "and a linear-convergence fallback - accepts the j2 stall-floor and "
                             "near-perfectly-plastic powder states that the reference solver treats "
                             "as converged (see experiments/solver/ABAQUS_SOLVER_NOTES.md).")
    parser.add_argument("--mechanics-acceptance-force-frac", type=float,
                        default=cfg(config, "mechanics_acceptance_force_frac", 0.005))
    parser.add_argument("--mechanics-acceptance-disp-frac", type=float,
                        default=cfg(config, "mechanics_acceptance_disp_frac", 0.01))
    parser.add_argument("--mechanics-acceptance-fallback-frac", type=float,
                        default=cfg(config, "mechanics_acceptance_fallback_frac", 0.02))
    parser.add_argument("--mechanics-acceptance-fallback-after", type=int,
                        default=cfg(config, "mechanics_acceptance_fallback_after", 9))
    parser.add_argument("--mechanics-temperature-floor", type=float,
                        default=cfg(config, "mechanics_temperature_floor", None),
                        help="Clamp the temperature seen by the mechanics chain (thermal strain and "
                             "material tables) to at least this value in K. Guard against activation "
                             "undershoot artifacts (G1) feeding sub-physical temperatures into full-stiffness "
                             "solid; a mitigation knob, not a fix — the thermal field itself stays unclamped.")
    parser.add_argument("--powder-elset", default=cfg(config, "powder_elset", None),
                        help="Name of an inp ELSET whose cells are PERMANENT powder: excluded from "
                             "substrate/support classification and from printing, thermally active "
                             "as powder from step 0. Combine with --powder-solid-E to make them "
                             "mechanically load-bearing (weak-solid powder, Kaess 2023 convention).")
    parser.add_argument("--powder-solid-E", type=float,
                        default=cfg(config, "powder_solid_E", None),
                        help="Weak-solid powder Young's modulus in Pa (Kaess 2023: 10e9). When set "
                             "(requires --powder-elset), permanent-powder cells join the mechanics "
                             "active set with this constant E, --powder-solid-yield, zero hardening "
                             "and zero thermal expansion; they are deactivated (depowdered) for the "
                             "release solve. None keeps legacy behavior (powder carries no load).")
    parser.add_argument("--powder-solid-yield", type=float,
                        default=cfg(config, "powder_solid_yield", 1.0e6),
                        help="Weak-solid powder yield stress in Pa (Kaess 2023: 1e6).")
    parser.add_argument("--powder-solid-hardening", type=float,
                        default=cfg(config, "powder_solid_hardening", 1.0e7),
                        help="Weak-solid powder hardening modulus in Pa. The reference model is "
                             "ideally plastic (H=0), but H=0 over ~30k permanently-yielded powder "
                             "cells makes the consistent tangent semi-definite and Newton stalls at "
                             "~1e-3 relative regardless of increment cutback (observed). The default "
                             "0.1%% of E adds <1 MPa at 10%% strain - a documented regularization "
                             "deviation, not physics.")
    parser.add_argument("--thermal-mass-lumping", action="store_true",
                        default=cfg(config, "thermal_mass_lumping", False),
                        help="Evaluate the thermal transient/source (mass-map) terms with the TET4 "
                             "vertex quadrature rule instead of the interior Gauss points. For linear "
                             "tets this is exactly row-sum capacitance lumping (Abaqus first-order "
                             "heat-transfer element behavior): the capacitance matrix becomes diagonal, "
                             "restoring the discrete maximum principle that suppresses activation "
                             "undershoot; conduction is unaffected (constant gradients). Requires "
                             "--quadrature-order 2 (4 equal-weight points).")
    parser.add_argument("--mechanics-bbar", choices=("auto", "on", "off"),
                        default=cfg(config, "mechanics_bbar", "auto"),
                        help="B-bar (element-average volumetric strain) for the mechanics element. "
                             "auto = on for HEX8, off for TET4. On HEX8 this is the Abaqus C3D8 "
                             "selective-reduced-integration behavior and prevents volumetric locking "
                             "under J2 flow (checkerboard hydrostatic pressure). On TET4 the strain "
                             "is element-constant, so B-bar is an exact no-op - TET4 locking needs a "
                             "hex mesh (or nodal averaging, not implemented).")
    parser.add_argument("--mechanics-max-cuts", type=int,
                        default=cfg(config, "mechanics_max_cuts", 0),
                        help="Abaqus-style automatic increment cutback for mechanics solves: on Newton "
                             "failure retry the thermal-load increment in 2,4,...,2**N equal substeps "
                             "(temperature interpolated from the last accepted mechanics state). Substeps "
                             "are pure Newton continuation - the final substep solves the exact original "
                             "problem and no plastic state is committed in between. 0 disables (legacy).")
    parser.add_argument("--thermal-output-every", type=int, default=cfg(config, "thermal_output_every", 0))
    parser.add_argument("--mechanics-output-every", type=int, default=cfg(config, "mechanics_output_every", 1))
    parser.add_argument("--summary-every", type=int, default=cfg(config, "summary_every", 1))
    parser.add_argument("--calibration-dir", default=cfg(config, "calibration_dir", None))
    parser.add_argument("--output-dir", default=cfg(config, "output_dir", "/home/user/work/159/output/inp_thermal_stress_oneway_layers"))
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = read_config(config_args.config)
    parser = build_parser(config)
    args = parser.parse_args()
    args.config = config_args.config
    return args
