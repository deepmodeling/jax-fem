# V1 — single-track melt-pool triangle verification (decision D-09)

**Goal**: verify the jax-fem solver's moving-heat-source thermal physics by a
three-way comparison on IN625 single laser tracks:

> **our solver  vs  NIST measurement  vs  Balbaa 2022 ABAQUS**

This is a *verification* rung (code-to-code + code-to-experiment at melt-pool
scale). It is NOT the AMB2018-01 validation — the 11-ridge CMM deflection
curve remains the only validation target of the main case.

## Ownership and quarantine (read first)

- This directory is owned by the **V1 session**. The main-line session does not
  edit it; the V1 session does not edit anything outside it (both share branch
  `test`; do not push).
- **Input quarantine**: reproducing Balbaa requires using *his* inputs
  (Table 1 thermophysical values, A = 0.62, Sih–Barlow powder model, phi = 0.4,
  epsilon = 0.4) even where they conflict with the main case's D-01/D-08
  sources. These values must never leak into the main-case inputs, and vice
  versa. Every V1 input file carries a `"provenance": "Balbaa2022 Table N"` or
  `"provenance": "NIST measurement"` field.
- Zero-calibration discipline applies: nothing is tuned to match either the
  NIST measurement or the ABAQUS numbers; discrepancies are reported, not fitted.

## Reference documents (already in the repo)

- `../../references/docs/Balbaa2022_multiscale-RS-LPBF-IN625_JMMP-6-2.pdf`
  (41 pp, CC BY 4.0, DOI 10.3390/jmmp6010002; aminer cache copy, sha256
  94c51f9c...b75; MDPI direct download returns 403 from this network)
- Balbaa's single-track model (paper Sec. 2, Figs. 7–10): exponential
  volumetric heat source with optical penetration depth 100 um (interpolated
  from pure-Ni data, Table 2), DRS-measured absorptivity 0.62 @ 1070 nm,
  powder-bed conductivity via Sih–Barlow (Eq. 12–13), powder cp/rho scaled by
  (1 - phi), phi = 0.4, emissivity 0.4, Table 1 bulk properties, solidus
  1290 C used as the melt-pool boundary. Predicted width 131 um (exponential
  source) vs experiment.
- Balbaa validated against: [68] Lane et al., "Measurements of melt pool
  geometry and cooling rates of individual laser traces on IN625 bare plates",
  IMMI (AMB2018-02 series) and [69] Dilip et al. 2017 (tracks with powder).
  NOTE: resolve carefully which cases are bare-plate and which have a powder
  layer — Balbaa modified his geometry to a 20 um layer for these comparisons.

## Work plan

1. **Transcribe** (machine-readable, with provenance fields):
   - `inputs/balbaa-model.json` — Table 1 (thermophysical), Table 2 (OPD),
     Table 3 (process parameters), heat-source definition, BCs, mesh/layer
     description as printed. Register anything the paper does not specify in
     `inputs/deviations.yaml` (Kaess discipline: deviation registered, not
     silently filled).
   - `inputs/nist-meltpool.json` — acquire the Lane et al. paper and/or the
     AMB2018-02 measurement tables from the NIST portal (NIST was reachable
     from this network on 2026-07-28); record laser power / speed cases, melt
     pool width/depth/length and cooling rates with uncertainties.
2. **Run** our solver on the same single-track configuration(s):
   entry point `python -m jax_fem_am.simulation.runner` (WSL conda env
   `jax-fem-env`); the hemispherical heat source exists as
   `source_model=paper_hemispherical` (thermal.py); an exponential-depth
   volumetric source may need implementing — if so it lands in solver code,
   coordinate with the main session before touching shared modules (solver
   code is shared; only this cases/ subdirectory is exclusive).
3. **Compare** (the triangle): melt pool width / depth / (length if measured)
   and cooling rate per case: ours vs NIST measured (with uncertainty) vs
   Balbaa's reported prediction. Report relative errors both ways; no fitting.
4. **Report**: `RESULTS.md` — tables + the honest discrepancy list.

## Acceptance framing

There is no pass/fail threshold to tune toward. Success =
(a) our thermal solver reproduces measured melt-pool dimensions with error
comparable to or better than Balbaa's ABAQUS (he reports ~131 um width vs
experiment at matching conditions), and (b) any systematic deviation is
explained and registered. If our physics disagrees strongly, that is a
finding about the solver, exactly what V1 exists to expose.

## Status

- 2026-07-29: workspace created, Balbaa PDF archived, spec written (main
  session). Transcription not started. NIST melt-pool data not yet acquired.
- 2026-07-29 (V1 session): transcription done (inputs/balbaa-model.json,
  inputs/nist-meltpool.json, deviations.yaml D-V1-01..20). Lane2020 acquired
  via Wayback copy of NIST TechPubs (nist.gov unreachable today, D-V1-13).
  Balbaa validation condition identified as AMB2018-02 CBM case B
  (195 W / 800 mm/s, spot D4sigma 100 um): his Exp[68] bars match Lane
  Table 4 CBM-B exactly. Solver needs NO shared-code change: source_model
  =legacy is Eq-18-identical (energy-ledger verified). One shared-code bug
  found and worked around, NOT fixed here (stepper.py:966 crashes with
  --mechanics-every 0 + thermal-only surrogate; flag
  --no-xla-thermal-only-mechanics-surrogate avoids it) -- main session to fix.
  Transcription finding: printed Sih-Barlow (Eq 12-13) is ~2x below the
  paper's own Fig 1 powder curve; Fig 1 digitization used (D-V1-20).
  CBM-B thermal run in progress.
- 2026-07-29 (V1 session, later): ALL RUNS DONE, RESULTS.md written.
  Triangle at CBM-B: width 141.0 um (NIST 133, Balbaa 131), depth below
  substrate top 90.6 um (NIST 91, Balbaa 44), length 575 um (NIST 780,
  Balbaa 378), CR 1290->1190C 2.22e6 (NIST 9.35e5), CR 1290->1000C 1.05e6
  (NIST ~9.5e5). Two-way A/C runs also done (depth exact at C, -24% at A =
  keyhole-regime signature). Sensitivity: rho bounding and powder-k
  variants both immaterial (<=4%). Remaining open: D-V1-13 (data.nist.gov
  diff when reachable), cp_powder double-count sensitivity (D-V1-15).
