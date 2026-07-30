# V2 — cube residual-stress code-to-code verification (decision D-09)

**Goal**: verify the jax-fem THERMO-MECHANICAL chain by reproducing Balbaa
2022's part-scale models and comparing:

> **our solver  vs  Balbaa 2022 ABAQUS prediction  vs  Balbaa's XRD measurement**

on IN625 cube coupons (in-depth residual-stress profiles). This is the
mechanics companion of V1 (which verified the thermal chain; see
`../v1-single-track/RESULTS.md`).

## Ownership and quarantine

- This directory is owned by the **V session**; the main-line session does not
  edit it, and V2 edits nothing outside it (shared branch `test`, no push).
- **Input quarantine** (same rule as V1): Balbaa's inputs are used verbatim
  here and never enter the main case; main-case inputs (D-01/D-08 genealogy)
  never enter V runs. NOTE an edge case now exists: the main case transcribed
  the Kaschnitz IJT 2019 property tables (commit f0ca038). Balbaa cites the
  Kaschnitz-family papers as his [49], so using those tables in V2 arguably
  IMPROVES parity — but it crosses the quarantine line as written. Registered
  as D-V2-04; **requires user approval before any V2 run consumes them**;
  default remains Balbaa-as-printed reconstruction (V1 approach).
- Zero-calibration: nothing is tuned toward the XRD, pyrometer, or ABAQUS
  numbers; discrepancies are reported, not fitted.

## Gate order (run sequence)

1. **Thermal gate** — multi-track model vs the two-color pyrometer trace
   (220 W, 650 mm/s, hatch 0.12 mm, layer 0.04 mm; elements > 1000 C inside a
   2 mm circle averaged at 10 ms increments). Pass/fail is not tuned; the
   gate exists so that a later RS discrepancy can be attributed to the
   mechanical chain rather than the thermal one.
2. **RS gate** — multi-layer cube model vs XRD in-depth profiles:
   two samples (140 W and 220 W, both 650 mm/s, hatch 0.12 mm), depth
   0 -> 1 mm in 0.1 mm steps, scan- and hatch-direction stresses, each value
   averaged over a 2 mm diameter spot (use the existing `xrd_vtu` gauge
   operator with exactly this protocol). Plus surface RS from Balbaa's [50]
   as secondary.

## Mesh parity principle (differs from V1 deliberately)

V1 could take a finer-than-Balbaa mesh (D-V1-14) because melt-pool geometry
converges monotonically with resolution. **V2 cannot**: part-scale RS is
sensitive to discretization (layer lumping, stress averaging volumes,
activation granularity), so code-to-code parity requires reproducing
Balbaa's meshes, initial conditions, and stepping as printed — deviations
only where his spec is under-determined, each registered. The meshes are
therefore specified separately and exactly, in `model/`:

### M1 — multi-track mesh (Balbaa Sec 2.6.2, Fig 3)
- powder layer 4 x 4 x 0.04 mm, DC3D8-equivalent C3D8, uniform 40 um
  (100 x 100 x 1 = 10,000 cells);
- substrate 4 x 4 x 0.4 mm (thickness = 10 x layer, footprint flush per
  Fig 3), vertically graded 40 -> 140 um. Grading law unstated ->
  geometric 5-sublayer ladder scaled to 400 um total:
  [38.8, 53.1, 72.7, 99.4, 136.0] um top->bottom (D-V2-06);
- in-plane 40 um everywhere: Balbaa tie-constrains a (possibly laterally
  coarser) substrate; our runner needs one conforming mesh, so we mesh the
  substrate conforming at 40 um in-plane (finer-or-equal, no tie; D-V2-05);
- total 60,000 cells (`make_v2_mesh_multitrack.py`).

### M2 — multi-layer cube mesh (Balbaa Sec 2.6.3, Fig 4)
- part 10 x 10 x 10 mm, uniform 200 um (= his DC3D8 200 um, each element a
  lump of 5 x 40 um layers): 50^3 = 125,000 cells;
- substrate: extent NOT stated; figure-derived ~30 x 30 x 6 mm (D-V2-07);
  his substrate mesh is ~2 mm with a tie constraint — a conforming
  structured alternative must either mesh the substrate at 200 um
  (675k substrate cells; ~800k total, memory check required before L3) or
  truncate the substrate; open decision D-V2-07, generator parameterized
  (`make_v2_mesh_cube.py`, use --count-only to size variants).

## Initial/boundary-condition parity checklist (as printed)

| item | Balbaa | our mapping |
|---|---|---|
| initial T, multi-track | 80 C whole model | --preheat-temperature 353.15 |
| bottom BC, multi-track thermal | 80 C fixed at substrate bottom | --bottom-thermal-bc fixed |
| top losses | convection + radiation, T_inf 40 C | convection_h + emissivity (values per V1 D-V1-19 discussion) |
| initial T, cube model | not restated (D-V2-12) | default 80 C, registered |
| stress-free reference | 1000 C imposed initial T for the stress model | activation/reset reference temperature = 1273.15 K |
| mechanics bottom BC | total fixation of substrate bottom (stated for cube; multi-track unstated D-V2-13) | --bottom-mechanics-bc fixed |
| recoat | 5 s per layer (cube) | --recoat-time 5.0 |
| final cooldown | 600 s after last layer | --cooling-steps/-dt to 600 s |
| substrate removal | ABAQUS model change after cooldown | --release-after-cooling |
| constitutive | J-C with A=650, B=1618, n=0.243 (measured, p.33 — NOT Table 4's 558/2201.3/0.8), C=2.09e-4, m=1.146, Tm=1290 C, Tr=20 C, edot0=1670/s | flow_curve tables generated from modified J-C at a registered reference strain rate (D-V2-02) |
| elastic modulus | measured 171 GPa RT; literature E(T) offset to match (Fig 41) | E_table digitized from Fig 41 (D-V2-03) |
| thermal properties | Table 1 (same as V1) | V1's quarantined config (linear reconstruction) unless D-V2-04 approved |

## Work plan

1. **Transcribe** (`inputs/balbaa-v2-model.json` done in skeleton; extend as
   figures are digitized): both model definitions, modified J-C set, XRD and
   pyrometer protocols. Digitize Fig 41 (E(T)) and the XRD profile figures
   when the RS gate is prepared. Register everything unstated in
   `inputs/deviations.yaml` (D-V2-xx; Kaess discipline).
2. **Meshes**: `model/make_v2_mesh_multitrack.py` (M1, committed),
   `model/make_v2_mesh_cube.py` (M2 generator; final substrate variant is an
   L3-time decision).
3. **Paths**: serpentine stripe generator (kaess `make_kaess_path.py` is the
   template; 0/90 alternation via rotation-deg 90). Two unknowns block final
   path files: stripe width (D-V2-09) and the 4x4-domain vs 10x10-exposure
   step-time paradox (D-V2-08) — the latter needs a reading decision before
   the thermal gate.
4. **Runs**: BLOCKED until the main line reaches the mechanics stage (L3)
   and freezes: layer-lumping semantics, T_cut/stress-relaxation definition,
   release protocol, and re-runs the post-merge mechanics regression
   (tests/benchmarks + kaess golden gate). Per D-09.
5. **Compare + report**: `RESULTS.md`, same discipline as V1 (relative
   errors both ways, honest discrepancy list, no fitting).

## Acceptance framing

No threshold to tune toward. Success = (a) thermal gate: pyrometer-band
average temperature reproduced with error explainable by registered
deviations; (b) RS gate: in-depth profiles reproduced with error comparable
to or better than Balbaa's own ABAQUS-vs-XRD gap, with every systematic
deviation explained and registered. A strong disagreement is a finding about
the mechanics chain — exactly what V2 exists to expose.

## Status

- 2026-07-30: workspace created by the V session on user request. SPEC +
  transcription skeleton + deviation seed + M1 mesh generator/mesh + M2
  generator committed. Constitutive trap found and registered (A/B/n
  replaced by measured tensile values, p.33). Runs remain gated on L3.
