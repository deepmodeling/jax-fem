# AM-Benchmark AMB2018-01 — prerequisite ledger

Single entry point for everything that must be settled before case design starts.
Status as of 2026-07-28.

Evidence classes follow the project constitution: `paper_text`, `paper_table`,
`figure_digitized`, `author_artifact`, `inferred`, `assumption`.

Governing paper (the benchmark's own reference):
Phan, Strantza, Hill, Gnaupel-Herold, Heigel, D'Elia, DeWald, Clausen, Pagan, Ko,
Brown, Levine (2019), "Elastic Residual Strain and Stress Measurements and
Corresponding Part Deflections of 3D Additive Manufacturing Builds of IN625
AM-Bench Artifacts Using Neutron Diffraction, Synchrotron X-Ray Diffraction, and
Contour Method", *Integrating Materials and Manufacturing Innovation* 8(3):318-334,
doi 10.1007/s40192-019-00149-0. Archived at
`references/docs/Phan2019_AMB2018-01_residual-strain-and-part-deflection_IMMI.pdf`
(sha256 5f8bc3132d06fda72015df5bbce7e62feca2f47f642851e5dcc2abf608a47562).

---

## 0. Decision log

| # | Decision | Status | Date |
|---|---|---|---|
| D-01 | **Material data source scope = Option A** — extend within the AM-Bench series: AMB2022-04 mechanical + Zhang 2019 powder conductivity + Keller 2017 CALPHAD recipe for cp / latent heat / solidus | **APPROVED** | 2026-07-28 |
| D-02 | **Scale reduction = layer lumping with per-layer instantaneous deposition.** Starting ratio **N = 10** (200 um computational layers), registered as a parameter to be **frozen by a convergence study on a reduced sub-domain**, never by matching the measured deflection | **APPROVED (N provisional)** | 2026-07-28 |
| D-03 | **Density-jump convention = compaction convention** (section D.2). Element represents the final consolidated solid volume and its mass: in-part powder `rho_solid`, lateral powder `phi*rho_solid`, future layers strictly void. Powder -> solid is a **conductivity switch only**; mass is exactly conserved and no source term is required | **APPROVED** | 2026-07-28 |
| D-04 | **Activation follows recoat, not scanning** (section D.3), so the inter-layer insulating powder blanket is present | **APPROVED** | 2026-07-28 |
| ~~D-05~~ | ~~Single part + substrate patch, outer boundary pinned at 73.5 C at 15 mm~~ | **SUPERSEDED by D-07** — the 15 mm figure was unsound, see the retraction below | 2026-07-28 |
| ~~D-06~~ | ~~Lateral powder margin coincides with the substrate patch (15-20 mm)~~ | **SUPERSEDED by D-07** — ceases to be an independent parameter | 2026-07-28 |
| D-07 | **Computational domain = one 20 mm periodic cell in y (the actual part pitch), substrate to its real x extent, Dirichlet 73.9 C on the substrate underside** | **APPROVED** | 2026-07-28 |

All design decisions are settled. Remaining open items are data gaps (section E)
and the registered conflicts (section B), not design choices.

### RETRACTION — why D-05 / D-06 were withdrawn

The 15 mm margin in D-05/D-06 was **not taken from any reference.** It was derived
in-session as the per-layer thermal diffusion length

`alpha = k/(rho*cp) = 20/(8440*550) = 4.31e-6 m2/s`,
`sqrt(alpha * 52 s) = 15.0 mm`

The arithmetic is correct and the property values are self-consistent at 500-600 C,
but the figure was used as a *boundary-placement* criterion, which it is not. For a
semi-infinite solid under a surface step, the residual disturbance at depth x is
`erfc(x / 2*sqrt(alpha*t))`:

| Distance | Argument | erfc | Residual disturbance |
|---|---|---|---|
| **1x sqrt(alpha t) = 15 mm** | 0.5 | 0.480 | **48 %** <- the withdrawn choice |
| 2x = 30 mm | 1.0 | 0.157 | 16 % |
| 3x = 45 mm | 1.5 | 0.034 | 3.4 % |
| 4x = 60 mm | 2.0 | 0.005 | 0.5 % |

Pinning a Dirichlet condition where roughly half the disturbance survives flattens a
genuinely non-zero gradient. A quiet boundary needs 3-4x, i.e. 45-60 mm — which does
not fit inside a 100 mm substrate that the 75 mm part already occupies.

The framing was also wrong. Over the 9.4 h build the diffusion length is 380 mm,
far exceeding the substrate, so the substrate problem is **quasi-steady** — set by
the balance between deposited power and the platform sink — not transient on a
52 s scale. A single-layer transient length scale is the wrong instrument for
truncating it.

Recorded here rather than overwritten, per the constitution: withdrawn decisions and
their reasons are evidence.

### Rationale for D-07

The replacement uses measured boundaries and real geometry instead of a derived
distance.

| Direction | Boundary | Basis |
|---|---|---|
| **z, underside** | **Dirichlet 73.9 C** | A physically real boundary: the IN625 substrate is bolted to a temperature-controlled steel platform, measured at 73.9 C by four thermocouples at build start. Measurement, not approximation. |
| **y** | **20 mm periodic / symmetry cell** | The actual part pitch — centres at y = 19.8 / 39.8 / 59.8 / 79.8 mm in `AMB2018_01_Build.STL`. A structural truncation, not a chosen distance. |
| **x** | The substrate's real extent | Measured from the STL: 20.3 mm of substrate beyond the part's left end, **only 4.66 mm beyond the right end**. There is no freedom here. |

Three independent facts support the y periodic cell:

1. All four parts are scanned within the same layer, 0.307-0.363 s apart, against a
   52 s layer time — effectively **in phase** at layer scale.
2. The measured specimens are `AMB2018-01-625-CBM-B1-P3` and
   `AMB2018-01-15.5-CBM-B3-P3` — **P3, an interior part**, with neighbours on both
   sides.
3. The x-stagger between parts is 0.5 mm out of 75 mm (0.7 %), which does not break
   the mirror approximation.

Consequence: the lateral powder margin is no longer an independent parameter. It
follows from the cell width — half the 15 mm inter-part gap on each side. One fewer
number requiring justification.

Cost versus the withdrawn version: y-domain 35 mm -> **20 mm**, roughly 40 % fewer
elements, and the inter-part thermal cross-talk that D-05 discarded is now
represented rather than dropped.

### Rationale recorded for D-02

The validation quantity is a bending response: cutting the legs releases a moment
`M = integral(sigma_xx(z) * z dz)` that lifts the free end by 1.276 mm. What must be
resolved is therefore the **through-thickness stress profile**, not the melt pool —
melt-pool detail is integrated away. The continuous load-bearing beam is the
constant-section bridge, z = 7.5 -> 12.5 mm, 5 mm thick (the 12 legs are
discontinuous and contribute stress but little continuous bending stiffness).

That gives a physical criterion for the lump ratio:

| N | Computational layer | Layers through the 5 mm bridge | Verdict |
|---|---|---|---|
| 5 | 100 um | 50 | ample |
| **10** | **200 um** | **25** | **resolves the profile and its moment** |
| 25 | 500 um | 10 | marginal |
| 50 | 1000 um | 5 | moment integral badly distorted |

Cost estimate at N = 10, in-plane 250 um (167 um if the 0.5 mm thin leg needs three
elements across): part 378 k - 567 k elements, plus ~30 % for the powder margin,
plus a graded build plate (~120 k; a uniform 250 um plate would be 8.2 M and is out
of the question). Total order **0.8 - 1.2 M elements**.

Time steps: 63 computational layers x 20-50 steps each ~ **1300-3200 steps** plus
cooldown and release — against ~6e7 steps for a path-resolved solve. Four orders of
magnitude cheaper. This, not mesh size, is what makes the case tractable.

**Coupling that must not be overlooked:** the lump ratio interacts with the
stress-relaxation cutoff temperature (gap E, highest threat). A coarser lump gives a
larger single thermal excursion, so more material exceeds the cutoff and has its
stress and plastic strain reset. Lump ratio and cutoff temperature must therefore be
swept **together**, not independently.

Known losses at N = 10: intra-lump thermal cycling (10 heat-cool cycles become 1);
the 45 degree overhang is described by only ~12 computational layers and is
staircased, which falls in a stress-concentration region.

---

## A. Settled — verified, no decision needed

### A.1 Geometry (measured directly from the official STL, `references/geometry/`)

`AMB2018_01_Part.STL` — binary, 376 triangles, 190 welded vertices, **watertight**
(564 edges each used by exactly 2 triangles, 0 boundary, 0 non-manifold), Euler
characteristic 2, orientation consistent, signed volume +3732.5 mm3, one connected
component, zero degenerate triangles. Units mm.

| Quantity | Value |
|---|---|
| Bounding box | 75.000 x 5.000 x 12.500 mm |
| Leg height | 0 -> 5.0 mm |
| 45 degree overhang | 5.0 -> 7.5 mm |
| Constant-section bridge | 7.5 -> 12.0 mm |
| Ridges | 12.0 -> 12.5 mm (0.5 mm tall) |

Legs at mid-height (z = 2.5 mm), from ray-cast cross section at y = 2.5 mm.
Gaps are uniformly 2.000 mm; the repeat period is 14.000 mm; 4 repeats.

| Type | Count | Width | Left edge x (mm) |
|---|---|---|---|
| Thick | 4 | 5.0 mm | 0.0, 14.0, 28.0, 42.0 |
| Thin | 4 | **0.5 mm** | 7.0, 21.0, 35.0, 49.0 |
| Medium | 4 | 2.5 mm | 9.5, 23.5, 37.5, 51.5 |
| End block | 1 | 19.0 mm at mid-plane | 56.0 (see tip note) |

The 0.5 mm thin leg is the minimum feature and sets the mesh floor.

**Tip correction (2026-07-28):** the "56.0 to 75.0" end-block extent above is a
mid-plane (y = 2.5) statement only. Ray casting across y shows the part's right end
is a **45-degree point in plan view**: max-x runs from 72.55 mm at the y edges to
75.00 mm at mid-plane, at every height from the legs through the bridge (the ridges
stop at x = 71.0). The part is **not** an extrusion of its mid-plane cross-section;
meshes and scan paths must model the tip. This matches the JRES scan tables (odd
base line lengths 16.49-18.94 mm; even 0.00-4.94 mm; "laser on time decreases in
tip", Fig 8) and is recorded in `inputs/scan-timing.json`.

Ridges (CMM measurement targets) — 11, each 1.000 mm wide, 7.000 mm pitch,
centres at x = 0.5, 7.5, 14.5, 21.5, 28.5, 35.5, 42.5, 49.5, 56.5, 63.5, 70.5 mm.

`AMB2018_01_Build.STL` — watertight, 2896 triangles, 5 components:
build plate 100 x 100 x 12.7 mm (z 4.035 -> 16.735) plus 4 parts sitting on it at
z 16.735 -> 29.235. Parts spaced 20 mm in y (centres 19.8 / 39.8 / 59.8 / 79.8),
staggered in x by 0.4-0.5 mm (left edges 21.318 / 21.718 / 22.218 / 22.718).
The plate is genus 4 — four ~6 mm through-holes at the corners
(x 10-16 / 86-92, y 9.3-15.3 / 85.3-91.3): the 1/4"-20 cap-screw mounting holes.
Not a defect.

### A.2 Process parameters

| Quantity | Value | Class | Source |
|---|---|---|---|
| Machine | EOSint M270D | `paper_text` | Heigel JRES 2020 |
| Infill power / speed | 195 W / 800 mm/s | `paper_text` | Phan 2019, JRES Table 2 |
| Contour power / speed | 100 W / 900 mm/s | `paper_text` | same |
| Layer height | 20 um | `paper_text` | same |
| Hatch spacing | 100 um | `paper_text` | same |
| Scan strategy | infill X-parallel odd layers, Y-parallel even layers; contour first | `paper_text` | Phan 2019 |
| Atmosphere | **nitrogen**, ~0.5 % O2 steady state; build aborts above 1.3 % | `paper_text` | JRES |
| Recoater | 80 mm/s; HSS blade for IN625 | `paper_text` | JRES |
| Build duration | 9 h 23 min / 9 h 22 min / 9 h 6 min (three builds) | `paper_table` | Heigel IMMI Table 1 |
| Layer time, layers 1-250 (legs) | **52 s**, of which ~26 s is scanning all four parts | `paper_text` | Heigel IMMI |
| Dwell placement | imposed **at the end of each layer** (so the NIST AMMT could replicate the build later) | `paper_text` | Heigel IMMI, verbatim |

Argon belongs to the AMMT, which built AMB2018-02. It is **not** this build. Do not mix.

### A.3 Thermal initial conditions

Build plate setpoint was 80 C **but was never reached**. Measured at build start
(14 thermocouples, Heigel IMMI Build3):

| Location | Temperature |
|---|---|
| IN625 substrate | **73.5 C** (8 TCs, 72.2-74.6) |
| Steel build plate | **73.9 C** (4 TCs, 73.1-74.5) |
| Chamber gas | 32 C (max 38 C during build) |
| Frame around build volume | 49.8 C |

**Use 73.5 C for the substrate initial condition, not 80 C.**

### A.4 Validation targets

Part deflection after wire-EDM separation, upward, per ridge. Recovered from the
live NIST figure `figure_2b.fw_.png` and cross-checked against Phan Fig. 16.

| x (mm) | IN625 (mm) | IN625 sigma | 15-5 PH (mm) |
|---|---|---|---|
| 0.5 | 1.276 | 0.002 | 1.168 |
| 7.5 | 0.997 | 0.001 | 0.914 |
| 14.5 | 0.754 | 0.002 | 0.695 |
| 21.5 | 0.551 | 0.003 | 0.516 |
| 28.5 | 0.387 | 0.003 | 0.358 |
| 35.5 | 0.250 | 0.001 | 0.231 |
| 42.5 | 0.146 | 0.000 | 0.133 |
| 49.5 | 0.065 | 0.000 | 0.076 |
| 56.5 | 0.012 | 0.001 | 0.008 |
| 63.5 | 0.000 | 0.001 | 0.002 |
| 70.5 | 0.003 | 0.001 | -0.001 |

sigma is statistical over the 3 CMM points per ridge and **excludes** the CMM
MPE of 0.005 mm (ISO 10360-2). The 15-5 column is single-sourced from the figure
and has no published uncertainties.

Residual elastic strain: neutron (IN625 + 15-5, X/Y/Z) and synchrotron EDXRD
(IN625, X/45/Z/shear) numeric files already downloaded and checksum-verified under
`references/measurements/`.

### A.5 Elastic constants used by the benchmark itself

From Phan 2019's own reference list — these are the **only** material properties
the governing paper cites:

| Quantity | Value | Ref |
|---|---|---|
| Single-crystal C11 / C12 / C44 | 243.6 / 156.7 / 117.8 GPa | [9] Wang et al. 2016, MSEA 674:406-412 |
| Isotropic E / nu (contour method) | 207 GPa / 0.278 | [20] Special Metals INCONEL 625 bulletin |

Both numeric values are printed in Phan's body text — the source papers are not
needed to obtain them.

Note: 207 GPa / 0.278 are **wrought annealed handbook** constants applied to an
as-built LPBF part. As-built LPBF IN625 is <001>-textured along the build
direction, so the true modulus differs. This is a benchmark simplification, and it
is already baked into the published contour-method stresses being validated
against. Use the same constants for consistency with the benchmark's own reduction.

---

## B. Conflicts — must be registered, not silently resolved

| # | Conflict | Evidence |
|---|---|---|
| B1 | **Layer count 624 vs 625** | JRES Table 2 says 624 (byte-verified); Phan 2019 says 625 (byte-verified). Arithmetic favours 625: 625 x 20 um = 12.500 mm, exactly the measured STL bridge height. 624 may be a thermography frame count. |
| B2 | **Laser spot size** | Phan and Heigel IMMI: 85 um D4-sigma contour / 100 um defocused infill. JRES Table 2: a single 0.10 mm vendor figure. AMB2022-04: 80 um D4-sigma contour. State which metric is used. |
| B3 | **Solidus** | Special Metals / Wikipedia melting range 1288-1349 C; NIST CALPHAD (Ghosh) solidus 1587 K = 1314 C. Also: Scheil (non-equilibrium, appropriate for LPBF) predicts a solidus tens to >100 K below equilibrium. Three different numbers depending on definition. |
| B4 | **Ridge numbering is inverted** between prose and figures | NIST prose calls the maximum-deflection end "ridge 1"; NIST Fig. 1 and Phan Fig. 9 both label that same end "Ridge 11". **Key the model on the x coordinate, never the ridge index.** |
| B5 | Bridge height | JRES text says 12 mm; IMMI says 12.5 mm; **STL measures 12.500 mm**. Use 12.5. |

---

## C. Awaiting approval — material data source scope

Phan 2019 cites **only** elastic constants (section A.5). It is a measurement
paper and runs no simulation, so it contains no thermophysical or plastic data.
That gap is structural, not a search failure. Running a thermo-mechanical model
therefore requires sources outside the governing paper. Three options:

| Option | Scope | Trade |
|---|---|---|
| **A** | Extend within the AM-Bench series: AMB2022-04 mechanical (specimens cut from a reserved AMB2018-01 bridge) + Zhang 2019 powder conductivity (NIST-measured) + Keller 2017 CALPHAD recipe for cp / latent heat / solidus | Highest provenance; mechanical data is same-build material. Keller is not in Phan's reference list. |
| **B** | Phan's references only, plus purchased measured data (Kaschnitz 2019 x2) | Most conservative. Both papers paywalled and **it is unverified whether they tabulate numbers or only plot them** — may be wasted money. |
| **C** | Register as `assumption`, compute thermophysics via CALPHAD in-house, cite database version | What NIST itself does. Defensible, but the evidence class is "assumption", not "measured". |

**DECIDED 2026-07-28: Option A** (decision D-01). Consequences:

- Mechanical properties come from AMB2022-04, whose specimens were machined from a
  reserved AMB2018-01 bridge — same machine, same powder lot, same scan pattern.
  Evidence class `author_artifact` within the benchmark series.
- Powder conductivity from Zhang et al. 2019 (NIST-measured). Class `paper_table`.
  Still to confirm: the measurement gas (the build was under nitrogen; Wei 2018
  shows ~200 % gas dependence) and whether extrapolation above 500 C is defensible.
- cp, latent heat and solidus/liquidus from a CALPHAD run following the Keller 2017
  recipe (Thermo-Calc TCNI + Scheil-Gulliver). Class `assumption`; the database
  version must be recorded.
- Keller 2017 is outside Phan's reference list. That extension is what this decision
  authorises, and it must be recorded as such in the source manifest.
- **Absorptivity is NOT inherited from Keller.** Section C.2 shows A = 0.50 there is
  unsourced and that NIST's own companion paper calls it too large. Register A as an
  `assumption` with a sensitivity bracket.

### C.1 Verified negatives (do not re-search these)

- AM-Bench published **no** IN625 thermophysical property data in 2018, 2022 or 2025.
  Levine et al.'s own Outcomes paper names "a lack of material property data" as a
  motivation for AM-Bench and lists powder/liquid/solid thermal conductivity as an
  *outstanding measurement need*.
- AMB2018-03 is polycarbonate (material extrusion); AMB2018-04 is Polyamide 12 (PBF).
  Both polymer benchmarks. Zero alloy data.
- AM-Bench 2025 is IN718 and Ti-6Al-4V. No IN625.
- AMB2022-04 (mechanical) and AMB2022-05 (microstructure) are explicit extensions of
  AMB2018-01 and **do** cover IN625 — but mechanically and microstructurally, never
  thermophysically.
- **Mills (2002) does not contain IN625.** Its only Ni superalloys are CMSX-4,
  Hastelloy X and IN718. Papers claiming "IN625 properties from Mills" are
  misciting — including Psihoyos & Labeas (doi 10.3934/matersci.2022027), which
  models this very benchmark and also states 600 mm/s and a 45-degree/90-degree
  rotating scan strategy, both wrong. Do not use that paper as a reference.
- Raw 55-point CMM deflection data was **never published** — the Wayback capture of
  2020-12-13 already shows "download link will be provided soon" with no hyperlink.
  Only the co-chairs have it. The 11 ridge averages in A.4 are sufficient.

### C.2 Resolved: the powder-conductivity substitution is safe

An earlier concern was that Keller 2017's absorptivity A = 0.50 might have been
co-calibrated with its assumed powder conductivity of 1.0-3.0 W/(m K), so that
substituting Zhang 2019's measured 0.65-1.02 W/(m K) would break a fitted pair.

Reading the arXiv LaTeX source of both Keller 2017 and Ghosh 2018 shows otherwise:

- A = 0.50 appears once, with **no citation and no justification**. The word
  "absorpt" occurs exactly once in the whole paper.
- Keller cites Ma et al. 2015 for "appropriate parameter values" — and **Ma used
  A = 0.12**, a factor of 4.2 lower.
- Ghosh et al. (NIST's own companion paper) state A = 0.5 is **too large** and
  causes systematic melt-pool depth over-prediction.
- Keller's validation case is a **bare plate with no powder at all** (verbatim:
  "FEA simulation results are compared against experimental laser scans on bare
  plates, without powder"), so k_powder took no part in it.

**A and k_powder have zero covariance. There is no calibration to break.**
Absorptivity should be registered as an `assumption` with a sensitivity bracket,
not inherited as 0.50.

---

## D. Awaiting approval — modelling conventions

### D.1 Powder / solid property design

Powder and solid are the **same alloy**. Proposed rule:

> Mass-normalised thermodynamic properties (cp, latent heat, transformation
> temperatures) take identical values in powder and solid. Volume-normalised
> properties (rho, rho*cp) scale with packing fraction phi. Thermal conductivity
> and emissivity must be taken independently and **must not** be derived by
> scaling the solid values.

Justification: powder conduction is dominated by inter-particle contact resistance
and interstitial gas, not by geometric dilution — measured k_powder is 4-7 % of
solid, where porosity scaling alone would give ~50 %. This is also why Wei et al.
2018 find a ~200 % difference between helium and argon.

This reduces the powder data requirement to exactly two quantities:
**k_powder(T)** (available, NIST-measured) and **emissivity_powder** (unavailable).

### D.2 Density jump at consolidation

Steady-state mass balance: `g = t / phi`, so ~40 um of loose powder
(phi ~ 0.5; 33-45 um for Zhang's measured phi = 0.44-0.60) is consumed to produce
each 20 um solid layer. The melt pool must therefore penetrate ~20 um into the
previous solid — consistent with the observed 2-3x layer-thickness melt depth.

**Proposed convention:** an element represents the **final consolidated solid
volume and its mass**.

| State | rho | k | Mechanical |
|---|---|---|---|
| Void (above the powder bed surface) | — | — | strictly zero contribution |
| Powder, in-part (activated, not yet melted) | **rho_solid** | k_powder | weak solid / none |
| Powder, lateral margin (never melts) | **phi * rho_solid** | k_powder | weak solid / none |
| Consolidated solid | rho_solid | k_solid | full |

Consequence: powder -> solid becomes a **conductivity switch only**. Mass is
conserved exactly, no source term is needed, and the convention is self-consistent
with "layer thickness = solid increment", which is what the machine parameter means.

The in-part / lateral asymmetry is deliberate: the two element classes represent
different things (future solid volume vs actual powder volume).

**Registered cost:** the ~40 um of loose powder above the solid surface is
compressed into a 20 um element, so within that element rho*cp is ~2x and thermal
diffusivity ~0.5x the true powder values, and the exposed surface sits ~20 um low.
Energy-to-melt is exact. Expected second-order for part-scale deflection,
first-order for melt-pool geometry.

### D.3 Activation timing — **not** a free choice

Activation must follow **recoat**, not scanning:

```
recoat  -> layer N activates as POWDER (rho_solid, k_powder, no load path)
scan    -> melts -> switches to SOLID (rho_solid, k_solid, full stiffness)
dwell   -> layer N+1 not yet recoated; the top surface is genuinely bare
recoat  -> layer N+1 activates as powder and re-covers the top
```

If layers are activated on scanning instead, the model never has the insulating
powder blanket over the part between layers, and top-surface radiative/convective
loss is systematically over-predicted across a 9.4 h, 625-layer build.

Cheap test before committing: run a few layers both ways and compare cumulative
top-surface heat loss. A few percent is ignorable; tens of percent is not.

### D.4 Domain (decision D-07)

```
x : substrate real extent — 20.34 mm beyond the part's left end,
    4.66 mm beyond its right end (measured from AMB2018_01_Build.STL)
y : 20.000 mm periodic cell, centred on the part; symmetry / periodic
    boundaries at +/- 10 mm from the part centreline
z : substrate 12.7 mm + part 12.5 mm; Dirichlet 73.9 C on the substrate
    underside (measured); powder fills to the current build height, above
    which elements are strictly void
```

Lateral powder margin is not an independent parameter — it is half the 15 mm
inter-part gap, fixed by the cell width.

**Mesh grading is mandatory, not an optimisation.** At a uniform 250 um in-plane
spacing the powder region alone would be ~3.5 M elements and the substrate ~8.2 M.
Graded, the expected totals are: part 378 k (378 k at 250 um, 567 k if the 0.5 mm
thin leg needs three elements across), powder ~220 k at ~1 mm, substrate ~42 k at
1-2 mm — order **0.6 - 1.0 M elements**.

### D.5 Still undecided

- **Release-stage domain.** Deflection was measured after the part, still attached
  to an EDM'd-out section of substrate, had its 12 legs cut. Whether the release
  step reuses the build domain or switches to that cut-out section is open.
- **Lump-ratio convergence study design.** D-02 fixes the method and the starting
  ratio; the sub-domain, the convergence metric and the acceptance threshold are
  not yet specified. Must be frozen before the full case runs.

---

## E. Data gaps — each needs a decision, not a search

### E.0 Option A acquisition status (2026-07-28)

All Option A resources that exist are now archived and hashed in
`source-manifest.json` (schema v2). Cross-check: the four files also fetched
independently by the earlier research agents match byte-for-byte.

| Resource | Where | Status |
|---|---|---|
| MaCTO mechanical data (3 files: raw curves, answers table, README) | `references/material/amb2022-04-macto/` | **acquired** (DOI 10.18434/mds2-2681) |
| Special Metals INCONEL 625 bulletin | `references/material/` | **acquired** |
| Gen3 CSP k/cp/alpha spreadsheet | `references/material/` | **acquired** |
| Zhang 2019 powder-conductivity paper | `references/docs/` | **acquired** (via Europe PMC render; author manuscript) |
| Keller 2017 CALPHAD-recipe preprint | `references/docs/` | **acquired** (arXiv PDF, 25 pp) |
| Kaschnitz / Heugenhauser measured rho(T) through melting | — | paywalled, NOT acquired; tabulation unverified |
| CALPHAD run (cp, latent heat, solidus/liquidus) | `derived/` (future) | **to do** — needs Thermo-Calc/JMatPro access, database version must be recorded |

Still open from the gap table below: sigma_y above 773 K (extrapolation +
relaxation cutoff), emissivity (solid and powder), rho(T) above RT, layer time
for layers 251+, and the Zhang measurement-gas check.

| Gap | Status | Candidate | Threat to residual stress |
|---|---|---|---|
| sigma_y(T), hardening, 773 K -> solidus | **nothing exists anywhere** | extrapolate MaCTO shape + stress-relaxation cutoff (Denlinger & Michaleris, doi 10.1016/j.addma.2016.06.011) | **Highest.** The cutoff temperature is a calibration parameter, not data. Sensitivity study mandatory. |
| Emissivity (solid and powder) | no usable value; NIST's own thermography paper admits it is unknown | treat surface radiation as calibrated | High over a 9.4 h build |
| Latent heat, self-consistent solidus/liquidus | CALPHAD only | Thermo-Calc TCNI + Scheil, cite database version | Medium; see conflict B3 |
| Layer time, layers 251+ | **never published** | only 52 s for layers 1-250 is known | Medium — this is 60 % of the build |
| k_solid(T) | available | Special Metals (measured at Battelle, -157 -> 982 C); Gen3 CSP xlsx (260-1000 C with 95 % CI, but k is derived as alpha*cp*rho with rho fixed at 8.44, so it drifts at high T) | Low |
| k_powder(T) | available, NIST-measured | Zhang et al. 2019, 0.65 W/(m K) @ 100 C -> 1.02 @ 500 C. **Confirm the measurement gas** — the build was under nitrogen, and Wei 2018 shows ~200 % gas dependence. Extrapolation above 500 C is unverified. | Low-medium |
| cp(T) | available | Special Metals 12 points (footnoted "Calculated", not measured); Gen3 CSP; Kaschnitz DSC | Low |
| rho(T) | RT only in datasheets | Heugenhauser & Kaschnitz doi 10.32908/hthp.v48.726, 150-1400 C solid + mushy + liquid (paywalled, tabulation unverified) | Low |
| 15-5 PH deflection uncertainties | never published | Phan is IN625-only; the promised 15-5 paper does not appear to exist | Low (only if 15-5 is modelled) |

### In-situ thermography (DOI 10.18434/M31935) — EVALUATED AND REJECTED

Data descriptor: Heigel, Lane, Levine, Phan, Whiting (2020), *J. Res. NIST* 125:125005,
doi 10.6028/jres.125.005, archived at
`references/docs/Heigel2020_AMB2018-01_in-situ-thermography_JRES-125-005.pdf`.

Dataset: 122 zip archives plus two MATLAB functions, covering all 624 layers of two
builds. A sample layer file was inspected directly:

| Property | Value |
|---|---|
| Format | MATLAB v5 `Layer` struct, readable with `scipy.io.loadmat` |
| Pixel size | 51.95 x 33.98 um |
| ROI | 126 x 360 px = **6.55 x 12.23 mm** — covers legs 7/8/9 only (x = 28-40 mm) |
| Frame rate | **1799 fps** (0.556 ms per frame); 2497 frames for 3.979 s of layer 1 |
| Field | `RadiantTemp`, uint16, 126 x 360 x 2497 = 113 M values, 227 MB uncompressed per layer |
| Calibration | Sakuma-Hattori coefficients `SHvariable_A/B/C` = 2.655 / -800.7 / 1.94e6 |
| Occupancy | **0.838 % non-zero.** >550 C: 0.81 %; >1050 C: 0.05 % (camera saturates 1050-1100 C) |
| Hot area | median 0.46 mm2 per active frame, max 3.55 mm2 |

**Not used.** Three candidate uses were considered and all fail:

1. *Heat-source / absorptivity calibration* — blocked. Converting radiant to true
   temperature requires an emissivity that has no measured value (gap E). Fitting
   absorptivity against this data means fitting A and epsilon to one dataset:
   unidentifiable, and precisely the coupling that discredited Keller's A = 0.50.
   Would need AMB2018-02 sectioned melt-pool geometry as an independent constraint.
2. *Scan-path verification* — **redundant.** Tables 3, 4 and 5 of the same paper
   specify contour timing per feature and per-line laser on/off timing for odd and
   even layers to the millisecond. The path is defined, not inferred.
3. *Cooling-rate extraction* — blocked by the same emissivity coupling.

Scale mismatch against the approved N = 10 model is also decisive: 34-52 um pixels
against 200 um computational layers, and 0.556 ms frames against ~26 s steps — five
orders of magnitude in time.

Recorded as a negative result so it is not re-evaluated. Physical relevance is not
the issue: the camera's 550-1050 C window sits squarely in the range where the
thermal-gradient mechanism generates residual stress. The instrument is simply the
wrong scale for a part-level lumped model.

### Note on the MaCTO mechanical data

AMB2022-04 specimens were machined from a **reserved AMB2018-01 bridge**, so their
stress-strain curves already contain whatever precipitation occurred during that
build. IN625 is solid-solution strengthened but precipitates M23C6, gamma'' and
delta between 650-875 C, and the lower part of a 9.4 h build cycles through that
window repeatedly.

Convenient: no precipitation-kinetics model is needed. But it introduces a weak
circularity — material data carrying thermal history A is used to predict the
process that produces thermal history A. Register it; do not assume it away.

---

## F. Environment

```
/home/user/miniconda3/envs/jax-fem-env/bin/python
```

jax 0.10.2 (CPU, x64), petsc4py 3.25.1, fenics-basix 0.10.0, pypardiso 0.4.7 +
MKL 2026.1.0 (verified: residual 0 on a 2x2 solve), gmsh 4.15.2, meshio 5.3.5,
pymupdf 1.28.0. jaxlib is CPU-only; the RTX 5080 is unused until `jax[cuda12]`
is installed.

Branch `test` at `bdd1671` (merge of `codex/r3-optimization`); rollback point
`backup/test-before-merge` at `79d416a`. The solver has **not** been run since the
merge — `tests/benchmarks/` (four cases with FEniCSx reference solutions) has not
been executed.
