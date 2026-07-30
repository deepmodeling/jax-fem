# V1 — single-track melt-pool triangle: results

**Triangle**: our jax-fem thermal solver vs NIST AMB2018-02 measurement vs
Balbaa 2022 ABAQUS, on IN625 single tracks. Zero-calibration: every input
traces to `inputs/balbaa-model.json` (Balbaa 2022 as printed),
`inputs/nist-meltpool.json` (Lane 2020 Tables 3/4), or a registered entry in
`inputs/deviations.yaml` (D-V1-01..20). Nothing was tuned against any number
in this report; extraction definitions were fixed a priori in
`model/analyze_v1.py`.

Run configuration (all runs): Balbaa validation variant — 20 um powder layer
on 280 um substrate, exponential-depth volumetric source (Eq 18 == our
`source_model=legacy`, energy-ledger-verified), A = 0.62, OPD d = 100 um,
r = 50 um (CBM spot D4sigma 100 um, provenance NIST), uniform 10 um hexes
(144k cells), preheat 80 C, mushy band 1563–1623 K, latent 290 kJ/kg,
thermal-only. Melt-pool boundary: 1290 C solidus (both references' own
convention). Solver command + used config archived in each run directory;
thermal energy ledger complete for every run (max relative balance error
<= 2e-8, absorbed power == 0.62 x P exactly).

## Figures

- `figures/v1_triangle_absolute.png` — absolute width / depth / length, all
  three cases, three sources side by side (NIST error bars: u_mean for
  width/depth, U(k=2) for length — the only quantity with a full published
  uncertainty budget, Lane2020 Table 5).
- `figures/v1_triangle_relative_caseB.png` — signed relative error vs the NIST
  measurement at case B, ours vs Balbaa.
- Regenerate: `python model/plot_v1_triangle.py` (values hard-coded from the
  tables below; the script plots, it does not compute).

## The triangle at CBM case B (195 W, 800 mm/s)

Balbaa's validation condition is not stated in his paper; it was identified
as AMB2018-02 CBM case B because his Exp[68] bars match Lane Table 4 CBM-B
exactly (133/91/780 um) — see D-V1-03. His predictions below are digitized
from Figs 8–10 (+/-2 um width/depth, +/-10 um length; the 131 um width is
stated in his text).

| quantity | ours | NIST CBM-B (u_mean) | Balbaa ABAQUS (exp. source) |
|---|---|---|---|
| width [um] | 141.0 | 133 (0.50) | 131 |
| depth below substrate top [um] | 90.6 | 91 (0.52) | 44 |
| depth below layer top [um] | 110.6 | — (bare plate) | (convention not stated) |
| length [um] | 575 | 780 (0.50; U_k2 = 87.4) | 378 |
| cooling rate 1290->1190 C [C/s] | 2.22e6 | 9.35e5 (1.82e4) | not reported per case |
| cooling rate 1290->1000 C [C/s] | 1.05e6 | 9.33–9.57e5 (Table 3 tracks) | not reported |
| peak T [K] | 3346 | — | ~2835 (Fig 7 contour max) |

Relative errors (signed):

| quantity | ours vs NIST | Balbaa vs NIST | ours vs Balbaa |
|---|---|---|---|
| width | +6.0% | −1.5% | +7.6% |
| depth (below substrate top) | **−0.5%** | −51.6% | +105.8% |
| length | −26.3% | −51.5% | +52.1% |
| CR 1290->1190 | +136.9% | — | — |
| CR 1290->1000 | +11.2% | — | — |

## Two-way (ours vs NIST) at CBM cases A and C

Balbaa published no predictions at these conditions; same model config,
only P and v changed.

| quantity | A: ours | A: NIST | err | C: ours | C: NIST | err |
|---|---|---|---|---|---|---|
| width [um] | 164.4 | 171 (0.82) | −3.8% | 106.2 | 100 (0.48) | +6.2% |
| depth (substrate top) [um] | 114.9 | 151 (5.75) | **−23.9%** | 60.0 | 60 (0.16) | **+0.0%** |
| length [um] | 525 | 659 (0.47) | −20.3% | 570 | 754 (0.68) | −24.4% |
| CR 1290->1190 [C/s] | 1.47e6 | 6.20e5 | +137% | 2.84e6 | 1.28e6 | +122% |
| CR 1290->1000 [C/s] | 6.89e5 | 5.35–5.84e5 | +23% | 1.54e6 | 1.34–1.54e6 | +9% |
| peak T [K] | 3757 | — | | 2778 | — | |

## Input-uncertainty sensitivity (registered deviations, quantified)

| variant | width | depth (sub.) | length | CR 1290->1190 | verdict |
|---|---|---|---|---|---|
| baseline (rho 8453, Fig-1 powder k) | 141.0 | 90.6 | 575 | 2.22e6 | — |
| rho = 7925 (D-V1-18 bounding) | 141.9 (+0.6%) | 94.3 (+4.1%) | 575 (0%) | 2.23e6 (+0.5%) | immaterial |
| Sih–Barlow-as-printed powder k (D-V1-20) | 141.3 (+0.2%) | 90.6 (0.0%) | 575 (0%) | 2.10e6 (−5.0%) | immaterial |

Mesh quantization: cell-history geometry is quantized to 10 um; nodal
sub-cell interpolation agrees within 1 um (141.0 vs 140.0), so discretization
is not a factor at the table's precision.

## Honest discrepancy list

1. **Depth at case A: −24%.** A has the highest line energy (0.375 J/mm) and
   NIST's measured depth/width ratio there is 0.88 — the transition toward a
   vapor-depression (keyhole-like) regime. Our model (and Balbaa's) is pure
   conduction with no vapor depression, so under-predicted depth at the
   highest line energy is the expected signature of that missing physics.
   Depths at B (−0.5%) and C (+0.0%) are at measurement precision.
2. **Length: −20% to −26% across all three cases.** Direction and magnitude
   are consistent, i.e. systematic. Candidates (registered, not fitted):
   no Marangoni-driven elongation in a conduction model (Balbaa attempted an
   effective-conductivity surrogate and still got −52%); trailing-edge latent
   release; NIST length comes from surface NIR imaging while ours is the
   volumetric solidus extent. Note internal consistency with (3).
3. **Near-solidus cooling rate: +122% to +137%, but wide-band (to 1000 C)
   only +9% to +23%.** A shorter predicted pool tail is the same physics as
   a steeper trailing temperature gradient: discrepancies (2) and (3) are one
   phenomenon, not two. The wide-band rates being within ~10–23% says the
   overall thermal field is right; the extreme near-solidus slope is where
   the missing melt-pool physics concentrates.
4. **Peak temperature exceeds boiling (3000 K) at A and B** (3757 / 3346 K):
   no evaporation heat sink in the model (same as Balbaa; his lower 2835 K
   peak is consistent with his artificial in-pool conductivity enhancement).
5. **Width +6% at B/C, −4% at A**: small but outside NIST's u_mean. The
   as-printed (1−phi)^2 powder heat-capacity double-count (D-V1-15,
   reproduced deliberately) slightly cheapens powder heating and could
   contribute; not tuned out. A cp_powder = cp_s sensitivity remains open.

## Verdict against the SPEC acceptance framing

(a) Our solver reproduces the measured melt-pool dimensions with error
comparable to or better than Balbaa's ABAQUS at the matching condition:
width comparable (+6.0% vs −1.5%), depth far better (−0.5% vs −51.6%),
length far better (−26% vs −52%). (b) The systematic deviations are
explained by identified missing physics (keyhole, Marangoni/evaporation)
and registered above — none was calibrated away.

## Transcription findings (feed back to main session / future work)

- **Balbaa's Fig 1 powder-conductivity curve is ~2x the value his printed
  Sih–Barlow equations produce** (with the ln term restored; the MDPI PDF
  text layer also silently drops that ln factor). We used the Fig 1 curve as
  primary input (D-V1-20).
- **Balbaa's [49] (Heugenhauser & Kaschnitz 2019, density/expansion paper)
  cannot be the actual source of his k and cp ranges** — provenance in his
  Table 1 is partially broken; only the range endpoints are usable (D-V1-01;
  Fig 1 confirms solid k is exactly the linear ramp between the endpoints).
- **Eq 14+15 as printed double-count porosity** in the powder volumetric
  heat capacity (D-V1-15).
- **Shared-code bug (for the main session, NOT fixed from V1):**
  `--mechanics-every 0` crashes at `stepper.py:966`
  (`set_flow_curve_active_mask` missing on `_ThermalOnlyMechanicsProblem`);
  workaround `--no-xla-thermal-only-mechanics-surrogate` used here. Also
  `run_audit` requires `release.vtu` and cannot audit thermal-only runs, and
  the scan dt is silently driven by path-row spacing, not `--dt`.
- **nist.gov and ncbi.nlm.nih.gov were unreachable on 2026-07-29** (local +
  remote fetcher); Lane 2020 was obtained from the Internet Archive copy of
  the NIST TechPubs PDF (sha256 39de5e69...640). Raw data.nist.gov
  CHAL-AMB2018-02 records still to be diffed against the transcription when
  the network allows (D-V1-13).

## Run inventory

| run | condition | output dir (under /home/user/work/159/output/) |
|---|---|---|
| CBM-B baseline | 195 W, 800 mm/s | v1_cbmB_P195_20260729T051047Z (30-step cooling; geometry only) |
| CBM-B long-cool | 195 W, 800 mm/s | v1_cbmB-longcool_P195_20260729T053849Z (authoritative) |
| CBM-A | 150 W, 400 mm/s | v1_cbmA_P150_20260729T061105Z |
| CBM-C | 195 W, 1200 mm/s | v1_cbmC_P195_20260729T064608Z |
| CBM-B rho bound | 195 W, 800 mm/s | v1_cbmB-rho7925_P195_20260729T071318Z |
| CBM-B SB powder k | 195 W, 800 mm/s | v1_cbmB-sbpowder_P195_20260729T074632Z |

Metrics per run: `<run>/v1_meltpool_metrics.json` (extractor:
`model/analyze_v1.py`).
