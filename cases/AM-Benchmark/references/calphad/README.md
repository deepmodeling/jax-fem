# CALPHAD source and trial — decision D-08 (approved 2026-07-29)

Replaces the "Thermo-Calc TCNI + Scheil" candidate in the E-table with a fully
reproducible open-source route: **pycalphad 0.11.2 + scheil 0.3.0 + MatCalc
open Ni database mc_ni_v2.036**.

## Source

| File | What it is |
|---|---|
| `mc_ni_v2036.tdb` | Original download, byte-identical to matcalc.at. sha256 `84ba8131…f305313ab` |
| `mc_ni_v2036_pycalphad.tdb` | Derived: pycalphad-compatible copy produced by `filter_tdb.py` |
| `filter_tdb.py` | The adaptation script (run on a Latin-1→UTF-8 transcoded copy) |
| `trial.py` / `post.py` | Trial computation and post-processing |
| `trial_results.json` | Full numeric results (equilibrium scan, Scheil path, cp scans) |

- **Origin**: https://www.matcalc.at/images/stories/Download/Database/mc_ni_v2036.tdb
  (MatCalc open databases page). Assessed at TU Wien by E. Povoden-Karadeniz;
  version dated 2024-08-20.
- **License**: Open Database License (ODbL) 1.0; contents DbCL 1.0. Free use
  and redistribution with attribution — stated in the file header.
- **Elements**: Ni, Al, B, C, Co, Cr, Cu, Fe, Hf, La, Mn, Mo, N, Nb, O, S, Si,
  Ti, V, W, Y, Zr. Phases include LIQUID, FCC_A1 (γ + MC via second composition
  set), GAMMA_PRIME, GAMMA_DP, DELTA (δ-Ni₃Nb), LAVES/LAV_C14, M23C6, M6C,
  σ/μ/P TCP phases.

## Adaptation (syntax only — zero thermodynamic edits)

`filter_tdb.py` produces `mc_ni_v2036_pycalphad.tdb` from the UTF-8 transcode:

1. Drops MatCalc-only metadata commands pycalphad's grammar rejects:
   `REFERENCE_ELEMENT` (1), `ADD_COMPOSITION_SET` (9), `ATTACH_CONTRIBUTION`
   (1), stray reference-section text (1).
2. Drops the 13 `HMVA` parameters (MatCalc vacancy-formation enthalpies,
   thermo-kinetic only).
3. Expands MatCalc abbreviations `CONST`→`CONSTITUENT` (3: DELTA, GAMMA_DP,
   GAMMA_PRIME) and `PARAM`→`PARAMETER` (1).
4. Repairs an upstream typo `6000.00.00`→`6000.00` (2 temperature limits).

All 3587 thermodynamic parameters (1443 G incl. the expanded `PARAM`, 1981 L,
96 TC, 67 BMAGN) are carried over unchanged. The database has **no molar-volume parameters** — ρ(T) cannot
come from it (E-table gap unchanged).

## Trial results (AMB2018-01 mill-cert composition, EOS lot M421601)

Composition wt%: Cr 20.61, Mo 8.82, Nb 3.97, Fe 0.81, Ti 0.39, Al 0.30,
Si 0.18, C 0.02, Ni bal. (traces Co/Mn/Ta/P/S/O/N dropped).

| Quantity | Value |
|---|---|
| Equilibrium solidus / liquidus | 1552 K / 1630 K (1279 / 1357 °C) |
| Latent heat | 259 kJ/kg |
| Scheil fs = 0.05 / 0.50 / 0.90 / 0.95 / 0.98 | 1354 / 1328 / 1202 / 1150 / 1129 °C |
| Scheil terminal transient | 871 °C |
| γ-frozen cp (FCC_A1 only) | 479 J/(kg·K) @300 °C → 584 @1000 °C |

### Gates

1. **Special Metals melting range 1288–1349 °C**: computed 1279–1357 °C, both
   bounds within ~9 °C — PASS.
2. **Ghosh (NIST) CALPHAD solidus 1587 K**: computed 1552 K, Δ = 35 K low.
   Partly attributable to the mill-cert composition (Nb 3.97 wt% near the top
   of spec vs nominal compositions used by Ghosh) — registered in conflict B3,
   not treated as failure.
3. **Gen3 CSP cp 260–1000 °C**: γ-frozen cp inside the 95 % CI at 6 of 9
   temperatures, ≤10 % high elsewhere — PASS with notes.

### Registered caveats

1. Database tested-composition window (header): Cr < 20, Mo < 5–8 wt%;
   mill cert has Cr 20.61, Mo 8.82 — slightly outside.
2. No molar volume ⇒ no ρ(T) (gap stays with Kaschnitz).
3. γ-frozen cp shows an isolated bump at 260 °C (540 J/(kg·K); likely the
   magnetic-model Curie anomaly or a numerical artifact); smooth ≥300 °C.
4. Equilibrium-cp scan has 5 non-converged points at 775–850 K (does not
   affect the γ-frozen cp used by the model).

## Conventions established (see PREREQUISITES.md D.6)

- Mushy-zone latent-heat release follows the **Scheil fs(T) curve** directly —
  no single "effective solidus" is picked (conflict B3 stays open as a
  definitional note).
- Model cp is the **γ-frozen (FCC-only) cp**, not equilibrium cp: equilibrium
  cp double-counts δ/carbide dissolution enthalpies (833 vs measured 469
  J/(kg·K) at 500 °C) that the as-built γ microstructure does not exhibit at
  heating-rate-relevant timescales.

## Reproduction

```bash
iconv -f LATIN1 -t UTF-8 mc_ni_v2036.tdb > mc_ni_v2036_utf8.tdb
python filter_tdb.py        # paths inside point at /tmp/calphad; adjust
python trial.py             # ~13 min on 24-core CPU (Scheil dominates)
python post.py
```

Environment: WSL conda env `jax-fem-env` (python 3.13), pycalphad 0.11.2,
scheil 0.3.0, symengine 0.13.0 — installed from the TUNA PyPI mirror.
