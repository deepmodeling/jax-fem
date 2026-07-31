# Solver mechanics-regression re-verification — 2026-07-30 (V session, user-directed)

**Scope**: item 5 of the L3 prerequisite list — first mechanics-chain
verification since the r3-optimization merge (bdd1671) and the surrogate fix
(e7acada). Two gates: the FEniCSx benchmark suite and the kaess golden
equivalence gate (protocol per `GOLDEN_EQUIVALENCE.txt` verdict 2026-07-22:
`MKL_NUM_THREADS=1` deterministic configuration).

## Gate 1 — FEniCSx benchmark suite: **PASS 5/5**

`python -m tests.benchmarks` (CPU, MKL single-thread), 15 s:
hyperelasticity, linear_elasticity_cube, linear_elasticity_cylinder,
linear_poisson, plasticity — all OK against stored FEniCSx gold
(plasticity: displacement diff 5.6e-8; stresses match to 5 decimals).
NOTE: ledger F-section says "four cases"; the suite contains five tests.

## Gate 2 — kaess golden equivalence: **NOT bitwise; both causes traced to documented post-golden physics changes**

Setup: golden command replayed verbatim on the new entry point
(`-m jax_fem_am.simulation.runner`), byte-identical inputs (mesh sha256
827b6ec1... matches the golden manifest; material config d22865b0...; path
file copied from the golden dir), MKL_NUM_THREADS=1. 247/247 steps, per-step
laser state (time/x/y/z/power/dt) identical in path_used.csv.

### Findings

| observable | golden (2026-07-22) | regression run | verdict |
|---|---|---|---|
| step-0 laser: commanded | 0.0075 J | 0.0075 J | identical |
| step-0 laser: deposited / capture | 3.3147e-3 J / 88.39 % | 3.4114e-3 J / 90.97 % | **behavior change #1** |
| step-0 T field | max 8590.6 K | max 8047.2 K; 30 nodes differ > 100 K, all under the laser spot | consequence of #1 |
| step-200 T field | — | 5686 nodes differ > 100 K (divergence has spread) | consequence of #1 |
| step-217 (first cooling) energy balance | **violated** (5.0e-4 J) | clean (1.4e-10 J) | **behavior change #2 — improvement** |
| release u vs golden | — | max diff 4.3e-4 | ≈ documented same-code MT noise band (4.2e-4); **not attributable** — see determinism section: under ST the same field is bitwise stable, so this difference is golden-side noise and/or the thermal change, not new nondeterminism |
| ST-vs-ST determinism (rerun b) | — | **bitwise identical**: T, u and max_temperature_history all max_abs = 0.0 at steps 0/200 AND at release; ledger digit-identical | **determinism PASS** → new-baseline candidate |

### Root causes (both intentional, both post-date the golden)

1. **Strict active domain** (`process/activation.py::uses_strict_active_domain`):
   for the SAME flag combination the golden used (`layer_on_scan` +
   `future_layer_mode=void`), the new code enforces "future, unprinted cells
   contribute exactly zero" — documented in-code as the paper-reproduction
   contract, replacing the historical ersatz behavior. Changes source capture
   at the scan front (88.39 → 90.97 % at step 0); all thermal-field
   divergence follows from this. Not toggleable by flag; the semantics of the
   flag pair themselves changed.
2. **Cooling temperature schedule** (`stepper.py`, derived config key
   `cooling_temperature_schedule: linear_k_over_n_to_final`): the golden run
   itself VIOLATED the energy-balance gate at the first cooling step
   (5.0e-4 J, `balance_within_solver_tolerance: false` recorded in the golden
   ledger); the new linear ramp fixes it (1.4e-10 J, gate true).

### Determinism of the current code (the protocol the 2026-07-22 verdict asked for)

Two independent full runs (a and b) of the identical command under
`MKL_NUM_THREADS=1`:

- **field level: exactly zero difference** — `T`, `u` and
  `max_temperature_history` all report `max_abs = 0.0` at step 0, step 200
  **and at release**. The release displacement field is the observable that
  carried the ±100 % reproducibility band in the 2026-07-22 multithreaded
  investigation; under the single-thread protocol it is bitwise stable even
  on the locked TET4 mesh;
- whole-run ledger summary maxima identical to all 16 significant digits
  (`maximum_absolute_balance_error_j` 4.698363942367591e-07,
  `maximum_assembly_identity_error_j` 6.661338147750939e-16,
  `maximum_relative_balance_error` 7.204132100383321e-05);
- the complete step-0 ledger row (every floating-point field, e.g.
  `laser_deposited_j` 0.0034113922427965408, `balance_error_j`
  8.430730682258865e-18, `storage_j` 0.003409958738609663) matches
  character-for-character;
- both runs report all four gate booleans true and 247/247 steps.

Because these summary values are aggregates over the entire 247-step
floating-point history, digit-for-digit agreement means the arithmetic path
was reproduced exactly. **The deterministic single-thread protocol therefore
holds for the current code**, which is what makes a re-baseline meaningful:
future gate runs can be judged bitwise instead of against a ±100 % noise
band.

Remaining used_config diffs are new keys with behavior-preserving defaults
(`mechanics_acceptance: legacy`, `phase_history_model: legacy_reset`,
`source_model: legacy`, `xla_pardiso_mode: None`).

### Verdict

- **The golden REFERENCE is stale, not the code broken**: every identified
  difference traces to one of two documented, deliberate physics corrections,
  one of which is provably an improvement (energy balance). No evidence of
  unintended drift was found beyond them — but complete attribution (proving
  the step-0 capture change is EXACTLY the strict-domain fix and nothing
  else) needs a code-level A/B with the strict-domain path disabled, which is
  solver-side work outside V-session scope.
- **Determinism restored and proven**: the current code reproduces bitwise
  under `MKL_NUM_THREADS=1`, including the release displacement field that
  previously had a ±100 % band. The 2026-07-22 recommendation is hereby
  validated in practice, and a re-baseline turns the golden gate from a
  noise-limited comparison into a bitwise one.
- Mechanics-side: benchmarks 5/5; the kaess u difference vs the golden cannot
  be separated from the thermal-driven change plus the golden's own
  multithreaded noise band (but is NOT new nondeterminism — see above). The
  HEX8+B-bar production path is not exercised by this gate (golden was c3d4);
  consider adding a c3d8 golden at re-baseline time.

### Recommendation to the main session

1. Confirm the two semantic changes are intended-and-kept (they look like
   r3-opt-era fixes; the cooling one is unambiguously better).
2. **Re-baseline the golden**: adopt the deterministic pair
   `kaess_golden_regression_20260730` / `_20260730_b` (MKL_NUM_THREADS=1,
   current entry point, bitwise-identical pair) as the new reference, update
   `GOLDEN_EQUIVALENCE.txt`, and ideally add a c3d8_powder_margin golden for
   the production element path.
3. Until re-baselined, do not treat the 7-22 golden as authoritative for the
   thermal chain (its cooling step violates the ledger gate that current code
   enforces).

## Addendum 2026-07-31 — c3d8 (HEX8+B-bar) golden, the production element path

The gate above runs on the c3d4 legacy arm because that is what the 2026-07-22
golden used. The V session therefore established a **c3d8 golden on the
production mesh** (`kaess_cantilever_c3d8_powder_margin.inp`, 29,568 cells,
paper-minimal release, phase23), same deterministic protocol
(`MKL_NUM_THREADS=1`).

Run a completed 247/247 (`solver_completed: true`, release.vtu present).
Run b (determinism pair) was killed by a session disconnect at step 5 and
was relaunched detached (`setsid`) on 2026-07-31; the pair verdict is
pending.

**Finding — activation undershoot survives on HEX8.** Run a's ledger reports
`all_temperature_invariants_valid: false`: exactly **one node at step 1**
falls below the plate temperature 423.15 K by more than the 1 mK ledger
tolerance. The same protocol on the TET4 mesh reports **zero** violations at
the same step. This matters because `--thermal-mass-lumping` was introduced
(G1 fix, 664d675) with the claim that it kills activation undershoot exactly
— "T_min == plate temperature bitwise" — and that claim was verified on TET4.
On HEX8 the lumping is a vertex-collocation permutation of the Gauss points
rather than a true row-sum lump, which is the likely reason it is not
exactly conservative here.

Scope of the finding: 1 node / 32,683, 1 step / 247, and the run's energy
ledger still passes every balance and assembly-identity gate
(max relative balance error 1.9e-4, dominated by other steps). It is a
diagnostic-gate failure, not a demonstrated physics error. Magnitude not yet
quantified — the run wrote fields only every 200 steps, so step 1 was not
saved; a targeted short run with `--thermal-output-every 1` will pin it down.

Consequence for the main line: the D-07 meshes are C3D8, so the production
path is the HEX8 one. (Noted in passing: the L0 command observed running on
2026-07-31 does not pass `--thermal-mass-lumping` at all, so it is not
exposed to this specific interaction — but it is also not getting the G1
undershoot fix.)

## Run inventory

| run | purpose | dir (under /home/user/work/159/output/) |
|---|---|---|
| golden reference (stale) | 2026-07-22 baseline, old entry point, multithreaded | kaess_p2_T150C_P250_c3d4_golden1L_tet4b |
| regression run | new entry, MKL 1 thread | kaess_golden_regression_20260730 |
| determinism rerun | ST-vs-ST reproducibility + re-baseline candidate | kaess_golden_regression_20260730_b |

Both regression runs: 247/247 steps, ledger `complete: true`, all four gate
booleans true. Golden (2026-07-22) ledger records
`balance_within_solver_tolerance: false` at step 217.

Comparator: field-level VTU + ledger-numerics script (excluded metadata),
archived at `compare_golden.py` in this directory.
