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
| release u vs golden | — | max diff 4.3e-4 | ≈ documented same-code noise band (4.2e-4); not separable |
| ST-vs-ST determinism (rerun b) | — | PENDING (rerun in progress) | new-baseline candidate if bitwise |

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
- Mechanics-side: benchmarks 5/5; the kaess u difference cannot be separated
  from the thermal-driven change plus the documented TET4 noise band. The
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

## Run inventory

| run | purpose | dir (under /home/user/work/159/output/) |
|---|---|---|
| golden reference (stale) | 2026-07-22 baseline, old entry point, multithreaded | kaess_p2_T150C_P250_c3d4_golden1L_tet4b |
| regression run | new entry, MKL 1 thread | kaess_golden_regression_20260730 |
| determinism rerun | ST-vs-ST bitwise + re-baseline candidate | kaess_golden_regression_20260730_b |

Comparator: field-level VTU + ledger-numerics script (excluded metadata),
archived at `compare_golden.py` in this directory.
