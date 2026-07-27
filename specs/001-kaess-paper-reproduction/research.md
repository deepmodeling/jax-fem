# Phase 0 Research: Kaess 2023 论文级数值复现

**Date**: 2026-07-23

**Status**: Initial source and implementation review complete; author-input gaps open

## Research Question

当前 JAX-FEM AM 实现距离 Kaess et al. (2023) 的公开数值模型还有哪些
高影响差异？什么证据足以支持“论文级复现”，以及 CPU/GPU 应如何进入
科学证据链？

## Primary Sources

| Source | Role | Local/online location |
|---|---|---|
| Kaess et al. 2023 paper | Authoritative published model and results | `cases/kaess_2023/references/cases/kaess_2023_paper.pdf` |
| Publisher article | Citation and public article | <https://doi.org/10.3390/ma16062321> |
| Extracted full text | Searchable local evidence | `cases/kaess_2023/references/cases/kaess_2023_fulltext.txt` |
| Benchmark metadata | Current parameter and digitization register | `cases/kaess_2023/references/cases/kaess_2023.json` |
| Existing benchmark plan | Historical decisions; not the new source of truth | `cases/kaess_2023/references/cases/kaess_2023_benchmark_plan.md` |
| Current formal-like launcher | Current implementation and documented deviations | `cases/kaess_2023/run_kaess_phase2.sh` |
| Medium launcher | Regression-only reduced-order case | `cases/kaess_2023/run_kaess_medium_fullheight.sh` |

## Confirmed Reference Model

- Weakly coupled implicit thermal and mechanical Abaqus workflow.
- 29,568 DC3D8/C3D8 elements.
- Layer activation via Abaqus MODEL CHANGE.
- Standard case: 10×30 µm, build plate 150°C, 250 W, 850 mm/s,
  50 µm beam radius, 100 µm hatch, 67° inter-layer rotation.
- Hemispherical volumetric Gaussian source with absorptivity.
- Weak-solid powder with temperature-dependent conductivity.
- Cooling followed by partial support removal and upward cantilever bending.
- Primary published comparisons are pre-release residual-stress depth profiles
  and post-release bending/maximum deflection.

## Current Matches

The repository already contains or approximates:

- reference total element count and a C3D8 powder-margin mesh;
- standard power, speed, preheat, beam radius, hatch, layer thickness and count;
- weak coupling, cooling and release pipeline stages;
- powder weak-solid modulus/yield parameters;
- thermal ledger, run audit, manifest, response gate and XRD operators;
- unit tests for J2, B-bar, thermal balance, lifecycle and provenance.

These matches are useful foundations but do not close the paper-parity claim.

## Confirmed P0 Gaps

| ID | Current behavior | Reference requirement | Decision |
|---|---|---|---|
| P0-BC | Full bottom `ux=uy=uz=0` clamp | Bottom `uz=0` with partial in-plane freedom | Add paper-style minimal-rigid-body BC and tests |
| P0-HS | Plane Gaussian × exponential depth | Hemispherical 3D Gaussian | Implement exact formula and power-integration gate |
| P0-ACT | Future elements retain full thermal mass and weak stiffness | MODEL CHANGE inactive domain | Implement exact zero-contribution activity semantics |
| P0-SURF | Static complete-mesh exterior faces | Current active exposed top surface | Rebuild exposed faces from activity each stage |
| P0-COOL | Bottom ramps; convection/radiation ambient remains preheat | Cooling environment switches under frozen protocol | Add stage-dependent ambient and schedule audit |
| P0-MAT | Constant powder conductivity and extra mushy/liquid scaling | Temperature-dependent powder and source-backed high-T behavior | Freeze tables; remove unsupported duplicate scaling |
| P0-J2 | Simplified linear hardening/scalar history | Published/Abaqus temperature-dependent plastic behavior | Digitize/obtain full curves and prove incremental update/tangent |
| P0-HIST | Stress-free/eqp reset options are enabled | No published support identified | Disable unless source or declared assumption justifies |
| P0-REL | Geometric release boxes and ersatz deletion | Exact support cell set from Figure 7/author input | Freeze, visualize and hash explicit cell set |

## Key Decisions

### Decision R-001 — Default claim is public-information code-to-code reproduction

**Rationale**: The paper does not publish enough information for exact Abaqus
input equivalence. Missing items include precise in-plane anchors, step times,
full material curves, scan ordering, user subroutines and cut element identities.

**Rejected alternative**: Silently infer these inputs and call the result exact.
That would make the claim stronger than the evidence.

### Decision R-002 — Existing 3×100 µm run remains regression-only

**Rationale**: It preserves full build height but changes the number of laser
passes, thermal cycles, activation events and plastic-history accumulation.

**Rejected alternative**: Treating equal total height as sufficient paper parity.

### Decision R-003 — Build-state validation precedes release

**Rationale**: Release displacement is a response to the accumulated
pre-release residual-stress field. Tuning the cut or anchors cannot repair an
incorrect upstream field.

### Decision R-004 — CPU float64 small samples are the numerical reference

**Rationale**: Current mechanics uses CPU PARDISO even in hybrid runs, and
ill-conditioned problems can amplify ordering differences. A deterministic
CPU float64 reference remains necessary, but it only needs to cover every
physical transition and representative matrix scale. Requiring two complete
10-layer CPU runs would duplicate the expensive campaign without adding a new
physics mode after small-scale equivalence has been established.

### Decision R-005 — Validation thresholds are preregistered engineering gates

**Rationale**: The paper does not supply acceptance criteria. Project thresholds
must be frozen before formal results and reported as project choices.

### Decision R-006 — Current accelerated route is hybrid, not full GPU

**Rationale**: Live source inspection shows JAX local kernels use the selected
GPU, then Jacobians are converted to host NumPy, assembled into a CPU PETSc
matrix, and solved by CPU MKL PARDISO. The runtime explicitly records
`full_loop_xla=false`. The correct name is
`hybrid_gpu_assembly_cpu_pardiso`.

`full_gpu` remains a separate future qualification requiring GPU sparse linear
solve, major state residence, no CPU PARDISO, and no unexpected fallback.

### Decision R-007 — Scientific equivalence and performance are separate tests

**Rationale**: A recent 20-step CPU run and its CPU repeat diverged strongly in
an all-node displacement maximum while their active/printed-domain fields were
close. Scientific reference runs therefore use single-thread MKL/OMP and native
float64 active-domain checkpoints. Performance tests use the same CPU thread
budget for CPU and hybrid, run sequentially, and do not redefine scientific
truth.

## Preregistered First-Pass Thresholds

| Metric | Proposed gate |
|---|---|
| Heat-source numerical integral | relative error ≤ 0.5% |
| Latent-heat integral | relative error ≤ 0.5% |
| Thermal energy closure | ≤ 1% |
| Active-domain vs physical deletion, small model | relative difference ≤ `1e-8` |
| Time/path refinement | key QoI difference ≤ 2% |
| Solver-tolerance tightening | key QoI difference ≤ 1% |
| Two CPU small-reference repeats | active-domain max displacement difference ≤ `max(0.1 µm, 1%)` |
| Figure 9 bending curve | NRMSE ≤ `max(10%, 2×digitization uncertainty)` |
| Figure 8 stress peak/trough | suggested amplitude error ≤ 15% |
| Figure 8 zero crossing | depth error ≤ one local element height |
| GPU thermal | temperature QoI ≤ 0.1%; events and phases identical |
| GPU/hybrid mechanics | stress QoI ≤ 1%; displacement ≤ `max(0.5 µm, 2%)` |
| Accelerated performance | proposed wall speedup ≥ 1.20× on two paired samples; linear solves +≤10% |

These values require user approval in the specification Review Gate.

## Existing Quantitative Anchors

The current reference JSON records:

- Figure 9a, 150°C maximum front bending: approximately
  `14.0 ± 0.3 µm`;
- Figure 8b, 150°C `σx`: surface about `250 MPa`, peak about `650 MPa`
  near `-0.15 mm`, zero crossing near `-0.21 mm`, and minimum about
  `-450 MPa` near `-0.30 mm`;
- all reported parameter sets bend upward after cut;
- increasing preheat reduces stress and bending;
- 30 and 60 µm layer-thickness responses are reported as close.

The metadata field `source.verification_status` still says figure values are
not digitized while later fields contain digitized values. This inconsistency
must be corrected during G0 rather than propagated into formal manifests.

## Unresolved Author Inputs

| Unknown | Why it matters | Resolution |
|---|---|---|
| Exact bottom in-plane anchor nodes | Can alter residual stress and release | Request author input; otherwise minimal-anchor sensitivity study |
| Recoat/cooling step durations | Controls thermal history | Request; otherwise preregister bounded sensitivity |
| Exact scan start/order | Changes local thermal cycles | Request user subroutine/path; otherwise freeze documented reconstruction |
| Full plastic/expansion tables | Controls residual stress | Obtain cited data/author tables; preserve digitization uncertainty |
| Melt/solidification history semantics | Controls plastic history | Inspect author subroutine; otherwise explicit assumption and cycle tests |
| Exact cut cell set | Controls bending response | Author input or Figure 7 digitization with cell-set sensitivity |

## CPU/GPU Research Boundary

The approved research sequence is:

1. prove the mathematical model on deterministic CPU material/element/patch
   tests;
2. run CPU small-domain, real-DOF 12–20-step prefix, and the smallest complete
   1-layer build/cooling/release reference plus a reduced-domain 3-layer
   history/release mini-cycle, with native float64 checkpoints;
3. compare each accelerated mode on the exact same Git identity, inputs,
   precision, active/printed mask and acceptance model;
4. qualify cooling and release before any formal warpage claim;
5. after qualification, run accelerated `3×30 → 5×60 → 10×30 µm`;
6. use the qualified accelerated backend for the standard paper comparison and
   parameter matrix; a complete 10-layer CPU run is optional audit evidence,
   not a mandatory predecessor;
7. keep `hybrid_gpu_assembly_cpu_pardiso`,
   `gpu_dominant_experimental`, and `full_gpu` as separate verdicts.

No parameter may be retuned for backend agreement.

## Live GPU Evidence on 2026-07-23

The newest same-config diagnostic pairs support continued GPU development, but
do not yet qualify formal release:

| Pair | CPU wall | Hybrid wall | Speedup | Linear solves |
|---|---:|---:|---:|---:|
| 20 scan steps | 607.277 s | 425.909 s | 1.426× | 330 vs 351 |
| 12 scan steps | 395.453 s | 262.208 s | 1.508× | 212 vs 213 |

For the 20-step pair, assembly fell from about 189.8 s to 42.9 s while the
PARDISO phase remained CPU-bound. VTU-level active/printed comparisons showed
identical temperature and activation events, stress/eqp differences below the
draft 1% gate, and displacement differences below the draft release gate.
However, these files are written as float32 and cannot establish internal
float64 equivalence.

The probes have three decisive limitations:

- only 12/20 scan rows; `cooling_steps=0` and no release;
- `liquid_mechanics_factor=1.0` was a runtime diagnostic override while the
  formal material file remains `0.0001`;
- the earlier partial full-scale throughput sample used a different checkout,
  Git identity and material factor, so it cannot be merged into one
  qualification chain.

Therefore the current evidence qualifies the route as a promising
GPU-assembly candidate, not a completed GPU fix, release qualification, or
full-GPU implementation.

## Research Output Needed Before Implementation

- source matrix with page/figure/formula locations;
- author-input request and response log;
- cleaned digitized CSV with uncertainty;
- approved assumption register;
- approved acceptance thresholds;
- approved calibration versus held-out parameter split;
- decision on repository-local material input versus external hash.
