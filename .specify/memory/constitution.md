# JAX-FEM AM Paper Reproduction Constitution

## Core Principles

### I. Evidence Traceability Is Non-Negotiable

Every scientific input, equation, comparison quantity, and acceptance threshold
MUST be traceable to one of the following evidence classes:
`paper_text`, `paper_table`, `figure_digitized`, `abaqus_semantics`,
`author_artifact`, `inferred`, or `assumption`.

Every `inferred` or `assumption` item MUST record its rationale, expected impact,
uncertainty range, and required sensitivity analysis. A high-impact input MUST
never enter a formal run as an undocumented default.

### II. Physics Parity Precedes Solver Tuning and Performance

The paper model is the scientific contract. Boundary conditions, heat source,
element activation, exposed surfaces, cooling schedule, material history, and
release geometry MUST be aligned or explicitly registered as deviations before
solver tolerances, quasi-Newton updates, factorization reuse, CPU/GPU placement,
or other performance work is accepted for formal results.

No numerical tolerance, absorptivity, hardening value, damping term, liquid
stiffness factor, or history reset may be used to compensate for a known
physics mismatch.

### III. Verification Gates Before Long Runs

Every non-trivial physics change MUST follow:

1. a failing analytical, unit, patch, or small-domain test;
2. the smallest implementation that satisfies the requirement;
3. targeted regression tests;
4. a checkpoint review before the next scale is attempted.

The scale ladder is split by purpose:

- CPU reference: `material point / element → patch / small domain → single
  track → real-DOF short prefix → 1-layer complete build/cooling/release →
  reduced-domain multi-layer history/release mini-cycle`;
- qualified accelerated execution: `3×30 µm → 5×60 µm → 10×30 µm →
  parameter matrix`.

An upstream gate failure blocks downstream formal execution. A completed
pipeline, zero exit code, manifest, or non-empty response gate is necessary
operational evidence, but is never sufficient scientific evidence.

### IV. CPU Verification Anchors; Qualified Accelerators May Be Formal

Float64 CPU runs with fixed threads, frozen inputs, and a recorded environment
are the authoritative **small-scale numerical references**. They MUST cover
every distinct physical transition used by the formal model, including
activation, thermal history, mechanical equilibrium, cooling, and release, but
a full 10×30 µm CPU run is not a mandatory prerequisite.

An accelerated backend MAY become the formal backend for the standard case and
parameter matrix after it passes progressively larger, identical-input
equivalence tests against the CPU references. Qualification MUST compare fields,
events, convergence behavior, and build/cooling/release responses, not only
scalar peaks or successful exit codes.

Backend names MUST describe actual device placement. A run with GPU JAX
assembly and CPU PARDISO is `hybrid_gpu_assembly_cpu_pardiso`, not `full_gpu`.
`full_gpu` requires GPU thermal work, mechanics assembly, linear solve, and
state residence, with no unexpected CPU fallback. Failure of `full_gpu` does
not invalidate a separately qualified hybrid backend. Physics parameters MUST
NOT be changed to make platforms agree.

### V. Claim Discipline and Reproducibility

Kaess 2023 is treated as a `code_to_code_benchmark`. Unless independent
experimental data and a separate validation protocol are supplied, project
outputs MUST NOT be described as experimental validation or proof of real-world
predictive accuracy.

Every accepted run MUST preserve the code identity, dirty state, environment,
hardware, command, input hashes, logs, convergence history, energy ledger,
checkpoints, raw fields, quantities of interest, and comparison results. Failed
runs and failed gates are evidence and MUST NOT be hidden or overwritten.

## Scientific Quality Gates

- **G0 — Source freeze**: claim level, evidence matrix, assumptions, quantities
  of interest, and thresholds are reviewed before implementation.
- **G1 — Physics parity**: paper-consistent boundary, source, activation,
  exposed-surface, cooling, material-history, and release behavior is proven.
- **G2 — Code verification**: analytical, unit, patch, and conservation tests
  pass.
- **G3 — CPU verification baseline**: CPU float64 analytical/small-domain,
  representative full-mesh prefix, 1-layer build/cooling/release, and a
  reduced-domain multi-layer history/release mini-cycle pass time, path, mesh,
  solver-tolerance, and repeatability gates.
- **G4 — Backend qualification and promotion**: each accelerated mode
  independently meets field-, event-, convergence-, and
  build/cooling/release-equivalence gates, records its true CPU/GPU placement,
  and passes the preregistered identical-input performance gate before it can
  be promoted as a formal accelerated backend.
- **G5 — Accelerated standard reproduction**: a qualified backend completes
  the 3×30 µm and 5×60 µm bridges and the formal 10×30 µm standard case; the
  preregistered Figure 8 and Figure 9 metrics pass before the formal parameter
  matrix is opened. Partial or failed comparisons are retained without
  post-hoc threshold changes and may continue only as explicitly diagnostic
  work.
- **G6 — Accelerated parameter matrix and performance report**: only promoted
  backends run the formal paper matrix; speedup and resource claims retain the
  G4 identical-physics evidence and transparent cold/warm timing.
- **G7 — Reproducibility package**: an independent clean-directory rerun
  regenerates the accepted quantities of interest and report inputs.

## Development Workflow

Work follows the Spec Kit lifecycle:

`constitution → spec → clarify → plan → checklist → tasks → analyze
→ implement → converge`.

- The specification is updated before requirements change.
- Tasks are dependency ordered, independently verifiable, and normally touch no
  more than five files.
- Implementation uses test-driven, incremental slices and keeps documentation,
  physics changes, solver changes, and performance changes reviewable.
- Formal runs are never launched from a draft configuration.
- Any threshold, quantity of interest, sampling path, or claim-level change
  requires explicit human approval and an updated specification.

## Governance

This constitution overrides convenience scripts, legacy case comments, and
ad-hoc tuning decisions for paper-reproduction work.

Amendments require:

1. a written rationale and affected requirements;
2. an updated version and amendment date;
3. review of downstream specifications, plans, tasks, and accepted evidence;
4. explicit approval before affected formal results are regenerated.

Versioning follows semantic intent:

- MAJOR: changes scientific claim boundaries or removes a core principle;
- MINOR: adds a principle, gate, or mandatory evidence class;
- PATCH: clarifies wording without changing obligations.

**Version**: 1.2.0 | **Ratified**: 2026-07-23 | **Last Amended**: 2026-07-23

**Amendment 1.1.0**: Replaced the mandatory full-scale CPU-first sequence with
CPU small-scale verification anchors followed by qualified accelerated formal
execution. This preserves CPU numerical authority while avoiding a redundant
full 10-layer CPU campaign and makes backend placement claims auditable.

**Amendment 1.2.0**: Required both numerical equivalence and preregistered
performance qualification before a backend is promoted for formal accelerated
reproduction. Numerical equivalence without speedup remains useful diagnostic
evidence but cannot support an acceleration claim.
