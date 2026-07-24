# T009/T016 Active-Domain Red–Green Evidence

**Date**: 2026-07-24
**Scope**: T009 failing-test gate and T016 strict zero-contribution active domain
**Green implementation commit**: `342cc35`
**Claim boundary**: P0 calculation-verification evidence, not a formal paper run

## Frozen semantics

The paper workflow is strict only when both conditions hold:

- `layer_activation_mode=layer_on_scan`
- `future_layer_mode=void`

In that mode, a future cell has exactly zero thermal storage, conductivity,
mechanical stiffness and residual contribution. The static mesh shape is kept
for JAX compilation. A node incident to a contributing cell remains a free
physical node; only nodes owned exclusively by noncontributing cells receive
temporary identity-row Dirichlet constraints. Other activation modes retain
the legacy ersatz-material behavior.

## RED evidence

### Physical-deletion equivalence

Commit `027b1bf` added the first T009 tests before the production helpers
existed. The isolated RED checkout produced:

```text
3 failed, 1 passed
```

The failures were physical, not import/path failures:

- the full masked thermal mesh differed from physical cell deletion by
  `7.0325313281937e-2`, above the frozen `1e-8` gate;
- 9 of 12 active mechanics residual entries differed, with maximum absolute
  difference `7.694391e-1`;
- the accelerated thermal material kernel returned inactive density `4.0`
  instead of exact zero.

### Dynamic boundary refresh

Commit `3c14e37` added a same-`Problem` partial→all-active transition. Before
cache invalidation was implemented, the second thermal solve reused stale
constraint rows and reached residual `8.735e49` after 100 iterations.

### Flat-BC cache version

Before the final cache fix, the new deterministic contract test mutated the
same host node/component/value arrays in place and incremented
`_dirichlet_bc_version`. `_single_var_bc_flat()` still returned old DOFs
`[1, 4]` instead of `[2, 5]`, proving that array identity alone was not a safe
dynamic-BC cache signature.

### All-inactive overlap and weak-solid acceleration

Additional RED tests exposed:

- duplicate all-inactive thermal DOFs
  `[0, 2, 3, 0, 1, 2, 3, 4]`, conflicting with JAX scatter
  `unique_indices=True`;
- strict accelerated weak-solid powder returned `active_factor=0` while the
  reference material path returned `1`.

## GREEN implementation

Commit `342cc35` closes the RED cases with:

- exact-zero future thermal and mechanical coefficients in reference and JIT
  material paths;
- cell participation derived from the final assembled coefficients, rather
  than from a printed label;
- inactive-exclusive node constraints with shared interface nodes retained;
- first-condition precedence when merging BCs, so physical BCs win and each
  constrained DOF is unique;
- explicit FE BC versioning in both PETSc tangent-row and flat residual-BC
  caches;
- strict surface-owner masking, preventing heat flux on future void cells;
- step-0 thermal and optional weak-solid mechanical participation for the
  permanent powder ELSET;
- exact-zero cut/depowder factors and removed-domain constraints for release;
- a thermal ledger that accepts exact-zero void coefficients while requiring
  positive conductivity and heat capacity on the contributing material
  domain.

## Reproducible verification

The task acceptance command passed:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
python -m pytest -q -p no:cacheprovider \
  tests/integration/test_active_domain_equivalence.py \
  tests/unit/test_v06_lifecycle.py
```

```text
13 passed
```

The expanded cache/material/ledger regression set also passed:

```text
76 passed
```

It covered:

- thermal full-mesh versus physical-deletion relative difference `<=1e-8`;
- exact equality after changing all inactive thermal and mechanical
  placeholder factors;
- zero inactive residual and tangent values;
- partial→all-active reuse of one `Problem`;
- all-inactive BC uniqueness and physical-BC precedence;
- base/reference versus accelerated weak-solid powder field equality;
- exact-zero void acceptance by the typed thermal ledger;
- current flat BCs used consistently by `assign_bc`, `apply_bc_vec` and the
  PETSc tangent cache.

An independent review recomputed the formal C3D8 topology:

- build: 23,168 contributing cells, 5,928 constrained inactive nodes and zero
  unconstrained zero-row nodes;
- release after cut/depowder: 9,920 contributing cells, 20,944 constrained
  removed nodes and zero unconstrained zero-row nodes;
- all 189 frozen box-anchor nodes belong to the retained domain.

## Supplementary official-runner smoke

The unmodified official runner completed a temporary three-HEX, two-scan-step
CPU float64 case:

```text
exit_code=0
steps=2
linear_iterations=2
wall_seconds=6.305129532
strict_active_domain=true
ledger.complete=true
maximum_absolute_balance_error_j=4.3782407387e-17
maximum_assembly_identity_error_j=2.7711166695e-17
```

Step 0 used `active=printed=[1,1,0]`; step 1 used `[1,1,1]`. Future-exclusive
nodes were constrained at step 0 and released on activation. Both VTUs had
finite fields, `T=300 K` and `u=0`, as required by the zero-source isothermal
setup. This smoke supplements, but does not replace, the repository tests.

## Review disposition and remaining scope

Two independent read-only reviews found no remaining T009/T016 correctness
blocker. The following items remain outside this closure:

- T020 must select or validate generic release anchors after the final
  cut/depowder contributing domain is known;
- repeated per-step BC reconstruction is a performance optimization target;
- T021 remains open until every P0 slice T007–T020 has its RED/GREEN evidence
  and the complete physics/solver regression is clean.
