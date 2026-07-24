# T010/T017 Dynamic Surface and Cooling-Schedule Evidence

**Date**: 2026-07-24
**Scope**: T010 failing-test gate and T017 active-domain thermal boundary
**Claim boundary**: P0 calculation-verification evidence, not a formal paper run

## Frozen semantics

For `surface_selection=exterior`, the thermal problem constructs one
fixed-shape upward-owner face superset at startup. This follows the paper's
"top active element layer" wording and the Spec's "真实暴露顶面" requirement.
Physical face normals are reconstructed from face nodes, oriented outward
using the owner-cell centroid, and filtered by
`normal · build_direction > 1e-10 |normal|`; side and downward faces are
excluded. Each candidate records its owner and adjacent cell (`-1` for the
mesh exterior). At every increment the surface mask is

```text
physical(owner) AND (outside OR NOT physical(neighbor))
```

where `physical` is the printed domain plus any permanent powder domain. Thus:

- an active/future-layer upward interface contributes exactly once;
- an active/active shared top face contributes zero;
- a void/void shared top face contributes zero;
- an active upward mesh-exterior face contributes once.

The face arrays, sparse assembly indices and JIT batch shapes remain fixed.
Only device-resident activity and ambient arrays change per step.

The frozen final-cooling protocol uses the same linear `k/N` schedule for the
fixed-bottom temperature and convection/radiation ambient:

```text
T(k) = T_process + k/N * (T_final - T_process),  k = 1..N
```

Every `StepState` stores both temperatures. `path_used.csv` records the exact
per-step values, while `used_config.json` records the schedule mode, endpoints
and cooling-step count.

## RED evidence

Before implementation, the T010 acceptance command produced:

```text
3 failed, 10 passed, 10 subtests passed
```

The failures were physical rather than import or fixture failures:

- the lower active HEX top at the active/future interface had area `0.0`
  instead of `1.0`;
- activation-stage top areas were `[0.0, 1.0]` instead of `[1.0, 1.0]`,
  so the first exact convection integral was also absent;
- cooling ambient remained `[423.15, 423.15, 423.15]` instead of the frozen
  `[423.15, 361.575, 300.0]` schedule.

The complete-mesh `exterior_only` filter permanently removed the internal
active/future face during `Problem` initialization. The previous owner-only
mask could disable an existing face but could not restore that deleted face.
Separately, `self.ambient` was captured as a static Python attribute during
the first JIT trace.

## GREEN implementation

The implementation adds:

- a host-built, manifold-checked owner/neighbor face topology;
- the opt-in `active_domain_exterior` selector, without changing the static
  `exterior_only` behavior used by other physics;
- a physical-outward-normal top-face selector that works for HEX8 and TET4,
  including either build direction;
- a fixed upward-face superset and per-step physical exposure mask;
- explicit cell-quadrature and face-quadrature ambient inputs to JAX kernels,
  avoiding a device-to-host synchronization or stale JIT closure;
- a frozen `StepState` bottom/ambient schedule;
- schedule columns in `path_used.csv` and structured schedule provenance in
  `used_config.json`;
- thermal-ledger support for the explicit ambient field, with legacy
  14-variable fixtures retained for backward compatibility.

The shared-face regression proves that a lower active top contributes area
`1`, then becomes zero when its upper neighbor activates. A second regression
proves that the two-HEX candidate set contains exactly two upward faces and
that the total active top areas for lower-only, upper-only, both-active and
both-inactive states are `[1, 1, 1, 0]`. This also prevents the fixed superset
from accidentally expanding to all side/internal faces.

An independent formal-mesh count found exactly 29,568 candidates for the
29,568-cell HEX8 model, one upward face per cell. This is 82.7% smaller than
the rejected all-direction superset (170,688 faces). Face quadrature is 50% of
volume quadrature, and estimated warm assembly overhead versus the old static
exterior is about 14--19%, so the dynamic implementation no longer blocks the
29,568-cell CPU sample gate. Formal GPU/G3 performance remains subject to its
own profile gate.

## Reproducible verification

The task acceptance command is:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
python -m pytest -q \
  tests/integration/test_active_surface_boundary.py \
  tests/unit/test_v06_thermal_balance.py
```

```text
17 passed, 10 subtests passed
```

The expanded thermal/source/ledger/active-domain/XLA contract set passed:

```text
144 passed, 10 subtests passed
```

After adding the explicit upward-face and shared-face regressions, the focused
thermal/source/ledger/active-domain set passed:

```text
50 passed, 10 subtests passed
```

The full repository run reached:

```text
534 passed, 2 skipped, 10 subtests passed
```

Its ten remaining failures are pre-existing or intentionally frozen RED gates:

- seven T018/T019 material-history and J2-curve/tangent failures;
- three legacy-v02 runner tests whose referenced
  `legacy/v02/run_ti64_material.py` file is absent.

No T017, thermal-ledger, active-domain, source, or accelerator-contract test
failed.
