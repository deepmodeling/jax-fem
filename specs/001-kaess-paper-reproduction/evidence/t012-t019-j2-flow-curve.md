# T012/T019 Flow-Curve and Canonical J2 Evidence

**Date**: 2026-07-27

**Scope**: T012 RED contract and T019 solver-capability implementation

**Claim boundary**: P0 code-verification evidence. The repository can load and
solve a frozen multi-temperature, multi-plastic-strain flow curve with a
residual/tangent-consistent J2 update. The available Kaess Figure 4(b)
digitization remains `pending_review`; this evidence is not proof of the
authors' original Abaqus plastic table.

## Source boundary

Kaess et al. (2023), Section 2.5 and Figure 4(b)
([DOI 10.3390/ma16062321](https://doi.org/10.3390/ma16062321)) state that
Abaqus linearly interpolates the supplied property ranges and keeps endpoint
values constant outside them. Figure 4(b) visibly supports the following
two-endpoint approximations:

| Temperature | Flow stress at `eqp=0` | Flow stress at `eqp=0.4` |
|---|---:|---:|
| 20 degC | 500 MPa | 600 MPa |
| 400 degC | 400 MPa | 500 MPa |
| 800 degC | 300 MPa | 400 MPa |
| 1000 degC | 150 MPa | 200 MPa |
| 1370 degC | approximately 20 MPa | approximately 20 MPa |

The room-temperature curve is rounded from reference 33; the higher
temperature curves are identified by the paper itself as assumptions. The
page image cannot resolve the former `1 MPa` hardening floor at 1370 degC.
The `1 MPa` molten minimum at 1400 degC and its higher-temperature extension
are solver realizations, not Figure 4(b) nodes.

The pending digitization records the page-image SHA-256, pixel-axis
calibration, per-curve source origin, unquantified reading-error status, and
solver-realization nodes in:

```text
cases/kaess_2023/candidates/g0-v2-t018/
  flow_curve_table.pending.csv
  flow_curve_table.pending.metadata.json
```

`A-MATERIAL-CURVES` and `KAESS-AUTH-004` remain open. The candidate is not
promotion-eligible until its reading error is independently frozen and the
material bundle is explicitly reapproved.

## RED evidence

The T012 gate first exposed two independent gaps in the pre-T019 path:

- the scalar `yield(T) + H(T) * eqp` interface could not preserve a generic
  rectangular nonlinear flow-curve grid in the constitutive update;
- the mechanics residual used a second return-map formula whose
  within-increment saturation crossing differed from the canonical J2 map,
  so the AD tangent and committed update were not guaranteed to be
  same-source.

Two additional fail-closed tests were introduced during review and failed
before their fixes:

```text
test_flow_curve_table_rejects_blank_source_provenance
  Failed: DID NOT RAISE ValueError

test_flow_curve_material_update_rejects_a_missing_selector
  Failed: DID NOT RAISE ValueError
```

These failures were physical/provenance contract failures, not import,
environment, or tolerance failures.

## Implemented input contract

`FlowCurveTable` reads the long-form columns:

```text
temperature_K,equivalent_plastic_strain,flow_stress_Pa,source
```

Host-side validation requires:

- at least two strictly increasing temperatures and plastic-strain nodes;
- a complete rectangular grid without duplicate nodes;
- plastic strain beginning at zero;
- finite, positive stresses that are nondecreasing with plastic strain;
- a non-empty source on every node;
- exclusive selection of the flow curve or legacy yield/hardening tables.

Paths resolve relative to the material-config directory, with the former
launch-directory behavior retained only as a legacy fallback. Provenance uses
the same resolution rule.

## Exact constitutive update

At fixed temperature, the tabulated curve is linearly interpolated to
`sigma_i(T)`. With old equivalent plastic strain `p_n`, trial equivalent
stress `q_trial`, and `A = 3 mu`, the consistency equation is:

```text
q_trial - A * (p - p_n) = sigma_y(p, T)
```

or:

```text
A * p + sigma_y(p, T) = q_trial + A * p_n
```

The left side is monotone and piecewise linear. The implementation builds its
values at every plastic-strain knot, locates the final segment once with
`searchsorted(method="compare_all")`, and solves that segment in closed form.
This permits one increment to cross multiple curve segments without a nested
Newton loop. The right endpoint is a constant-flow-stress plateau, matching
the frozen clamp contract.

An optional global saturation cap is not applied by clipping input knots.
Instead, an independent closed-form cap root is selected only when the raw
curve at that root has reached the cap. This preserves the exact kink if the
cap crosses inside a segment and avoids `inf` arithmetic through a finite
safe-cap value.

## One source for residual, tangent, and state

`ThermoMechanical._material_point_update()` is now the single constitutive
entry point used by:

- raw mechanics residual and committed `eqp`;
- v06 state-safe residual, stress postprocessing, `eqp`, and tensor
  `eps_p` commit;
- plain and B-bar element kernels;
- build, cut, depowder, cooling, and release mechanics instances.

The existing nine public mechanics parameter slots remain unchanged.
Flow-curve selection is an internal per-quadrature field. Solid, mushy,
liquid, substrate, and support points select the solid curve; permanent
weak-solid powder, void, cut, and inactive points do not. A configured curve
with an unbound or missing selector now fails closed rather than silently
falling back to scalar material data.

The accelerated path does not contain a second J2 constitutive kernel. J2
continues to use the canonical material function inside the compiled
element-residual/Jacobian path.

## Pending material candidate and hash chain

The pending candidate replaces simultaneous scalar yield/hardening inputs
with the `7 x 2` flow-curve grid. The final two temperature rows are explicitly
marked as solver realizations. The scalar CSV files remain in the bundle only
as superseded, unloaded references.

```text
candidate material config
  023253394aef32e0245943c73bb2bddbaca4b31410ac2502fa986818525891fb
material bundle manifest
  fa401cee52ee2c82be833d64848ea43de805ee8cf14c7ed7304d0719cc0889b4
reapproval request
  5bcd2dd3fba950fe8c9fa0a9314560d1526e1d76cccfd16fafab8eda967ef6e3
pending flow-curve table
  ae9b0fe8c8e30e715618ff0b6c2dcc82d1565545742b63c0578c63a8e91f8c3d
digitization metadata
  0c7c8637cbdd97f0eb51dbfca6b123cb17b1dfe351e0d6b018af6882165dee0c
```

An executable contract test recomputes the complete config-to-manifest and
manifest-to-request chain, verifies the canonical G0 reference is unchanged,
and loads the candidate after changing to an unrelated temporary directory.
Every approval field remains pending and `promotion_eligible` remains false.

## Reproducible verification

Frozen CPU focused regression:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
python -m pytest -q \
  tests/unit/test_kaess_j2_tangent.py \
  tests/unit/test_kaess_flow_curve_table.py \
  tests/unit/test_v06_material_validation.py \
  tests/unit/test_v06_j2_kernel.py \
  tests/contract/test_v06_adapter.py \
  tests/contract/test_kaess_material_candidate.py \
  tests/regression/test_v05_plastic_history.py \
  tests/unit/test_v06_lifecycle.py \
  tests/unit/test_v03_bbar_hex8.py \
  tests/unit/test_v03_weak_solid_powder.py \
  tests/integration/test_v03_physics_fixes.py \
  tests/integration/test_v06_provenance.py \
  tests/integration/test_active_domain_equivalence.py \
  tests/unit/test_kaess_accelerated_history.py
```

```text
121 passed in 24.84s
```

Independent mathematical review additionally reported:

- `62` focused CPU tests and `20` RTX 5080 tests passed;
- `2000` random cross-segment/saturation probes had maximum normalized
  consistency error `1.61e-14`;
- a real one-cell HEX8 B-bar flow-curve residual completed JIT with finite
  residual and AD tangent;
- no Critical or Important mathematical finding remained.

The reviewer found two initially green tests that did not hit their named
branches. They were replaced with exact-root constructions: `p=0.075` crosses
a knot and ends inside the later segment; a `495 MPa` cap is tested at
`p=0.04` before and `p=0.08` after its within-segment crossing.

An independent integration/interface review found no remaining Critical,
Important, or Minor issue after the final fail-closed guard. Its additional
CPU sets passed `31` and `42` tests, its CUDA set passed `31` tests, and direct
raw/v06 plain/B-bar selector, residual, AD-tangent, cutback, and release probes
all passed.

This closes the T019 solver-capability implementation. It does not close the
paper-source deviation or approve the pending material candidate.
