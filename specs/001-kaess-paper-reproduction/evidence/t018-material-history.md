# T018 Material, Enthalpy, and Phase-History Evidence

**Date**: 2026-07-27

**Scope**: T018 runtime and candidate-material implementation

**Claim boundary**: P0 code-verification evidence; the G0-v2 material bundle is
still `pending_review` and is not a formal paper input

## Source-backed contract

Kaess et al. (2023), Section 2.5 and Table 2
([DOI 10.3390/ma16062321](https://doi.org/10.3390/ma16062321)) establish:

- latent heat `L = 280 kJ/kg` over `1643.15--1673.15 K`;
- powder conductivity of `0.15 W/(m K)` at `293.15 K` and
  `0.60 W/(m K)` at `1643.15 K`, with linear interpolation and endpoint
  clamping;
- powder specific heat and thermal expansion equal to the corresponding
  temperature-dependent solid values;
- an irreversible field-variable change from powder to solid after first
  melting;
- high-temperature solid curves that already represent molten response.

The paper does not report a second liquid/mushy stiffness multiplier, a
stress-relaxation reference reset, or an equivalent-plastic-strain reset.
`A-PHASE-HISTORY` therefore keeps those mechanisms disabled and treats the
first solidification reference as an explicit, sensitivity-requiring solver
realization.

## RED evidence

The expanded material-history test first produced:

```text
7 failed, 3 passed
```

The failures demonstrated that:

- an increment crossing the complete melting interval represented zero latent
  heat because its coefficient was frozen at the old temperature;
- the thermal weak form did not contain an enthalpy difference;
- a later melt cycle overwrote the first reference temperature;
- the legacy reset flag erased `eqp`;
- the legacy relaxation temperature replaced the local reference;
- the tabulated high-temperature modulus was multiplied by additional
  liquid/mushy factors;
- weak-solid powder incorrectly used zero thermal expansion.

Two additional RED checks isolated adjacent issues:

- the energy ledger rejected the phase-change inputs and could not reproduce
  the same enthalpy increment as the weak form;
- fractional values around the activity threshold produced fractional shared
  face weights (`0.250001`, `0.499999`) instead of a binary exposed surface.

## Implemented runtime semantics

The thermal residual now uses

```text
rho * [cp * (T_new - T_old)
       + L * (f_liquid(T_new) - f_liquid(T_old)) * physical] / dt
```

where `f_liquid` is the clipped linear fraction over the frozen melting
interval. This captures the complete `280 kJ/kg` even when one nonlinear
increment crosses both interval endpoints. The solver-facing thermal ledger
uses the identical storage expression, including its pre-solve state-override
audit. The legacy apparent-capacity branch remains available only when no
valid latent-heat interval is configured.

`thermal_material_quads(..., T_new_quad=...)` also exposes a secant apparent
capacity for material-point audits. The accelerated wrapper forwards that
argument to the canonical implementation rather than silently using its
old-temperature-only kernel.

The `paper_irreversible` phase-history update now:

- permits powder to enter the initial melt transient and become solid once;
- treats `STATE_SOLID` as an absorbing history state thereafter;
- latches `T_ref` only at first solidification;
- preserves `eqp` and tensor `eps_p` on later melt events;
- uses the same canonical function on CPU and JIT paths;
- treats the old reset/relaxation switches as inert compatibility inputs.

Weak-solid powder uses the solid `alpha(T)` curve. When `E(T)` is present,
liquid and mushy active factors are exactly one, preventing duplicate
softening. The CLI default remains `legacy_reset`, preserving the historical
reversible melt, relaxation-reference, and plastic-reset behavior for old
workflows. Both Kaess launchers explicitly select
`--phase-history-model paper_irreversible` and
`--no-reset-plastic-on-melt`.

Material table paths now resolve relative to the material-config directory,
with the former launch-directory convention retained only as a compatibility
fallback.

## Shared-face boundary hardening

The dynamic top-surface mask now first converts the physical cell indicator to
a strict binary value at `> 0.5`. For two cells sharing an upward interface,
the exact owner/neighbor states are:

```text
lower physical, upper void  -> lower shared top = 1, upper top = 0
lower void, upper physical  -> lower shared top = 0, upper top = 1
both physical               -> lower shared top = 0, upper exterior top = 1
both void                   -> both = 0
```

The topology remains manifold-checked and requires a conforming mesh whose
shared faces reuse the same node identifiers. Geometrically coincident faces
with disjoint node IDs are topological cracks and must be rejected by the
formal mesh-quality gate rather than silently merged.

## Unapproved G0-v2 candidate

The non-canonical bundle is stored at:

```text
cases/kaess_2023/candidates/g0-v2-t018/
```

It contains a bundle-local material JSON, all referenced CSV files, the exact
powder-conductivity table, the pending Figure 4(b) flow-curve digitization and
metadata, a per-file SHA-256 manifest, and a reapproval request. After adding
the explicit paper-history selector, the T019 flow-curve candidate, and
rebuilding the content-addressed chain:

```text
candidate material config
  023253394aef32e0245943c73bb2bddbaca4b31410ac2502fa986818525891fb
material bundle manifest
  fa401cee52ee2c82be833d64848ea43de805ee8cf14c7ed7304d0719cc0889b4
reapproval request
  5bcd2dd3fba950fe8c9fa0a9314560d1526e1d76cccfd16fafab8eda967ef6e3
powder conductivity table
  a4e762578fd7a7dd585a315157e0423b150c4f15e235076925f1b50315a89eab
pending flow-curve table
  ae9b0fe8c8e30e715618ff0b6c2dcc82d1565545742b63c0578c63a8e91f8c3d
flow-curve digitization metadata
  0c7c8637cbdd97f0eb51dbfca6b123cb17b1dfe351e0d6b018af6882165dee0c
```

Every approval field remains absent and both `status` and `decision` remain
`pending_review`. No canonical G0 record, paper-parity config, source manifest,
or external material file was changed. T018 and PAR019/PAR024/PAR025 therefore
remain open until the material bundle is independently reviewed, explicitly
approved, and bound into the formal provenance chain.

An executable contract test recomputes every load-bearing hash, verifies the
canonical-approval reference, and loads the candidate through the formal
parser after changing to an unrelated temporary working directory. Both a
manual repository-root run and the cross-CWD test resolved every referenced
table inside the candidate directory, selected the `7 x 2` flow-curve grid,
and left the superseded scalar yield/hardening tables unloaded.

## Reproducible verification

Focused runtime, accelerator, material-validation, surface, and ledger tests:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
python -m pytest -q \
  tests/unit/test_kaess_material_history.py \
  tests/unit/test_kaess_accelerated_history.py \
  tests/unit/test_v06_material_validation.py \
  tests/integration/test_active_surface_boundary.py \
  tests/unit/test_v06_thermal_ledger.py
```

```text
38 passed
```

The broader T018 compatibility set passed:

```text
175 passed
```

Independent final review additionally passed `160` focused CPU tests and `35`
RTX 5080 GPU tests. It found no remaining Critical or Important issue.

At the time this T018 implementation slice was first reviewed, the
full-repository run reached:

```text
553 passed, 2 skipped, 10 subtests passed
```

Its five remaining failures were outside the T018 implementation:

- two intentional T019 J2 flow-curve/tangent RED gates;
- three legacy-v02 runner tests whose referenced
  `legacy/v02/run_ti64_material.py` file is absent.

T019 and the legacy runner were subsequently closed. The clean T021
regression at commit `8f1603f` passed `588` tests with no failure; see
`specs/001-kaess-paper-reproduction/evidence/t021-p0-regression.json`.
