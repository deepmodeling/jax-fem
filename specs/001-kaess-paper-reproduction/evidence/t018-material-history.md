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
powder-conductivity table, the pending Figure 4(b) flow-curve digitization,
independent PDF-vector calibrations for Figure 4(a--e), a reproducible PyMuPDF
extractor/spec pair, a per-file SHA-256 manifest, and a reapproval request.
After rebuilding the content-addressed chain:

```text
candidate material config
  899a912609db10490872bfe8d1a738a40b5c68e03b37dafac8d4c659b8bca178
material bundle manifest
  c7cc552afca8c3ebad26160e41c6aa701fc099f3b8abfd6182a6360e7258b061
reapproval request
  b94be55ed173186bea9dcacfd5fbd4044afef3187dc72fd1aa72e00ba1a211c2
powder conductivity table
  a4e762578fd7a7dd585a315157e0423b150c4f15e235076925f1b50315a89eab
pending flow-curve table
  e97a5f79fbeaf98621ac502038a8ba615ac278df5629bc04167f04b069bb68f6
flow-curve digitization metadata
  8444d400235d43143624b9da634ff7fb07fb7055e125a81b37f3d2fd1596b1b6
independent PDF vector calibration
  224ee3351e9b33a5125b9f579791639d954e75748ce714dbaaeab84448ec6aaf
Figure 4(a/c/d/e) vector calibration
  42a10c4ad615de23e7b4e7835994db64cc476d94c4dd98d25a52f8eef5822f47
reproducible PyMuPDF extractor
  a187f278e4760207a8daa2f3c4dd55bdaee408db138185265c0d72d2e9e669d1
frozen vector-extraction spec
  200ef6c1b3d7d0f9142ec4402f391ceb0796bb6ef2298967597b2485c95b08ac
```

The vector calibration fits the original PDF axes with a maximum
`0.513 MPa` stress residual and the curve endpoints with a maximum
`0.486 MPa` rounding residual. It freezes conservative digitization errors of
`±1 MPa` and `±0.0032` equivalent plastic strain. It also corrects the 1370 °C
curve from the preliminary `20 → 20 MPa` raster reading to the vector-supported
`20 → 30 MPa`. The unpublished original input table and the 1673.15/1873.15 K
solver-realization nodes remain separate model-input uncertainties.

The independently rerunnable Figure 4(a/c/d/e) calibration freezes reading
limits of `±2 K`, `±0.5 GPa`, `±0.1 W/(m K)`, `±2 J/(kg K)`, and
`±0.1e-6 /K`. It corrects the solid specific heat at the `1370 °C`
solidus from the previous interpolated value of about `730 J/(kg K)` to the
vector-supported `670 J/(kg K)`. The `20 °C` thermal-expansion value is
explicitly labelled as a linear extrapolation, not a plotted author node.
Thermal-expansion nodes above `500 °C` and the high-temperature Figure 4(b)
curves are separately labelled as paper-author assumptions rather than
experimental source data.

The immutable manifest and request intentionally keep `status` and `decision`
at `pending_review`; neither file was rewritten after review. Conditional
authority is recorded in a separate hash-bound approval envelope. The
canonical G0-v1 record, paper-parity config, source manifest, and external
material file remain unchanged. T018 and PAR019/PAR024/PAR025 remain open
because validation-only authority is not a superseding final material
approval.

## Formal launcher identity safety

The phase-1 and phase-2 launchers now resolve the material through the G0
environment name `KAESS_MATERIAL_CONFIG`. The older `MATERIAL_CONFIG` name is
accepted only as a compatibility alias; conflicting values fail before an
output directory is created.

Before either launcher creates a run directory, it invokes
`jax_fem_am.verification.material_identity`. The gate binds:

- the exact material-config SHA-256;
- the approved paper-parity record;
- the SHA-256 of the G0 approval record;
- the protocol ID and approved material-freeze object.

The currently approved external material bytes pass the legacy G0-v1 gate.
The G0-v2 candidate still fails that gate with a material SHA-256 mismatch and
cannot accidentally enter a formal run. A separate public G0-v2 entry point
validates the repository bundle only against the fixed conditional-approval
path and SHA-256, and only for explicit CPU validation scopes.

The G0-v1 gate still freezes only the top-level external JSON, because that is
the scope of the formal approval. The conditional G0-v2 gate additionally
binds manifest status, config identity, every dependency file, repository
source evidence, request/canonical/parity references, scope, and the actual
JAX backend. It is an overlay for validation work, not a replacement for the
formal external mode.

`EXTRA_ARGS` is parsed into an argv array before the identity check and is
restricted to an explicit allowlist of reviewed operational options; phase 1
additionally permits a smoke-only `--layers` override, which is not a formal
case. Duplicate `--config`, material-table/scalar, powder, phase-history and
reset overrides are rejected before run-directory creation. The selected
material path is canonicalized to one absolute path before the gate, solver
and manifest consume it.

The identity module executes from the launcher's own repository root, so a
different checkout in the caller's working directory cannot shadow the
reviewed gate. Plate temperature conversion passes the environment value as a
`float()` argument instead of interpolating it into Python source. Operational
cadence values are range checked (`summary_every >= 1`), and phase-2
`PATH_ARGS` uses a separate generator-option allowlist that cannot override
the launcher's content-addressed output path.

The phase-2 paper-parity powder arm also defaults
`powder_solid_hardening` to `0 Pa`, matching the candidate/reference
ideal-plastic specification instead of silently injecting the prior
`1e7 Pa` numerical regularization. A nonzero diagnostic override must now be
explicit; such an effective-parameter deviation is not promotion evidence.

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

The later formal-material identity slice passed:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
/home/user/miniforge3/envs/jax-fem-env/bin/python -m pytest -q \
  tests/contract/test_kaess_material_identity_gate.py \
  tests/contract/test_kaess_parity_config.py \
  tests/contract/test_kaess_material_candidate.py \
  tests/unit/test_kaess_material_history.py \
  tests/unit/test_kaess_accelerated_history.py \
  tests/unit/test_v06_material_validation.py
```

```text
54 passed
```

Both launchers passed `bash -n`. A direct phase-2 launcher probe using the
unapproved candidate failed at the identity gate and confirmed that no output
directory was created.

The repository-wide CPU regression after the final fail-closed status check
also passed:

```text
630 passed, 2 skipped, 16 subtests passed
```

## G0-v2 conditional approval

The project owner issued the exact decision
`按上述范围条件批准 G0-v2` at `2026-07-27T07:07:13Z`. The independent
approval envelope is:

```text
path:
  cases/kaess_2023/inputs/g0-v2-material-conditional-approval.json
sha256:
  4c917871ea433b6589ad13ec681c09d4067d8710d0771b99fadcd1681cbc123b
size_bytes:
  3078
decision:
  conditionally_approved
authorization:
  validation_only
jax_platform:
  cpu
formal_eligible:
  false
promotion_eligible:
  false
```

The envelope freezes the unchanged identities approved by the owner:

| Artifact | SHA-256 |
|---|---|
| paper-parity config | `7e777d73f736d72578bcfccb80199e36163d9b2f60fdb08e9e8b3d2fa56320f7` |
| canonical G0-v1 approval | `206d7af567f0ee4d9113e8780b67b503936030538f266114834aa63052390839` |
| G0-v2 reapproval request | `b94be55ed173186bea9dcacfd5fbd4044afef3187dc72fd1aa72e00ba1a211c2` |
| material-bundle manifest | `c7cc552afca8c3ebad26160e41c6aa701fc099f3b8abfd6182a6360e7258b061` |
| material config | `899a912609db10490872bfe8d1a738a40b5c68e03b37dafac8d4c659b8bca178` |

Only `g1_cpu_validation`, `g2_cpu_validation`, and
`sensitivity_analysis` are authorized. The public gate derives its repository,
manifest, and request from the reviewed source checkout, requires the fixed
approval SHA-256, rejects self-signed chains, validates strict JSON and Draft
2020-12 schemas, verifies every content-addressed file and source artifact,
and checks the declared `cpu` platform against `jax.default_backend()`.
Content-addressed text uses repository-enforced LF checkout semantics.

The following items were expressly excluded from this approval:

- Abaqus total/mean thermal-expansion runtime implementation;
- activation-reference-temperature semantics.

Final G0-v2 approval additionally requires both excluded semantics to be
verified, the flow-curve solver-realization sensitivity to pass, and a new
superseding owner decision. The immutable manifest/request remain
`pending_review`, and formal phase-1/phase-2 launchers remain on G0-v1.

The approval-gate slice passed `352` contract tests. A single monolithic pytest
process was interrupted by a WSL instance restart without a test assertion;
complete directory-isolated CPU reruns covered the same repository suite:

```text
352 contract passed
136 integration passed, 1 skipped
164 unit passed, 1 skipped, 10 subtests passed
27 benchmark/regression passed, 6 subtests passed
total: 679 passed, 2 skipped, 16 subtests passed
```

These are software and provenance regressions, not a substitute for the
remaining G1/G2 material-point, element, and sensitivity simulations.
