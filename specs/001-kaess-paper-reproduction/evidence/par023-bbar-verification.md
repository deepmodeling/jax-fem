# PAR023 B-bar Verification Evidence

**Date**: 2026-07-27

**Runtime**: WSL `jax-fem-env`, CPU backend, native JAX `float64`

**Scope**: full-integration (`2×2×2`) HEX8 B-bar mechanics

## Required checks

The three clauses in PAR023 are exercised by real
`ThermoMechanical.compute_residual()` paths in
`tests/unit/test_v03_bbar_hex8.py`:

1. `test_hex8_bbar_matches_plain_on_affine_field` is the uniform-strain
   patch test. B-bar and the plain tensor-map residual agree for an affine
   displacement field.
2. `test_elastic_near_incompressible` loads a `ν=0.49` HEX8 beam. B-bar
   increases the locking-suppressed tip response and reduces the hydrostatic
   pressure checkerboard by the frozen margins.
3. `test_hex8_bbar_has_only_the_six_rigid_body_zero_modes` differentiates the
   assembled B-bar residual of a mildly distorted, unconstrained HEX8 using
   `jax.jacfwd`. It checks tangent symmetry, positive semidefiniteness,
   exactly six rigid-body null modes, exactly eighteen positive modes, a
   scale-aware spectral gap, and the explicit translation/rotation basis.

The modal test intentionally uses no Dirichlet boundary conditions, zero
thermal load, elastic material response, and an isoparametric distortion. It
therefore measures the B-bar/integration kernel without concealing null modes
through boundary-row replacement or plastic tangent effects.

## Modal result

The observed distorted-element spectrum was the following diagnostic
snapshot. The executable contract uses dtype- and scale-aware invariants
rather than pinning these implementation-dependent values:

| Metric | Result |
|---|---:|
| Tangent dimension | `24 × 24` |
| dtype | `float64` |
| Quadrature points | `8` |
| Minimum positive `JxW` | `1.2265901090729471e-10 m³` |
| Spectral scale | `2.4188207762750787e8` |
| Scale-aware rank tolerance | `1.2890066487658978e-5` |
| Symmetry defect, spectral norm | `3.1328351976014992e-8` |
| Zero modes | `6` |
| Positive modes | `18` |
| Negative modes | `0` |
| Smallest positive eigenvalue | `1.142513876166703e7` |
| Smallest-positive / largest ratio | `4.72343336626265e-2` |
| Rigid-basis residual / spectral scale | `1.4468754640288052e-16` |

The positive spectrum is separated from the numerical nullspace by much more
than the required `100 × rank_tolerance`; there is no additional hourglass or
negative-energy mode.

## Verification

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
/home/user/miniforge3/envs/jax-fem-env/bin/python -m pytest -q \
  tests/unit/test_v03_bbar_hex8.py
```

```text
10 passed
```

This closes PAR023 for the implemented full-integration HEX8 B-bar kernel. It
does not certify mesh-quality sensitivity or the later CPU/GPU formal-run
parity gates.
