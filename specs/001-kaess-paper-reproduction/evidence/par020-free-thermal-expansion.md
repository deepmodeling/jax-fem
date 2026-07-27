# PAR020 Free-Thermal-Expansion Evidence

**Date**: 2026-07-27

**Runtime**: WSL `jax-fem-env`, CPU backend, native JAX `float64`

**Scope**: real three-dimensional element assembly, stress recovery and
Newton solution for a uniform free thermal strain

## Executable contract

`tests/unit/test_kaess_free_thermal_expansion.py` exercises the production
`jax_fem_am.physics.mechanics.ThermoMechanical` class through all four
element/kernel combinations:

1. full-integration HEX8 with the plain tensor-map kernel;
2. full-integration HEX8 with the B-bar universal kernel;
3. TET4 with the plain tensor-map kernel;
4. TET4 with the B-bar universal kernel.

Both heating (`dT = +100 K`) and cooling (`dT = -137 K`) are tested. The HEX8
is mildly distorted so the result is not restricted to a Cartesian mapping.
Every element has positive `JxW` at every quadrature point.

The analytical displacement field is

```text
u(x) = alpha * dT * (x - x0)
```

and the model uses a minimal 3-2-1 constraint:

- the origin node is fixed in `x`, `y` and `z`;
- the node on the local x axis is fixed in `y` and `z`;
- a third node in the local xy plane is fixed in `z`.

These six scalar constraints remove the rigid-body modes without constraining
the analytical expansion or contraction field.

## Non-vacuous checks

For every arm, the test verifies all of the following:

- the raw assembled residual at the analytical field is finite and negligible;
- every component at every `stress_quad` point is finite and negligible;
- the corresponding zero-displacement thermal-load control produces nonzero
  residual and stress, proving that the configured `active=1` response,
  `alpha`, `dT`, stiffness, assembly and recovery are live;
- the free-field error is no larger than
  `128 * eps(float64)` times the same-model locked control;
- a real Newton solve from zero displacement, using the declared Dirichlet
  constraints and SciPy sparse direct linear solve, recovers the analytical
  field within the same scale-aware tolerance.

The locked control prevents false passes from zero activity or a missing
thermal load. Checking the full integration-point stress tensor prevents
hydrostatic or sign-cancelling stress from being hidden by a von Mises or
cell-average-only assertion.

## Verification

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
/home/user/miniforge3/envs/jax-fem-env/bin/python -m pytest -q \
  tests/unit/test_kaess_free_thermal_expansion.py
```

```text
8 passed
```

The focused mechanics cross-check was:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
/home/user/miniforge3/envs/jax-fem-env/bin/python -m pytest -q \
  tests/unit/test_kaess_free_thermal_expansion.py \
  tests/unit/test_v03_bbar_hex8.py \
  tests/unit/test_kaess_j2_tangent.py \
  tests/regression/test_local_stress_postprocessing.py
```

```text
42 passed
```

The final full regression result was:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
/home/user/miniforge3/envs/jax-fem-env/bin/python -m pytest -q
```

```text
601 passed, 2 skipped, 16 subtests passed
```

This evidence closes PAR020 for element-level free thermal expansion. It does
not close PAR025 tensor-history reset semantics, Figure 8 postprocessing, or
later CPU/GPU field-parity gates; those remain separate obligations.
