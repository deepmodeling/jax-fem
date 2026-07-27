# T021 P0 Full-Regression Evidence

**Date**: 2026-07-27

**Tested commit**: `8f1603f4ee69a15b0049ad04724aba654db7e740`

**Scope**: complete repository CPU regression after T014--T020 and legacy-v02
compatibility restoration

## Clean-run result

The source tree was clean before execution. The WSL `jax-fem-env` runtime used
Python `3.13.13`, JAX `0.10.2`, NumPy `2.4.6`, and the CPU backend:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
/home/user/miniforge3/envs/jax-fem-env/bin/python -m pytest -q
```

```text
588 passed, 2 skipped, 16 subtests passed in 111.22s
```

There were no failures or collection errors. The two conditional skips were
then isolated with `pytest -q -rs`:

- the legacy Newton-stall exception probe skipped because that machine/problem
  combination did not reach its stall floor;
- the optional direct v03 parser probe skipped because the legacy parser does
  not accept a positional argv argument.

Neither skip is an unexecuted T007--T020 P0 physics gate.

The exact Physics and Solver command listed in `quickstart.md` was also run:

```text
99 passed, 2 skipped, 10 subtests passed in 22.09s
```

The machine-readable command, runtime identity, counts, skip classification,
and T007--T020 red/green mapping are stored in
`t021-p0-regression.json`.

## Legacy-v02 closure

The three former collection/regression failures came from the absent
`legacy/v02/run_ti64_material.py`. The restored launcher is based on the
previously validated Ti-6Al-4V runner and now:

- resolves historical project-relative and bundle-local material tables;
- checks every declared material table before a run;
- prepends `legacy/v01` and the repository root to the child `PYTHONPATH`;
- requires an explicit `--` before solver passthrough arguments;
- prevents passthrough from overriding validated material/run options through
  exact names, aliases, reverse switches, `--name=value`, or argparse
  abbreviations;
- runs the solver through an argv list without `shell=True`.

Its focused regression passed:

```text
6 passed, 6 subtests passed
```

A dry run against the real
`/home/user/work/159/materials/Ti-6Al-4V` pack also resolved the tables,
interpreter, solver, and `PYTHONPATH` successfully. This remains a legacy
compatibility tool, not a Kaess formal-run launcher.

## Gate decision

T021 is complete: the P0 code suite has no unexpected failure and every
T007--T020 RED/GREEN slice has a machine-readable status.

This does not approve G1/G2. T018 remains open because the new material bundle
and Figure 4(b) flow curve are still `pending_review`; anchor sensitivity and
the remaining unchecked PAR items also remain required.
