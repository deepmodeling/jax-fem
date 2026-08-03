# Known issues

Registered defects and behavioral surprises that are not yet fixed. Add new
entries at the top with date and reporting session; move fixed ones to the
bottom section with the fixing commit.

## Open

### --scan-steps-per-layer is per hatch line, not per layer (reported 2026-07-31, main session)

In hatch-raster mode (`--hatch-lines-per-layer N`), `--scan-steps-per-layer S`
produces N x S scan steps per layer (S steps along EACH line), not S total.
With per-step energy fixed by `--dt` and laser power, adding hatch lines
multiplies the deposited layer energy. L0 run 5 deposited 4x the intended
energy this way and diverged. Either rename the flag (`--scan-steps-per-line`)
or divide S across lines. Until then, scale power or steps to conserve the
per-layer energy budget (see `cases/AM-Benchmark/tools/run_l0.sh` rev 3 note).

### Macro consolidation without --powder-elset fuses powder into solid (reported 2026-07-31, main session)

With `liquidus == solidus` (macro consolidation-on-activation) and
`--layer-activation-mode layer_on_scan` + centroid geometry, activation covers
the entire layer plane, and the macro branch solidifies every activated
non-fixture cell — including cells that are physically permanent powder (gap
and margin regions). L0 run 6 fused 43810 powder cells into a slab this way.
`--powder-elset <SET>` prevents it (permanent-powder cells are excluded from
activation each step). Wanted: a startup warning when macro mode is active,
a powder-mode mesh has unprinted regions, and no `--powder-elset` is given.

### run_audit cannot audit thermal-only runs (reported 2026-07-29, V1 session)

`run_audit` assumes mechanics outputs exist; a run with `--mechanics-every 0`
produces no mechanics artifacts and the audit fails rather than auditing the
thermal side alone. Wanted: a thermal-only audit mode (energy ledger, field
statistics) that skips mechanics checks.

### Scan-phase dt is governed by path line spacing, not --dt (reported 2026-07-29, V1 session)

During scanning, the effective time step comes from the path generator's
segment/line spacing; the `--dt` argument is silently overridden. Either
document this as intended (dt = segment traversal time) or honor `--dt` by
subdividing segments. Until then, convergence-in-dt studies during scanning
must vary the path discretization, not `--dt`.

## Fixed

### Thermal-only mechanics surrogate crashed at stepper.py:966 (reported 2026-07-29, V1 session)

With `--mechanics-every 0` and the v04 thermal-only mechanics surrogate
enabled (default), the stepper's unconditional
`mechanics.set_flow_curve_active_mask(...)` hit an AttributeError because
`_ThermalOnlyMechanicsProblem` did not implement the method. Workaround was
`--no-xla-thermal-only-mechanics-surrogate` (paying full mechanics Problem
construction). Fixed by adding a no-op `set_flow_curve_active_mask` to the
surrogate — thermal-only runs never assemble mechanics, so there is nothing
to bind.
