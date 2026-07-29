# Known issues

Registered defects and behavioral surprises that are not yet fixed. Add new
entries at the top with date and reporting session; move fixed ones to the
bottom section with the fixing commit.

## Open

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
