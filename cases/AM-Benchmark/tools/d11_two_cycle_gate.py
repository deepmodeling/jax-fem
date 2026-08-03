"""D-11 two-cycle gate test (PREREQUISITES.md section D.7, user addition 2).

Drives the PRODUCTION v06 semantics chain (install_v06_adapter on the v03
stepper: lifecycle wrapper -> REGISTRY relaxation events -> run_mechanics
reference capture -> state-safe J2 kernel) through two identical thermal
cycles T0 -> T_cut+100 K -> T0 on a biaxially constrained one-element patch.

Constraint note: the registered analytic estimate E*alpha*dT/(1-nu) is the
BIAXIAL formula (a fully displacement-constrained isotropic patch under
uniform dT carries pure hydrostatic stress and can never yield in J2), so
the patch is constrained in x and y everywhere and in z on the bottom face.

Criteria (registered):
  (i)   just above T_cut the stress magnitude is below the yield-tolerance
        floor (reported against 1 % of sigma_y);
  (ii)  residual stress at T0 after cycle 2 equals cycle 1 within 1e-10
        relative (mechanical memory fully wiped by the crossing);
  (iii) the returned stress matches min(sigma_y(T0), E*alpha*(T_cut-T0)/(1-nu))
        within 2 %.

Variant A (H ~ 0) is the instrument for (i)+(iii): perfect plasticity makes
the analytic min() exact. Variant B (H = 2 GPa, L0-like) is the instrument
for (ii): visible hardening exposes any eqp history surviving the crossing.
eqp at both cycle ends is reported for the D-11 "complete reset" semantics.

Usage: python d11_two_cycle_gate.py  (writes derived/d11/gate-test.json)
"""
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp

from types import SimpleNamespace

from jax_fem.generate_mesh import Mesh
from jax_fem_am.simulation import stepper
from jax_fem_am.simulation.runner import REGISTRY, install_v06_adapter
from jax_fem_am.materials.phases import STATE_SOLID

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "derived", "d11")

T0 = 347.05
T_CUT = 1273.15
PEAK = T_CUT + 100.0
JUST_ABOVE = T_CUT + 0.01

E_MOD = 200.0e9
POISSON = 0.3
ALPHA = 1.28e-5
YIELD = 630.0e6

LENGTH = 1.0e-3


def temperature_cycle():
    up = list(onp.arange(T0, T_CUT, 100.0)) + [JUST_ABOVE, PEAK]
    down = [JUST_ABOVE] + list(onp.arange(T_CUT - 100.0, T0, -100.0)) + [T0]
    return up + down


def build_problem():
    pts = LENGTH * onp.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=onp.float64)
    cells = onp.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=onp.int64)

    def everywhere(_p):
        return True

    def bottom(p):
        return jnp.isclose(p[2], 0.0, rtol=0.0, atol=1e-12)

    def zero(_p):
        return 0.0

    # biaxial constraint: x and y fixed everywhere, z fixed on the bottom
    bc = [[everywhere, everywhere, bottom], [0, 1, 2], [zero, zero, zero]]
    problem = stepper.ThermoMechanical(
        mesh=Mesh(pts, cells, ele_type="HEX8"),
        vec=3, dim=3, ele_type="HEX8", quadrature_order=2,
        dirichlet_bc_info=bc,
        additional_info=("j2_plastic", None, 0.0, 0.0, (), False, None),
    )
    return problem


def run_variant(name, hardening):
    REGISTRY.reset()
    problem = build_problem()
    shape = (len(problem.fes[0].cells), problem.fes[0].num_quads, 1)

    def full(v):
        return jnp.full(shape, v, dtype=jnp.float64)

    args = SimpleNamespace(
        phase_history_model="legacy_reset",
        stress_relaxation_temperature=T_CUT,
        reset_plastic_on_melt=False,
        solidus_temperature=1552.0,
        liquidus_temperature=1552.0,
    )
    REGISTRY.args = args

    phase = full(float(STATE_SOLID))
    # Production convention (events.py + lifecycle.py): with relaxation
    # enabled, T_ref IS the relaxation temperature, and positive dT above it
    # is clipped to zero thermal strain. Initialize as freshly consolidated
    # material (the macro-mode state after deposition).
    T_ref = full(T_CUT)
    eqp = full(0.0)
    active = full(1.0)
    u = [jnp.zeros((problem.fes[0].num_total_nodes, 3))]
    # absolute tol accepts machine-zero residuals (fully-clipped hot steps
    # start at equilibrium, where the relative criterion is 0/0)
    newton = {"tol": 1.0e-6, "rel_tol": 1.0e-10, "max_iter": 50,
              "line_search_flag": True}

    record = {"name": name, "hardening": hardening, "steps": []}
    cycle_end_stress, cycle_end_eqp, hot_vm = [], [], None
    schedule = temperature_cycle()
    n_cycle = len(schedule)
    T_prev = T0

    for cycle in (1, 2):
        for k, T_now in enumerate(schedule):
            T_quad = full(T_now)
            phase, T_ref, eqp, _, _ = stepper.update_phase_reference_and_eqp(
                T_quad, active, phase, T_ref, eqp, args)
            dT_quad = T_quad - T_ref
            params = [T_quad, dT_quad, active, full(E_MOD), full(ALPHA),
                      full(POISSON), full(YIELD), full(hardening), eqp]
            u = stepper.run_mechanics(problem, u, params, newton)
            eqp = problem.compute_eqp_update(u[0], params)
            params[-1] = eqp
            stress = onp.asarray(
                problem.compute_cell_stress(u[0], params)["stress_quad"])
            vm = float(onp.max(onp.asarray(
                stepper.von_mises_from_stress(jnp.asarray(stress)))))
            record["steps"].append(
                {"cycle": cycle, "T": T_now, "vm_MPa": vm / 1e6,
                 "eqp_max": float(onp.max(onp.asarray(eqp)))})
            T_prev = T_now
        cycle_end_stress.append(stress.copy())
        cycle_end_eqp.append(float(onp.max(onp.asarray(eqp))))

    # criterion (i): the cooling-leg JUST_ABOVE step of cycle 1 (the schedule
    # visits JUST_ABOVE twice; the second occurrence is the cooling leg)
    hot_steps = [s for s in record["steps"]
                 if s["cycle"] == 1 and s["T"] == JUST_ABOVE]
    hot_vm = hot_steps[-1]["vm_MPa"]

    s1, s2 = cycle_end_stress
    denom = float(onp.max(onp.abs(s1)))
    rel_cycle = float(onp.max(onp.abs(s2 - s1))) / denom if denom else 0.0

    vm_end = record["steps"][-1]["vm_MPa"] * 1e6
    analytic = min(YIELD, E_MOD * ALPHA * (T_CUT - T0) / (1.0 - POISSON))
    rel_analytic = abs(vm_end - analytic) / analytic

    record["criteria"] = {
        "i_hot_vm_MPa": hot_vm,
        "i_floor_MPa": 0.01 * YIELD / 1e6,
        "i_pass": bool(hot_vm < 0.01 * YIELD / 1e6),
        "ii_rel_cycle_diff": rel_cycle,
        "ii_pass": bool(rel_cycle <= 1.0e-10),
        "iii_vm_end_MPa": vm_end / 1e6,
        "iii_analytic_MPa": analytic / 1e6,
        "iii_rel_err": rel_analytic,
        "iii_pass": bool(rel_analytic <= 0.02),
        "eqp_cycle_ends": cycle_end_eqp,
    }
    return record


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not install_v06_adapter(stepper):
        raise RuntimeError("v06 adapter installation failed")
    results = {
        "spec": "PREREQUISITES.md D.7 two-cycle gate; biaxial patch; "
                "cycles T0->T_cut+100->T0",
        "T0": T0, "T_cut": T_CUT, "peak": PEAK,
        "variants": [run_variant("A_perfect_plastic", 1.0e7),
                     run_variant("B_hardening_L0", 2.0e9)],
    }
    a, b = results["variants"]
    print("=== D-11 two-cycle gate ===")
    for v, crits in ((a, ("i", "iii")), (b, ("ii",))):
        c = v["criteria"]
        print(f"variant {v['name']}:")
        print(f"  (i)   hot vm {c['i_hot_vm_MPa']:.4f} MPa "
              f"(floor {c['i_floor_MPa']:.1f}) -> {c['i_pass']}")
        print(f"  (ii)  cycle rel diff {c['ii_rel_cycle_diff']:.3e} "
              f"-> {c['ii_pass']}")
        print(f"  (iii) vm_end {c['iii_vm_end_MPa']:.2f} vs analytic "
              f"{c['iii_analytic_MPa']:.2f} MPa "
              f"(rel {c['iii_rel_err']:.4f}) -> {c['iii_pass']}")
        print(f"  eqp cycle ends: {c['eqp_cycle_ends']}")
    gate = (a["criteria"]["i_pass"] and a["criteria"]["iii_pass"]
            and b["criteria"]["ii_pass"])
    results["gate_pass"] = bool(gate)
    print(f"GATE: {'PASS' if gate else 'FAIL'}")
    out = os.path.join(OUT_DIR, "gate-test.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
        f.write("\n")
    print(f"wrote {out}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
