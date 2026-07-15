"""v05 wrapper: incremental J2 plasticity with stored plastic-strain tensor.

The v03/v04 mechanics keeps only the scalar equivalent plastic strain (eqp,
the hardening state) but NOT the plastic strain tensor. Stress is therefore a
total-deformation quantity: with a uniform relaxation reference temperature
and a uniformly cooled final state, the thermal strain field is compatible,
the release solve shrinks the body uniformly, and the residual stress
vanishes (observed: released vm ~= 20 MPa vs 1004 MPa constrained). Layerwise
stress history cannot lock in either.

v05 upgrades the material model to a proper small-strain incremental J2
formulation (radial return, isotropic hardening + saturation):

    elastic_eps = eps(u) - alpha*dT*I - eps_p_old
    trial       -> radial return -> sigma, delta_eqp
    eps_p_new   = eps_p_old + 1.5*delta_eqp * dev_trial/seq_trial
    eqp_new     = eqp_old + delta_eqp

eps_p is stored per quadrature point (6 components, tensor shear convention)
and is updated after every mechanics solve; the release solve inherits the
build-phase eps_p, so locked-in incompatible strain produces genuine residual
stress and springback (Mercelis & Kruth 2006 M-profile mechanism).

Usage: identical CLI to the v04 wrapper; this module composes ON TOP of v04
(all XLA/profiling/physics patches apply), e.g.

    python 159_local/v05/am_thermal_stress_macro_intersection_mech100_v05.py \
        <all v04/v03 flags>
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V04_WRAPPER_PATH = (
    REPO_ROOT / "159_local" / "v04"
    / "am_thermal_stress_macro_intersection_mech100_XLA.py"
)


def load_v04_wrapper():
    name = "macro_mech100_v04_xla_for_v05"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V04_WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _EpsPRegistry:
    """Carries the build-phase eps_p to the release problem instance."""

    def __init__(self):
        self.build_problem = None
        self.eps_p = None

    def reset(self):
        self.build_problem = None
        self.eps_p = None


REGISTRY = _EpsPRegistry()


def install_plastic_history_patch(base_module, profiler=None) -> bool:
    """Replace base.ThermoMechanical with the eps_p-carrying subclass."""
    np = base_module.np
    jax = base_module.jax
    BaseMech = getattr(
        base_module, "_v05_original_thermo_mechanical",
        base_module.ThermoMechanical,
    )
    base_module._v05_original_thermo_mechanical = BaseMech

    def eps_p_to_tensor(eps_p6):
        return np.array([
            [eps_p6[0], eps_p6[3], eps_p6[5]],
            [eps_p6[3], eps_p6[1], eps_p6[4]],
            [eps_p6[5], eps_p6[4], eps_p6[2]],
        ])

    class PlasticHistoryThermoMechanical(BaseMech):
        def custom_init(self, mechanics_model, yield_saturation=None, foundation_stiffness=0.0,
                        *extra):
            # *extra forwards newer base options (powder foundation, plane
            # axes) without hard-coding their signature here.
            BaseMech.custom_init(self, mechanics_model, yield_saturation, foundation_stiffness,
                                 *extra)
            shape = (len(self.fes[0].cells), self.fes[0].num_quads, 6)
            self._eps_p_state = np.zeros(shape)
            self.internal_vars = list(self.internal_vars) + [self._eps_p_state]

        # --- incremental radial return -----------------------------------
        def _return_map(self, u_grad, dT, young, alpha, poisson, yield_stress,
                        hardening, eqp_old, eps_p_old):
            eps = 0.5 * (u_grad + u_grad.T)
            nu = np.clip(poisson[0], -0.49, 0.49)
            E = young[0]
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            thermal_eps = alpha[0] * dT[0] * np.eye(self.dim)
            elastic_eps = eps - thermal_eps - eps_p_to_tensor(eps_p_old)
            sigma_trial = lmbda * np.trace(elastic_eps) * np.eye(self.dim) + 2.0 * mu * elastic_eps

            if self.mechanics_model != "j2_plastic":
                zeros6 = np.zeros(6)
                return sigma_trial, 0.0, zeros6

            hydro = np.trace(sigma_trial) / 3.0 * np.eye(self.dim)
            dev = sigma_trial - hydro
            seq = np.sqrt(1.5 * np.sum(dev * dev) + 1e-30)
            hardened_yield = yield_stress[0] + hardening[0] * eqp_old[0]
            if self.yield_saturation is not None and self.yield_saturation > 0.0:
                sat = self.yield_saturation
                current_yield = np.maximum(np.minimum(hardened_yield, sat), 1e-12)
                hardening_eff = np.where(hardened_yield < sat, hardening[0], 0.0)
            else:
                current_yield = np.maximum(hardened_yield, 1e-12)
                hardening_eff = hardening[0]
            delta_eqp = np.maximum(seq - current_yield, 0.0) / (3.0 * mu + hardening_eff + 1e-12)
            scale = 1.0 - 3.0 * mu * delta_eqp / seq
            sigma = hydro + scale * dev
            # plastic flow direction: n = 1.5 * dev / seq  (d eqp consistent)
            flow = 1.5 * delta_eqp / seq * dev
            d_eps_p = np.array([
                flow[0, 0], flow[1, 1], flow[2, 2],
                flow[0, 1], flow[1, 2], flow[0, 2],
            ])
            return sigma, delta_eqp, d_eps_p

        def stress_fn(self, u_grad, T, dT, active_factor, young, alpha, poisson,
                      yield_stress, hardening, eqp_old, eps_p_old):
            sigma, _, _ = self._return_map(
                u_grad, dT, young, alpha, poisson, yield_stress, hardening,
                eqp_old, eps_p_old,
            )
            return active_factor[0] * sigma

        # --- state plumbing ----------------------------------------------
        def set_params(self, params):
            self.internal_vars = list(params) + [self._eps_p_state]

        def _u_grads(self, sol):
            g = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] \
                * self.fes[0].shape_grads[:, :, :, None, :]
            return np.sum(g, axis=2)

        def compute_cell_stress(self, sol, params):
            (T_quad, dT_quad, active_factor_quad, young_quad, alpha_quad,
             poisson_quad, yield_quad, hardening_quad, eqp_old_quad) = params
            u_grads = self._u_grads(sol)
            sigmas = jax.vmap(jax.vmap(self.stress_fn))(
                u_grads, T_quad, dT_quad, active_factor_quad, young_quad,
                alpha_quad, poisson_quad, yield_quad, hardening_quad,
                eqp_old_quad, self._eps_p_state,
            )
            return {
                "stress_quad": sigmas,
                "vm_quad": base_module.von_mises_from_stress(sigmas),
            }

        def compute_eqp_update(self, sol, params):
            (T_quad, dT_quad, active_factor_quad, young_quad, alpha_quad,
             poisson_quad, yield_quad, hardening_quad, eqp_old_quad) = params
            u_grads = self._u_grads(sol)

            def one_quad(u_grad, T, dT, active_factor, young, alpha, poisson,
                         yield_stress, hardening, eqp_old, eps_p_old):
                _, delta_eqp, d_eps_p = self._return_map(
                    u_grad, dT, young, alpha, poisson, yield_stress,
                    hardening, eqp_old, eps_p_old,
                )
                active = np.where(active_factor[0] > 0.5, 1.0, 0.0)
                return (
                    np.array([eqp_old[0] + active * delta_eqp]),
                    eps_p_old + active * d_eps_p,
                )

            eqp_new, eps_p_new = jax.vmap(jax.vmap(one_quad))(
                u_grads, T_quad, dT_quad, active_factor_quad, young_quad,
                alpha_quad, poisson_quad, yield_quad, hardening_quad,
                eqp_old_quad, self._eps_p_state,
            )
            if self.mechanics_model == "j2_plastic":
                self._eps_p_state = eps_p_new
                REGISTRY.eps_p = eps_p_new
                if REGISTRY.build_problem is None:
                    REGISTRY.build_problem = self
                return eqp_new
            return params[-1]

    base_module.ThermoMechanical = PlasticHistoryThermoMechanical

    original_run_mechanics = getattr(
        base_module, "_v05_original_run_mechanics", base_module.run_mechanics
    )
    base_module._v05_original_run_mechanics = original_run_mechanics

    def run_mechanics_with_state_pickup(mechanics, u_guess, params, newton_overrides=None):
        # A fresh problem (the release solve) inherits the build-phase eps_p.
        if (
            isinstance(mechanics, PlasticHistoryThermoMechanical)
            and mechanics is not REGISTRY.build_problem
            and REGISTRY.eps_p is not None
            and not getattr(mechanics, "_v05_state_adopted", False)
            and REGISTRY.eps_p.shape == mechanics._eps_p_state.shape
        ):
            mechanics._eps_p_state = REGISTRY.eps_p
            mechanics._v05_state_adopted = True
            print("v05: release problem adopted build-phase plastic strain state")
        return original_run_mechanics(mechanics, u_guess, params, newton_overrides)

    base_module.run_mechanics = run_mechanics_with_state_pickup
    if profiler is not None:
        profiler.meta["v05_plastic_history"] = "eps_p tensor per quad, incremental radial return"
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    v04 = load_v04_wrapper()
    REGISTRY.reset()

    original_load = v04.load_base_solver

    def load_base_solver_with_v05(*args: Any, **kwargs: Any):
        base = original_load(*args, **kwargs)
        install_plastic_history_patch(base)
        print("v05: incremental plastic-history patch installed")
        return base

    v04.load_base_solver = load_base_solver_with_v05
    try:
        return v04.main(argv)
    finally:
        v04.load_base_solver = original_load


if __name__ == "__main__":
    raise SystemExit(main())
