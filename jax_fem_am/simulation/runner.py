"""v06 compatibility driver: verified J2 state on the v03/v04 runtime.

This is the single compatibility boundary while the paper model is migrated.
The v03 time loop and v04 performance instrumentation are reused; v05 is kept
only as a frozen comparison baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence


from jax_fem_am.materials.j2 import (  # noqa: E402
    elastic_strain_from_stress,
)
from jax_fem_am.domain.lifecycle import (  # noqa: E402
    effective_thermal_increment,
    update_stress_free_reference,
)
from jax_fem_am.materials.material_validation import validate_material_inputs  # noqa: E402
from jax_fem_am.verification.thermal_ledger import (  # noqa: E402
    EnergyLedgerRecorder,
    extract_solver_step,
)


class _StateRegistry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.build_problem = None
        self.eps_p = None
        self.eqp = None
        self.eps_ref = None
        self.pending_reference = None
        self.pending_cold_reset = None
        self.relaxation_mask = None
        self.relaxation_hot = None
        self.args = None
        self.step_states = None
        self.thermal_ledger = None
        self.previous_thermal_solution = None


REGISTRY = _StateRegistry()


def load_v04_wrapper():
    import jax_fem_am.simulation.acceleration as acceleration

    return acceleration


def install_phase_lifecycle_wrapper(base_module):
    """Wrap the currently installed phase kernel with v06 state side effects.

    v04 may replace the phase function with a JIT kernel after the base module
    is loaded, so this hook is intentionally safe to call again immediately
    before entering the v03 time loop. ``paper_irreversible`` has one
    lifecycle event (first solidification); ``legacy_reset`` retains the
    historical relaxation/remelt side effects for non-paper workflows.
    """
    current = base_module.update_phase_reference_and_eqp
    if getattr(current, "_v06_phase_lifecycle_wrapper", False):
        return False
    np = base_module.np

    def update_phase_with_lifecycle(
        T_quad,
        active_quad,
        phase_quad,
        T_ref_quad,
        eqp_quad,
        args,
    ):
        result = current(
            T_quad,
            active_quad,
            phase_quad,
            T_ref_quad,
            eqp_quad,
            args,
        )
        REGISTRY.eqp = result[2]
        newly_solidified = result[3]
        paper_irreversible = (
            getattr(args, "phase_history_model", "legacy_reset")
            == "paper_irreversible"
        )
        if paper_irreversible:
            no_relaxation = np.zeros_like(
                newly_solidified,
                dtype=bool,
            )
            REGISTRY.relaxation_mask = no_relaxation
            REGISTRY.relaxation_hot = no_relaxation
            reference_event = newly_solidified
        else:
            active = active_quad > 0.5
            fixture = (
                (phase_quad == base_module.STATE_SUBSTRATE)
                | (phase_quad == base_module.STATE_SUPPORT)
            )
            managed = active & (~fixture)
            relax_temperature = getattr(
                args,
                "stress_relaxation_temperature",
                None,
            )
            relaxation_enabled = bool(
                relax_temperature is not None
                and relax_temperature > 0.0
            )
            relaxation_mask = (
                managed
                if relaxation_enabled
                else np.zeros_like(managed, dtype=bool)
            )
            REGISTRY.relaxation_mask = relaxation_mask
            relaxation_hot = (
                managed & (T_quad >= float(relax_temperature))
                if relaxation_enabled
                else np.zeros_like(managed, dtype=bool)
            )
            previous_hot = REGISTRY.relaxation_hot
            if (
                previous_hot is None
                or previous_hot.shape != relaxation_hot.shape
            ):
                previous_hot = np.zeros_like(
                    relaxation_hot,
                    dtype=bool,
                )
            became_relaxation_hot = relaxation_hot & (~previous_hot)
            # D-11 complete history reset (PREREQUISITES.md D.7, user
            # addition 1): on the cooling crossing of T_cut the point's
            # mechanical history is wiped -- eqp := 0 here; the plastic
            # strain tensor and the stress-free reference are handled at
            # the next mechanics call via pending_cold_reset.
            became_relaxation_cold = previous_hot & (~relaxation_hot)
            if bool(np.any(became_relaxation_cold)):
                eqp_wiped = np.where(
                    became_relaxation_cold,
                    np.zeros_like(result[2]),
                    result[2],
                )
                result = (
                    result[0],
                    result[1],
                    eqp_wiped,
                    result[3],
                    result[4],
                )
                REGISTRY.eqp = eqp_wiped
                if REGISTRY.pending_cold_reset is None:
                    REGISTRY.pending_cold_reset = became_relaxation_cold
                else:
                    REGISTRY.pending_cold_reset = (
                        REGISTRY.pending_cold_reset | became_relaxation_cold
                    )
            REGISTRY.relaxation_hot = relaxation_hot

            entered_melted = result[4]
            old_melted = (
                (phase_quad == base_module.STATE_LIQUID)
                | (phase_quad == base_module.STATE_MUSHY)
            )
            new_melted = (
                (result[0] == base_module.STATE_LIQUID)
                | (result[0] == base_module.STATE_MUSHY)
            )
            became_melted = new_melted & (~old_melted)
            reference_event = (
                newly_solidified
                | became_melted
                | became_relaxation_hot
            )
            if (
                getattr(args, "reset_plastic_on_melt", False)
                and REGISTRY.build_problem is not None
            ):
                REGISTRY.build_problem.reset_remelted_state(
                    entered_melted
                )
        if REGISTRY.pending_reference is None:
            REGISTRY.pending_reference = reference_event
        else:
            REGISTRY.pending_reference = (
                REGISTRY.pending_reference | reference_event
            )
        return result

    update_phase_with_lifecycle._v06_phase_lifecycle_wrapper = True
    update_phase_with_lifecycle._v06_wrapped_phase_function = current
    base_module.update_phase_reference_and_eqp = update_phase_with_lifecycle
    return True


def install_mechanics_event_wrapper(base_module):
    """Force a mechanics update when a reference-reset event is pending."""
    current = base_module.should_run_mechanics
    if getattr(current, "_v06_mechanics_event_wrapper", False):
        return False
    onp = base_module.onp

    def should_run_mechanics_with_events(global_step, args):
        scheduled = bool(current(global_step, args))
        pending = REGISTRY.pending_reference
        event_due = bool(
            getattr(args, "mechanics_every", 0) > 0
            and pending is not None
            and onp.any(onp.asarray(pending))
        )
        return scheduled or event_due

    should_run_mechanics_with_events._v06_mechanics_event_wrapper = True
    should_run_mechanics_with_events._v06_wrapped_should_run = current
    base_module.should_run_mechanics = should_run_mechanics_with_events
    return True


def install_output_state_wrapper(base_module):
    """Append v06 tensor state and elastic strain to VTU cell fields."""
    current = base_module.make_quad_stress_cell_infos
    if getattr(current, "_v06_output_state_wrapper", False):
        return False
    np = base_module.np

    def append_tensor(cell_infos, name, tensor, expected_shape):
        if tensor is None:
            return
        tensor = np.asarray(tensor)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} tensor shape {tensor.shape} does not match "
                f"stress tensor shape {expected_shape}"
            )
        num_quads = tensor.shape[1]
        for quad_idx in range(num_quads):
            for suffix, row, col in base_module.STRESS_COMPONENTS:
                value = tensor[:, quad_idx, row, col]
                if row != col:
                    value = 0.5 * (
                        value + tensor[:, quad_idx, col, row]
                    )
                field_name = base_module.quad_field_name(
                    f"{name}_quad_{suffix}", quad_idx, num_quads
                )
                cell_infos.append((field_name, value))

    def make_quad_state_cell_infos(quad_stress):
        cell_infos = list(current(quad_stress))
        stress_shape = np.asarray(quad_stress["stress_quad"]).shape
        elastic = quad_stress.get("elastic_strain_quad")
        append_tensor(
            cell_infos,
            "elastic_strain",
            elastic,
            stress_shape,
        )
        append_tensor(
            cell_infos,
            "eps_p",
            REGISTRY.eps_p,
            stress_shape,
        )
        append_tensor(
            cell_infos,
            "eps_ref",
            REGISTRY.eps_ref,
            stress_shape,
        )
        return cell_infos

    make_quad_state_cell_infos._v06_output_state_wrapper = True
    make_quad_state_cell_infos._v06_wrapped_output_function = current
    base_module.make_quad_stress_cell_infos = make_quad_state_cell_infos
    return True


def install_thermal_ledger_wrapper(base_module):
    """Wrap the final v04 solver boundary and audit accepted thermal solves."""
    current = base_module.solver
    if getattr(current, "_v06_thermal_ledger_wrapper", False):
        return False
    thermal_class = getattr(
        base_module,
        "_v04_original_TransientThermal_for_surrogate",
        getattr(
            base_module,
            "_v06_original_transient_thermal",
            getattr(base_module, "TransientThermal", None),
        ),
    )
    if not isinstance(thermal_class, type):
        raise TypeError("v06 could not resolve the concrete TransientThermal class")

    def solver_with_thermal_ledger(problem, *args, **kwargs):
        solution = current(problem, *args, **kwargs)
        if not isinstance(problem, thermal_class):
            return solution
        if REGISTRY.args is None or REGISTRY.step_states is None:
            raise RuntimeError("thermal ledger is missing parsed arguments or step states")
        if REGISTRY.thermal_ledger is None:
            REGISTRY.thermal_ledger = EnergyLedgerRecorder(
                Path(REGISTRY.args.output_dir),
                expected_steps=len(REGISTRY.step_states),
            )
        step_index = len(REGISTRY.thermal_ledger.rows)
        if step_index >= len(REGISTRY.step_states):
            raise RuntimeError("thermal solver call count exceeds generated step states")
        solver_options = kwargs.get("solver_options")
        if solver_options is None and args and isinstance(args[0], dict):
            solver_options = args[0]
        solver_options = solver_options or {}
        newton_options = solver_options.get("newton", solver_options)
        residual_tolerance = float(newton_options.get("tol", 1.0e-6))
        row = extract_solver_step(
            problem,
            solution,
            step_index=step_index,
            step_state=REGISTRY.step_states[step_index],
            absorptivity=REGISTRY.args.absorptivity,
            previous_solution=REGISTRY.previous_thermal_solution,
            temperature_atol_k=1.0e-3,
            solver_residual_tolerance_w=residual_tolerance,
        )
        REGISTRY.thermal_ledger.append(row)
        REGISTRY.previous_thermal_solution = solution[0]
        return solution

    solver_with_thermal_ledger._v06_thermal_ledger_wrapper = True
    solver_with_thermal_ledger._v06_wrapped_solver = current
    base_module.solver = solver_with_thermal_ledger
    return True


def install_step_state_capture_wrappers(base_module):
    """Capture the exact generated thermal schedule for ledger row identity."""
    changed = False
    for function_name in (
        "generate_raster_step_states",
        "generate_path_file_step_states",
    ):
        current = getattr(base_module, function_name)
        if getattr(current, "_v06_step_state_capture_wrapper", False):
            continue

        def capture(*args, __current=current, **kwargs):
            result = __current(*args, **kwargs)
            REGISTRY.step_states = list(result[0])
            return result

        capture._v06_step_state_capture_wrapper = True
        capture._v06_wrapped_step_generator = current
        setattr(base_module, function_name, capture)
        changed = True
    return changed


def install_v06_adapter(base_module, profiler=None):
    """Install the v06 state-safe constitutive adapter on a v03 base module."""
    if getattr(base_module, "_v06_adapter_installed", False):
        return True

    np = base_module.np
    jax = base_module.jax
    BaseMech = base_module.ThermoMechanical
    original_thermal = base_module.TransientThermal
    original_phase_update = base_module.update_phase_reference_and_eqp
    original_run_mechanics = base_module.run_mechanics
    original_save_step = base_module.save_step
    original_load_property_tables = base_module.load_property_tables
    base_module._v06_original_thermo_mechanical = BaseMech
    base_module._v06_original_phase_update = original_phase_update
    base_module._v06_original_run_mechanics = original_run_mechanics
    base_module._v06_original_save_step = original_save_step
    base_module._v06_original_transient_thermal = original_thermal
    base_module._v06_original_load_property_tables = original_load_property_tables

    def load_validated_property_tables(args):
        tables = original_load_property_tables(args)
        validate_material_inputs(args, tables)
        REGISTRY.args = args
        return tables

    class StateSafeThermoMechanical(BaseMech):
        def custom_init(
            self,
            mechanics_model,
            yield_saturation=None,
            foundation_stiffness=0.0,
            *extra,
        ):
            # *extra forwards newer base options (powder lateral foundation,
            # plane axes) without hard-coding their signature here.
            BaseMech.custom_init(
                self, mechanics_model, yield_saturation, foundation_stiffness,
                *extra,
            )
            curve_selector = None
            if self.flow_curve is not None:
                curve_selector = self.internal_vars.pop()
            shape = (len(self.fes[0].cells), self.fes[0].num_quads, 3, 3)
            self._eps_p_state = np.zeros(shape)
            self._eps_ref_state = np.zeros(shape)
            self._relaxation_mask = np.zeros(shape[:-2] + (1,), dtype=bool)
            self.internal_vars = list(self.internal_vars) + [
                self._eps_p_state,
                self._eps_ref_state,
            ]
            if curve_selector is not None:
                self.internal_vars.append(curve_selector)

        def _return_map(
            self,
            u_grad,
            dT,
            young,
            alpha,
            poisson,
            yield_stress,
            hardening,
            eqp_old,
            eps_p_old,
            eps_ref_old=None,
            T=None,
            active_factor=None,
            flow_curve_active=None,
        ):
            if eps_ref_old is None:
                eps_ref_old = np.zeros((self.dim, self.dim))
            if T is None and getattr(self, "flow_curve", None) is not None:
                raise ValueError(
                    "temperature is required for a flow-curve return map"
                )
            if T is None:
                T = np.asarray([0.0])
            if active_factor is None:
                active_factor = np.asarray([1.0])
            update = BaseMech._material_point_update(
                self,
                u_grad,
                T,
                dT,
                active_factor,
                young,
                alpha,
                poisson,
                yield_stress,
                hardening,
                eqp_old,
                eps_p_old=eps_p_old,
                eps_ref_old=eps_ref_old,
                flow_curve_active=flow_curve_active,
            )
            return (
                update.stress,
                update.delta_eqp,
                update.state.eps_p - eps_p_old,
            )

        def stress_fn(
            self,
            u_grad,
            T,
            dT,
            active_factor,
            young,
            alpha,
            poisson,
            yield_stress,
            hardening,
            eqp_old,
            eps_p_old,
            eps_ref_old,
            flow_curve_active=None,
        ):
            stress, _, _ = self._return_map(
                u_grad,
                dT,
                young,
                alpha,
                poisson,
                yield_stress,
                hardening,
                eqp_old,
                eps_p_old,
                eps_ref_old,
                T=T,
                active_factor=active_factor,
                flow_curve_active=flow_curve_active,
            )
            return active_factor[0] * stress

        def _effective_dT(self, dT_quad):
            return effective_thermal_increment(
                dT_quad,
                self._relaxation_mask,
            )

        def set_params(self, params):
            BaseMech._require_flow_curve_mask_bound(self)
            effective_params = list(params)
            effective_params[1] = self._effective_dT(effective_params[1])
            self.internal_vars = effective_params + [
                self._eps_p_state,
                self._eps_ref_state,
            ]
            if self.flow_curve is not None:
                self.internal_vars.append(
                    self._flow_curve_active_mask
                )

        # _u_grads is inherited from the v03 base class: it applies B-bar
        # (element-average volumetric strain) when the problem was built
        # with bbar=True, keeping the state-update strain measure identical
        # to the residual's.

        def capture_stress_free_reference(
            self,
            sol,
            event_mask,
            alpha_quad,
            dT_quad,
        ):
            u_grads = self._u_grads(sol)
            total_strain = 0.5 * (
                u_grads + np.swapaxes(u_grads, -1, -2)
            )
            effective_dT = self._effective_dT(dT_quad)
            thermal_scalar = alpha_quad * effective_dT
            thermal_strain = thermal_scalar[..., None] * np.eye(self.dim)
            self._eps_ref_state = update_stress_free_reference(
                self._eps_ref_state,
                total_strain,
                thermal_strain,
                self._eps_p_state,
                event_mask,
            )
            REGISTRY.eps_ref = self._eps_ref_state

        def compute_cell_stress(self, sol, params):
            BaseMech._require_flow_curve_mask_bound(self)
            (
                T_quad,
                dT_quad,
                active_quad,
                young_quad,
                alpha_quad,
                poisson_quad,
                yield_quad,
                hardening_quad,
                eqp_old_quad,
            ) = params
            dT_quad = self._effective_dT(dT_quad)
            selector = (
                self._flow_curve_active_mask
                if self.flow_curve is not None
                else np.zeros_like(active_quad[..., :1])
            )
            stress = jax.vmap(jax.vmap(self.stress_fn))(
                self._u_grads(sol),
                T_quad,
                dT_quad,
                active_quad,
                young_quad,
                alpha_quad,
                poisson_quad,
                yield_quad,
                hardening_quad,
                eqp_old_quad,
                self._eps_p_state,
                self._eps_ref_state,
                selector,
            )
            elastic_strain = elastic_strain_from_stress(
                stress,
                young_quad,
                poisson_quad,
            )
            return {
                "stress_quad": stress,
                "vm_quad": base_module.von_mises_from_stress(stress),
                "elastic_strain_quad": elastic_strain,
            }

        def compute_eqp_update(self, sol, params):
            BaseMech._require_flow_curve_mask_bound(self)
            if self.mechanics_model != "j2_plastic":
                return params[-1]
            (
                T_quad,
                dT_quad,
                active_quad,
                young_quad,
                alpha_quad,
                poisson_quad,
                yield_quad,
                hardening_quad,
                eqp_old_quad,
            ) = params
            dT_quad = self._effective_dT(dT_quad)

            def one_quad(
                u_grad,
                T,
                dT,
                active,
                young,
                alpha,
                poisson,
                yield_stress,
                hardening,
                eqp_old,
                eps_p_old,
                eps_ref_old,
                flow_curve_active,
            ):
                _, delta_eqp, delta_eps_p = self._return_map(
                    u_grad,
                    dT,
                    young,
                    alpha,
                    poisson,
                    yield_stress,
                    hardening,
                    eqp_old,
                    eps_p_old,
                    eps_ref_old,
                    T=T,
                    active_factor=active,
                    flow_curve_active=flow_curve_active,
                )
                mask = np.where(active[0] > 0.5, 1.0, 0.0)
                return (
                    np.array([eqp_old[0] + mask * delta_eqp]),
                    eps_p_old + mask * delta_eps_p,
                )

            selector = (
                self._flow_curve_active_mask
                if self.flow_curve is not None
                else np.zeros_like(active_quad[..., :1])
            )
            eqp_new, eps_p_new = jax.vmap(jax.vmap(one_quad))(
                self._u_grads(sol),
                T_quad,
                dT_quad,
                active_quad,
                young_quad,
                alpha_quad,
                poisson_quad,
                yield_quad,
                hardening_quad,
                eqp_old_quad,
                self._eps_p_state,
                self._eps_ref_state,
                selector,
            )
            self._eps_p_state = eps_p_new
            REGISTRY.eps_p = eps_p_new
            REGISTRY.eqp = eqp_new
            if REGISTRY.build_problem is None:
                REGISTRY.build_problem = self
            return eqp_new

        def reset_remelted_state(self, remelted):
            mask = np.asarray(remelted, dtype=bool)[..., 0]
            self._eps_p_state = np.where(
                mask[..., None, None], 0.0, self._eps_p_state
            )
            REGISTRY.eps_p = self._eps_p_state

    def run_mechanics_with_state(
        mechanics,
        u_guess,
        params,
        newton_overrides=None,
    ):
        is_state_safe = isinstance(mechanics, StateSafeThermoMechanical)
        if is_state_safe and REGISTRY.build_problem is None:
            REGISTRY.build_problem = mechanics
            REGISTRY.eps_p = mechanics._eps_p_state
            REGISTRY.eps_ref = mechanics._eps_ref_state
        is_release = (
            is_state_safe
            and REGISTRY.build_problem is not None
            and mechanics is not REGISTRY.build_problem
        )
        if is_state_safe and REGISTRY.relaxation_mask is not None:
            if REGISTRY.relaxation_mask.shape != mechanics._relaxation_mask.shape:
                raise ValueError(
                    "relaxation_mask shape does not match mechanics state shape"
                )
            mechanics._relaxation_mask = REGISTRY.relaxation_mask

        build_problem = REGISTRY.build_problem
        cold_reset = REGISTRY.pending_cold_reset
        if (
            is_state_safe
            and cold_reset is not None
            and bool(np.any(cold_reset))
        ):
            # D-11 complete history reset, tensor side: wipe the plastic
            # strain of points that cooled through T_cut by folding it into
            # the stress-free reference (stress = C : (total - thermal -
            # eps_p - eps_ref) is unchanged by eps_p -> 0, eps_ref +=
            # eps_p), so the anchor stays at the last above-T_cut capture
            # while the plastic memory is erased. eqp was already zeroed in
            # the lifecycle wrapper.
            cold_cell_mask = np.asarray(cold_reset, dtype=bool)[..., 0]
            fold = cold_cell_mask[..., None, None]
            mechanics._eps_ref_state = np.where(
                fold,
                mechanics._eps_ref_state + mechanics._eps_p_state,
                mechanics._eps_ref_state,
            )
            mechanics._eps_p_state = np.where(
                fold, 0.0, mechanics._eps_p_state
            )
            REGISTRY.eps_p = mechanics._eps_p_state
            REGISTRY.eps_ref = mechanics._eps_ref_state
            REGISTRY.pending_cold_reset = np.zeros_like(
                cold_reset, dtype=bool
            )
        reference_mask = None
        for name, event_mask in (
            ("pending_reference", REGISTRY.pending_reference),
            ("relaxation_hot", REGISTRY.relaxation_hot),
        ):
            if event_mask is None or not bool(np.any(event_mask)):
                continue
            if not hasattr(build_problem, "capture_stress_free_reference"):
                raise ValueError(f"{name} exists without a v06 build problem")
            if event_mask.shape != build_problem._relaxation_mask.shape:
                raise ValueError(
                    f"{name} shape does not match build mechanics state shape"
                )
            reference_mask = (
                event_mask
                if reference_mask is None
                else reference_mask | event_mask
            )
        if reference_mask is not None:
            build_problem.capture_stress_free_reference(
                u_guess[0],
                reference_mask,
                params[4],
                params[1],
            )
            if REGISTRY.pending_reference is not None:
                REGISTRY.pending_reference = np.zeros_like(
                    REGISTRY.pending_reference, dtype=bool
                )
            if REGISTRY.pending_cold_reset is not None:
                REGISTRY.pending_cold_reset = np.zeros_like(
                    REGISTRY.pending_cold_reset, dtype=bool
                )

        adopted = False
        if is_release and not getattr(mechanics, "_v06_state_adopted", False):
            for name, state, attribute in (
                ("eps_p", REGISTRY.eps_p, "_eps_p_state"),
                ("eps_ref", REGISTRY.eps_ref, "_eps_ref_state"),
            ):
                expected_shape = getattr(mechanics, attribute).shape
                if state is None:
                    raise ValueError(f"release is missing {name} build state")
                if state.shape != expected_shape:
                    raise ValueError(
                        f"release {name} state shape {state.shape} does not "
                        f"match {expected_shape}"
                    )
                setattr(mechanics, attribute, state)
            mechanics._v06_state_adopted = True
            adopted = True
            print("v06: release adopted build mechanical tensor state")
        solution = original_run_mechanics(
            mechanics, u_guess, params, newton_overrides
        )
        if adopted and mechanics.mechanics_model == "j2_plastic":
            eqp_new = mechanics.compute_eqp_update(solution[0], params)
            params[-1] = eqp_new
            REGISTRY.eqp = eqp_new
            REGISTRY.eps_ref = mechanics._eps_ref_state
            mechanics._v06_release_committed = True
            print("v06: release plastic state committed")
        return solution

    def save_step_with_committed_release_state(*args, **kwargs):
        positional = list(args)
        mode_id = positional[19] if len(positional) > 19 else kwargs.get("mode_id")
        if (
            mode_id == base_module.MODE_TO_ID["release"]
            and REGISTRY.eqp is not None
        ):
            if len(positional) > 16:
                positional[16] = REGISTRY.eqp
            else:
                kwargs["eqp_quad"] = REGISTRY.eqp
        kwargs["quad_cell_info_factory"] = (
            base_module.make_quad_stress_cell_infos
        )
        return original_save_step(*positional, **kwargs)

    base_module.ThermoMechanical = StateSafeThermoMechanical
    install_phase_lifecycle_wrapper(base_module)
    install_mechanics_event_wrapper(base_module)
    install_output_state_wrapper(base_module)
    install_step_state_capture_wrappers(base_module)
    base_module.run_mechanics = run_mechanics_with_state
    base_module.save_step = save_step_with_committed_release_state
    base_module.load_property_tables = load_validated_property_tables
    base_module._v06_adapter_installed = True
    if profiler is not None:
        profiler.meta["v06_constitutive_model"] = (
            "state-safe tensor J2 with exact capped-hardening return"
        )
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    v04 = load_v04_wrapper()
    REGISTRY.reset()
    original_load = v04.load_base_solver
    original_report = v04.ProfilingReport

    class V06ProfilingReport(original_report):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.meta.update(
                {
                    "v06_driver": str(Path(__file__).resolve()),
                    "v06_constitutive_model": (
                        "state-safe tensor J2 with exact capped-hardening return"
                    ),
                    "v06_model_boundary": "v03-loop + v04-performance + v06-state",
                    "v06_claim_level": "numerical_model_output_only",
                    "v06_thermal_energy_ledger": "enabled",
                }
            )

    def load_base_solver_with_v06(*args: Any, **kwargs: Any):
        base = original_load(*args, **kwargs)
        install_v06_adapter(base)
        original_base_main = base.main

        def base_main_with_phase_refresh(*main_args: Any, **main_kwargs: Any):
            phase_refreshed = install_phase_lifecycle_wrapper(base)
            mechanics_refreshed = install_mechanics_event_wrapper(base)
            ledger_refreshed = install_thermal_ledger_wrapper(base)
            if phase_refreshed:
                print("v06: lifecycle wrapper refreshed after v04 JIT patches")
            if mechanics_refreshed:
                print("v06: mechanics-event wrapper refreshed after v04 caches")
            if ledger_refreshed:
                print("v06: thermal energy ledger installed after v04 solver patch")
            return original_base_main(*main_args, **main_kwargs)

        base.main = base_main_with_phase_refresh
        print("v06: paper-validation constitutive adapter installed")
        return base

    v04.load_base_solver = load_base_solver_with_v06
    v04.ProfilingReport = V06ProfilingReport
    completed = False
    try:
        result = v04.main(argv)
        completed = int(result or 0) == 0
        return result
    finally:
        if REGISTRY.thermal_ledger is not None:
            REGISTRY.thermal_ledger.finalize(completed=completed)
        v04.load_base_solver = original_load
        v04.ProfilingReport = original_report


if __name__ == "__main__":
    raise SystemExit(main())
