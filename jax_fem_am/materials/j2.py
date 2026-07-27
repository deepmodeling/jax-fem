"""Small-strain associative J2 plasticity with capped isotropic hardening.

The six-component storage used by v05 is intentionally not exposed here.
The constitutive kernel uses symmetric 3x3 tensors, which avoids engineering-
shear conversion ambiguity at the solver boundary.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class PlasticState(NamedTuple):
    """Path-dependent state at one material point."""

    eqp: jnp.ndarray
    eps_p: jnp.ndarray


class J2Update(NamedTuple):
    """Stress and committed state returned by one radial-return operation."""

    stress: jnp.ndarray
    state: PlasticState
    delta_eqp: jnp.ndarray


class FlowCurve(NamedTuple):
    """Immutable rectangular flow-stress data used by JAX return maps."""

    temperatures: jnp.ndarray
    plastic_strains: jnp.ndarray
    stresses: jnp.ndarray


def _interpolate_clamped(x, knots, values):
    """Linearly interpolate one sorted axis with endpoint clamping."""
    x = jnp.clip(jnp.asarray(x), knots[0], knots[-1])
    upper = jnp.clip(
        jnp.searchsorted(
            knots,
            x,
            side="right",
            method="compare_all",
        ),
        1,
        knots.size - 1,
    )
    lower = upper - 1
    x0 = knots[lower]
    x1 = knots[upper]
    weight = (x - x0) / (x1 - x0)
    return values[lower] + weight * (values[upper] - values[lower])


def flow_stress_from_curve(
    temperature,
    equivalent_plastic_strain,
    temperatures,
    plastic_strains,
    flow_stresses,
):
    """Evaluate a rectangular temperature/plastic-strain flow-stress grid.

    Temperature interpolation and endpoint clamping match the Kaess/Abaqus
    material description. Plastic-strain endpoint clamping is an explicit
    solver assumption for values outside the frozen input curve.
    """
    temperatures = jnp.asarray(temperatures)
    plastic_strains = jnp.asarray(plastic_strains)
    flow_stresses = jnp.asarray(flow_stresses)
    stress_at_temperature = _interpolate_clamped(
        temperature,
        temperatures,
        flow_stresses,
    )
    return _interpolate_clamped(
        equivalent_plastic_strain,
        plastic_strains,
        stress_at_temperature,
    )


def equivalent_stress(stress):
    """Return von Mises equivalent stress for a symmetric 3x3 tensor.

    The sqrt is guarded at exactly zero deviatoric stress: d(sqrt)/dx is
    infinite at x=0, and both AD modes turn that into NaN (0*inf) for every
    quadrature point whose trial stress is exactly zero - e.g. freshly
    activated powder with zero thermal strain. The guard picks the physically
    correct elastic-branch subgradient (0) instead.
    """
    stress = jnp.asarray(stress)
    dev = stress - jnp.trace(stress, axis1=-2, axis2=-1)[..., None, None] * (
        jnp.eye(3) / 3.0
    )
    sq = 1.5 * jnp.sum(dev * dev, axis=(-2, -1))
    positive = sq > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, sq, 1.0)), 0.0)


def elastic_strain_from_stress(stress, young, poisson):
    """Invert isotropic Hooke elasticity for scalar or batched SI inputs."""
    stress = 0.5 * (
        jnp.asarray(stress) + jnp.swapaxes(jnp.asarray(stress), -1, -2)
    )
    young = jnp.asarray(young)
    poisson = jnp.asarray(poisson)
    while young.ndim < stress.ndim:
        young = young[..., None]
    while poisson.ndim < stress.ndim:
        poisson = poisson[..., None]
    trace = jnp.trace(stress, axis1=-2, axis2=-1)[..., None, None]
    return (
        (1.0 + poisson) * stress / young
        - poisson * trace * jnp.eye(3) / young
    )


def _plastic_increment(q_trial, eqp_old, shear, yield0, hardening, saturation):
    """Solve the piecewise-linear consistency equation exactly."""
    three_mu = 3.0 * shear
    linear_yield_old = yield0 + hardening * eqp_old
    finite_cap = jnp.isfinite(saturation)
    current_yield = jnp.where(
        finite_cap, jnp.minimum(linear_yield_old, saturation), linear_yield_old
    )

    linear_delta = jnp.maximum(
        (q_trial - linear_yield_old) / (three_mu + hardening), 0.0
    )
    safe_hardening = jnp.where(hardening > 0.0, hardening, 1.0)
    cap_eqp = (saturation - yield0) / safe_hardening
    saturation_delta = jnp.maximum((q_trial - saturation) / three_mu, 0.0)

    has_linear_cap = finite_cap & (hardening > 0.0) & (saturation > yield0)
    crosses_cap = eqp_old + linear_delta > cap_eqp
    capped_delta = jnp.where(crosses_cap, saturation_delta, linear_delta)

    constant_yield = jnp.where(finite_cap, jnp.minimum(yield0, saturation), yield0)
    constant_delta = jnp.maximum((q_trial - constant_yield) / three_mu, 0.0)
    no_linear_hardening = finite_cap & ~has_linear_cap
    delta = jnp.where(
        has_linear_cap,
        capped_delta,
        jnp.where(no_linear_hardening, constant_delta, linear_delta),
    )
    return jnp.where(q_trial > current_yield, delta, 0.0)


def _plastic_increment_from_curve(
    q_trial,
    eqp_old,
    shear,
    temperature,
    flow_curve,
    saturation,
):
    """Solve a monotone piecewise-linear consistency equation exactly."""
    three_mu = 3.0 * shear
    stresses = _interpolate_clamped(
        temperature,
        flow_curve.temperatures,
        flow_curve.stresses,
    )
    strains = flow_curve.plastic_strains
    consistency_knots = three_mu * strains + stresses
    target = q_trial + three_mu * eqp_old

    upper = jnp.clip(
        jnp.searchsorted(
            consistency_knots,
            target,
            side="left",
            method="compare_all",
        ),
        1,
        strains.size - 1,
    )
    lower = upper - 1
    segment_fraction = (
        (target - consistency_knots[lower])
        / (consistency_knots[upper] - consistency_knots[lower])
    )
    segment_root = strains[lower] + segment_fraction * (
        strains[upper] - strains[lower]
    )
    below_root = (target - stresses[0]) / three_mu
    beyond_root = (target - stresses[-1]) / three_mu
    raw_root = jnp.where(
        target < consistency_knots[0],
        below_root,
        jnp.where(
            target > consistency_knots[-1],
            beyond_root,
            segment_root,
        ),
    )
    raw_root = jnp.maximum(raw_root, eqp_old)

    current_curve_yield = flow_stress_from_curve(
        temperature,
        eqp_old,
        flow_curve.temperatures,
        flow_curve.plastic_strains,
        flow_curve.stresses,
    )
    finite_cap = jnp.isfinite(saturation)
    safe_cap = jnp.where(finite_cap, saturation, 0.0)
    current_yield = jnp.where(
        finite_cap,
        jnp.minimum(current_curve_yield, safe_cap),
        current_curve_yield,
    )
    cap_root = (target - safe_cap) / three_mu
    cap_root_yield = flow_stress_from_curve(
        temperature,
        cap_root,
        flow_curve.temperatures,
        flow_curve.plastic_strains,
        flow_curve.stresses,
    )
    use_cap = (
        finite_cap
        & (cap_root >= eqp_old)
        & (cap_root_yield >= safe_cap)
    )
    effective_root = jnp.where(use_cap, cap_root, raw_root)
    delta = jnp.maximum(effective_root - eqp_old, 0.0)
    return jnp.where(q_trial > current_yield, delta, 0.0)


def radial_return(
    *,
    strain,
    thermal_strain,
    state: PlasticState,
    young,
    poisson,
    yield_stress,
    hardening,
    saturation=jnp.inf,
    temperature=None,
    flow_curve=None,
    flow_curve_active=True,
):
    """Return stress and committed state for one total-strain increment.

    Parameters are SI scalars and require young > 0, hardening >= 0 and
    -1 < poisson < 0.5. An infinite saturation value disables the cap. When
    ``flow_curve`` is provided, its piecewise-linear consistency equation
    replaces the scalar yield/hardening pair and ``temperature`` is required.
    """
    strain = 0.5 * (jnp.asarray(strain) + jnp.asarray(strain).T)
    thermal_strain = 0.5 * (
        jnp.asarray(thermal_strain) + jnp.asarray(thermal_strain).T
    )
    eps_e_trial = strain - thermal_strain - state.eps_p
    shear = young / (2.0 * (1.0 + poisson))
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    trial = lame * jnp.trace(eps_e_trial) * jnp.eye(3) + 2.0 * shear * eps_e_trial
    hydro = jnp.trace(trial) * jnp.eye(3) / 3.0
    dev_trial = trial - hydro
    q_trial = equivalent_stress(trial)
    scalar_delta_eqp = _plastic_increment(
        q_trial,
        state.eqp,
        shear,
        yield_stress,
        hardening,
        saturation,
    )
    if flow_curve is None:
        delta_eqp = scalar_delta_eqp
    else:
        if temperature is None:
            raise ValueError(
                "temperature is required when flow_curve is provided"
            )
        curve_delta_eqp = _plastic_increment_from_curve(
            q_trial,
            state.eqp,
            shear,
            temperature,
            flow_curve,
            saturation,
        )
        delta_eqp = jnp.where(
            flow_curve_active,
            curve_delta_eqp,
            scalar_delta_eqp,
        )
    # Guard the plastic correction with where() on BOTH the primal and the
    # denominator: q_trial can be exactly zero (fresh stress-free material),
    # and a subnormal floor like finfo.tiny underflows to zero when the AD
    # chain squares it (d(1/q)/du ~ q_dot/q^2), turning the whole jacobian
    # row into NaN. In the plastic branch q_trial >= yield >> 0, so using
    # q_trial itself as the divisor is safe there.
    is_plastic_step = delta_eqp > 0.0
    q_div = jnp.where(is_plastic_step, q_trial, 1.0)
    scale = jnp.where(
        is_plastic_step,
        jnp.clip(1.0 - 3.0 * shear * delta_eqp / q_div, 0.0, 1.0),
        1.0,
    )
    delta_eps_p = jnp.where(
        is_plastic_step, 1.5 * delta_eqp / q_div, 0.0
    ) * dev_trial
    new_state = PlasticState(
        eqp=state.eqp + delta_eqp,
        eps_p=state.eps_p + delta_eps_p,
    )
    return J2Update(
        stress=hydro + scale * dev_trial,
        state=new_state,
        delta_eqp=delta_eqp,
    )


def reset_on_remelt(state: PlasticState, remelted):
    """Reset both hardening and tensor history at remelted quadrature points."""
    remelted = jnp.asarray(remelted, dtype=bool)
    return PlasticState(
        eqp=jnp.where(remelted, 0.0, state.eqp),
        eps_p=jnp.where(remelted[..., None, None], 0.0, state.eps_p),
    )
