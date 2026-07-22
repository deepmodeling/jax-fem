import jax
import jax.numpy as jnp
from jax import lax


def thermal_step(T, laser, dt, k):
    return T + dt * (k * (laser - T))


def mechanics_step(u, E, nu):
    strain = u * 0.01
    return u + E * strain / (1.0 + nu)


def fem_step(carry, state):

    T, u, params, cells = carry

    laser = state["laser"]
    dt = state["dt"]

    # thermal
    T_new = thermal_step(T, laser, dt, params["k"])

    # mechanics
    u_new = mechanics_step(u, params["E"], params["nu"])

    # coupling
    T_new = T_new + laser * params["absorptivity"]

    return (T_new, u_new, params, cells), (T_new, u_new)