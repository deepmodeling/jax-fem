import jax
from jax import lax
from .fem_core import fem_step


def run_xla_fem(mesh, step_states, T0, u0, params):

    init = (T0, u0, params, mesh.cells)

    def step(carry, state):
        return fem_step(carry, state)

    final, history = lax.scan(step, init, step_states)

    return final, history