import jax.numpy as jnp


def build_states(step_states):

    def convert(s):
        return {
            "laser": float(s.laser_power),
            "dt": float(s.dt),
            "layer": int(s.layer_idx),
            "hatch": int(s.hatch_idx),
        }

    return [convert(s) for s in step_states]