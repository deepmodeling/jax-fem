import jax
import jax.numpy as np
import jax.profiler
import numpy as onp

def func1(x):
  return np.tile(x, 10) * 0.5

def func2(x):
  y = func1(x)
  return y, np.tile(x, 10) + 1

x = jax.random.normal(jax.random.PRNGKey(42), (1000, 1000))
y, z = func2(x)

k = np.ones((10000, 10000))

del k

z.block_until_ready()

jax.profiler.save_device_memory_profile(f"modules/fem/experiments/data/memory.prof")
