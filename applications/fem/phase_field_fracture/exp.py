import jax
import jax.numpy as jnp
import numpy as np
import os

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')
numpy_dir = os.path.join(data_dir, 'numpy')
file_path = os.path.join(numpy_dir, 'tmp.npy')

@jax.custom_vjp
def f(x, y):
    return jnp.sin(x) * y

def f_fwd(x, y):
    jnp.save(file_path, jnp.cos(x))
    return f(x, y) + 1e3, (jnp.sin(x), y)

def f_bwd(res, g):
    cos_x = jnp.load(file_path)
    sin_x, y = res
    return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)

print(jax.value_and_grad(f)(1., 2.))
print(jax.grad(f)(1., 2.))
print(f(1., 2.))