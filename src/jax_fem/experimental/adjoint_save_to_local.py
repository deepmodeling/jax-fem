import jax
import jax.numpy as np
import os

jax.config.update("jax_enable_x64", True)

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')
numpy_dir = os.path.join(data_dir, 'numpy')
file_path = os.path.join(numpy_dir, 'tmp.npy')


def raw_f(x, y):
    return np.sin(x) * y

@jax.custom_vjp
def f(x, y):
    return np.sin(x) * y

def f_fwd(x, y):
    np.save(file_path, np.cos(x))
    return f(x, y), (np.sin(x), y)

def f_bwd(res, g):
    cos_x = np.load(file_path)
    sin_x, y = res
    return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)

print(f(1., 2.))
print(raw_f(1., 2.))

print(jax.grad(f)(1., 2.))
print(jax.grad(raw_f)(1., 2.))








 