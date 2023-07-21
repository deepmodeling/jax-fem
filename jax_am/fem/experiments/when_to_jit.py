import numpy as np
import jax.numpy as jnp
import jax
import time


@jax.jit
def g(x):
    def f():  # function we're benchmarking (works in both NumPy & JAX)
        return x.T @ (x - x.mean(axis=0))
    return f()

x = jnp.ones((1000, 1000), dtype=jnp.float32)  # same as JAX default dtype


@jax.jit
def h(x):  
    return x.T @ (x - x.mean(axis=0))
    



def s(x):
    def f():  # function we're benchmarking (works in both NumPy & JAX)
        return np.sum(x**2)
    return jax.jit(f)()
 
# for i in range(10):
#     start = time.time()
#     a = g(x).block_until_ready()
#     # a = h(x).block_until_ready()
#     print(time.time() - start)


print(jax.grad(s)(1.))
print(jax.grad(s)(2.))

 