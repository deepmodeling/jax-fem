import jax.numpy as np
import jax
import time

x = np.ones((1000, 1000), dtype=np.float32)  


@jax.jit
def f(x):
    return x.T @ (x - x.mean(axis=0))

def f1(x):
    return f(x)


def f2(x):
    def f(x): 
        return x.T @ (x - x.mean(axis=0))
    f = jax.jit(f)
    return f(x)


@jax.jit
def g(x):
    return x.T @ (x - x.mean(axis=0))

def f3(x, h):
    return h(x)


print(f"\nTest f1...")
for i in range(10):
    start = time.time()
    a = f1(x).block_until_ready()
    print(time.time() - start)


print(f"\nTest f2...")
for i in range(10):
    start = time.time()
    a = f2(x).block_until_ready()
    print(time.time() - start)


print(f"\nTest f3...")
for i in range(10):
    start = time.time()
    a = f3(x, g).block_until_ready()
    print(time.time() - start)