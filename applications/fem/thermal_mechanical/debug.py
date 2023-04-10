import jax.numpy as np
import jax

def fn():

    def f(x):
        return a*x

    def h(x):

        
        a = 1
        x = f(x)
        a = 2
        x = f(x)
        a = 3
        x = f(x)
        return x

    print(h(1.))
    print(jax.grad(h)(1.))

fn()