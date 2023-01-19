import jax
import jax.numpy as np
import numpy as onp


def exp():

    a = onp.array([1., 2., 3.])

    @jax.custom_jvp
    def simple_fn(x):
        y = x**2
        return y

    @simple_fn.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        v, = tangents
        y = simple_fn(x)
        return y, a
        # return y, v*2*x

    print(jax.jacfwd(simple_fn)(np.array([10., 11., 12.])))



if __name__ == "__main__":
    exp()