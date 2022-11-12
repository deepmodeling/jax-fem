import numpy as onp
import jax
import jax.numpy as np
import os
 

crt_dir = os.path.dirname(__file__)


def exp():
    def body(x):
      return 0.1 * x

    def true_fun(arg):
      x, pred = arg
      return jax.lax.while_loop(lambda x: pred & ((x <= 0) | (x > 1e-6)), body, x)

    def false_fun(arg):
      x, pred = arg
      return jax.lax.while_loop(lambda x: ~pred & (x < -1e-6), body, x)

    def branching(x):
      cond = x > 0
      return jax.lax.cond(cond, true_fun, false_fun, (x, cond))

    a = np.arange(10.) - 5
    print(jax.vmap(branching)(a))


def exp_cp():
    input_slip_sys = onp.loadtxt(os.path.join(crt_dir, 'input_slip_sys.txt'))
    print(input_slip_sys)
    

if __name__ == "__main__":
    exp_cp()
