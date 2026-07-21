import jax.numpy as np
import jax

class A:
    def __init__(self):
        self.f = self.get_fn()

    def get_fn(self):
        @jax.jit
        def f():
            return self.E
        return f

    def set_params(self, E):
        self.E = E

a = A()

def test_fn_a(E):
    a.set_params(E)
    return a.f()

E_a = 10.

print(test_fn_a(E_a))
print(jax.grad(test_fn_a)(E_a))


class B:
    def f(self):
        return self.E

    def set_params(self, E):
        self.E = E

b = B()

@jax.jit
def test_fn_b(E):
    b.set_params(E)
    return b.f()

E_b = 10.

print(test_fn_b(E_b))
print(jax.grad(test_fn_b)(E_b))

E_b = 20.

print(test_fn_b(E_b))
print(jax.grad(test_fn_b)(E_b))