"""
Only under forward mode, safe_sqrt works. (for JAX version 0.4.26)
"""
import jax.numpy as np
import jax


def safe_sqrt(x):  
    safe_x = np.where(x > 0., np.sqrt(x), 0.)
    return safe_x


print(jax.jacrev(np.sqrt)(0.)) # Expected inf
print(jax.jacfwd(np.sqrt)(0.)) # Expected inf
print(jax.jacrev(safe_sqrt)(0.)) # Expected nan
print(jax.jacfwd(safe_sqrt)(0.)) # Expected 0.