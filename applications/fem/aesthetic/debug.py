import jax 
from jax.config import config
import jax.numpy as np

config.update("jax_enable_x64", True)

a = np.array(1.)
print(a.dtype)

config.update("jax_enable_x64", False)


b = np.array(1.)
print(b.dtype)


c = a + b
print(c.dtype)


 

config.update("jax_enable_x64", True)


print(a.dtype)
print(b.dtype) 
c = a + b
print(c.dtype)



 