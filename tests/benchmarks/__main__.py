import unittest
# from jax_fem.tests import __path__ # also works
from . import __path__

suite = unittest.TestLoader().discover(__path__[0])
print(f"Found test suite = {suite}")
unittest.TextTestRunner(verbosity=2).run(suite)
