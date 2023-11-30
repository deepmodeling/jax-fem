import unittest
from . import __path__

suite = unittest.TestLoader().discover(__path__[0])
unittest.TextTestRunner(verbosity=2).run(suite)
