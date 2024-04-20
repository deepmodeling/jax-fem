from pyfiglet import Figlet

f = Figlet(font='starwars')
print(f.renderText('JAX - FEM'))

from .logger_setup import setup_logger
# LOGGING
logger = setup_logger(__name__)

# TODO: Be automatic
__version__ = "0.0.4"