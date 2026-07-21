# logger.py
import logging


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Ignore TensorFlow and JAX warnings
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)

    # Create a handler
    handler = logging.StreamHandler()

    # Create a formatter
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    # Add the formatter to the handler
    handler.setFormatter(formatter)

    # Check if the logger already has handlers. If the logger doesn't have any
    # handlers, add the new handler and set propagate to False to prevent the
    # log messages from being passed to the root logger and possibly being
    # duplicated.

    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False

    return logger