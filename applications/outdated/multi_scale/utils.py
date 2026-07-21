import numpy as onp
import jax
import jax.numpy as np


def flat_to_tensor(X_flat):
    return np.array([[X_flat[0], X_flat[3], X_flat[4]],
                     [X_flat[3], X_flat[1], X_flat[5]],
                     [X_flat[4], X_flat[5], X_flat[2]]])


def tensor_to_flat(X_tensor):
    return np.array([X_tensor[0, 0], X_tensor[1, 1], X_tensor[2, 2], X_tensor[0, 1], X_tensor[0, 2], X_tensor[1, 2]])
