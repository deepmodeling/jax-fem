from typing import Any, Mapping

import haiku as hk
import jax.numpy as jnp
from jax import tree_util

__all__ = ["reduce_loss_tree",
           "weighted_loss",
           "split_loss_tree",
           "calculate_losses"]


def reduce_loss_tree(loss_tree: Mapping) -> jnp.array:
    """Reduces a loss tree to a scalar (i.e. jnp.array w/ size 1)."""
    return tree_util.tree_reduce(lambda x, y: x + y, loss_tree)


def weighted_loss(loss_tree: Mapping, weights: Mapping) -> Any:
    """Updates a loss tree by applying weights at the leaves."""
    return hk.data_structures.map(
        # m: module_name, n: param name, v: param value
        lambda m, n, v: weights[n] * v,
        loss_tree)


def split_loss_tree(loss_tree: Mapping):
    """Splits a loss tree into content and style loss trees."""
    return hk.data_structures.partition(
        lambda m, n, v: n == "content_loss",
        loss_tree)


def calculate_losses(loss_tree: Mapping, weights: Mapping):
    """Returns a tuple of current content loss and style loss."""
    # obtain content and style trees
    content_tree, style_tree = split_loss_tree(loss_tree)

    # reduce and return losses
    return weights['content_loss']*reduce_loss_tree(content_tree), weights['style_loss']*reduce_loss_tree(style_tree)
