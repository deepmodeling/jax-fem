import copy
import os
import time
from typing import Tuple, List

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import app
from absl import flags

from image_utils import load_image, checkpoint
from models import augmented_vgg19
from modules import imagenet_mean, imagenet_std
from tree_utils import weighted_loss, calculate_losses, reduce_loss_tree

OptimizerUpdate = Tuple[hk.Params, optax.OptState, hk.State]

# Syntax: Name, default value, help string
POOLING = flags.DEFINE_string("pooling", "avg", "Pooling method to use.")
NUM_STEPS = flags.DEFINE_integer("num_steps", 500, "Number of training steps.")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3,
                                   "Learning rate of the Adam optimizer.")
IMAGE_SIZE = flags.DEFINE_integer("image_size", 512, "Target size of the "
                                                     "images in pixels.")
SAVE_IMAGE_EVERY = flags.DEFINE_integer("save_image_every", 50,
                                        "Saves the image every n steps "
                                        "to monitor progress.")
CONTENT_WEIGHT = flags.DEFINE_float("content_weight", 1.,
                                    "Content loss weight.")
STYLE_WEIGHT = flags.DEFINE_float("style_weight", 1e4, "Style loss weight.")
CONTENT_LAYERS = flags.DEFINE_list("content_layers", "conv_4",
                                   "Names of network layers for which to "
                                   "capture content loss.")
STYLE_LAYERS = flags.DEFINE_list("style_layers",
                                 "conv_1,conv_2,conv_3,conv_4,conv_5",
                                 "Names of network layers for which to "
                                 "capture style loss.")
OUT_DIR = flags.DEFINE_string("out_dir", "images", "Output directory to save "
                                                   "the styled images to.")
FLAGS = flags.FLAGS


def validate_argv_inputs(argv: List):
    if len(argv) < 4:
        raise app.UsageError("Usage: python main.py CONTENT_IMAGE "
                             "STYLE_IMAGE MODEL_WEIGHTS [--flags]")

    if not os.path.isdir(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
    # TODO: Expand input validation


def style_transfer(argv):
    # validate_argv_inputs(argv)

    # first arg is Python file name
    # content_fp, style_fp, model_fp = argv[1:]

    model_fp = "models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # content_fp = "contents/dancing.jpg" 
    content_fp = "contents/structure.jpg"
    style_fp = "styles/voronoi.png"
    

    content_image = load_image(content_fp, "content", FLAGS.image_size)
    style_image = load_image(style_fp, "style", FLAGS.image_size)

    weights = {"content_loss": FLAGS.content_weight,
               "style_loss": FLAGS.style_weight}

    def net_fn(image: jnp.ndarray, is_training: bool = False):
        vgg = augmented_vgg19(fp=model_fp,
                              style_image=style_image,
                              content_image=content_image,
                              mean=imagenet_mean,
                              std=imagenet_std,
                              content_layers=FLAGS.content_layers,
                              style_layers=FLAGS.style_layers,
                              pooling=FLAGS.pooling)
        return vgg(image, is_training)

    def loss(trainable_params: hk.Params,
             non_trainable_params: hk.Params,
             current_state: hk.State,
             image: jnp.ndarray):

        merged_params = hk.data_structures.merge(trainable_params,
                                                 non_trainable_params)

        # stateful apply call, state contains the losses
        _, new_state = net.apply(merged_params, current_state,
                                 None, image, is_training=True)

        w_loss = weighted_loss(new_state, weights=weights)

        loss_val = reduce_loss_tree(w_loss)

        return loss_val, new_state

    @jax.jit
    def update(trainable_params: hk.Params,
               non_trainable_params: hk.Params,
               c_opt_state: optax.OptState,
               c_state: hk.State,
               image: jnp.ndarray) -> OptimizerUpdate:
        """Learning rule (stochastic gradient descent)."""
        (_, new_state), trainable_grads = (
            jax.value_and_grad(loss, has_aux=True)(trainable_params,
                                                   non_trainable_params,
                                                   c_state,
                                                   image))

        # update trainable params
        updates, new_opt_state = opt.update(trainable_grads,
                                            c_opt_state,
                                            trainable_params)

        new_params = optax.apply_updates(trainable_params, updates)

        return new_params, new_opt_state, new_state

    net = hk.transform_with_state(net_fn)
    opt = optax.adam(learning_rate=FLAGS.learning_rate)

    input_image = copy.deepcopy(content_image)

    # Initialize network and optimiser; we supply an input to get shapes.
    full_params, state = net.init(None, input_image, False)

    # split params into trainable and non-trainable
    t_params, nt_params = hk.data_structures.partition(
        lambda m, n, v: m == "norm",
        full_params
    )

    opt_state = opt.init(t_params)

    num_params = hk.data_structures.tree_size(full_params)
    num_t_params = hk.data_structures.tree_size(t_params)
    mem = hk.data_structures.tree_bytes(full_params)

    print(f"Total number of parameters: {num_params}")
    print(f"Number of trainable parameters: {num_t_params}")
    print(f"Number of non-trainable parameters: {num_params - num_t_params}")
    print(f"Memory footprint of network parameters: {mem / 1e6:.2f} MB")

    start = time.time()
    print("Starting style transfer optimization loop.")
    # Style transfer loop.
    # TODO: Think about changing to jax.lax control flow
    for step in range(FLAGS.num_steps + 1):
        # Do SGD on the same input image over and over again.
        t_params, opt_state, state = update(t_params, nt_params,
                                            opt_state, state, input_image)

        if step % 10 == 0:
            c_loss, s_loss = calculate_losses(state)

            print(f"Iteration: {step} Content loss: {c_loss:.4f} "
                  f"Style loss: {s_loss:.4f}")

        if step % FLAGS.save_image_every == 0:
            # save current image to check progress
            checkpoint(t_params, FLAGS.out_dir, f"styled_it{step}.jpg")

    print(f"Style transfer finished. Took {(time.time() - start):.2f} secs.")


if __name__ == '__main__':
    app.run(style_transfer)
