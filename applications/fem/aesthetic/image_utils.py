import os
import os.path
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageOps
from jax import tree_util


# TODO: Make target size a tuple for controlling aspect ratio
def load_image(fp: str, img_type: str, target_size: int = 512, dtype=None, reverse = False):
    if not os.path.exists(fp):
        raise ValueError(f"File {fp} does not exist.")

    print(f'Loading {img_type} image...')

    image = Image.open(fp)

    image = ImageOps.grayscale(image).convert("RGB")

    image.save(fp[:-3] + 'jpg')

    image = image.resize((target_size, target_size))

    image = image.rotate(-90)

    image = jnp.array(image, dtype=dtype)
    image = image / 255.

    image = np.where(image < 0.5, 0., 1.)

    image = jnp.clip(image, 0., 1.)
    image = jnp.expand_dims(jnp.moveaxis(image, -1, 0), 0)

    if reverse:
        image = 1. - image

    print(f"{img_type.capitalize()} image loaded successfully. "
          f"Shape: {image.shape}")

    return image


def save_image(params: hk.Params, out_fp: str, reverse = False):
    im_data = tree_util.tree_leaves(params)[0]
    # clip values to avoid overflow problems in uint8 conversion
    im_data = jnp.clip(im_data, 0., 1.)

    if reverse:
        im_data = 1. - im_data

    # undo transformation block, squeeze off the batch dimension
    image: np.ndarray = np.squeeze(np.asarray(im_data))
    image = image * 255
    image = image.astype(np.uint8)
    image = np.moveaxis(image, 0, -1)

    # TODO: This needs to change for a tiled image
    im = Image.fromarray(image, mode="RGB")

    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    im.save(out_fp)


def checkpoint(params: hk.Params, out_dir: str, filename: str, reverse = False):
    """Saves the image at a checkpoint given by step."""
    out_fp = os.path.join(out_dir, filename)

    save_image(params, out_fp, reverse)
