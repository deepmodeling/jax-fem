import sys
sys.path.append("../../..")

import copy
import os
import time
from typing import Tuple, List
import haiku as hk
import numpy as np
import jax
from jax import tree_util
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from jax_fem.generate_mesh import rectangle_mesh

from jax.config import config

from applications.fem.aesthetic.arguments import args, bcolors
from applications.fem.aesthetic.image_utils import load_image, checkpoint
from applications.fem.aesthetic.models import augmented_vgg19
from applications.fem.aesthetic.modules import imagenet_mean, imagenet_std
from applications.fem.aesthetic.tree_utils import weighted_loss, calculate_losses, reduce_loss_tree


np.set_printoptions(threshold=1000, linewidth=75, suppress=False, precision=8)

input_path = os.path.join(os.path.dirname(__file__), 'input') 

def style_transfer(problem, rho_initial, image_path, reverse):
    model_fp = os.path.join(input_path, "models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")

    # content_fp = os.path.join(input_path, "styles/tree.png")
    # style_fp = os.path.join(input_path, "styles/tree.png")

    content_fp = os.path.join(input_path, image_path)
    style_fp = os.path.join(input_path, image_path)
    content_image = load_image(content_fp, "content", args.image_size, reverse=reverse)
    style_image = load_image(style_fp, "style", args.image_size, reverse=reverse)


    style_image_grey = np.repeat(np.mean(style_image, axis=1)[:, None, :, :], 3, axis=1)


    # debug_img = np.array([[0., 0., 0.], 
    #                       [0., 0., 1.], 
    #                       [1., 0., 1.]])
    # debug_img = np.repeat(debug_img[None, None, :, :], 3, axis=1)
    # checkpoint(debug_img, os.path.join(args.output_path, 'jpg'), f"debug.jpg", reverse=False)

    checkpoint(style_image_grey, os.path.join(args.output_path, 'jpg'), f"style_basis.jpg", reverse=True)

    weights = {"content_loss": args.content_weight,
               "style_loss": args.style_weight}
 
    def net_fn(image: jnp.ndarray, is_training: bool = False):
        vgg = augmented_vgg19(fp=model_fp,
                              style_image=style_image,
                              content_image=content_image,
                              mean=imagenet_mean,
                              std=imagenet_std,
                              content_layers=args.content_layers,
                              style_layers=args.style_layers,
                              pooling=args.pooling)
        return vgg(image, is_training)

    @jax.jit
    def smooth_loss(image_theta):
        image_theta = np.mean(image_theta, axis=(0, 1))
        
        x_grad_loss = (image_theta[:-2, 1:-1] + image_theta[2:, 1:-1] - 2*image_theta[1:-1, 1:-1])**2
        y_grad_loss = (image_theta[1:-1, :-2] + image_theta[1:-1, 2:] - 2*image_theta[1:-1, 1:-1])**2
        grad_loss = np.sum((x_grad_loss + y_grad_loss)**1.25)
        alpha = 0.1
        result = alpha*grad_loss

        result = 0.

        return result

    @jax.jit
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

        image_theta = tree_util.tree_leaves(trainable_params)[0]
        grad_loss = smooth_loss(image_theta)

        total_loss = loss_val + grad_loss

        return total_loss, new_state

    net = hk.transform_with_state(net_fn)
    input_image = copy.deepcopy(content_image)
    full_params, state = net.init(None, input_image, False)
    t_params, nt_params = hk.data_structures.partition(lambda m, n, v: m == "norm", full_params)
 
    Nx_image, Ny_image = args.image_size, args.image_size
    image_meshio_mesh = rectangle_mesh(Nx=Nx_image, Ny=Ny_image, domain_x=args.Lx, domain_y=args.Ly)
    image_points = image_meshio_mesh.points
    image_cells = image_meshio_mesh.cells_dict['quad']
    image_cell_centroids = np.mean(np.take(image_points, image_cells, axis=0), axis=1)
    xc_image = image_cell_centroids[:, 0].reshape((Nx_image, Ny_image))
    yc_image = image_cell_centroids[:, 1].reshape((Nx_image, Ny_image))

    cell_centroids = np.mean(np.take(problem.points, problem.cells, axis=0), axis=1)
    xc = cell_centroids[:, 0].reshape((args.Nx, args.Ny))
    yc = cell_centroids[:, 1].reshape((args.Nx, args.Ny))


    def resize_fwd(to_data):
        print("resize fwd...")
        to_data = np.array(to_data)
        interp_fwd = RegularGridInterpolator((xc[:, 0], yc[0, :]), to_data, method='linear', bounds_error=False, fill_value=None)
        return interp_fwd(image_cell_centroids)

    def resize_bwd(image_data):
        print("resize bwd...")
        image_data = np.array(image_data)
        interp_bwd = RegularGridInterpolator((xc_image[:, 0], yc_image[0, :]), image_data, method='linear', bounds_error=False, fill_value=None)
        return interp_bwd(cell_centroids)

    def style_vg_and_state(image_theta):
        t_params = {'norm': {'image': image_theta}}
        (value, state), grads = (
            jax.value_and_grad(loss, has_aux=True)(t_params,
                                                   nt_params,
                                                   None,
                                                   input_image))
        return (value, grads), state

    def style_value_and_grad(rho, step):
        config.update("jax_enable_x64", False)

        theta = rho.reshape((args.Nx, args.Ny))
        image_theta = resize_fwd(theta)
        image_theta = image_theta.reshape((args.image_size, args.image_size))[None, None, :, :]
        image_theta = jnp.repeat(image_theta, 3, axis=1)
        image_theta = jnp.array(image_theta, dtype=jnp.float32)
 
        (value, grads_image), state = style_vg_and_state(image_theta)

        grads_image = tree_util.tree_leaves(grads_image)[0]
        grads_image_grey = np.mean(grads_image, axis=(0, 1))
        grads = resize_bwd(grads_image_grey).reshape(rho.shape)

        c_loss, s_loss = calculate_losses(state, weights)
        g_loss = smooth_loss(image_theta)
        print(f"{bcolors.HEADER}Iteration: {step} Content loss: {c_loss:.4f} Style loss: {s_loss:.4f} Smooth loss: {g_loss:.4f}{bcolors.ENDC}")

        if step % args.save_image_every == 0:
            checkpoint(image_theta, os.path.join(args.output_path, 'jpg'), f"styled_it{step:03d}.jpg", reverse=True)

        config.update("jax_enable_x64", True)

        return value, grads

    initial_loss, _ = style_value_and_grad(rho_initial, 0)

    return style_value_and_grad, initial_loss
