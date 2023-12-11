import numpy as onp
import jax
import jax.numpy as np
import torch
import argparse
import os
import sys
import numpy as onp
import matplotlib.pyplot as plt
from jax.config import config

torch.manual_seed(0)

# Set numpy printing format
onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)

# Manage arguments
parser = argparse.ArgumentParser()
parser.add_argument('--L', type=float, default=1.)
parser.add_argument('--num_hex', type=int, default=10)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--E_in', type=float, default=1e3)
parser.add_argument('--E_out', type=float, default=1e2)
parser.add_argument('--nu_in', type=float, default=0.3)
parser.add_argument('--nu_out', type=float, default=0.4)
parser.add_argument('--ratio', type=float, default=0.3)


parser.add_argument('--activation', choices=['tanh', 'selu', 'relu', 'sigmoid', 'softplus'], default='tanh')
parser.add_argument('--width_hidden', type=int, default=64)
parser.add_argument('--n_hidden', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--input_size', type=int, default=6)

args = parser.parse_args()


# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})