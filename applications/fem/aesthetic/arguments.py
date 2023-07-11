import argparse
import os

parser = argparse.ArgumentParser()
parser.parse_args()

parser.add_argument("--pooling", type=str, default="max", help="Pooling method to use.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of training steps.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate of the Adam optimizer.")
parser.add_argument("--image_size", type=int, default=512, help="Target size of the images in pixels.")
parser.add_argument("--save_image_every", type=int, default=1, help="Saves the image every n steps to monitor progress.")

parser.add_argument("--content_weight", type=float, default=1., help="Content loss weight.")
parser.add_argument("--style_weight", type=float, default=1., help="Style loss weight.")

output_path = os.path.join(os.path.dirname(__file__), 'output') 
parser.add_argument("--output_path", type=str, default=output_path, help="Output directory to save the styled images to.")
# parser.add_argument("--content_layers", default=['conv_4'], 
#     help="Names of network layers for which to capture content loss.")
parser.add_argument("--content_layers", default={"conv_14": 0.1}, 
    help="Names of network layers for which to capture content loss.")

parser.add_argument("--style_layers", default={"conv_3": 1e1, "conv_5": 1e1}, 
    help="Names of network layers for which to capture style loss.")


# parser.add_argument("--style_layers", default={"conv_1": 1, 
#                                                "conv_2": 1, 
#                                                "conv_3": 1, 
#                                                "conv_4": 1, 
#                                                "conv_5": 1,
#                                                "conv_6": 1,
#                                                "conv_7": 1,
#                                                "conv_8": 1,
#                                                "conv_9": 1,
#                                                "conv_10": 1,
#                                                "conv_11": 1,
#                                                "conv_12": 1,
#                                                "conv_13": 1,
#                                                "conv_14": 1,
#                                                "conv_15": 1,
#                                                "conv_16": 1},
#     help="Names of network layers for which to capture style loss.")

parser.add_argument("--Lx", type=float, default=1., help="Length of domain.")
parser.add_argument("--Ly", type=float, default=1., help="Width of domain.")
parser.add_argument("--Nx", type=int, default=200, help="Number of elements along x-direction")
parser.add_argument("--Ny", type=int, default=200, help="Number of elements along y-direction.")

args = parser.parse_args()

# TODO
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
