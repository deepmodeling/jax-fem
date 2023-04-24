import os
import meshio
import numpy as onp
import json
import yaml


def json_parse(json_filepath):
    with open(json_filepath) as f:
        args = json.load(f)
    json_formatted_str = json.dumps(args, indent=4)
    print(json_formatted_str)
    return args


def yaml_parse(yaml_filepath):     
    with open(yaml_filepath) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print(f"YAML parameters:")
        # TODO: These are just default parameters
        print(yaml.dump(args, default_flow_style=False))
        print(f"These are default parameters")
    return args


def box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z):
    dim = 3
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    z = onp.linspace(0, domain_z, Nz + 1)
    xv, yv, zv = onp.meshgrid(x, y, z, indexing='ij')
    points_xyz = onp.stack((xv, yv, zv), axis=dim) 
    points = points_xyz.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xyz = points_inds.reshape(Nx + 1, Ny + 1, Nz + 1)
    inds1 = points_inds_xyz[:-1, :-1, :-1]
    inds2 = points_inds_xyz[1:, :-1, :-1]
    inds3 = points_inds_xyz[1:, 1:, :-1]
    inds4 = points_inds_xyz[:-1, 1:, :-1]
    inds5 = points_inds_xyz[:-1, :-1, 1:]
    inds6 = points_inds_xyz[1:, :-1, 1:]
    inds7 = points_inds_xyz[1:, 1:, 1:]
    inds8 = points_inds_xyz[:-1, 1:, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), axis=dim).reshape(-1, 8)
    out_mesh = meshio.Mesh(points=points, cells={'hexahedron': cells})
    return out_mesh


def rectangle_mesh(Nx, Ny, domain_x, domain_y):
    dim = 2
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing='ij')
    points_xy = onp.stack((xv, yv), axis=dim) 
    points = points_xy.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xy = points_inds.reshape(Nx + 1, Ny + 1)
    inds1 = points_inds_xy[:-1, :-1]
    inds2 = points_inds_xy[1:, :-1]
    inds3 = points_inds_xy[1:, 1:]
    inds4 = points_inds_xy[:-1, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4), axis=dim).reshape(-1, 4)
    out_mesh = meshio.Mesh(points=points, cells={'quad': cells})
    return out_mesh


def make_video(data_dir):
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite

    # TODO
    os.system(f'ffmpeg -y -framerate 10 -i {data_dir}/png/tmp/u.%04d.png -pix_fmt yuv420p -vf \
               "crop=trunc(iw/2)*2:trunc(ih/2)*2" {data_dir}/mp4/test.mp4')
