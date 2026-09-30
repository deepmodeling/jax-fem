"""Render the actual voxel labels, with no axes, mesh edges or smoothing."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def grain_colors(count):
    # Match the blue/gray/red family in the crystal-plasticity example.
    palette = plt.get_cmap('coolwarm')(np.linspace(0., 1., count))[:, :3]
    rng = np.random.default_rng(17)
    return palette[rng.permutation(count)]


def _visible_faces(labels):
    """Visible surfaces of a cube with a corner removed for display only."""
    shape = np.array(labels.shape)
    keep = np.ones(labels.shape, dtype=bool)
    keep[shape[0]//2:, :shape[1]//2, shape[2]//2:] = False
    vertices, grains, normals = [], [], []
    # Camera is in the +x,-y,+z octant; other normals face away from it.
    for axis, sign in [(0, 1), (1, -1), (2, 1)]:
        neighbor = np.zeros_like(keep)
        target, source = [slice(None)]*3, [slice(None)]*3
        target[axis] = slice(None, -1) if sign > 0 else slice(1, None)
        source[axis] = slice(1, None) if sign > 0 else slice(None, -1)
        neighbor[tuple(target)] = keep[tuple(source)]
        ijk = np.argwhere(keep & ~neighbor)
        others = [k for k in range(3) if k != axis]
        corners = np.zeros((4, 3), dtype=int)
        corners[:, axis] = int(sign > 0)
        corners[:, others] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        vertices.append(ijk[:, None] + corners)
        grains.append(labels[tuple(ijk.T)])
        normal = np.zeros(3)
        normal[axis] = sign
        normals.append(np.tile(normal, (len(ijk), 1)))
    return np.concatenate(vertices), np.concatenate(grains), np.concatenate(normals)


def _boundaries(vertices, grains, normals, shape):
    node_ids = np.ravel_multi_index(vertices.reshape(-1, 3).T, np.array(shape)+1).reshape(-1, 4)
    edges = node_ids[:, [[0, 1], [1, 2], [2, 3], [3, 0]]].reshape(-1, 2)
    owners = np.repeat(np.arange(len(vertices)), 4)
    keys = np.sort(edges, axis=1)
    order = np.lexsort(keys.T[::-1])
    keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1))+1]
    counts = np.diff(np.r_[starts, len(keys)])
    selected = list(order[starts[counts == 1]])
    shared = starts[counts == 2]
    a, b = order[shared], order[shared+1]
    different = (grains[owners[a]] != grains[owners[b]]) | np.any(normals[owners[a]] != normals[owners[b]], axis=1)
    selected.extend(a[different])
    return np.stack(np.unravel_index(edges[selected].ravel(), np.array(shape)+1), axis=1).reshape(-1, 2, 3)


def render_comparison(fields, labels, destination, n_grains):
    """Three consistent orthographic views; labels are the only annotation."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    background = '#ffffff'
    colors = grain_colors(n_grains)
    fig = plt.figure(figsize=(12, 4.8), facecolor=background)
    light = np.array([-.3, -.45, .84])
    light /= np.linalg.norm(light)
    for panel, (field, label) in enumerate(zip(fields, labels)):
        ax = fig.add_axes([panel/3, .075, 1/3, .91], projection='3d', facecolor=background)
        ax.set_proj_type('ortho')
        vertices, ids, normals = _visible_faces(field)
        coordinates = vertices/np.array(field.shape) - .5
        shade = .75 + .25*np.maximum(normals @ light, 0.)
        rgb = colors[ids]*shade[:, None]
        collection = Poly3DCollection(coordinates, facecolors=rgb, edgecolors='none',
                                      antialiased=False, rasterized=True, zsort='average')
        ax.add_collection3d(collection)
        lines = _boundaries(vertices, ids, normals, field.shape)/np.array(field.shape) - .5
        ax.add_collection3d(Line3DCollection(lines, colors='#26365c', linewidths=.25,
                                            alpha=.58, rasterized=True))
        ax.set(xlim=(-.53, .53), ylim=(-.53, .53), zlim=(-.53, .53))
        ax.set_box_aspect((1, 1, 1), zoom=1.06)
        ax.view_init(elev=24, azim=-56)
        ax.set_axis_off()
        fig.text((panel+.5)/3, .058, label, ha='center', va='center', fontsize=15,
                 fontfamily='DejaVu Sans', color='#26365c')
    fig.savefig(destination.with_suffix('.png'), dpi=240, facecolor=background)
    fig.savefig(destination.with_suffix('.pdf'), dpi=240, facecolor=background)
    plt.close(fig)
