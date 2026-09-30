"""Voxel polycrystal geometry with a common anisotropic Voronoi metric."""
import numpy as np
from scipy.spatial import cKDTree


def _box(bounds, divisions):
    bounds = np.asarray(bounds, dtype=float)
    divisions = np.asarray(divisions)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(np.diff(bounds, axis=1) <= 0):
        raise ValueError('bounds must be three finite increasing intervals')
    if divisions.shape != (3,) or divisions.dtype.kind not in 'iu' or np.any(divisions < 1):
        raise ValueError('divisions must be three positive integers')
    return bounds, tuple(int(n) for n in divisions)


def generate_seeds(bounds, count, seed=0, randomness=0.35, max_attempts=None):
    """Reproducible seeds with minimum separation, including periodic images.

    The minimum distance is (1-randomness)*sqrt(6)/2*(V/(sqrt(2)*count))**(1/3).
    Only seed spacing uses periodic distances; the tessellation is not periodic.
    """
    bounds, _ = _box(bounds, (1, 1, 1))
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('count must be a positive integer')
    if not np.isfinite(randomness) or not 0 <= randomness <= 1:
        raise ValueError('randomness must lie in [0, 1]')
    budget = 2000*count if max_attempts is None else max_attempts
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError('max_attempts must be a positive integer')
    rng = np.random.default_rng(seed)
    lengths = bounds[:, 1] - bounds[:, 0]
    distance = (1-randomness)*np.sqrt(6)/2*(np.prod(lengths)/(np.sqrt(2)*count))**(1/3)
    seeds = np.empty((count, 3))
    filled = 0
    for _ in range(budget):
        candidate = rng.uniform(bounds[:, 0], bounds[:, 1])
        delta = np.abs(seeds[:filled] - candidate)
        delta = np.minimum(delta, lengths - delta)
        if np.any(np.sum(delta**2, axis=1) <= distance**2):
            continue
        seeds[filled] = candidate
        filled += 1
        if filled == count:
            return seeds
    raise RuntimeError(f'Placed {filled}/{count} seeds in {budget} attempts; increase randomness or budget')


def _tree(seeds, stretch, direction):
    seeds = np.asarray(seeds, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if (seeds.ndim != 2 or seeds.shape[1] != 3 or not len(seeds)
            or not np.isfinite(seeds).all() or len(np.unique(seeds, axis=0)) != len(seeds)):
        raise ValueError('seeds must be distinct finite (n, 3) coordinates')
    if not np.isfinite(stretch) or stretch <= 0:
        raise ValueError('stretch must be positive and finite')
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError('direction must be a finite nonzero vector')
    axis = direction/np.linalg.norm(direction)
    transform = np.eye(3) + (1/stretch - 1)*np.outer(axis, axis)
    origin = seeds.mean(axis=0)
    return cKDTree((seeds-origin) @ transform), origin, transform


def assign_grains(centers, seeds, stretch=1., direction=(0., 0., 1.)):
    """Assign element centers to seeds; IDs index the original seed array.

    Distance to a seed is ||T (x-s)||, T=I+(1/stretch-1) a a^T. Both query
    points and seeds use the same metric. Seed positions are held fixed when
    stretch/direction change. The input stretch is not a measured grain ratio.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError('centers must be finite (n, 3) coordinates')
    tree, origin, transform = _tree(seeds, stretch, direction)
    return tree.query((centers-origin) @ transform, workers=1)[1].astype(np.int32)


def voxel_grains(bounds, divisions, seeds, stretch=1., direction=(0., 0., 1.)):
    """Grain IDs on a regular HEX8 grid, stored in x/y/z (C) order.

    Query in batches to avoid a full (nx*ny*nz, 3) coordinate allocation.
    This creates geometry labels only; no FEM solve or material is involved.
    """
    bounds, divisions = _box(bounds, divisions)
    tree, origin, transform = _tree(seeds, stretch, direction)
    spacing = (bounds[:, 1] - bounds[:, 0])/divisions
    size = int(np.prod(divisions))
    labels = np.empty(size, dtype=np.int32)
    for start in range(0, size, 65536):
        indices = np.arange(start, min(start+65536, size))
        ijk = np.stack(np.unravel_index(indices, divisions), axis=1)
        centers = bounds[:, 0] + (ijk + .5)*spacing
        labels[start:start+len(indices)] = tree.query((centers-origin) @ transform, workers=1)[1]
    return labels.reshape(divisions)


def create_mesh(bounds, divisions):
    """Coordinates/connectivity in JAX-FEM/meshio HEX8 ordering.

    Construct a JAX-FEM mesh with Mesh(points, cells, ele_type='HEX8').
    Cell order matches voxel_grains(...).ravel(). Nodes are shared across grains.
    """
    bounds, divisions = _box(bounds, divisions)
    shape = tuple(n+1 for n in divisions)
    if np.prod(shape, dtype=np.int64) > np.iinfo(np.int32).max:
        raise ValueError('mesh exceeds the int32 connectivity limit')
    axes = [np.linspace(*limits, n+1) for limits, n in zip(bounds, divisions)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    nodes = np.arange(len(points), dtype=np.int32).reshape(shape)
    cells = np.stack([nodes[:-1, :-1, :-1], nodes[1:, :-1, :-1],
                      nodes[1:, 1:, :-1], nodes[:-1, 1:, :-1],
                      nodes[:-1, :-1, 1:], nodes[1:, :-1, 1:],
                      nodes[1:, 1:, 1:], nodes[:-1, 1:, 1:]], axis=-1).reshape(-1, 8)
    return points, cells
