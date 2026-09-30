import unittest

import jax
import numpy as np
import numpy.testing as npt

from jax_fem.generate_mesh import Mesh
from jax_fem.fe import FiniteElement
from applications.polycrystal.model import generate_seeds, assign_grains, voxel_grains, create_mesh

jax.config.update('jax_enable_x64', True)


class TestPolycrystal(unittest.TestCase):
    def test_seed_spacing_and_limit(self):
        seeds = generate_seeds([[0., 1.]]*3, 32, seed=19)
        npt.assert_array_equal(seeds, generate_seeds([[0., 1.]]*3, 32, seed=19))
        delta = np.abs(seeds[:, None] - seeds[None])
        distances = np.linalg.norm(np.minimum(delta, 1-delta), axis=2)
        np.fill_diagonal(distances, np.inf)
        minimum = .65*np.sqrt(6)/2*(1/(np.sqrt(2)*32))**(1/3)
        self.assertGreater(distances.min(), minimum)
        with self.assertRaises(RuntimeError):
            generate_seeds([[0., 1.]]*3, 2, max_attempts=1)

    def test_metric_against_direct_distances(self):
        rng = np.random.default_rng(14)
        seeds, centers = rng.random((12, 3)), rng.random((100, 3))
        axis = np.array([1., 2., 3.])/np.sqrt(14)
        delta = centers[:, None] - seeds[None]
        projection = delta @ axis
        squared_distance = np.sum(delta**2, axis=2) + (1/4**2-1)*projection**2
        expected = squared_distance.argmin(axis=1)
        npt.assert_array_equal(assign_grains(centers, seeds, 4., axis), expected)
        npt.assert_array_equal(assign_grains(centers, seeds, 4., -axis), expected)
        npt.assert_array_equal(assign_grains(centers+300., seeds+300., 4., axis), expected)
        isotropic = assign_grains(centers, seeds)
        npt.assert_array_equal(assign_grains(centers, seeds, 1., axis), isotropic)

    def test_voxels_and_actual_jax_fem_geometry(self):
        bounds, divisions = [[0., 2.], [0., 3.], [0., 4.]], (2, 3, 4)
        points, cells = create_mesh(bounds, divisions)
        seeds = np.array([[.25, 1.5, 2.], [1.75, 1.5, 2.]])
        labels = voxel_grains(bounds, divisions, seeds)
        npt.assert_array_equal(labels[0], 0)
        npt.assert_array_equal(labels[1], 1)
        npt.assert_array_equal(labels.ravel(), assign_grains(points[cells].mean(axis=1), seeds))
        mesh = Mesh(points, cells, ele_type='HEX8')
        fe = FiniteElement(mesh, vec=1, dim=3, ele_type='HEX8')
        gradients, weights = fe.get_shape_grads()
        self.assertTrue(np.all(np.asarray(weights) > 0))
        self.assertTrue(np.isfinite(gradients).all())
        npt.assert_allclose(np.sum(weights, axis=1), 1., atol=1e-14)
        # Adjacent grains use the same four nodes on their shared face.
        self.assertEqual(len(np.intersect1d(cells[0], cells[12])), 4)

    def test_refined_planar_interface(self):
        seeds = np.array([[.25, .5, .5], [.75, .5, .5]])
        for n in [4, 8, 16]:
            with self.subTest(divisions=n):
                labels = voxel_grains([[0., 1.]]*3, (n,)*3, seeds)
                npt.assert_allclose(np.bincount(labels.ravel())/labels.size, [.5, .5])

    def test_invalid_inputs(self):
        for divisions in [(0, 2, 2), (-1, 2, 2), (1.5, 2, 2), (True, True, True)]:
            with self.subTest(divisions=divisions), self.assertRaises(ValueError):
                create_mesh([[0., 1.]]*3, divisions)
        for stretch, axis in [(0., [0, 0, 1]), (np.nan, [0, 0, 1]), (1., [0, 0, 0])]:
            with self.subTest(stretch=stretch, axis=axis), self.assertRaises(ValueError):
                assign_grains([[0., 0., 0.]], [[0., 0., 0.]], stretch, axis)
        with self.assertRaises(ValueError):
            assign_grains([[0., 0., 0.]], [[0., 0., 0.]]*2)


if __name__ == '__main__':
    unittest.main()
