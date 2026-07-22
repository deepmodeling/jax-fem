import unittest

import numpy as np

try:
    import pypardiso  # noqa: F401
except ImportError as exc:  # pragma: no cover
    PARDISO_IMPORT_ERROR = exc
else:
    PARDISO_IMPORT_ERROR = None

from jax_fem_am.simulation.acceleration import _PardisoCustomSolver


class _CsrMatrix:
    def __init__(self, data):
        self.indptr = np.array([0, 2, 4], dtype=np.int32)
        self.indices = np.array([0, 1, 0, 1], dtype=np.int32)
        self.data = np.asarray(data, dtype=np.float64)

    def getValuesCSR(self):
        return self.indptr, self.indices, self.data


@unittest.skipIf(
    PARDISO_IMPORT_ERROR is not None,
    f"pypardiso unavailable: {PARDISO_IMPORT_ERROR}",
)
class PardisoPhase23Test(unittest.TestCase):
    def test_reuses_symbolic_analysis_when_matrix_values_change(self):
        solver = _PardisoCustomSolver("phase23")
        rhs = np.array([1.0, 2.0])

        first_dense = np.array([[4.0, 1.0], [1.0, 3.0]])
        first = solver(
            _CsrMatrix([4.0, 1.0, 1.0, 3.0]), rhs, None, {}
        )
        np.testing.assert_allclose(first, np.linalg.solve(first_dense, rhs))

        second_dense = np.array([[5.0, 1.0], [1.0, 2.0]])
        second = solver(
            _CsrMatrix([5.0, 1.0, 1.0, 2.0]), rhs, None, {}
        )
        np.testing.assert_allclose(second, np.linalg.solve(second_dense, rhs))

        self.assertEqual(solver._v07_variant._stats["analyze_calls"], 1)
        self.assertEqual(solver._v07_variant._stats["solves"], 2)


if __name__ == "__main__":
    unittest.main()
