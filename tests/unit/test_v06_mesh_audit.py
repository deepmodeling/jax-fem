import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.mesh_audit import summarize_tet_mesh  # noqa: E402


def regular_tetrahedron(offset):
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
        ]
    ) + np.asarray(offset)


class MeshAuditSummaryTest(unittest.TestCase):
    def test_summary_is_json_serializable_and_identifies_worst_cell(self):
        regular = regular_tetrahedron([0.0, 0.0, 0.0])
        sliver = regular_tetrahedron([2.0, 0.0, 0.0])
        sliver[3, 2] = 1.0e-7
        points = np.vstack([regular, sliver])
        cells = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])

        summary = summarize_tet_mesh(points, cells, thresholds=(0.05, 0.1))

        json.dumps(summary)
        self.assertEqual(summary["mesh"]["num_cells"], 2)
        self.assertEqual(summary["quality"]["below_threshold"]["0.05"], 1)
        self.assertEqual(summary["quality"]["below_threshold"]["0.1"], 1)
        self.assertEqual(summary["quality"]["worst_cells"][0]["cell_id"], 1)
        self.assertEqual(summary["validity"]["inverted_count"], 0)

    def test_summary_reports_inverted_and_degenerate_cells(self):
        regular = regular_tetrahedron([0.0, 0.0, 0.0])
        points = np.vstack([regular, [[0.25, 0.25, 0.0]]])
        cells = np.asarray([[0, 2, 1, 3], [0, 1, 2, 4]])

        summary = summarize_tet_mesh(points, cells)

        self.assertEqual(summary["validity"]["inverted_count"], 1)
        self.assertEqual(summary["validity"]["degenerate_count"], 1)
        self.assertFalse(summary["validity"]["valid_for_fem"])

    def test_v06_smoke_fixture_has_complete_fixed_base_and_valid_tets(self):
        path = (
            ROOT / "jax_fem_am" / "verification" / "fixtures"
            / "unit_cube_6tet.inp"
        )
        mesh = meshio.read(path)
        points = np.asarray(mesh.points)
        cells = np.asarray(mesh.cells_dict["tetra"])

        summary = summarize_tet_mesh(points, cells)
        self.assertTrue(summary["validity"]["valid_for_fem"])
        self.assertEqual(summary["mesh"]["num_cells"], 6)
        self.assertEqual(int(np.count_nonzero(np.isclose(points[:, 2], 0.0))), 4)

        face_counts = Counter()
        for cell in cells:
            for face in (
                (cell[0], cell[1], cell[2]),
                (cell[0], cell[1], cell[3]),
                (cell[0], cell[2], cell[3]),
                (cell[1], cell[2], cell[3]),
            ):
                face_counts[tuple(sorted(int(node) for node in face))] += 1
        base_faces = [
            face
            for face, count in face_counts.items()
            if count == 1 and np.allclose(points[list(face), 2], 0.0)
        ]
        self.assertEqual(len(base_faces), 2)


if __name__ == "__main__":
    unittest.main()
