import json
import sys
import tempfile
import unittest
from pathlib import Path

import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.xrd_vtu import predict_vtu_gauges  # noqa: E402


class XrdVtuPipelineTest(unittest.TestCase):
    def _write_cube_vtu(
        self,
        path,
        mode_id=5.0,
        temperature_k=300.0,
        invert_first_cell=False,
    ):
        source = (
            ROOT
            / "jax_fem_am/verification/fixtures/unit_cube_6tet.inp"
        )
        mesh = meshio.read(source)
        cells = np.asarray(mesh.cells_dict["tetra"]).copy()
        if invert_first_cell:
            cells[0, [0, 1]] = cells[0, [1, 0]]
        strain = np.diag([1200.0e-6, -100.0e-6, 50.0e-6])
        cell_data = {
            "elastic_strain_quad_xx": [np.full(len(cells), strain[0, 0])],
            "elastic_strain_quad_yy": [np.full(len(cells), strain[1, 1])],
            "elastic_strain_quad_zz": [np.full(len(cells), strain[2, 2])],
            "elastic_strain_quad_xy": [np.zeros(len(cells))],
            "elastic_strain_quad_yz": [np.zeros(len(cells))],
            "elastic_strain_quad_xz": [np.zeros(len(cells))],
            "printed": [np.ones(len(cells))],
            "mechanics_valid": [np.ones(len(cells))],
            "mode_id": [np.full(len(cells), mode_id)],
        }
        meshio.write(
            path,
            meshio.Mesh(
                points=np.asarray(mesh.points),
                cells=[("tetra", cells)],
                point_data={
                    "T": np.full(len(mesh.points), temperature_k),
                },
                cell_data=cell_data,
            ),
        )

    def _protocol(self):
        return {
            "schema_version": "v06.xrd-gauges/1",
            "required_state": "attached_to_build_plate_before_EDM",
            "measurement_temperature_k": 300.0,
            "temperature_tolerance_k": 1.0,
            "maximum_registration_rms_fraction_of_min_gauge": 0.25,
            "mesh_to_specimen": {
                "scale_m_per_mesh_unit": 1.0,
                "rotation": np.eye(3).tolist(),
                "translation_m": [0.0, 0.0, 0.0],
                "registration_rms_m": 0.0,
            },
            "gauges": [
                {
                    "id": "cube_eps_xx",
                    "geometry_model": "rectangular_box",
                    "center_m": [0.0005, 0.0005, 0.0005],
                    "size_m": [0.001, 0.001, 0.001],
                    "rotation_gauge_to_specimen": np.eye(3).tolist(),
                    "direction_specimen": [1.0, 0.0, 0.0],
                }
            ],
        }

    def test_attached_state_vtu_produces_coverage_gated_microstrain(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_000002_cooling.vtu"
            self._write_cube_vtu(path)

            report = predict_vtu_gauges(path, self._protocol())

        self.assertEqual(report["claim_level"], "measurement_operator_prediction_only")
        self.assertEqual(report["quadrature_projection"], "equal_weight_P0")
        self.assertEqual(report["gauges"][0]["status"], "ok")
        self.assertAlmostEqual(
            report["gauges"][0]["predicted_microstrain"], 1200.0
        )

    def test_release_vtu_is_rejected_for_attached_xrd_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "release.vtu"
            self._write_cube_vtu(path, mode_id=7.0)

            with self.assertRaisesRegex(ValueError, "attached"):
                predict_vtu_gauges(path, self._protocol())

    def test_attached_protocol_requires_final_cooling_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_000000_scan.vtu"
            self._write_cube_vtu(path, mode_id=1.0)

            with self.assertRaisesRegex(ValueError, "cooling"):
                predict_vtu_gauges(path, self._protocol())

    def test_unknown_measurement_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_000002_cooling.vtu"
            self._write_cube_vtu(path)
            protocol = self._protocol()
            protocol["required_state"] = "unknown_state"

            with self.assertRaisesRegex(ValueError, "required_state"):
                predict_vtu_gauges(path, protocol)

    def test_measurement_temperature_and_registration_rms_are_gated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_000002_cooling.vtu"
            self._write_cube_vtu(path, temperature_k=350.0)

            with self.assertRaisesRegex(ValueError, "measurement temperature"):
                predict_vtu_gauges(path, self._protocol())

            self._write_cube_vtu(path, temperature_k=300.0)
            protocol = self._protocol()
            protocol["mesh_to_specimen"]["registration_rms_m"] = 3.0e-4
            with self.assertRaisesRegex(ValueError, "registration RMS"):
                predict_vtu_gauges(path, protocol)

    def test_inverted_tetrahedron_is_rejected_before_gauge_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_000002_cooling.vtu"
            self._write_cube_vtu(path, invert_first_cell=True)

            with self.assertRaisesRegex(ValueError, "inverted"):
                predict_vtu_gauges(path, self._protocol())


if __name__ == "__main__":
    unittest.main()
