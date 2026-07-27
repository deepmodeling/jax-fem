import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_fem_am.verification.provenance import (  # noqa: E402
    audit_artifacts_match,
    build_manifest,
    response_gate_is_valid,
    sha256_file,
    source_tree_record,
    thermal_ledger_is_valid,
    xrd_gauge_ids_match,
    xrd_inputs_match,
    xrd_report_is_valid,
)


class V06ProvenanceTest(unittest.TestCase):
    def test_response_gate_hashes_must_match_current_evidence_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            names = (
                "used_config.json",
                "v06_run_audit.json",
                "xrd_operator_smoke.json",
                "thermal_energy_ledger.jsonl",
            )
            for name in names:
                (run_dir / name).write_text(name, encoding="utf-8")
            (run_dir / "used_config.json").write_text(
                json.dumps({"laser_power": 0.0}), encoding="utf-8"
            )
            roles = {
                "used_config": names[0],
                "run_audit": names[1],
                "xrd_prediction": names[2],
                "thermal_ledger": names[3],
            }
            report = {
                "schema_version": "v06.response-gate/1",
                "valid": True,
                "required": False,
                "status": "zero_input_invariant_smoke",
                "claim_level": "zero_input_invariant_smoke_only",
                "inputs": {
                    role: {
                        "path": str(run_dir / name),
                        "sha256": sha256_file(run_dir / name),
                    }
                    for role, name in roles.items()
                },
            }

            self.assertTrue(response_gate_is_valid(report, run_dir))
            (run_dir / "v06_run_audit.json").write_text(
                "tampered", encoding="utf-8"
            )
            self.assertFalse(response_gate_is_valid(report, run_dir))

            (run_dir / "v06_run_audit.json").write_text(
                names[1], encoding="utf-8"
            )
            (run_dir / "used_config.json").write_text(
                json.dumps({"laser_power": 1.0}), encoding="utf-8"
            )
            report["inputs"]["used_config"]["sha256"] = sha256_file(
                run_dir / "used_config.json"
            )
            self.assertFalse(response_gate_is_valid(report, run_dir))

    def test_thermal_ledger_must_match_audit_step_count_and_all_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "thermal_energy_ledger.jsonl"
            rows = [
                {
                    "schema_version": "v06.thermal-energy-ledger-step/1",
                    "step_index": index,
                    "balance_within_solver_tolerance": True,
                    "assembly_identity_within_tolerance": True,
                    "state_override_within_tolerance": True,
                    "temperature_invariants_valid": True,
                }
                for index in range(2)
            ]
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = {
                "schema_version": "v06.thermal-energy-ledger-summary/1",
                "complete": True,
                "recorded_step_count": 2,
                "expected_step_count": 2,
                "all_balance_steps_within_tolerance": True,
                "all_assembly_identities_within_tolerance": True,
                "all_pre_solve_state_overrides_within_tolerance": True,
                "all_temperature_invariants_valid": True,
            }
            audit = {"transient": {"step_count": 2}}

            self.assertTrue(thermal_ledger_is_valid(summary, ledger, audit))
            rows[1]["balance_within_solver_tolerance"] = False
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertFalse(thermal_ledger_is_valid(summary, ledger, audit))

    def test_audit_hashes_must_match_current_vtu_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            step = run_dir / "step_000000_scan.vtu"
            release = run_dir / "release.vtu"
            step.write_bytes(b"step-v1")
            release.write_bytes(b"release-v1")
            report = {
                "transient": {
                    "step_count": 1,
                    "steps": [
                        {
                            "name": step.name,
                            "sha256": sha256_file(step),
                        }
                    ],
                },
                "latest_constrained": {
                    "source": {
                        "path": str(step),
                        "sha256": sha256_file(step),
                    }
                },
                "release": {
                    "source": {
                        "path": str(release),
                        "sha256": sha256_file(release),
                    }
                },
            }

            self.assertTrue(audit_artifacts_match(report, run_dir))
            step.write_bytes(b"tampered-step")
            self.assertFalse(audit_artifacts_match(report, run_dir))

    def test_xrd_hashes_and_gauge_ids_must_match_current_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            vtu = run_dir / "step_000001_cooling.vtu"
            protocol = root / "protocol.json"
            vtu.write_bytes(b"vtu-v1")
            protocol.write_text(
                json.dumps({"gauges": [{"id": "g0"}, {"id": "g1"}]}),
                encoding="utf-8",
            )
            report = {
                "inputs": {
                    "vtu": {"path": str(vtu), "sha256": sha256_file(vtu)},
                    "protocol": {
                        "path": str(protocol),
                        "sha256": sha256_file(protocol),
                    },
                },
                "gauges": [{"id": "g0"}, {"id": "g1"}],
            }

            self.assertTrue(xrd_inputs_match(report, run_dir, protocol))
            self.assertTrue(xrd_gauge_ids_match(report, protocol))
            vtu.write_bytes(b"tampered-vtu")
            self.assertFalse(xrd_inputs_match(report, run_dir, protocol))
            report["gauges"] = [{"id": "g0"}]
            self.assertFalse(xrd_gauge_ids_match(report, protocol))

    def test_xrd_artifact_must_have_only_successful_gauges(self):
        self.assertTrue(
            xrd_report_is_valid(
                {
                    "claim_level": "measurement_operator_prediction_only",
                    "gauges": [{"id": "g0", "status": "ok"}],
                }
            )
        )
        self.assertFalse(
            xrd_report_is_valid(
                {
                    "claim_level": "measurement_operator_prediction_only",
                    "gauges": [{"id": "g0", "status": "low_material_coverage"}],
                }
            )
        )

    def test_sha256_file_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.bin"
            path.write_bytes(b"v06\x00paper-validation")

            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_runtime_source_tree_digest_changes_with_dependency_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "jax_fem"
            package.mkdir()
            source = package / "solver.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")

            before = source_tree_record([package], root)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            after = source_tree_record([package], root)

        self.assertEqual(before["python_file_count"], 1)
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_manifest_records_inputs_material_tables_and_claim_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_root = Path(temporary) / "work"
            repo_root = work_root / "jax-fem"
            run_dir = work_root / "output" / "case"
            material_dir = work_root / "materials"
            repo_root.mkdir(parents=True)
            run_dir.mkdir(parents=True)
            material_dir.mkdir(parents=True)

            mesh = repo_root / "mesh.inp"
            mesh.write_text("*HEADING\n", encoding="utf-8")
            table = material_dir / "E.csv"
            table.write_text("T,E\n300,1.0e11\n", encoding="utf-8")
            config = material_dir / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            release_cell_set = repo_root / "release-cellset.json"
            release_cell_set.write_text(
                '{"schema_version":"kaess.release-cellset/1"}\n',
                encoding="utf-8",
            )
            release_hash = sha256_file(release_cell_set)
            (run_dir / "used_config.json").write_text(
                json.dumps(
                    {
                        "E_table": "materials/E.csv",
                        "layers": 1,
                        "release_cell_set": str(release_cell_set),
                        "release_cell_set_sha256": release_hash,
                        "derived": {
                            "release_selection_mode": "exact_cell_set",
                            "paper_release_gate_eligible": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "v06_run_audit.json").write_text(
                "{}\n", encoding="utf-8"
            )
            expected_mesh_hash = sha256_file(mesh)

            manifest = build_manifest(
                repo_root=repo_root,
                run_dir=run_dir,
                mesh=mesh,
                material_config=config,
                label="unit-test",
            )

        self.assertEqual(manifest["schema_version"], "jax_fem_am.verification.provenance/2")
        self.assertEqual(manifest["claim_level"], "forensic_manifest_only")
        self.assertEqual(manifest["run_status"], "incomplete_or_invalid")
        self.assertEqual(manifest["label"], "unit-test")
        self.assertEqual(
            manifest["inputs"]["mesh"]["sha256"], expected_mesh_hash
        )
        self.assertEqual(
            manifest["material_tables"]["E_table"]["sha256"],
            hashlib.sha256(b"T,E\n300,1.0e11\n").hexdigest(),
        )
        self.assertIn("v06_run_audit", manifest["artifacts"])
        self.assertIn("used_config", manifest["inputs"])
        self.assertEqual(
            manifest["inputs"]["release_cell_set"]["sha256"],
            release_hash,
        )
        self.assertTrue(manifest["paper_release_gate"]["eligible"])

    def test_geometric_release_box_is_not_paper_gate_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary) / "jax-fem"
            run_dir = Path(temporary) / "run"
            repo_root.mkdir()
            run_dir.mkdir()
            mesh = repo_root / "mesh.inp"
            config = repo_root / "material.json"
            mesh.write_text("*HEADING\n", encoding="utf-8")
            config.write_text("{}\n", encoding="utf-8")
            (run_dir / "used_config.json").write_text(
                json.dumps(
                    {
                        "release_cut_box": [0, 1, 0, 1, 0, 1],
                        "derived": {
                            "release_selection_mode": (
                                "geometric_box_diagnostic"
                            ),
                            "paper_release_gate_eligible": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_manifest(
                repo_root=repo_root,
                run_dir=run_dir,
                mesh=mesh,
                material_config=config,
                label="diagnostic-release",
            )

        self.assertFalse(manifest["paper_release_gate"]["eligible"])
        self.assertEqual(
            manifest["paper_release_gate"]["selection_mode"],
            "geometric_box_diagnostic",
        )

    def test_material_tables_resolve_relative_to_material_config_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary) / "jax-fem"
            run_dir = Path(temporary) / "run"
            bundle = repo_root / "cases" / "candidate"
            run_dir.mkdir(parents=True)
            bundle.mkdir(parents=True)
            mesh = repo_root / "mesh.inp"
            mesh.write_text("*HEADING\n", encoding="utf-8")
            flow_curve = bundle / "flow.csv"
            flow_curve.write_text(
                "temperature_K,equivalent_plastic_strain,"
                "flow_stress_Pa,source\n"
                "300,0,5e8,test\n"
                "300,0.1,6e8,test\n"
                "800,0,3e8,test\n"
                "800,0.1,3.4e8,test\n",
                encoding="utf-8",
            )
            expected_flow_hash = sha256_file(flow_curve)
            config = bundle / "material.json"
            config.write_text(
                '{"flow_curve_table":"flow.csv"}\n',
                encoding="utf-8",
            )
            (run_dir / "used_config.json").write_text(
                '{"flow_curve_table":"flow.csv"}\n',
                encoding="utf-8",
            )

            manifest = build_manifest(
                repo_root=repo_root,
                run_dir=run_dir,
                mesh=mesh,
                material_config=config,
                label="config-relative-table",
            )

        record = manifest["material_tables"]["flow_curve_table"]
        self.assertNotIn("missing", record)
        self.assertEqual(record["sha256"], expected_flow_hash)

    def test_malformed_used_config_degrades_to_forensic_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary) / "jax-fem"
            run_dir = Path(temporary) / "run"
            repo_root.mkdir()
            run_dir.mkdir()
            mesh = repo_root / "mesh.inp"
            config = repo_root / "material.json"
            mesh.write_text("*HEADING\n", encoding="utf-8")
            config.write_text("{}\n", encoding="utf-8")
            (run_dir / "used_config.json").write_text(
                '{"truncated":', encoding="utf-8"
            )

            manifest = build_manifest(
                repo_root=repo_root,
                run_dir=run_dir,
                mesh=mesh,
                material_config=config,
                label="malformed-config",
            )

        self.assertFalse(manifest["completeness"]["used_config_valid"])
        self.assertEqual(manifest["claim_level"], "forensic_manifest_only")


if __name__ == "__main__":
    unittest.main()
