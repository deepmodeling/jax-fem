import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import meshio

from jax_fem_am.mesh.readers import read_solid_inp
from jax_fem_am.physics import release as release_module
from jax_fem_am.io import vtu as vtu_module
from jax_fem_am.simulation import stepper
from jax_fem_am.verification.mesh_quality import audit_hex_mesh


ROOT = Path(__file__).resolve().parents[2]
FORMAL_MESH = (
    ROOT / "cases" / "kaess_2023"
    / "kaess_cantilever_c3d8_powder_margin.inp"
)
FORMAL_RELEASE_SET = (
    ROOT / "cases" / "kaess_2023" / "inputs" / "release-cellset.json"
)


def _legacy_load_release_cell_set(
    path,
    *,
    expected_mesh_sha256,
    num_cells,
):
    """RED baseline: emulate the current unverified geometric-box contract."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    removed = np.asarray(document.get("removed_cell_ids", []), dtype=np.int64)
    mask = np.zeros(num_cells, dtype=bool)
    mask[removed[(removed >= 0) & (removed < num_cells)]] = True
    return SimpleNamespace(
        document=document,
        removed_cell_ids=removed,
        retained_root_cell_ids=np.asarray(
            document.get("retained_root_cell_ids", []),
            dtype=np.int64,
        ),
        cell_mask=mask,
        artifact_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


def _legacy_validate_release_cell_set(
    release_set,
    *,
    cells,
    points,
    removable_cell_mask,
    protected_cell_mask,
    anchor_node_ids,
):
    del cells, points, removable_cell_mask, protected_cell_mask, anchor_node_ids
    return release_set.cell_mask


def _legacy_validate_release_direction(
    displacement,
    *,
    measurement_node_ids,
    build_axis,
    expected_sign,
    minimum_magnitude,
):
    del expected_sign
    values = np.asarray(displacement)[measurement_node_ids, build_axis]
    magnitude = float(np.max(np.abs(values)))
    if magnitude < minimum_magnitude:
        raise ValueError("release displacement is too small")
    return magnitude


load_release_cell_set = getattr(
    release_module,
    "load_release_cell_set",
    _legacy_load_release_cell_set,
)
validate_release_cell_set = getattr(
    release_module,
    "validate_release_cell_set",
    _legacy_validate_release_cell_set,
)
validate_release_direction = getattr(
    release_module,
    "validate_release_direction",
    _legacy_validate_release_direction,
)


def _dirichlet_dof_pairs(points, bc):
    locations, components, values = bc
    assert len(locations) == len(components) == len(values)
    pairs = []
    for location, component in zip(locations, components):
        for node_id, point in enumerate(points):
            try:
                selected = location(point, node_id)
            except TypeError:
                selected = location(point)
            if bool(np.asarray(selected)):
                pairs.append((node_id, int(component)))
    return np.asarray(pairs, dtype=np.int64)


def _release_document(**overrides):
    document = {
        "schema_version": "kaess.release-cellset/1",
        "protocol_id": "kaess-2023-public-v1",
        "mesh_sha256": "a" * 64,
        "mesh_num_cells": 3,
        "cell_id_basis": "solver_zero_based",
        "source_class": "inferred",
        "source_locator": "Kaess 2023 Figure 7 registered to the frozen mesh",
        "removed_cell_ids": [0],
        "retained_root_cell_ids": [2],
        "expected_removed_count": 1,
    }
    document.update(overrides)
    return document


def _write_release_document(tmp_path, **overrides):
    path = tmp_path / "release-cellset.json"
    path.write_text(
        json.dumps(_release_document(**overrides), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _synthetic_partition():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ],
        dtype=np.int64,
    )
    removable = np.asarray([True, False, True])
    protected = np.asarray([False, True, False])
    return points, cells, removable, protected


def test_release_cell_set_binds_exact_mesh_identity(tmp_path):
    path = _write_release_document(tmp_path)
    loaded = load_release_cell_set(
        path,
        expected_mesh_sha256="a" * 64,
        num_cells=3,
    )
    np.testing.assert_array_equal(loaded.removed_cell_ids, [0])

    with pytest.raises(ValueError, match="mesh.*SHA|identity"):
        load_release_cell_set(
            path,
            expected_mesh_sha256="b" * 64,
            num_cells=3,
        )


def test_stepper_rejects_exact_release_outside_strict_active_domain(
    monkeypatch,
):
    monkeypatch.setattr(
        stepper,
        "parse_args",
        lambda: SimpleNamespace(
            layer_activation_mode="front",
            future_layer_mode="powder",
            release_cell_set="release-cellset.json",
            release_cut_box=None,
            release_after_cooling=True,
        ),
    )
    with pytest.raises(ValueError, match="strict|void"):
        stepper.main()


def test_exact_release_mask_forces_zero_mechanics_contribution():
    factors = np.asarray(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
        ]
    )
    removed_quad = np.asarray(
        [
            [[1.0], [1.0]],
            [[0.0], [0.0]],
            [[1.0], [1.0]],
        ]
    )
    result = release_module.zero_exact_release_cells(
        factors,
        removed_quad,
    )
    np.testing.assert_array_equal(
        result,
        [
            [[0.0], [0.0]],
            [[3.0], [4.0]],
            [[0.0], [0.0]],
        ],
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"removed_cell_ids": []}, "non-empty"),
        ({"removed_cell_ids": [0, 0], "expected_removed_count": 2}, "unique"),
        ({"removed_cell_ids": [3]}, "range"),
        ({"mesh_num_cells": 4}, "cell count"),
        ({"cell_id_basis": "abaqus_one_based"}, "solver_zero_based"),
    ],
)
def test_release_cell_set_rejects_invalid_identity_and_range(
    tmp_path,
    overrides,
    message,
):
    path = _write_release_document(tmp_path, **overrides)
    with pytest.raises(ValueError, match=message):
        load_release_cell_set(
            path,
            expected_mesh_sha256="a" * 64,
            num_cells=3,
        )


def test_release_set_cannot_cut_protected_part_cells(tmp_path):
    path = _write_release_document(
        tmp_path,
        removed_cell_ids=[1],
        expected_removed_count=1,
    )
    loaded = load_release_cell_set(
        path,
        expected_mesh_sha256="a" * 64,
        num_cells=3,
    )
    points, cells, removable, protected = _synthetic_partition()

    with pytest.raises(ValueError, match="protected|part"):
        validate_release_cell_set(
            loaded,
            cells=cells,
            points=points,
            removable_cell_mask=removable,
            protected_cell_mask=protected,
            anchor_node_ids=[4, 5, 6, 7],
        )


def test_release_set_retains_root_and_full_rank_anchor(tmp_path):
    points, cells, removable, protected = _synthetic_partition()
    with pytest.raises(ValueError, match="root"):
        load_release_cell_set(
            _write_release_document(
                tmp_path,
                removed_cell_ids=[2],
                retained_root_cell_ids=[2],
                expected_removed_count=1,
            ),
            expected_mesh_sha256="a" * 64,
            num_cells=3,
        )

    valid = load_release_cell_set(
        _write_release_document(tmp_path),
        expected_mesh_sha256="a" * 64,
        num_cells=3,
    )
    with pytest.raises(ValueError, match="anchor|rigid"):
        validate_release_cell_set(
            valid,
            cells=cells,
            points=points,
            removable_cell_mask=removable,
            protected_cell_mask=protected,
            anchor_node_ids=[2],
        )

    with pytest.raises(ValueError, match="rigid"):
        validate_release_cell_set(
            valid,
            cells=cells,
            points=points,
            removable_cell_mask=removable,
            protected_cell_mask=protected,
            anchor_node_ids=[2, 3, 4, 5],
            anchor_dof_pairs=[[2, 0], [3, 0], [4, 0], [5, 0]],
        )

    with pytest.raises(ValueError, match="root"):
        validate_release_cell_set(
            valid,
            cells=cells,
            points=points,
            removable_cell_mask=removable,
            protected_cell_mask=protected,
            anchor_node_ids=[1, 2, 3, 4],
        )

    corner_connected_cells = cells.copy()
    corner_connected_cells[2] = [4, 5, 6, 7]
    with pytest.raises(ValueError, match="connected|floating"):
        validate_release_cell_set(
            valid,
            cells=corner_connected_cells,
            points=points,
            removable_cell_mask=removable,
            protected_cell_mask=protected,
            anchor_node_ids=[4, 5, 6, 7],
        )

    disconnected_cells = cells.copy()
    disconnected_cells[2] = [0, 5, 6, 7]
    with pytest.raises(ValueError, match="connected|floating"):
        validate_release_cell_set(
            valid,
            cells=disconnected_cells,
            points=points,
            removable_cell_mask=removable,
            protected_cell_mask=protected,
            anchor_node_ids=[0, 5, 6, 7],
        )

    removed = validate_release_cell_set(
        valid,
        cells=cells,
        points=points,
        removable_cell_mask=removable,
        protected_cell_mask=protected,
        anchor_node_ids=[2, 3, 4, 5],
    )
    np.testing.assert_array_equal(removed, [True, False, False])


def test_analytic_cantilever_release_direction_is_upward():
    x = np.linspace(0.0, 1.0, 5)
    displacement = np.zeros((len(x), 3))
    displacement[:, 2] = x**2 * (3.0 - x)
    magnitude = validate_release_direction(
        displacement,
        measurement_node_ids=np.arange(len(x)),
        build_axis=2,
        expected_sign=1,
        minimum_magnitude=1.0e-12,
    )
    assert magnitude > 0.0

    displacement[:, 2] *= -1.0
    with pytest.raises(ValueError, match="direction|sign"):
        validate_release_direction(
            displacement,
            measurement_node_ids=np.arange(len(x)),
            build_axis=2,
            expected_sign=1,
            minimum_magnitude=1.0e-12,
        )


def test_frozen_kaess_release_artifact_matches_mesh_and_retained_root():
    mesh_sha256 = hashlib.sha256(FORMAL_MESH.read_bytes()).hexdigest()
    mesh = meshio.read(FORMAL_MESH)
    cells = np.asarray(mesh.cells_dict["hexahedron"], dtype=np.int64)
    points = np.asarray(mesh.points, dtype=np.float64)
    solver_points, solver_cells, selected_cells, ele_type = read_solid_inp(
        FORMAL_MESH,
        max_cells=0,
    )
    assert selected_cells == len(cells)
    assert ele_type == "HEX8"
    np.testing.assert_array_equal(solver_points, points)
    np.testing.assert_array_equal(solver_cells, cells)
    quality = audit_hex_mesh(solver_points, solver_cells)
    assert quality.inverted_count == 0
    assert quality.degenerate_count == 0
    assert quality.below_threshold_count == 0
    loaded = load_release_cell_set(
        FORMAL_RELEASE_SET,
        expected_mesh_sha256=mesh_sha256,
        num_cells=len(cells),
    )

    assert len(loaded.removed_cell_ids) == 1920
    assert len(loaded.retained_root_cell_ids) == 1920
    assert not np.intersect1d(
        loaded.removed_cell_ids,
        loaded.retained_root_cell_ids,
    ).size

    powder = np.zeros(len(cells), dtype=bool)
    powder[
        np.asarray(
            mesh.cell_sets_dict["POWDER"]["hexahedron"],
            dtype=np.int64,
        )
    ] = True
    centroids = np.mean(points[cells], axis=1)
    support = (~powder) & (centroids[:, 2] <= 2.999e-4)
    protected = (~powder) & (~support)
    retained_root = np.zeros(len(cells), dtype=bool)
    retained_root[loaded.retained_root_cell_ids] = True
    np.testing.assert_array_equal(
        loaded.cell_mask | retained_root,
        support,
    )
    anchor_nodes = np.flatnonzero(
        np.all(
            (points >= np.asarray([7.75e-4, 0.0, -1.0e-9]))
            & (points <= np.asarray([9.75e-4, 5.0e-4, 1.0e-9])),
            axis=1,
        )
    )

    removed = validate_release_cell_set(
        loaded,
        cells=cells,
        points=points,
        removable_cell_mask=support,
        protected_cell_mask=protected,
        anchor_node_ids=anchor_nodes,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(removed),
        loaded.removed_cell_ids,
    )


@pytest.mark.parametrize(
    ("anchor_corner", "expected_anchor_node_ids"),
    (
        ("min_min", [231, 239]),
        ("max_min", [239, 231]),
        ("max_max", [1219, 1211]),
        ("min_max", [1211, 1219]),
    ),
)
def test_frozen_release_uses_only_root_normal_plus_three_rigid_dofs(
    anchor_corner,
    expected_anchor_node_ids,
):
    mesh_sha256 = hashlib.sha256(FORMAL_MESH.read_bytes()).hexdigest()
    mesh = meshio.read(FORMAL_MESH)
    cells = np.asarray(mesh.cells_dict["hexahedron"], dtype=np.int64)
    points = np.asarray(mesh.points, dtype=np.float64)
    release_set = load_release_cell_set(
        FORMAL_RELEASE_SET,
        expected_mesh_sha256=mesh_sha256,
        num_cells=len(cells),
    )

    bc, metadata = release_module.make_root_minimal_release_mechanics_bc(
        points,
        cells,
        release_set.retained_root_cell_ids,
        build_axis_id=2,
        plane_axis_ids=(0, 1),
        base_coord=0.0,
        base_tolerance=1.0e-12,
        anchor_corner=anchor_corner,
        return_metadata=True,
    )
    pairs = _dirichlet_dof_pairs(points, bc)

    assert metadata["mode"] == "paper_minimal_root"
    assert metadata["retained_root_cell_count"] == 1_920
    assert metadata["bottom_node_count"] == 189
    assert metadata["constrained_dof_count"] == 192
    assert metadata["anchor_node_ids"] == expected_anchor_node_ids
    anchor_protocol = release_set.document["anchor_protocol"]
    assert anchor_protocol["mode"] == "paper_minimal_root"
    assert anchor_protocol["root_bottom_node_ids"] == (
        metadata["root_bottom_node_ids"]
    )
    assert anchor_protocol["expected_root_bottom_node_count"] == 189
    assert anchor_protocol["expected_physical_release_dof_count"] == 192
    variant = anchor_protocol["variants"][anchor_corner]
    assert variant["anchor_node_ids"] == expected_anchor_node_ids
    assert variant["rigid_body_rank"] == 6
    assert variant["in_plane_dof_pairs"] == [
        pair for pair in metadata["constrained_dof_pairs"]
        if pair[1] != 2
    ]
    assert len(pairs) == len(np.unique(pairs, axis=0)) == 192
    assert np.count_nonzero(pairs[:, 1] == 2) == 189
    assert np.count_nonzero(pairs[:, 1] != 2) == 3

    powder = np.zeros(len(cells), dtype=bool)
    powder[
        np.asarray(
            mesh.cell_sets_dict["POWDER"]["hexahedron"],
            dtype=np.int64,
        )
    ] = True
    centroids = np.mean(points[cells], axis=1)
    support = (~powder) & (centroids[:, 2] <= 2.999e-4)
    protected = (~powder) & (~support)
    validate_release_cell_set(
        release_set,
        cells=cells,
        points=points,
        removable_cell_mask=support,
        protected_cell_mask=protected,
        anchor_node_ids=np.unique(pairs[:, 0]),
        anchor_dof_pairs=pairs,
    )
    in_plane_pair_ids = np.flatnonzero(pairs[:, 1] != 2)
    assert len(in_plane_pair_ids) == 3
    for pair_id in in_plane_pair_ids:
        with pytest.raises(ValueError, match="rigid"):
            validate_release_cell_set(
                release_set,
                cells=cells,
                points=points,
                removable_cell_mask=support,
                protected_cell_mask=protected,
                anchor_node_ids=np.unique(pairs[:, 0]),
                anchor_dof_pairs=np.delete(pairs, pair_id, axis=0),
            )

    full_bottom_node_ids = np.flatnonzero(
        np.isclose(points[:, 2], 0.0, rtol=0.0, atol=1.0e-12)
    )
    build_bc, build_metadata = (
        release_module.make_paper_minimal_bottom_mechanics_bc(
            points,
            None,
            build_axis_id=2,
            plane_axis_ids=(0, 1),
            anchor_corner=anchor_corner,
            bottom_node_ids=full_bottom_node_ids,
            anchor_candidate_node_ids=metadata["root_bottom_node_ids"],
            return_metadata=True,
        )
    )
    build_pairs = _dirichlet_dof_pairs(points, build_bc)
    assert build_metadata["bottom_node_count"] == 1_421
    assert build_metadata["anchor_node_ids"] == expected_anchor_node_ids
    assert len(build_pairs) == 1_424
    assert {tuple(pair) for pair in pairs.tolist()} <= {
        tuple(pair) for pair in build_pairs.tolist()
    }
    assert {
        tuple(pair) for pair in pairs[pairs[:, 1] != 2].tolist()
    } == {
        tuple(pair) for pair in build_pairs[build_pairs[:, 1] != 2].tolist()
    }


@pytest.mark.parametrize("anchor_mode", ("rigid_body", "box"))
def test_exact_release_rejects_nonminimal_anchor_modes(anchor_mode):
    args = SimpleNamespace(
        release_cell_set="release-cellset.json",
        release_cut_box=None,
        release_after_cooling=True,
        release_anchor_mode=anchor_mode,
        release_anchor_box=(
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
            if anchor_mode == "box"
            else None
        ),
    )

    with pytest.raises(ValueError, match="paper_minimal_root"):
        stepper.validate_release_configuration(
            args,
            strict_active_domain=True,
        )


def test_exact_release_requires_continuous_build_anchor_mode():
    args = SimpleNamespace(
        release_cell_set="release-cellset.json",
        release_cut_box=None,
        release_after_cooling=True,
        release_anchor_mode="paper_minimal_root",
        release_anchor_box=None,
        bottom_mechanics_bc="fixed",
    )
    with pytest.raises(ValueError, match="bottom-mechanics-bc|continuous"):
        stepper.validate_release_configuration(
            args,
            strict_active_domain=True,
        )

    args.bottom_mechanics_bc = "paper_minimal"
    stepper.validate_release_configuration(
        args,
        strict_active_domain=True,
    )


def test_release_anchor_protocol_is_fail_closed_on_tampering():
    mesh_sha256 = hashlib.sha256(FORMAL_MESH.read_bytes()).hexdigest()
    mesh = meshio.read(FORMAL_MESH)
    cells = np.asarray(mesh.cells_dict["hexahedron"], dtype=np.int64)
    points = np.asarray(mesh.points, dtype=np.float64)
    release_set = load_release_cell_set(
        FORMAL_RELEASE_SET,
        expected_mesh_sha256=mesh_sha256,
        num_cells=len(cells),
    )
    _, metadata = release_module.make_root_minimal_release_mechanics_bc(
        points,
        cells,
        release_set.retained_root_cell_ids,
        build_axis_id=2,
        plane_axis_ids=(0, 1),
        base_coord=0.0,
        base_tolerance=1.0e-12,
        anchor_corner="min_min",
        return_metadata=True,
    )
    release_module.validate_release_anchor_protocol(
        release_set,
        metadata,
        anchor_corner="min_min",
    )

    tampered_document = json.loads(json.dumps(release_set.document))
    tampered_document["anchor_protocol"]["variants"]["min_min"][
        "anchor_node_ids"
    ][0] += 1
    tampered = SimpleNamespace(document=tampered_document)
    with pytest.raises(ValueError, match="registered|differ"):
        release_module.validate_release_anchor_protocol(
            tampered,
            metadata,
            anchor_corner="min_min",
        )


def test_release_vtu_contains_visualizable_removed_cell_mask(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def capture_save(_fe, _solution, _path, *, point_infos, cell_infos):
        captured["point_infos"] = dict(point_infos)
        captured["cell_infos"] = dict(cell_infos)

    monkeypatch.setattr(vtu_module, "save_sol", capture_save)
    num_cells = 3
    num_quads = 1
    vtu_module.save_step(
        SimpleNamespace(num_cells=num_cells),
        np.zeros((4, 1)),
        np.zeros((4, 3)),
        tmp_path / "release.vtu",
        np.zeros((num_cells, num_quads, 1)),
        None,
        np.ones(num_cells),
        np.ones(num_cells),
        np.zeros(num_cells),
        np.zeros(num_cells),
        np.zeros(num_cells),
        np.zeros(num_cells),
        np.zeros(num_cells),
        np.zeros(num_cells),
        np.zeros(num_cells),
        np.zeros(num_cells),
        np.zeros((num_cells, num_quads, 1)),
        1.0,
        0,
        4,
        np.asarray([True, False, True]),
        {
            "release_bottom_uz": np.asarray([1.0, 1.0, 0.0, 0.0]),
            "release_anchor_ux": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "release_anchor_uy": np.asarray([1.0, 0.0, 1.0, 0.0]),
        },
    )

    np.testing.assert_array_equal(
        captured["cell_infos"]["release_removed"],
        [1.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        captured["point_infos"]["release_bottom_uz"],
        [1.0, 1.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        captured["point_infos"]["release_anchor_ux"],
        [1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        captured["point_infos"]["release_anchor_uy"],
        [1.0, 0.0, 1.0, 0.0],
    )
