import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_fem_am.physics import release as release_module


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
            [4, 5, 6, 7],
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
    removed_root = load_release_cell_set(
        _write_release_document(
            tmp_path,
            removed_cell_ids=[2],
            retained_root_cell_ids=[2],
            expected_removed_count=1,
        ),
        expected_mesh_sha256="a" * 64,
        num_cells=3,
    )
    with pytest.raises(ValueError, match="root"):
        validate_release_cell_set(
            removed_root,
            cells=cells,
            points=points,
            removable_cell_mask=removable,
            protected_cell_mask=protected,
            anchor_node_ids=[4, 5, 6, 7],
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
            anchor_node_ids=[0],
        )


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
