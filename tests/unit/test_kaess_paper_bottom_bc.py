import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem_am.config.schema import build_parser
from jax_fem_am.io.vtu import write_used_config
from jax_fem_am.mesh.model import make_box_locations
from jax_fem_am.mesh.readers import read_solid_inp
from jax_fem_am.physics import release


REPO_ROOT = Path(__file__).resolve().parents[2]


def _cube_points(shift=(0.0, 0.0, 0.0)):
    axes = [np.asarray([0.0, 1.0]) + offset for offset in shift]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, 3)


def _bottom_location(build_axis_id, coordinate):
    def bottom(point):
        return np.isclose(point[build_axis_id], coordinate)

    return bottom


def _make_paper_bc(
    points,
    bottom,
    build_axis_id,
    plane_axis_ids,
    anchor_corner,
):
    factory = getattr(
        release,
        "make_paper_minimal_bottom_mechanics_bc",
        None,
    )
    if factory is None:
        return release.make_full_bottom_mechanics_bc(bottom)
    return factory(
        points,
        bottom,
        build_axis_id=build_axis_id,
        plane_axis_ids=plane_axis_ids,
        anchor_corner=anchor_corner,
    )


def _constrained_dofs(points, bc):
    locations, components, values = bc
    assert len(locations) == len(components) == len(values)
    constrained = []
    for location, component in zip(locations, components):
        selected = [
            node_id
            for node_id, point in enumerate(points)
            if bool(
                np.asarray(
                    location(point, node_id)
                    if location.__code__.co_argcount == 2
                    else location(point)
                )
            )
        ]
        assert selected
        constrained.extend(3 * node_id + int(component) for node_id in selected)
    assert len(constrained) == len(set(constrained))
    return constrained


def _rigid_body_modes(points):
    modes = np.zeros((3 * len(points), 6), dtype=np.float64)
    origin = np.mean(points, axis=0)
    for node_id, point in enumerate(points):
        row = slice(3 * node_id, 3 * node_id + 3)
        modes[row, :3] = np.eye(3)
        radius = point - origin
        for rotation_axis in range(3):
            unit_rotation = np.eye(3)[rotation_axis]
            modes[row, 3 + rotation_axis] = np.cross(
                unit_rotation, radius
            )
    return modes


def _constraint_matrix(num_dofs, constrained_dofs):
    matrix = np.zeros((len(constrained_dofs), num_dofs), dtype=np.float64)
    matrix[np.arange(len(constrained_dofs)), constrained_dofs] = 1.0
    return matrix


def _complete_axial_spring_tangent(points):
    num_dofs = 3 * len(points)
    tangent = np.zeros((num_dofs, num_dofs), dtype=np.float64)
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            direction = points[right] - points[left]
            direction /= np.linalg.norm(direction)
            compatibility = np.zeros(num_dofs, dtype=np.float64)
            compatibility[3 * left : 3 * left + 3] = -direction
            compatibility[3 * right : 3 * right + 3] = direction
            tangent += np.outer(compatibility, compatibility)
    return tangent


def test_paper_minimal_constrains_bottom_normal_and_only_three_in_plane_dofs():
    points = _cube_points()
    bottom = _bottom_location(build_axis_id=2, coordinate=0.0)

    bc = _make_paper_bc(
        points,
        bottom,
        build_axis_id=2,
        plane_axis_ids=(0, 1),
        anchor_corner="min_min",
    )
    constrained = set(_constrained_dofs(points, bc))

    bottom_ids = {
        node_id for node_id, point in enumerate(points) if point[2] == 0.0
    }
    expected = {3 * node_id + 2 for node_id in bottom_ids}
    anchor0 = int(
        np.flatnonzero(np.all(points == np.asarray([0.0, 0.0, 0.0]), axis=1))[0]
    )
    anchor1 = int(
        np.flatnonzero(np.all(points == np.asarray([1.0, 0.0, 0.0]), axis=1))[0]
    )
    expected.update((3 * anchor0, 3 * anchor0 + 1, 3 * anchor1 + 1))

    assert constrained == expected
    assert len(constrained) == len(bottom_ids) + 3


@pytest.mark.parametrize("build_axis_id", [0, 1, 2])
def test_paper_minimal_eliminates_all_six_rigid_body_modes(build_axis_id):
    points = _cube_points(shift=(0.3, -0.2, 0.4))
    plane_axis_ids = tuple(axis for axis in range(3) if axis != build_axis_id)
    base_coordinate = float(np.min(points[:, build_axis_id]))
    bottom = _bottom_location(build_axis_id, base_coordinate)

    bc = _make_paper_bc(
        points,
        bottom,
        build_axis_id=build_axis_id,
        plane_axis_ids=plane_axis_ids,
        anchor_corner="min_min",
    )
    constrained = _constrained_dofs(points, bc)
    constraint_matrix = _constraint_matrix(3 * len(points), constrained)

    assert np.linalg.matrix_rank(
        constraint_matrix @ _rigid_body_modes(points)
    ) == 6


def test_paper_minimal_makes_a_connected_tangent_full_rank():
    points = _cube_points()
    bc = _make_paper_bc(
        points,
        _bottom_location(build_axis_id=2, coordinate=0.0),
        build_axis_id=2,
        plane_axis_ids=(0, 1),
        anchor_corner="min_min",
    )
    constrained = set(_constrained_dofs(points, bc))
    free = sorted(set(range(3 * len(points))) - constrained)
    free_tangent = _complete_axial_spring_tangent(points)[np.ix_(free, free)]

    assert np.linalg.matrix_rank(free_tangent, tol=1.0e-10) == len(free)


@pytest.mark.parametrize(
    "anchor_corner",
    ["min_min", "max_min", "max_max", "min_max"],
)
def test_anchor_variants_allow_uniform_in_plane_thermal_contraction(
    anchor_corner,
):
    shift = np.asarray([0.3, -0.2, 0.4])
    points = _cube_points(shift=shift)
    bottom = _bottom_location(build_axis_id=2, coordinate=shift[2])
    paper_bc = _make_paper_bc(
        points,
        bottom,
        build_axis_id=2,
        plane_axis_ids=(0, 1),
        anchor_corner=anchor_corner,
    )

    x_side, y_side = anchor_corner.split("_")
    anchor = np.asarray(
        [
            np.min(points[:, 0]) if x_side == "min" else np.max(points[:, 0]),
            np.min(points[:, 1]) if y_side == "min" else np.max(points[:, 1]),
            shift[2],
        ]
    )
    contraction = -1.0e-3 * (points - anchor)
    paper_dofs = _constrained_dofs(points, paper_bc)
    full_dofs = _constrained_dofs(
        points, release.make_full_bottom_mechanics_bc(bottom)
    )

    assert np.allclose(contraction.reshape(-1)[paper_dofs], 0.0)
    assert np.max(np.abs(contraction.reshape(-1)[full_dofs])) > 0.0


def test_paper_minimal_rejects_a_collinear_bottom():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="non-collinear"):
        _make_paper_bc(
            points,
            _bottom_location(build_axis_id=2, coordinate=0.0),
            build_axis_id=2,
            plane_axis_ids=(0, 1),
            anchor_corner="min_min",
        )


def test_python_boolean_bottom_predicate_uses_host_fallback():
    points = _cube_points()

    def python_bottom(point):
        return bool(point[2] == 0.0)

    bc, metadata = release.make_paper_minimal_bottom_mechanics_bc(
        points,
        python_bottom,
        build_axis_id=2,
        plane_axis_ids=(0, 1),
        return_metadata=True,
    )

    assert metadata["bottom_node_count"] == 4
    resolved_bottom_mask = np.asarray(
        jax.vmap(bc[0][0])(
            jnp.asarray(points),
            jnp.arange(len(points)),
        )
    )
    np.testing.assert_array_equal(
        resolved_bottom_mask,
        points[:, 2] == 0.0,
    )
    assert len(_constrained_dofs(points, bc)) == 7


def test_resolved_anchor_metadata_is_written_to_used_config(tmp_path):
    points = _cube_points()
    _, metadata = release.make_paper_minimal_bottom_mechanics_bc(
        points,
        _bottom_location(build_axis_id=2, coordinate=0.0),
        build_axis_id=2,
        plane_axis_ids=(0, 1),
        return_metadata=True,
    )

    write_used_config(
        SimpleNamespace(paper_minimal_resolved_bc=metadata),
        tmp_path,
        derived={},
    )
    used_config = json.loads(
        (tmp_path / "used_config.json").read_text(encoding="utf-8")
    )

    assert used_config["paper_minimal_resolved_bc"] == metadata


def test_frozen_kaess_mesh_resolves_exact_auditable_anchor_nodes():
    mesh_path = (
        REPO_ROOT
        / "cases"
        / "kaess_2023"
        / "kaess_cantilever_c3d8_powder_margin.inp"
    )
    points, cells, _, ele_type = read_solid_inp(mesh_path, max_cells=0)
    locations = make_box_locations(
        points,
        build_axis="z",
        base_side="min",
    )
    bottom = locations[2]
    build_axis_id = locations[5]
    plane_axis_ids = locations[6]

    bc, metadata = release.make_paper_minimal_bottom_mechanics_bc(
        points,
        bottom,
        build_axis_id=build_axis_id,
        plane_axis_ids=plane_axis_ids,
        anchor_corner="min_min",
        return_metadata=True,
    )

    assert ele_type == "HEX8"
    assert len(cells) == 29_568
    assert metadata["bottom_node_count"] == 1_421
    assert len(_constrained_dofs(points, bc)) == 1_424
    assert metadata["anchor_node_ids"] == [0, 48]
    np.testing.assert_allclose(
        metadata["anchor_coordinates"],
        [
            [-1.0e-4, -1.0e-4, 0.0],
            [1.1e-3, -1.0e-4, 0.0],
        ],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert metadata["rotation_component"] == 1

    assumptions = json.loads(
        (
            REPO_ROOT
            / "cases"
            / "kaess_2023"
            / "inputs"
            / "assumptions.yaml"
        ).read_text(encoding="utf-8")
    )
    anchor_assumption = next(
        record
        for record in assumptions["assumptions"]
        if record["assumption_id"] == "A-BC-ANCHORS"
    )
    registered_primary = anchor_assumption["range"]["primary"]
    assert registered_primary["anchor_corner"] == "min_min"
    assert registered_primary["anchor_domain"].startswith(
        "retained W3 root bottom"
    )
    assert registered_primary["root_bottom_node_count"] == 189
    assert registered_primary["physical_release_dof_count"] == 192
    assert registered_primary["anchor_node_ids"] == [231, 239]
    np.testing.assert_allclose(
        registered_primary["anchor_coordinates_m"],
        [[7.75e-4, 0.0, 0.0], [9.75e-4, 0.0, 0.0]],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert anchor_assumption["range"]["g2_sensitivity_variants"] == [
        "min_min",
        "max_min",
        "max_max",
        "min_max",
    ]


def test_cli_keeps_fixed_default_and_exposes_paper_minimal_anchor_variants():
    parser = build_parser()

    default_args = parser.parse_args([])
    paper_args = parser.parse_args(
        [
            "--bottom-mechanics-bc",
            "paper_minimal",
            "--paper-minimal-anchor-corner",
            "max_max",
        ]
    )

    assert default_args.bottom_mechanics_bc == "fixed"
    assert default_args.paper_minimal_anchor_corner == "min_min"
    assert paper_args.bottom_mechanics_bc == "paper_minimal"
    assert paper_args.paper_minimal_anchor_corner == "max_max"


def test_cli_exposes_paper_minimal_root_release_mode():
    parser = build_parser()
    args = parser.parse_args(
        ["--release-anchor-mode", "paper_minimal_root"]
    )

    assert args.release_anchor_mode == "paper_minimal_root"


@pytest.mark.parametrize(
    "launcher_name",
    ["run_kaess_phase1.sh", "run_kaess_phase2.sh"],
)
def test_kaess_launchers_explicitly_select_the_frozen_paper_minimal_bc(
    launcher_name,
):
    launcher = (
        REPO_ROOT / "cases" / "kaess_2023" / launcher_name
    ).read_text(encoding="utf-8")

    assert "--bottom-mechanics-bc paper_minimal" in launcher
    assert "--paper-minimal-anchor-corner min_min" in launcher
    assert "--bottom-mechanics-bc fixed" not in launcher
