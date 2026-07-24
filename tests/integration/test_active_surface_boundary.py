import csv
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.generate_mesh import Mesh
from jax_fem_am.physics.thermal import TransientThermal
from jax_fem_am.simulation import stepper
from jax_fem_am.verification.thermal_ledger import integrate_surface_exchange


jax.config.update("jax_enable_x64", True)


TWO_STACKED_HEX_INP = """*HEADING
two unit HEX8 cells stacked in the build direction
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
9, 0.0, 0.0, 2.0
10, 1.0, 0.0, 2.0
11, 1.0, 1.0, 2.0
12, 0.0, 1.0, 2.0
*ELEMENT, TYPE=C3D8, ELSET=PART
1, 1, 2, 3, 4, 5, 6, 7, 8
2, 5, 6, 7, 8, 9, 10, 11, 12
"""


def _stacked_hex_mesh():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    cells = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8, 9, 10, 11],
        ],
        dtype=np.int32,
    )
    return points, cells


def _single_tet_mesh():
    points = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    return points, cells


def _make_static_exterior_problem():
    points, cells = _stacked_hex_mesh()

    def exterior_above_base(point):
        return point[2] > 1.0e-10

    # The static face superset includes active/future interfaces. The thermal
    # problem applies an owner/neighbor activity mask at each step.
    exterior_above_base.exterior_only = True
    exterior_above_base.active_domain_exterior = True
    exterior_above_base.active_domain_top_only = True
    exterior_above_base.active_domain_build_axis = 2
    exterior_above_base.active_domain_build_sign = 1.0
    return TransientThermal(
        mesh=Mesh(points, cells, ele_type="HEX8"),
        vec=1,
        dim=3,
        ele_type="HEX8",
        quadrature_order=2,
        dirichlet_bc_info=None,
        location_fns=[exterior_above_base],
        additional_info=(
            2.0,  # convection coefficient
            300.0,  # process ambient
            0.0,  # isolate convection from radiation
            5.670374419e-8,
            2,  # build axis
            0,
            1,
            1.0,
            1,
            0.0,
            1.0,
            False,
            "paper_hemispherical",
        ),
    )


def _make_tet_top_problem(build_sign):
    points, cells = _single_tet_mesh()

    def top_candidates(point):
        return jnp.isfinite(point[2])

    top_candidates.exterior_only = True
    top_candidates.active_domain_exterior = True
    top_candidates.active_domain_top_only = True
    top_candidates.active_domain_build_axis = 2
    top_candidates.active_domain_build_sign = build_sign
    return TransientThermal(
        mesh=Mesh(points, cells, ele_type="TET4"),
        vec=1,
        dim=3,
        ele_type="TET4",
        quadrature_order=2,
        dirichlet_bc_info=None,
        location_fns=[top_candidates],
        additional_info=(
            2.0,
            300.0,
            0.0,
            5.670374419e-8,
            2,
            0,
            1,
            build_sign,
            1,
            0.0,
            1.0,
            False,
            "paper_hemispherical",
        ),
    )


def _set_uniform_surface_state(problem, physical_cells):
    physical_cells = jnp.asarray(physical_cells, dtype=jnp.float64)
    shape = (problem.fes[0].num_cells, problem.fes[0].num_quads, 1)
    physical_quad = physical_cells[:, None, None] * jnp.ones(shape)
    zero_quad = jnp.zeros(shape)
    problem.set_params(
        [
            400.0 * jnp.ones((problem.fes[0].num_total_nodes, 1)),
            1.0,
            jnp.zeros(3),
            0.0,
            1.0,
            1.0,
            0.0,
            physical_quad,
            physical_quad,
            physical_quad,
            physical_quad,
            zero_quad,
            zero_quad,
            0.0,
            physical_quad,
        ]
    )


def _front_area_and_loss(problem, front_z):
    area = 0.0
    loss = 0.0
    for boundary_inds, surface_points, nanson, surface_vars in zip(
        problem.boundary_inds_list,
        problem.physical_surface_quad_points,
        problem.nanson_scale,
        problem.internal_vars_surfaces,
    ):
        if not len(boundary_inds):
            continue
        face_is_front = np.all(
            np.isclose(np.asarray(surface_points)[..., 2], front_z),
            axis=1,
        )
        face_active = np.asarray(surface_vars[0])[..., 0]
        surface_jxw = np.asarray(nanson)[:, 0, :]
        selected_active = face_active[face_is_front]
        selected_jxw = surface_jxw[face_is_front]
        area += float(np.sum(selected_active * selected_jxw))
        loss += integrate_surface_exchange(
            temperature_face=400.0 * np.ones_like(selected_jxw),
            surface_jxw=selected_jxw,
            active=selected_active,
            convection_h=problem.convection_h,
            ambient_k=problem.ambient,
            emissivity=problem.emissivity,
            stefan_boltzmann=problem.stefan_boltzmann,
            dt_s=1.0,
        )
    return area, loss


def _total_active_surface_area(problem):
    area = 0.0
    for nanson, surface_vars in zip(
        problem.nanson_scale,
        problem.internal_vars_surfaces,
    ):
        if not surface_vars:
            continue
        face_active = np.asarray(surface_vars[0])[..., 0]
        surface_jxw = np.asarray(nanson)[:, 0, :]
        area += float(np.sum(face_active * surface_jxw))
    return area


def test_future_layer_does_not_hide_current_active_top_surface():
    problem = _make_static_exterior_problem()

    _set_uniform_surface_state(problem, [1.0, 0.0])
    area, _ = _front_area_and_loss(problem, front_z=1.0)

    assert area == pytest.approx(1.0, rel=5.0e-3)


def test_active_top_area_and_convection_integral_follow_layer_activation():
    problem = _make_static_exterior_problem()
    areas = []
    losses = []

    for physical_cells, front_z in (([1.0, 0.0], 1.0), ([1.0, 1.0], 2.0)):
        _set_uniform_surface_state(problem, physical_cells)
        area, loss = _front_area_and_loss(problem, front_z)
        areas.append(area)
        losses.append(loss)

    # Unit top area in both stages. With h=2 W/(m2 K), T=400 K,
    # T_ambient=300 K and dt=1 s, each stage loses exactly 200 J.
    assert areas == pytest.approx([1.0, 1.0], rel=5.0e-3)
    assert losses == pytest.approx([200.0, 200.0], rel=5.0e-3)


def test_upward_shared_face_is_exposed_only_until_neighbor_activates():
    problem = _make_static_exterior_problem()

    areas = []
    for physical_cells in (
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ):
        _set_uniform_surface_state(problem, physical_cells)
        area, _ = _front_area_and_loss(problem, front_z=1.0)
        areas.append(area)

    assert areas == pytest.approx([1.0, 0.0, 0.0], rel=5.0e-3)


def test_dynamic_candidate_set_contains_only_upward_faces():
    problem = _make_static_exterior_problem()
    assert len(problem.boundary_inds_list[0]) == 2

    total_areas = []
    for physical_cells in (
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ):
        _set_uniform_surface_state(problem, physical_cells)
        total_areas.append(_total_active_surface_area(problem))

    assert total_areas == pytest.approx(
        [1.0, 1.0, 1.0, 0.0],
        rel=5.0e-3,
    )


@pytest.mark.parametrize("build_sign", [1.0, -1.0])
def test_tet_top_selection_uses_physical_outward_normal(build_sign):
    problem = _make_tet_top_problem(build_sign)
    fe = problem.fes[0]
    expected_local_face = 0 if build_sign > 0.0 else 3

    assert int(np.sum(fe.get_build_direction_face_flags(2, build_sign))) == 1
    np.testing.assert_array_equal(
        np.asarray(problem.boundary_inds_list[0]),
        np.asarray([[0, expected_local_face]], dtype=np.int32),
    )


def test_cooling_ramps_surface_ambient_with_the_frozen_bottom_schedule(
    tmp_path,
    monkeypatch,
):
    inp_path = tmp_path / "stacked_hex.inp"
    output_dir = tmp_path / "output"
    inp_path.write_text(TWO_STACKED_HEX_INP, encoding="utf-8")
    ambient_trace = []
    surface_exchange_trace = []

    def solver_probe(problem, solver_options=None):
        assert isinstance(problem, TransientThermal)
        ambient_trace.append(float(problem.ambient))
        uniform_temperature = 423.15 * jnp.ones(
            (problem.fes[0].num_total_nodes, 1)
        )
        residual = problem.compute_residual([uniform_temperature])[0]
        surface_exchange_trace.append(float(jnp.sum(residual)))
        return [uniform_temperature]

    monkeypatch.setattr(stepper, "solver", solver_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "active-surface-test",
            "--inp",
            str(inp_path),
            "--output-dir",
            str(output_dir),
            "--build-axis",
            "z",
            "--base-side",
            "min",
            "--layer-thickness",
            "2.0",
            "--layers",
            "1",
            "--scan-steps-per-layer",
            "1",
            "--hatch-lines-per-layer",
            "1",
            "--laser-power",
            "0",
            "--dt",
            "0.1",
            "--layer-activation-mode",
            "layer_on_scan",
            "--layer-activation-geometry",
            "intersection",
            "--future-layer-mode",
            "void",
            "--surface-selection",
            "exterior",
            "--quadrature-order",
            "2",
            "--ambient",
            "423.15",
            "--preheat-temperature",
            "423.15",
            "--bottom-thermal-bc",
            "fixed",
            "--bottom-temperature",
            "423.15",
            "--cooling-steps",
            "2",
            "--cooling-dt",
            "1.0",
            "--final-cooldown-temperature",
            "300.0",
            "--mechanics-every",
            "0",
            "--thermal-output-every",
            "1000",
            "--mechanics-output-every",
            "1000",
            "--summary-every",
            "1000",
        ],
    )

    stepper.main()

    assert ambient_trace == pytest.approx([423.15, 361.575, 300.0])
    assert surface_exchange_trace[0] == pytest.approx(0.0, abs=1.0e-10)
    assert surface_exchange_trace[1] > 0.0
    assert surface_exchange_trace[2] > surface_exchange_trace[1]
    with (output_dir / "path_used.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        path_rows = list(csv.DictReader(stream))
    assert [
        float(row["ambient_temperature"]) for row in path_rows
    ] == pytest.approx([423.15, 361.575, 300.0])
    assert [
        float(row["bottom_temperature"]) for row in path_rows
    ] == pytest.approx([423.15, 361.575, 300.0])
