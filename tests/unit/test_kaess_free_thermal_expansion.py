"""Real-FE free-thermal-expansion verification for Kaess PAR020.

The legacy regression only probes the constitutive map directly.  These tests
exercise the current ThermoMechanical element assembly and stress recovery for
both supported three-dimensional element types, with and without B-bar.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import pytest

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver
from jax_fem_am.physics.mechanics import ThermoMechanical


LENGTH = 1.0e-3
YOUNG = 200.0e9
POISSON = 0.3
ALPHA = 1.2e-5


def _element_geometry(ele_type):
    if ele_type == "HEX8":
        points = LENGTH * onp.array(
            [
                [0.00, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [1.00, 1.00, 0.00],
                [0.00, 1.00, 0.00],
                [0.00, 0.00, 1.00],
                [1.04, 0.02, 0.97],
                [1.13, 0.92, 1.11],
                [0.00, 1.00, 1.00],
            ],
            dtype=onp.float64,
        )
        cells = onp.array(
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            dtype=onp.int64,
        )
        xy_plane_node = 3
    elif ele_type == "TET4":
        points = LENGTH * onp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=onp.float64,
        )
        cells = onp.array([[0, 1, 2, 3]], dtype=onp.int64)
        xy_plane_node = 2
    else:  # pragma: no cover - guarded by test parameters
        raise ValueError(f"Unsupported element type: {ele_type}")
    return points, cells, xy_plane_node


def _node_selector(coordinate):
    target = jnp.asarray(coordinate)

    def selected(point):
        return jnp.all(
            jnp.isclose(point, target, rtol=0.0, atol=1.0e-12)
        )

    return selected


def _build_problem(ele_type, bbar):
    points, cells, xy_plane_node = _element_geometry(ele_type)
    origin = _node_selector(points[0])
    x_axis = _node_selector(points[1])
    xy_plane = _node_selector(points[xy_plane_node])

    def zero(_point):
        return 0.0

    # Minimal 3-2-1 constraint: remove the six rigid-body modes while leaving
    # u = alpha*dT*(x-x0) admissible.
    dirichlet_bc_info = [
        [origin, origin, origin, x_axis, x_axis, xy_plane],
        [0, 1, 2, 1, 2, 2],
        [zero, zero, zero, zero, zero, zero],
    ]
    problem = ThermoMechanical(
        mesh=Mesh(points, cells, ele_type=ele_type),
        vec=3,
        dim=3,
        ele_type=ele_type,
        quadrature_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(
            "linear_elastic",
            None,
            0.0,
            0.0,
            (),
            bbar,
        ),
    )
    return problem, jnp.asarray(points)


def _uniform_params(problem, delta_t):
    shape = (
        len(problem.fes[0].cells),
        problem.fes[0].num_quads,
        1,
    )

    def full(value):
        return jnp.full(shape, value, dtype=jnp.float64)

    return [
        full(400.0),
        full(delta_t),
        full(1.0),
        full(YOUNG),
        full(ALPHA),
        full(POISSON),
        full(1.0e30),
        full(0.0),
        full(0.0),
    ]


@pytest.mark.parametrize(
    ("ele_type", "bbar", "expected_num_quads"),
    [
        pytest.param("HEX8", False, 8, id="hex8-plain"),
        pytest.param("HEX8", True, 8, id="hex8-bbar"),
        pytest.param("TET4", False, 4, id="tet4-plain"),
        pytest.param("TET4", True, 4, id="tet4-bbar"),
    ],
)
@pytest.mark.parametrize(
    "delta_t",
    [
        pytest.param(100.0, id="heating"),
        pytest.param(-137.0, id="cooling"),
    ],
)
def test_free_thermal_expansion_is_stress_free_in_real_fe_assembly(
    ele_type,
    bbar,
    expected_num_quads,
    delta_t,
):
    problem, points = _build_problem(ele_type, bbar)
    params = _uniform_params(problem, delta_t)
    problem.set_params(params)

    exact_displacement = ALPHA * delta_t * (points - points[0])
    zero_displacement = jnp.zeros_like(exact_displacement)

    free_residual = onp.asarray(
        problem.compute_residual([exact_displacement])[0]
    )
    locked_residual = onp.asarray(
        problem.compute_residual([zero_displacement])[0]
    )
    free_stress = onp.asarray(
        problem.compute_cell_stress(
            exact_displacement,
            params,
        )["stress_quad"]
    )
    locked_stress = onp.asarray(
        problem.compute_cell_stress(
            zero_displacement,
            params,
        )["stress_quad"]
    )

    assert problem.fes[0].num_quads == expected_num_quads
    assert onp.all(onp.asarray(problem.fes[0].JxW) > 0.0)
    assert free_residual.dtype == onp.dtype(onp.float64)
    assert free_stress.dtype == onp.dtype(onp.float64)
    assert onp.all(onp.isfinite(free_residual))
    assert onp.all(onp.isfinite(free_stress))

    # The locked control proves that the element, material and thermal load are
    # all live.  Scale the round-off tolerance from the same assembled model.
    residual_scale = float(onp.max(onp.abs(locked_residual)))
    stress_scale = float(onp.max(onp.abs(locked_stress)))
    assert residual_scale > 0.0
    assert stress_scale > 0.0

    relative_tolerance = 128.0 * onp.finfo(free_stress.dtype).eps
    assert float(onp.max(onp.abs(free_residual))) <= (
        relative_tolerance * residual_scale
    )
    assert float(onp.max(onp.abs(free_stress))) <= (
        relative_tolerance * stress_scale
    )

    # Exercise the declared 3-2-1 constraints, global tangent assembly and
    # linear solve: starting from zero must recover the admissible analytical
    # expansion/contraction field in one linear-elastic Newton correction.
    solved_displacement = onp.asarray(
        solver(
            problem,
            solver_options={
                "newton": {
                    "tol": 0.0,
                    "rel_tol": relative_tolerance,
                    "max_iter": 3,
                    "line_search_flag": False,
                    "linear": {"spsolve_solver": {}},
                },
            },
        )[0]
    )
    displacement_scale = float(
        onp.max(onp.abs(onp.asarray(exact_displacement)))
    )
    assert displacement_scale > 0.0
    assert float(
        onp.max(
            onp.abs(
                solved_displacement - onp.asarray(exact_displacement)
            )
        )
    ) <= relative_tolerance * displacement_scale
