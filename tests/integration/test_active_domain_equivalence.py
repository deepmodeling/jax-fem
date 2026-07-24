import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver
from jax_fem_am.materials.phases import (
    STATE_POWDER,
    STATE_SOLID,
    STATE_VOID,
    make_quad_scalar,
    mechanics_material_quads,
    thermal_material_quads,
)
from jax_fem_am.physics.mechanics import ThermoMechanical
from jax_fem_am.physics.thermal import TransientThermal
from jax_fem_am.simulation import acceleration

try:
    from jax_fem_am.process.activation import (
        contributing_cell_mask,
        make_inactive_node_dirichlet_bc,
        merge_dirichlet_bcs,
        physical_node_mask,
        resolve_surface_active_mask,
    )
except ImportError:
    # RED baseline: express the expected active-domain contract locally so the
    # existing ersatz material fails for its physical contribution, not merely
    # because the production helpers have not been introduced yet.
    def physical_node_mask(cells, physical_cell, num_nodes=None):
        cells = np.asarray(cells, dtype=np.int64)
        physical_cell = np.asarray(physical_cell, dtype=bool)
        if num_nodes is None:
            num_nodes = int(cells.max()) + 1
        mask = np.zeros(num_nodes, dtype=bool)
        mask[np.unique(cells[physical_cell])] = True
        return mask

    def contributing_cell_mask(*quad_fields):
        mask = np.zeros(len(quad_fields[0]), dtype=bool)
        for field in quad_fields:
            values = np.asarray(field)
            mask |= np.any(
                values != 0,
                axis=tuple(range(1, values.ndim)),
            )
        return mask

    def make_inactive_node_dirichlet_bc(
        inactive_node_mask,
        *,
        vec,
        value,
    ):
        mask = jnp.asarray(inactive_node_mask, dtype=bool)

        def inactive_node(_point, node_id):
            return mask[node_id]

        def prescribed_value(_point):
            return value

        return [
            [inactive_node for _ in range(vec)],
            list(range(vec)),
            [prescribed_value for _ in range(vec)],
        ]

    def merge_dirichlet_bcs(*conditions):
        merged = [[], [], []]
        for condition in conditions:
            if condition is None:
                continue
            for target, values in zip(merged, condition):
                target.extend(values)
        return merged if merged[0] else None

    def resolve_surface_active_mask(args):
        strict = (
            args.layer_activation_mode == "layer_on_scan"
            and args.future_layer_mode == "void"
        )
        if strict and args.surface_active_mask is False:
            raise ValueError
        if strict:
            return True
        if args.surface_active_mask is not None:
            return args.surface_active_mask
        return args.surface_selection == "exterior"


POINTS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
CELLS = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
PHYSICAL_CELL = np.asarray([True, False])


def _thermal_args(
    inactive_mass_factor,
    inactive_thermal_factor=1.0e-4,
):
    return SimpleNamespace(
        rho=4.0,
        cp=5.0,
        conductivity=6.0,
        rho_solid=4.0,
        cp_solid=5.0,
        conductivity_solid=6.0,
        rho_liquid=None,
        cp_liquid=None,
        conductivity_liquid=None,
        rho_powder=2.0,
        cp_powder=3.0,
        conductivity_powder=1.0,
        inactive_thermal_factor=inactive_thermal_factor,
        inactive_mass_factor=inactive_mass_factor,
        old_layer_thermal_factor=1.0,
        solidus_temperature=0.0,
        liquidus_temperature=0.0,
        latent_heat=0.0,
        layer_activation_mode="layer_on_scan",
        future_layer_mode="void",
        powder_mode="powder",
    )


def _thermal_tables():
    return {
        key: None
        for key in (
            "cp_solid",
            "k_solid",
            "cp_powder",
            "k_powder",
            "cp_liquid",
            "k_liquid",
        )
    }


def _bottom_temperature_bc():
    def on_x_min(point):
        return jnp.isclose(point[0], 0.0)

    def one(_point):
        return 1.0

    return [[on_x_min], [0], [one]]


def _make_thermal(points, cells, boundary):
    return TransientThermal(
        mesh=Mesh(points, cells, ele_type="TET4"),
        vec=1,
        dim=3,
        ele_type="TET4",
        dirichlet_bc_info=boundary,
        location_fns=[],
        additional_info=(
            0.0,
            0.0,
            0.0,
            5.670374419e-8,
            2,
            0,
            1,
            1.0,
            0,
            0.0,
            1.0,
            False,
            "paper_hemispherical",
        ),
    )


def _set_thermal_state(
    problem,
    old_temperature,
    physical_cell,
    inactive_mass_factor,
    inactive_thermal_factor=1.0e-4,
):
    physical_cell = np.asarray(physical_cell, dtype=bool)
    active_quad = make_quad_scalar(
        physical_cell.astype(np.float64),
        problem.fes[0].num_quads,
    )
    phase_quad = make_quad_scalar(
        np.where(physical_cell, STATE_SOLID, STATE_VOID),
        problem.fes[0].num_quads,
    )
    rho, cp, conductivity, latent_cp = thermal_material_quads(
        jnp.zeros_like(active_quad),
        active_quad,
        phase_quad,
        _thermal_args(
            inactive_mass_factor,
            inactive_thermal_factor,
        ),
        _thermal_tables(),
        printed_quad=active_quad,
    )
    problem.set_params(
        [
            jnp.asarray(old_temperature, dtype=jnp.float64),
            0.1,
            jnp.zeros(3, dtype=jnp.float64),
            0.0,
            1.0,
            1.0,
            0.0,
            active_quad,
            rho,
            cp,
            conductivity,
            latent_cp,
            jnp.zeros_like(active_quad),
            0.0,
            active_quad,
        ]
    )
    return rho, conductivity


def _solve_thermal(
    points,
    cells,
    physical_cell,
    inactive_mass_factor,
    inactive_thermal_factor=1.0e-4,
):
    physical_cell = np.asarray(physical_cell, dtype=bool)
    node_mask = physical_node_mask(
        cells,
        physical_cell,
        num_nodes=len(points),
    )
    inactive_bc = make_inactive_node_dirichlet_bc(
        ~node_mask,
        vec=1,
        value=0.0,
    )
    problem = _make_thermal(
        points,
        cells,
        merge_dirichlet_bcs(_bottom_temperature_bc(), inactive_bc),
    )
    rho, conductivity = _set_thermal_state(
        problem,
        jnp.zeros((len(points), 1), dtype=jnp.float64),
        physical_cell,
        inactive_mass_factor,
        inactive_thermal_factor,
    )
    solution = solver(
        problem,
        solver_options={"newton": {"linear": {"spsolve_solver": {}}}},
    )[0]
    return np.asarray(solution), (np.asarray(rho), np.asarray(conductivity))


def _mechanics_args(inactive_mechanics_factor=1.0e-3):
    return SimpleNamespace(
        young=1_000.0,
        alpha=0.0,
        poisson=0.3,
        mechanics_model="linear_elastic",
        inactive_mechanics_factor=inactive_mechanics_factor,
        mushy_mechanics_factor=1.0e-3,
        liquid_mechanics_factor=1.0e-6,
        powder_solid_E=None,
        layer_activation_mode="layer_on_scan",
        future_layer_mode="void",
    )


def _make_mechanics(points, cells):
    return ThermoMechanical(
        mesh=Mesh(points, cells, ele_type="TET4"),
        vec=3,
        dim=3,
        ele_type="TET4",
        dirichlet_bc_info=None,
        additional_info=("linear_elastic", None, 0.0, 0.0, (), False),
    )


def _mechanics_residual_and_tangent(
    points,
    cells,
    physical_cell,
    displacement,
    inactive_mechanics_factor=1.0e-3,
):
    problem = _make_mechanics(points, cells)
    active_quad = make_quad_scalar(
        np.asarray(physical_cell, dtype=np.float64),
        problem.fes[0].num_quads,
    )
    phase_quad = make_quad_scalar(
        np.where(physical_cell, STATE_SOLID, STATE_VOID),
        problem.fes[0].num_quads,
    )
    (
        active_factor,
        young,
        alpha,
        poisson,
        yield_stress,
        hardening,
    ) = mechanics_material_quads(
        jnp.zeros_like(active_quad),
        active_quad,
        phase_quad,
        _mechanics_args(inactive_mechanics_factor),
        {"E": None, "alpha": None, "poisson": None},
    )
    problem.set_params(
        [
            jnp.zeros_like(active_quad),
            jnp.zeros_like(active_quad),
            active_factor,
            young,
            alpha,
            poisson,
            yield_stress,
            hardening,
            jnp.zeros_like(active_quad),
        ]
    )
    residual = problem.newton_update([jnp.asarray(displacement)])[0]
    ndof = len(points) * 3
    tangent = scipy.sparse.coo_matrix(
        (problem.V, (problem.I, problem.J)),
        shape=(ndof, ndof),
    ).tocsr()
    return (
        np.asarray(residual).reshape(-1),
        tangent,
        np.asarray(active_factor),
    )


def test_strict_thermal_domain_matches_physical_cell_deletion():
    full, (rho, conductivity) = _solve_thermal(
        POINTS,
        CELLS,
        PHYSICAL_CELL,
        inactive_mass_factor=1.0,
    )
    deleted, _ = _solve_thermal(
        POINTS[:4],
        CELLS[:1],
        np.asarray([True]),
        inactive_mass_factor=1.0e-12,
    )
    full_with_tiny_placeholder, _ = _solve_thermal(
        POINTS,
        CELLS,
        PHYSICAL_CELL,
        inactive_mass_factor=1.0e-12,
        inactive_thermal_factor=1.0e-12,
    )

    relative_error = np.linalg.norm(full[:4] - deleted) / np.linalg.norm(deleted)
    assert relative_error <= 1.0e-8
    np.testing.assert_array_equal(full_with_tiny_placeholder, full)
    assert np.count_nonzero(rho[1]) == 0
    assert np.count_nonzero(conductivity[1]) == 0
    assert full[4, 0] == 0.0


def test_strict_mechanics_domain_has_zero_residual_and_tangent_contribution():
    displacement = 1.0e-3 * POINTS
    full_displacement = displacement.copy()
    full_displacement[4] = [3.0, -2.0, 5.0]
    full_residual, full_tangent, active_factor = (
        _mechanics_residual_and_tangent(
            POINTS,
            CELLS,
            PHYSICAL_CELL,
            full_displacement,
        )
    )
    alternate_residual, alternate_tangent, _ = (
        _mechanics_residual_and_tangent(
            POINTS,
            CELLS,
            PHYSICAL_CELL,
            full_displacement,
            inactive_mechanics_factor=1.0,
        )
    )
    deleted_residual, deleted_tangent, _ = _mechanics_residual_and_tangent(
        POINTS[:4],
        CELLS[:1],
        np.asarray([True]),
        displacement[:4],
    )

    active_dofs = np.arange(12)
    inactive_dofs = np.arange(12, 15)
    np.testing.assert_allclose(
        full_residual[active_dofs],
        deleted_residual,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        full_tangent[active_dofs][:, active_dofs].toarray(),
        deleted_tangent.toarray(),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        full_residual[inactive_dofs],
        0.0,
    )
    # COO/CSR assembly may retain structural entries whose stored value is
    # exactly zero. The active-domain contract concerns numerical
    # contribution, not sparse-pattern compaction.
    np.testing.assert_array_equal(
        full_tangent[inactive_dofs].data,
        0.0,
    )
    np.testing.assert_array_equal(
        full_tangent[:, inactive_dofs].data,
        0.0,
    )
    assert np.count_nonzero(active_factor[1]) == 0
    np.testing.assert_array_equal(alternate_residual, full_residual)
    np.testing.assert_array_equal(
        alternate_tangent.toarray(),
        full_tangent.toarray(),
    )


def test_inactive_constraint_mask_contains_only_exclusive_nodes():
    node_mask = physical_node_mask(
        CELLS,
        PHYSICAL_CELL,
        num_nodes=len(POINTS),
    )
    np.testing.assert_array_equal(
        node_mask,
        [True, True, True, True, False],
    )

    bc = make_inactive_node_dirichlet_bc(
        ~node_mask,
        vec=1,
        value=0.0,
    )
    selected = np.asarray(
        jax.vmap(bc[0][0])(
            jnp.asarray(POINTS),
            jnp.arange(len(POINTS)),
        )
    )
    np.testing.assert_array_equal(
        selected,
        [False, False, False, False, True],
    )


def test_strict_domain_requires_surface_flux_masking():
    strict = SimpleNamespace(
        layer_activation_mode="layer_on_scan",
        future_layer_mode="void",
        surface_selection="box",
        surface_active_mask=None,
    )
    assert resolve_surface_active_mask(strict) is True

    strict.surface_active_mask = False
    with np.testing.assert_raises_regex(
        ValueError,
        "incompatible",
    ):
        resolve_surface_active_mask(strict)

    legacy = SimpleNamespace(
        layer_activation_mode="front",
        future_layer_mode="void",
        surface_selection="box",
        surface_active_mask=None,
    )
    assert resolve_surface_active_mask(legacy) is False


def test_mechanics_dofs_follow_nonzero_stiffness_not_printed_label():
    active = make_quad_scalar(
        np.asarray([1.0, 1.0]),
        num_quads=1,
    )
    phase = make_quad_scalar(
        np.asarray([STATE_SOLID, STATE_POWDER]),
        num_quads=1,
    )
    active_factor, *_ = mechanics_material_quads(
        jnp.zeros_like(active),
        active,
        phase,
        _mechanics_args(),
        {"E": None, "alpha": None, "poisson": None},
    )

    contributing_cell = contributing_cell_mask(active_factor)
    np.testing.assert_array_equal(contributing_cell, [True, False])
    np.testing.assert_array_equal(
        physical_node_mask(
            CELLS,
            contributing_cell,
            num_nodes=len(POINTS),
        ),
        [True, True, True, True, False],
    )


def test_permanent_powder_is_physical_only_when_its_model_contributes():
    active = make_quad_scalar(np.ones(2), num_quads=1)
    phase = make_quad_scalar(
        np.asarray([STATE_SOLID, STATE_POWDER]),
        num_quads=1,
    )
    thermal_args = _thermal_args(inactive_mass_factor=1.0)
    rho, cp, conductivity, _ = thermal_material_quads(
        jnp.zeros_like(active),
        active,
        phase,
        thermal_args,
        _thermal_tables(),
        printed_quad=active,
    )
    np.testing.assert_array_equal(
        contributing_cell_mask(rho * cp, conductivity),
        [True, True],
    )
    assert float(rho[1, 0, 0]) == thermal_args.rho_powder
    assert float(conductivity[1, 0, 0]) == (
        thermal_args.conductivity_powder
    )

    mechanics_args = _mechanics_args()
    mechanics_args.powder_solid_E = 10.0
    mechanics_args.powder_solid_yield = 1.0
    mechanics_args.powder_solid_hardening = 0.1
    active_factor, *_ = mechanics_material_quads(
        jnp.zeros_like(active),
        active,
        phase,
        mechanics_args,
        {"E": None, "alpha": None, "poisson": None},
    )
    np.testing.assert_array_equal(
        contributing_cell_mask(active_factor),
        [True, True],
    )


def test_dynamic_activation_refreshes_linear_system_constraint_rows():
    initially_physical = np.asarray([True, False])
    initial_nodes = physical_node_mask(
        CELLS,
        initially_physical,
        num_nodes=len(POINTS),
    )
    problem = _make_thermal(
        POINTS,
        CELLS,
        merge_dirichlet_bcs(
            _bottom_temperature_bc(),
            make_inactive_node_dirichlet_bc(
                ~initial_nodes,
                vec=1,
                value=0.0,
            ),
        ),
    )
    _set_thermal_state(
        problem,
        np.zeros((len(POINTS), 1)),
        initially_physical,
        inactive_mass_factor=1.0,
    )
    first_solution = solver(
        problem,
        solver_options={"newton": {"linear": {"spsolve_solver": {}}}},
    )[0]

    all_physical = np.asarray([True, True])
    problem.fes[0].update_Dirichlet_boundary_conditions(
        _bottom_temperature_bc()
    )
    _set_thermal_state(
        problem,
        first_solution,
        all_physical,
        inactive_mass_factor=1.0,
    )
    reused_solution = solver(
        problem,
        solver_options={"newton": {"linear": {"spsolve_solver": {}}}},
    )[0]

    fresh_problem = _make_thermal(
        POINTS,
        CELLS,
        _bottom_temperature_bc(),
    )
    _set_thermal_state(
        fresh_problem,
        first_solution,
        all_physical,
        inactive_mass_factor=1.0,
    )
    fresh_solution = solver(
        fresh_problem,
        solver_options={"newton": {"linear": {"spsolve_solver": {}}}},
    )[0]

    np.testing.assert_allclose(
        np.asarray(reused_solution),
        np.asarray(fresh_solution),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_all_inactive_merge_has_unique_dofs_and_preserves_physical_bc():
    inactive_bc = make_inactive_node_dirichlet_bc(
        np.ones(len(POINTS), dtype=bool),
        vec=1,
        value=0.0,
    )
    problem = _make_thermal(
        POINTS,
        CELLS,
        merge_dirichlet_bcs(_bottom_temperature_bc(), inactive_bc),
    )
    fe = problem.fes[0]
    flat_dofs = np.concatenate(
        [
            np.asarray(nodes) * fe.vec + np.asarray(components)
            for nodes, components in zip(
                fe.node_inds_list,
                fe.vec_inds_list,
            )
        ]
    )
    values = np.concatenate([np.asarray(item) for item in fe.vals_list])

    assert len(flat_dofs) == len(np.unique(flat_dofs))
    value_by_dof = dict(zip(flat_dofs.tolist(), values.tolist()))
    assert value_by_dof == {
        0: 1.0,
        1: 0.0,
        2: 1.0,
        3: 1.0,
        4: 0.0,
    }


def test_accelerated_material_kernels_keep_strict_zero_semantics():
    base = SimpleNamespace(
        jax=jax,
        np=jnp,
        STATE_VOID=STATE_VOID,
        STATE_POWDER=STATE_POWDER,
        STATE_SOLID=STATE_SOLID,
        STATE_MUSHY=3.0,
        STATE_LIQUID=4.0,
        STATE_SUBSTRATE=5.0,
        STATE_SUPPORT=6.0,
    )
    active = jnp.asarray([[[1.0]], [[0.0]]], dtype=jnp.float64)
    phase = jnp.asarray(
        [[[STATE_SOLID]], [[STATE_VOID]]],
        dtype=jnp.float64,
    )
    temperature = jnp.zeros_like(active)

    thermal_args = _thermal_args(inactive_mass_factor=1.0)
    thermal_key = acceleration._thermal_material_key(thermal_args, base)
    thermal_kernel = acceleration._make_jit_thermal_material_kernel(
        base,
        thermal_key,
    )
    rho, cp, conductivity, _ = thermal_kernel(
        temperature,
        active,
        phase,
        active,
        jnp.zeros_like(active),
    )
    np.testing.assert_array_equal(np.asarray(rho[1]), 0.0)
    np.testing.assert_array_equal(np.asarray(cp[1]), 0.0)
    np.testing.assert_array_equal(np.asarray(conductivity[1]), 0.0)

    mechanics_args = _mechanics_args()
    mechanics_key = acceleration._mechanics_material_key(
        mechanics_args,
        base,
    )
    mechanics_kernel = acceleration._make_jit_mechanics_material_kernel(
        base,
        mechanics_key,
    )
    active_factor, *_ = mechanics_kernel(temperature, active, phase)
    np.testing.assert_array_equal(np.asarray(active_factor[1]), 0.0)


def test_accelerated_weak_solid_powder_matches_reference_material_path():
    base = SimpleNamespace(
        jax=jax,
        np=jnp,
        STATE_VOID=STATE_VOID,
        STATE_POWDER=STATE_POWDER,
        STATE_SOLID=STATE_SOLID,
        STATE_MUSHY=3.0,
        STATE_LIQUID=4.0,
        STATE_SUBSTRATE=5.0,
        STATE_SUPPORT=6.0,
    )
    args = _mechanics_args()
    args.powder_solid_E = 10.0
    args.powder_solid_yield = 1.0
    args.powder_solid_hardening = 0.1
    temperature = jnp.zeros((1, 1, 1), dtype=jnp.float64)
    active = jnp.ones_like(temperature)
    phase = STATE_POWDER * jnp.ones_like(temperature)

    expected = mechanics_material_quads(
        temperature,
        active,
        phase,
        args,
        {"E": None, "alpha": None, "poisson": None},
    )
    key = acceleration._mechanics_material_key(args, base)
    kernel = acceleration._make_jit_mechanics_material_kernel(base, key)
    actual = kernel(temperature, active, phase)

    for actual_field, expected_field in zip(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_field),
            np.asarray(expected_field),
            rtol=0.0,
            atol=0.0,
        )
