import unittest
from unittest import mock
from types import SimpleNamespace


try:
    import jax_fem.problem as problem_module
    import jax.numpy as jnp
    import numpy as onp
    from jax_fem.problem import Problem
except ImportError as exc:  # pragma: no cover - depends on local runtime
    Problem = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class ProblemCellAssemblyCutCountTest(unittest.TestCase):
    def setUp(self):
        self.original_num_cuts = Problem.cell_assembly_num_cuts
        self.original_target_batch_size = Problem.cell_assembly_target_batch_size
        Problem.cell_assembly_num_cuts = 20
        Problem.cell_assembly_target_batch_size = None

    def tearDown(self):
        Problem.cell_assembly_num_cuts = self.original_num_cuts
        Problem.cell_assembly_target_batch_size = self.original_target_batch_size

    @staticmethod
    def make_problem(num_cells):
        problem = object.__new__(Problem)
        problem.num_cells = num_cells
        return problem

    def test_default_cut_count_preserves_legacy_twenty_cut_limit(self):
        self.assertEqual(self.make_problem(80)._cell_assembly_cut_count(), 20)

    def test_default_cut_count_is_capped_by_cell_count(self):
        self.assertEqual(self.make_problem(8)._cell_assembly_cut_count(), 8)

    def test_class_level_override_changes_cut_count(self):
        Problem.cell_assembly_num_cuts = 4

        self.assertEqual(self.make_problem(80)._cell_assembly_cut_count(), 4)

    def test_target_batch_size_computes_cut_count_from_cell_count(self):
        Problem.cell_assembly_target_batch_size = 2048

        self.assertEqual(self.make_problem(80)._cell_assembly_cut_count(), 1)
        self.assertEqual(self.make_problem(4096)._cell_assembly_cut_count(), 2)
        self.assertEqual(self.make_problem(4097)._cell_assembly_cut_count(), 3)

    def test_target_batch_size_is_capped_by_cell_count(self):
        Problem.cell_assembly_target_batch_size = 1

        self.assertEqual(self.make_problem(8)._cell_assembly_cut_count(), 8)

    def test_invalid_target_batch_size_fails_before_assembly(self):
        problem = self.make_problem(80)
        problem.cell_assembly_target_batch_size = 0

        with self.assertRaisesRegex(ValueError, "positive integer"):
            problem._cell_assembly_cut_count()

    def test_instance_override_wins_over_class_default(self):
        Problem.cell_assembly_num_cuts = 10
        problem = self.make_problem(80)
        problem.cell_assembly_num_cuts = 2

        self.assertEqual(problem._cell_assembly_cut_count(), 2)

    def test_invalid_cut_count_fails_before_assembly(self):
        problem = self.make_problem(80)
        problem.cell_assembly_num_cuts = 0

        with self.assertRaisesRegex(ValueError, "positive integer"):
            problem._cell_assembly_cut_count()

    def make_split_problem(self, jac_flag=False):
        problem = self.make_problem(3)
        problem.cell_assembly_target_batch_size = 2048
        problem.physical_quad_points = onp.arange(3)
        problem.shape_grads = onp.arange(3) + 10
        problem.JxW = onp.arange(3) + 20
        problem.v_grads_JxW = onp.arange(3) + 30

        def kernel(cells_sol_flat, *_args):
            return cells_sol_flat + 1

        def kernel_jac(cells_sol_flat, *_args):
            return cells_sol_flat + 1, cells_sol_flat + 2

        problem.kernel = kernel
        problem.kernel_jac = kernel_jac
        return problem

    def test_single_cut_split_returns_kernel_output_without_vstack(self):
        problem = self.make_split_problem()
        cells_sol_flat = onp.asarray([1.0, 2.0, 3.0])

        class NoVstack:
            @staticmethod
            def vstack(_values):
                raise AssertionError("single-cut path should not vstack")

        result = problem.split_and_compute_cell(
            cells_sol_flat,
            NoVstack,
            jac_flag=False,
            internal_vars=[],
        )

        onp.testing.assert_array_equal(result, cells_sol_flat + 1)

    def test_single_cut_jacobian_split_returns_kernel_output_without_vstack(self):
        problem = self.make_split_problem(jac_flag=True)
        cells_sol_flat = onp.asarray([1.0, 2.0, 3.0])

        class NoVstack:
            @staticmethod
            def vstack(_values):
                raise AssertionError("single-cut path should not vstack")

        values, jacs = problem.split_and_compute_cell(
            cells_sol_flat,
            NoVstack,
            jac_flag=True,
            internal_vars=[],
        )

        onp.testing.assert_array_equal(values, cells_sol_flat + 1)
        onp.testing.assert_array_equal(jacs, cells_sol_flat + 2)

    def make_empty_face_problem(self):
        problem = self.make_problem(2)
        problem.boundary_inds_list = [onp.empty((0, 2), dtype=onp.int64)]

        def fail_kernel(*_args):
            raise AssertionError("empty face sets should not call the face kernel")

        problem.kernel_face = [fail_kernel]
        problem.kernel_jac_face = [fail_kernel]
        return problem

    def test_empty_face_residual_returns_empty_values_without_kernel(self):
        problem = self.make_empty_face_problem()
        cells_sol_flat = onp.asarray([[1.0, 2.0], [3.0, 4.0]])

        values = problem.compute_face(
            cells_sol_flat,
            onp,
            jac_flag=False,
            internal_vars_surfaces=[[]],
        )

        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].shape, (0, 2))
        self.assertEqual(values[0].dtype, cells_sol_flat.dtype)

    def test_empty_face_jacobian_returns_empty_values_without_kernel(self):
        problem = self.make_empty_face_problem()
        cells_sol_flat = onp.asarray([[1.0, 2.0], [3.0, 4.0]])

        values, jacs = problem.compute_face(
            cells_sol_flat,
            onp,
            jac_flag=True,
            internal_vars_surfaces=[[]],
        )

        self.assertEqual(len(values), 1)
        self.assertEqual(len(jacs), 1)
        self.assertEqual(values[0].shape, (0, 2))
        self.assertEqual(jacs[0].shape, (0, 2, 2))
        self.assertEqual(values[0].dtype, cells_sol_flat.dtype)
        self.assertEqual(jacs[0].dtype, cells_sol_flat.dtype)

    def test_residual_scatter_skips_empty_face_sets(self):
        problem = self.make_problem(1)
        problem.num_vars = 1
        problem.fes = [SimpleNamespace(num_total_nodes=2, vec=1)]
        problem.cells_list = [jnp.asarray([[0, 1]])]
        problem.cells_list_face_list = [[onp.empty((0, 2), dtype=onp.int64)]]
        problem.unflatten_fn_dof = lambda x: [x.reshape((2, 1))]

        class FakeEmptyFaceValues:
            shape = (0, 2)

        result = problem.compute_residual_vars_helper(
            jnp.asarray([[1.0, 2.0]]),
            [FakeEmptyFaceValues()],
        )

        onp.testing.assert_array_equal(onp.asarray(result[0]), [[1.0], [2.0]])

    def test_single_var_residual_scatter_avoids_unflatten_for_cells(self):
        problem = self.make_problem(2)
        problem.num_vars = 1
        problem.fes = [SimpleNamespace(num_total_nodes=3, num_nodes=2, vec=1)]
        problem.cells_list = [jnp.asarray([[0, 1], [1, 2]])]
        problem.cells_list_face_list = []

        def fail_unflatten(_value):
            raise AssertionError("single-var scatter should reshape directly")

        problem.unflatten_fn_dof = fail_unflatten

        result = problem.compute_residual_vars_helper(
            jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            [],
        )

        onp.testing.assert_array_equal(onp.asarray(result[0]), [[1.0], [5.0], [4.0]])

    def test_single_var_residual_scatter_avoids_unflatten_for_faces(self):
        problem = self.make_problem(1)
        problem.num_vars = 1
        problem.fes = [SimpleNamespace(num_total_nodes=3, num_nodes=2, vec=1)]
        problem.cells_list = [jnp.asarray([[0, 1]])]
        problem.cells_list_face_list = [[jnp.asarray([[1, 2]])]]

        def fail_unflatten(_value):
            raise AssertionError("single-var face scatter should reshape directly")

        problem.unflatten_fn_dof = fail_unflatten

        result = problem.compute_residual_vars_helper(
            jnp.asarray([[1.0, 2.0]]),
            [jnp.asarray([[3.0, 4.0]])],
        )

        onp.testing.assert_array_equal(onp.asarray(result[0]), [[1.0], [5.0], [4.0]])

    def test_single_var_cells_sol_flat_reshapes_without_ravel_pytree(self):
        problem = self.make_problem(2)
        problem.num_vars = 1
        cells_sol_list = [jnp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]])]

        with mock.patch.object(
            problem_module.jax.flatten_util,
            "ravel_pytree",
            side_effect=AssertionError("single-var flatten should reshape directly"),
        ):
            cells_sol_flat = problem._flatten_cells_sol(cells_sol_list)

        onp.testing.assert_array_equal(
            onp.asarray(cells_sol_flat),
            [[1.0, 2.0], [3.0, 4.0]],
        )

    def test_multi_var_cells_sol_flat_matches_legacy_ravel_order(self):
        problem = self.make_problem(2)
        problem.num_vars = 2
        cells_sol_list = [
            jnp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]]),
            jnp.asarray([[[10.0, 11.0]], [[12.0, 13.0]]]),
        ]

        cells_sol_flat = problem._flatten_cells_sol(cells_sol_list)

        onp.testing.assert_array_equal(
            onp.asarray(cells_sol_flat),
            [[1.0, 2.0, 10.0, 11.0], [3.0, 4.0, 12.0, 13.0]],
        )

    def test_single_var_cells_flat_reshapes_without_ravel_pytree(self):
        problem = self.make_problem(2)
        problem.num_vars = 1
        problem.num_cells = 2
        cells_list = [jnp.asarray([[0, 2], [1, 3]])]

        with mock.patch.object(
            problem_module.jax.flatten_util,
            "ravel_pytree",
            side_effect=AssertionError("single-var cells should reshape directly"),
        ):
            cells_flat = problem._flatten_cells(cells_list)

        onp.testing.assert_array_equal(onp.asarray(cells_flat), [[0, 2], [1, 3]])

    def test_multi_var_cells_flat_matches_legacy_ravel_order(self):
        problem = self.make_problem(2)
        problem.num_vars = 2
        problem.num_cells = 2
        cells_list = [
            jnp.asarray([[0, 2], [1, 3]]),
            jnp.asarray([[10], [11]]),
        ]

        cells_flat = problem._flatten_cells(cells_list)

        onp.testing.assert_array_equal(onp.asarray(cells_flat), [[0, 2, 10], [1, 3, 11]])

    def test_single_var_cell_dof_indices_uses_direct_index_math(self):
        problem = self.make_problem(2)
        problem.num_vars = 1
        problem.fes = [SimpleNamespace(vec=3)]
        problem.offset = [0]
        cells_list = [jnp.asarray([[0, 2], [1, 3]])]

        with mock.patch.object(
            problem_module.jax,
            "vmap",
            side_effect=AssertionError("single-var dof indices should be direct"),
        ):
            inds = problem._cell_dof_indices(cells_list)

        onp.testing.assert_array_equal(
            inds,
            [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]],
        )

    def test_single_var_cell_dof_indices_handles_empty_face_set(self):
        problem = self.make_problem(0)
        problem.num_vars = 1
        problem.fes = [SimpleNamespace(vec=3)]
        problem.offset = [0]
        cells_list = [jnp.empty((0, 2), dtype=jnp.int32)]

        inds = problem._cell_dof_indices(cells_list)

        self.assertEqual(inds.shape, (0, 6))

    def test_multi_var_cell_dof_indices_matches_legacy_order(self):
        problem = self.make_problem(2)
        problem.num_vars = 2
        problem.fes = [SimpleNamespace(vec=1), SimpleNamespace(vec=2)]
        problem.offset = [0, 4]
        cells_list = [
            jnp.asarray([[0, 1], [2, 3]]),
            jnp.asarray([[0], [1]]),
        ]

        inds = problem._cell_dof_indices(cells_list)

        onp.testing.assert_array_equal(
            inds,
            [[0, 1, 4, 5], [2, 3, 6, 7]],
        )

    def test_single_var_unflatten_initialization_uses_direct_reshape(self):
        problem = self.make_problem(0)
        problem.num_vars = 1
        problem.fes = [SimpleNamespace(num_nodes=2, num_total_nodes=3, vec=2)]

        with mock.patch.object(
            problem_module.jax.flatten_util,
            "ravel_pytree",
            side_effect=AssertionError("single-var unflatten should be direct"),
        ):
            problem._init_unflatten_fns()

        dof = problem.unflatten_fn_dof(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
        sol = problem.unflatten_fn_sol_list(
            jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        )

        self.assertEqual(problem.num_total_dofs_all_vars, 6)
        onp.testing.assert_array_equal(
            onp.asarray(dof[0]),
            [[1.0, 2.0], [3.0, 4.0]],
        )
        onp.testing.assert_array_equal(
            onp.asarray(sol[0]),
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        )

    def test_multi_var_unflatten_initialization_matches_legacy_structure(self):
        problem = self.make_problem(0)
        problem.num_vars = 2
        problem.fes = [
            SimpleNamespace(num_nodes=2, num_total_nodes=3, vec=1),
            SimpleNamespace(num_nodes=1, num_total_nodes=2, vec=2),
        ]

        problem._init_unflatten_fns()

        dof = problem.unflatten_fn_dof(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
        sol = problem.unflatten_fn_sol_list(
            jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        )

        self.assertEqual(problem.num_total_dofs_all_vars, 7)
        self.assertEqual(len(dof), 2)
        self.assertEqual(len(sol), 2)
        onp.testing.assert_array_equal(onp.asarray(dof[0]), [[1.0], [2.0]])
        onp.testing.assert_array_equal(onp.asarray(dof[1]), [[3.0, 4.0]])
        onp.testing.assert_array_equal(onp.asarray(sol[0]), [[1.0], [2.0], [3.0]])
        onp.testing.assert_array_equal(onp.asarray(sol[1]), [[4.0, 5.0], [6.0, 7.0]])

    def test_single_var_compute_residual_vars_uses_direct_cells_sol_flatten(self):
        problem = self.make_problem(2)
        problem.num_vars = 1
        problem.cells_list = [jnp.asarray([[0, 1], [1, 2]])]
        sol_list = [jnp.asarray([[1.0], [2.0], [3.0]])]
        captured = {}

        def split_and_compute_cell(cells_sol_flat, _np_version, jac_flag, _internal_vars):
            self.assertFalse(jac_flag)
            captured["cells_sol_flat"] = cells_sol_flat
            return cells_sol_flat + 10.0

        problem.split_and_compute_cell = split_and_compute_cell
        problem.compute_face = lambda *_args: []
        problem.compute_residual_vars_helper = lambda weak_form, _faces: [weak_form]

        with mock.patch.object(
            problem_module.jax.flatten_util,
            "ravel_pytree",
            side_effect=AssertionError("single-var residual should reshape directly"),
        ):
            result = problem.compute_residual_vars(sol_list, [], [])

        onp.testing.assert_array_equal(
            onp.asarray(captured["cells_sol_flat"]),
            [[1.0, 2.0], [2.0, 3.0]],
        )
        onp.testing.assert_array_equal(
            onp.asarray(result[0]),
            [[11.0, 12.0], [12.0, 13.0]],
        )

    def test_single_var_compute_newton_vars_uses_direct_cells_sol_flatten(self):
        problem = self.make_problem(2)
        problem.num_vars = 1
        problem.cells_list = [jnp.asarray([[0, 1], [1, 2]])]
        sol_list = [jnp.asarray([[1.0], [2.0], [3.0]])]
        captured = {}

        def split_and_compute_cell(cells_sol_flat, _np_version, jac_flag, _internal_vars):
            self.assertTrue(jac_flag)
            captured["cells_sol_flat"] = cells_sol_flat
            return cells_sol_flat + 10.0, jnp.zeros((2, 2, 2))

        problem.split_and_compute_cell = split_and_compute_cell
        problem.compute_face = lambda *_args: ([], [])
        problem.compute_residual_vars_helper = lambda weak_form, _faces: [weak_form]

        with mock.patch.object(
            problem_module.jax.flatten_util,
            "ravel_pytree",
            side_effect=AssertionError("single-var newton should reshape directly"),
        ):
            result = problem.compute_newton_vars(sol_list, [], [])

        onp.testing.assert_array_equal(
            onp.asarray(captured["cells_sol_flat"]),
            [[1.0, 2.0], [2.0, 3.0]],
        )
        onp.testing.assert_array_equal(
            onp.asarray(result[0]),
            [[11.0, 12.0], [12.0, 13.0]],
        )
        onp.testing.assert_array_equal(problem.V, onp.zeros(8))


if __name__ == "__main__":
    unittest.main()
