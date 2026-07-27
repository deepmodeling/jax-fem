import unittest
from types import SimpleNamespace
from unittest import mock


try:
    import numpy as onp
    import jax.numpy as jnp
    from jax_fem import solver as jax_fem_solver
except ImportError as exc:  # pragma: no cover - depends on local runtime
    onp = None
    jnp = None
    jax_fem_solver = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class FakePetscMat:
    def __init__(self):
        self.indptr = onp.array([0, 1, 2], dtype=onp.int32)
        self.indices = onp.array([0, 1], dtype=onp.int32)
        self.data = onp.array([1.0, 1.0], dtype=onp.float64)
        self.shape = (2, 2)

    def getValuesCSR(self):
        return (
            self.indptr,
            self.indices,
            self.data,
        )

    def getSize(self):
        return self.shape


class NoAttributePetscMat:
    def __getattribute__(self, name):
        if name == "_jax_fem_bcoo_cache":
            raise RuntimeError("extension object does not expose attributes")
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "_jax_fem_bcoo_cache":
            raise RuntimeError("extension object does not expose attributes")
        object.__setattr__(self, name, value)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class JaxSolvePreconditionerTest(unittest.TestCase):
    def setUp(self):
        jax_fem_solver._BCOO_STRUCTURE_CACHE_BY_ID.clear()
        if hasattr(jax_fem_solver, "_BCOO_STRUCTURE_PATTERN_CACHE"):
            jax_fem_solver._BCOO_STRUCTURE_PATTERN_CACHE.clear()

    def make_single_var_bc_problem(self):
        def fail_unflatten(_dofs):
            raise AssertionError("single-var BC helpers should use flat indices")

        return SimpleNamespace(
            num_vars=1,
            offset=[0],
            num_total_dofs_all_vars=6,
            fes=[
                SimpleNamespace(
                    vec=2,
                    node_inds_list=[jnp.asarray([0, 2])],
                    vec_inds_list=[jnp.asarray([1, 0])],
                    vals_list=[jnp.asarray([10.0, 20.0])],
                )
            ],
            unflatten_fn_sol_list=fail_unflatten,
        )

    def test_single_var_apply_bc_vec_uses_flat_dof_indices(self):
        problem = self.make_single_var_bc_problem()
        res_vec = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dofs = jnp.asarray([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

        result = jax_fem_solver.apply_bc_vec(res_vec, dofs, problem)

        onp.testing.assert_array_equal(
            onp.asarray(result),
            [1.0, 91.0, 3.0, 4.0, 84.0, 6.0],
        )

    def test_single_var_apply_bc_vec_caches_jit_kernel_by_bc_and_scale(self):
        problem = self.make_single_var_bc_problem()
        res_vec = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dofs = jnp.asarray([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        original_jit = jax_fem_solver.jax.jit
        compiled = []

        def fake_jit(fn=None, **_kwargs):
            if fn is None:
                return lambda actual_fn: fake_jit(actual_fn, **_kwargs)
            compiled.append(fn)

            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapped

        try:
            jax_fem_solver.jax.jit = fake_jit
            first = jax_fem_solver.apply_bc_vec(res_vec, dofs, problem)
            second = jax_fem_solver.apply_bc_vec(res_vec, dofs, problem)
            scaled = jax_fem_solver.apply_bc_vec(
                res_vec,
                dofs,
                problem,
                scale=2.0,
            )
        finally:
            jax_fem_solver.jax.jit = original_jit

        self.assertEqual(len(compiled), 2)
        onp.testing.assert_array_equal(
            onp.asarray(first),
            [1.0, 91.0, 3.0, 4.0, 84.0, 6.0],
        )
        onp.testing.assert_array_equal(onp.asarray(second), onp.asarray(first))
        onp.testing.assert_array_equal(
            onp.asarray(scaled),
            [1.0, 81.0, 3.0, 4.0, 64.0, 6.0],
        )

    def test_single_var_apply_bc_vec_jit_cache_rebuilds_when_bc_lists_change(self):
        problem = self.make_single_var_bc_problem()
        res_vec = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dofs = jnp.asarray([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        original_jit = jax_fem_solver.jax.jit
        compiled = []

        def fake_jit(fn=None, **_kwargs):
            if fn is None:
                return lambda actual_fn: fake_jit(actual_fn, **_kwargs)
            compiled.append(fn)

            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapped

        try:
            jax_fem_solver.jax.jit = fake_jit
            first = jax_fem_solver.apply_bc_vec(res_vec, dofs, problem)
            problem.fes[0].vals_list = [jnp.asarray([30.0, 40.0])]
            second = jax_fem_solver.apply_bc_vec(res_vec, dofs, problem)
        finally:
            jax_fem_solver.jax.jit = original_jit

        self.assertEqual(len(compiled), 2)
        onp.testing.assert_array_equal(
            onp.asarray(first),
            [1.0, 91.0, 3.0, 4.0, 84.0, 6.0],
        )
        onp.testing.assert_array_equal(
            onp.asarray(second),
            [1.0, 71.0, 3.0, 4.0, 64.0, 6.0],
        )

    def test_single_var_assign_and_copy_bc_use_flat_dof_indices(self):
        problem = self.make_single_var_bc_problem()
        dofs = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        assigned = jax_fem_solver.assign_bc(dofs, problem)
        ones = jax_fem_solver.assign_ones_bc(dofs, problem)
        zeros = jax_fem_solver.assign_zeros_bc(dofs, problem)
        copied = jax_fem_solver.copy_bc(dofs, problem)

        onp.testing.assert_array_equal(
            onp.asarray(assigned),
            [1.0, 10.0, 3.0, 4.0, 20.0, 6.0],
        )
        onp.testing.assert_array_equal(
            onp.asarray(ones),
            [1.0, 1.0, 3.0, 4.0, 1.0, 6.0],
        )
        onp.testing.assert_array_equal(
            onp.asarray(zeros),
            [1.0, 0.0, 3.0, 4.0, 0.0, 6.0],
        )
        onp.testing.assert_array_equal(
            onp.asarray(copied),
            [0.0, 2.0, 0.0, 0.0, 5.0, 0.0],
        )

    def test_single_var_bc_flat_cache_rebuilds_when_bc_lists_change(self):
        problem = self.make_single_var_bc_problem()
        dofs = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        first = jax_fem_solver.assign_bc(dofs, problem)
        onp.testing.assert_array_equal(
            onp.asarray(first),
            [1.0, 10.0, 3.0, 4.0, 20.0, 6.0],
        )

        problem.fes[0].vals_list = [jnp.asarray([30.0, 40.0])]
        second = jax_fem_solver.assign_bc(dofs, problem)
        onp.testing.assert_array_equal(
            onp.asarray(second),
            [1.0, 30.0, 3.0, 4.0, 40.0, 6.0],
        )

        problem.fes[0].node_inds_list = [jnp.asarray([1])]
        problem.fes[0].vec_inds_list = [jnp.asarray([1])]
        problem.fes[0].vals_list = [jnp.asarray([70.0])]
        third = jax_fem_solver.assign_bc(dofs, problem)
        onp.testing.assert_array_equal(
            onp.asarray(third),
            [1.0, 2.0, 3.0, 70.0, 5.0, 6.0],
        )

    def test_single_var_bc_flat_cache_tracks_explicit_bc_version(self):
        problem = self.make_single_var_bc_problem()
        fe = problem.fes[0]
        fe.node_inds_list = [onp.asarray([0, 2], dtype=onp.int32)]
        fe.vec_inds_list = [onp.asarray([1, 0], dtype=onp.int32)]
        fe.vals_list = [onp.asarray([10.0, 20.0])]
        fe._dirichlet_bc_version = 0
        dofs = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        residual = jnp.zeros_like(dofs)

        first_flat = jax_fem_solver._single_var_bc_flat(problem)
        onp.testing.assert_array_equal(onp.asarray(first_flat[0]), [1, 4])
        onp.testing.assert_array_equal(onp.asarray(first_flat[1]), [10.0, 20.0])

        # Dynamic FE updates may reuse the same host/JAX array objects. The
        # explicit version is the authoritative cache invalidation signal.
        fe.node_inds_list[0][...] = [1, 2]
        fe.vec_inds_list[0][...] = [0, 1]
        fe.vals_list[0][...] = [30.0, 40.0]
        fe._dirichlet_bc_version += 1

        second_flat = jax_fem_solver._single_var_bc_flat(problem)
        assigned = jax_fem_solver.assign_bc(dofs, problem)
        applied = jax_fem_solver.apply_bc_vec(
            residual,
            dofs,
            problem,
        )

        onp.testing.assert_array_equal(onp.asarray(second_flat[0]), [2, 5])
        onp.testing.assert_array_equal(
            onp.asarray(second_flat[1]),
            [30.0, 40.0],
        )
        onp.testing.assert_array_equal(
            onp.asarray(assigned),
            [1.0, 2.0, 30.0, 4.0, 5.0, 40.0],
        )
        onp.testing.assert_array_equal(
            onp.asarray(applied),
            [0.0, 0.0, -27.0, 0.0, 0.0, -34.0],
        )

    def test_single_var_bc_zero_seed_cache_rebuilds_when_bc_lists_change(self):
        problem = self.make_single_var_bc_problem()
        dofs = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        first = jax_fem_solver._assign_bc_zero_seed(dofs, problem)
        second = jax_fem_solver._assign_bc_zero_seed(dofs, problem)
        self.assertIs(first, second)
        onp.testing.assert_array_equal(
            onp.asarray(first),
            [0.0, 10.0, 0.0, 0.0, 20.0, 0.0],
        )

        problem.fes[0].vals_list = [jnp.asarray([30.0, 40.0])]
        third = jax_fem_solver._assign_bc_zero_seed(dofs, problem)
        self.assertIsNot(first, third)
        onp.testing.assert_array_equal(
            onp.asarray(third),
            [0.0, 30.0, 0.0, 0.0, 40.0, 0.0],
        )

    def test_single_var_newton_step_builds_x0_without_copy_bc(self):
        problem = self.make_single_var_bc_problem()
        dofs = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res_vec = jnp.zeros_like(dofs)
        captured = {}
        original_linear_solver = jax_fem_solver.linear_solver
        original_copy_bc = jax_fem_solver.copy_bc

        def fake_linear_solver(_A, _b, x0, _linear_options, timing=None):
            captured["x0"] = x0
            return jnp.zeros_like(x0)

        def fail_copy_bc(_dofs, _problem):
            raise AssertionError("single-var Newton x0 should avoid copy_bc")

        try:
            jax_fem_solver.linear_solver = fake_linear_solver
            jax_fem_solver.copy_bc = fail_copy_bc
            updated, _linear_s = jax_fem_solver.newton_step(
                problem,
                res_vec,
                object(),
                dofs,
                {"linear": {}},
                captured,
            )
        finally:
            jax_fem_solver.linear_solver = original_linear_solver
            jax_fem_solver.copy_bc = original_copy_bc

        onp.testing.assert_array_equal(onp.asarray(updated), onp.asarray(dofs))
        onp.testing.assert_array_equal(
            onp.asarray(captured["x0"]),
            [0.0, 8.0, 0.0, 0.0, 15.0, 0.0],
        )
        self.assertGreater(captured["bc_initial_guess"], 0.0)

    def test_single_var_flatten_residual_list_uses_direct_reshape(self):
        problem = self.make_single_var_bc_problem()
        residual = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        original_ravel_pytree = jax_fem_solver.jax.flatten_util.ravel_pytree

        def fail_ravel_pytree(_value):
            raise AssertionError("single-var residual flatten should reshape")

        try:
            jax_fem_solver.jax.flatten_util.ravel_pytree = fail_ravel_pytree
            flat = jax_fem_solver._flatten_residual_list([residual], problem)
        finally:
            jax_fem_solver.jax.flatten_util.ravel_pytree = original_ravel_pytree

        onp.testing.assert_array_equal(
            onp.asarray(flat),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

    def test_flatten_residual_list_falls_back_for_multivar_problem(self):
        problem = SimpleNamespace(num_vars=2, fes=[object(), object()])
        residuals = [
            jnp.asarray([1.0, 2.0]),
            jnp.asarray([3.0, 4.0]),
        ]
        original_ravel_pytree = jax_fem_solver.jax.flatten_util.ravel_pytree
        captured = {}

        def fake_ravel_pytree(value):
            captured["value"] = value
            return jnp.asarray([9.0, 10.0]), None

        try:
            jax_fem_solver.jax.flatten_util.ravel_pytree = fake_ravel_pytree
            flat = jax_fem_solver._flatten_residual_list(residuals, problem)
        finally:
            jax_fem_solver.jax.flatten_util.ravel_pytree = original_ravel_pytree

        self.assertIs(captured["value"], residuals)
        onp.testing.assert_array_equal(onp.asarray(flat), [9.0, 10.0])

    def test_jax_solve_accepts_preconditioner_disabled(self):
        result = jax_fem_solver.jax_solve(
            FakePetscMat(),
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
        )

        onp.testing.assert_allclose(onp.asarray(result), onp.array([1.0, 2.0]))

    def test_jax_solve_accepts_method_and_tolerance_options(self):
        result = jax_fem_solver.jax_solve(
            FakePetscMat(),
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
            tol=1e-8,
            atol=1e-8,
            maxiter=20,
        )

        onp.testing.assert_allclose(onp.asarray(result), onp.array([1.0, 2.0]))

    def test_jax_solve_records_conversion_and_kernel_timing(self):
        timing = {}

        result = jax_fem_solver.jax_solve(
            FakePetscMat(),
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
            timing=timing,
        )

        onp.testing.assert_allclose(onp.asarray(result), onp.array([1.0, 2.0]))
        self.assertGreater(timing["sparse_conversion"], 0.0)
        self.assertGreater(timing["linear_kernel"], 0.0)
        self.assertGreater(timing["linear_residual_check"], 0.0)
        self.assertTrue(timing["_last_linear_internal_breakdown"])

    def test_jax_solve_can_skip_explicit_residual_check(self):
        timing = {}

        result = jax_fem_solver.jax_solve(
            FakePetscMat(),
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
            timing=timing,
            check_residual=False,
        )

        onp.testing.assert_allclose(onp.asarray(result), onp.array([1.0, 2.0]))
        self.assertGreater(timing["sparse_conversion"], 0.0)
        self.assertGreater(timing["linear_kernel"], 0.0)
        self.assertNotIn("linear_residual_check", timing)

    def test_jax_solve_direct_spsolve_uses_csr_path_without_residual_check(self):
        timing = {}
        mat = FakePetscMat()
        mat.data = onp.array([2.0, 3.0], dtype=onp.float64)

        result = jax_fem_solver.jax_solve(
            mat,
            jnp.array([4.0, 9.0]),
            None,
            precond=True,
            method="spsolve",
            timing=timing,
            check_residual=False,
        )

        onp.testing.assert_allclose(onp.asarray(result), onp.array([2.0, 3.0]))
        self.assertGreater(timing["sparse_conversion"], 0.0)
        self.assertGreater(timing["linear_kernel"], 0.0)
        self.assertEqual(timing["jax_spsolve_calls"], 1)
        self.assertNotIn("linear_residual_check", timing)
        self.assertNotIn("bcoo_cache_misses", timing)
        self.assertFalse(hasattr(mat, "_jax_fem_bcoo_cache"))

    def test_jax_solve_direct_spsolve_can_run_explicit_residual_check(self):
        timing = {}
        mat = FakePetscMat()
        mat.data = onp.array([2.0, 3.0], dtype=onp.float64)

        result = jax_fem_solver.jax_solve(
            mat,
            jnp.array([4.0, 9.0]),
            None,
            precond=False,
            method="spsolve",
            timing=timing,
        )

        onp.testing.assert_allclose(onp.asarray(result), onp.array([2.0, 3.0]))
        self.assertGreater(timing["linear_residual_check"], 0.0)
        self.assertEqual(timing["jax_spsolve_calls"], 1)
        self.assertEqual(timing["bcoo_cache_misses"], 1)
        self.assertTrue(hasattr(mat, "_jax_fem_bcoo_cache"))

    def test_jax_solve_rejects_nonzero_info_when_residual_check_is_skipped(self):
        timing = {}

        with mock.patch.object(
            jax_fem_solver.jax.scipy.sparse.linalg,
            "cg",
            return_value=(jnp.array([1.0, 2.0]), 7),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "JAX cg solver failed to converge",
            ):
                jax_fem_solver.jax_solve(
                    FakePetscMat(),
                    jnp.array([1.0, 2.0]),
                    None,
                    precond=False,
                    method="cg",
                    timing=timing,
                    check_residual=False,
                )

        self.assertGreater(timing["linear_kernel"], 0.0)
        self.assertNotIn("linear_residual_check", timing)

    def test_jax_solve_reuses_cached_sparse_structure_with_new_values(self):
        mat = FakePetscMat()
        mat.data = onp.array([2.0, 3.0], dtype=onp.float64)

        first = jax_fem_solver.jax_solve(
            mat,
            jnp.array([4.0, 9.0]),
            None,
            precond=True,
            method="cg",
        )

        mat.data = onp.array([4.0, 5.0], dtype=onp.float64)
        second = jax_fem_solver.jax_solve(
            mat,
            jnp.array([8.0, 10.0]),
            None,
            precond=True,
            method="cg",
        )

        onp.testing.assert_allclose(onp.asarray(first), onp.array([2.0, 3.0]))
        onp.testing.assert_allclose(onp.asarray(second), onp.array([2.0, 2.0]))
        cache = mat._jax_fem_bcoo_cache
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)

    def test_jax_solve_records_sparse_cache_counters(self):
        mat = FakePetscMat()
        timing = {}

        jax_fem_solver.jax_solve(
            mat,
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
            timing=timing,
            check_residual=False,
        )
        jax_fem_solver.jax_solve(
            mat,
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
            timing=timing,
            check_residual=False,
        )

        self.assertEqual(timing["bcoo_cache_misses"], 1)
        self.assertEqual(timing["bcoo_cache_hits"], 1)

    def test_jax_solve_reuses_sparse_structure_across_matrix_objects(self):
        timing = {}

        jax_fem_solver.jax_solve(
            FakePetscMat(),
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
            timing=timing,
            check_residual=False,
        )
        jax_fem_solver.jax_solve(
            FakePetscMat(),
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
            timing=timing,
            check_residual=False,
        )

        self.assertEqual(timing["bcoo_cache_misses"], 1)
        self.assertEqual(timing["bcoo_cache_hits"], 1)

    def test_fallback_id_cache_is_bounded_for_extension_mats(self):
        original_limit = jax_fem_solver._BCOO_STRUCTURE_CACHE_BY_ID_LIMIT
        jax_fem_solver._BCOO_STRUCTURE_CACHE_BY_ID_LIMIT = 2
        try:
            for _ in range(4):
                jax_fem_solver._set_cached_bcoo_structure(
                    NoAttributePetscMat(),
                    object(),
                )
            self.assertLessEqual(
                len(jax_fem_solver._BCOO_STRUCTURE_CACHE_BY_ID),
                2,
            )
        finally:
            jax_fem_solver._BCOO_STRUCTURE_CACHE_BY_ID_LIMIT = original_limit
            jax_fem_solver._BCOO_STRUCTURE_CACHE_BY_ID.clear()

    def test_jax_solve_rebuilds_sparse_cache_when_pattern_changes(self):
        mat = FakePetscMat()

        jax_fem_solver.jax_solve(
            mat,
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
        )
        first_cache = mat._jax_fem_bcoo_cache

        mat.indptr = onp.array([0, 2, 3], dtype=onp.int32)
        mat.indices = onp.array([0, 1, 1], dtype=onp.int32)
        mat.data = onp.array([1.0, 0.0, 1.0], dtype=onp.float64)
        result = jax_fem_solver.jax_solve(
            mat,
            jnp.array([1.0, 2.0]),
            None,
            precond=False,
            method="cg",
        )

        onp.testing.assert_allclose(onp.asarray(result), onp.array([1.0, 2.0]))
        self.assertIsNot(mat._jax_fem_bcoo_cache, first_cache)
        self.assertEqual(mat._jax_fem_bcoo_cache.misses, 1)
        self.assertEqual(mat._jax_fem_bcoo_cache.hits, 0)

    def test_jax_solve_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "unknown JAX linear solver method"):
            jax_fem_solver.jax_solve(
                FakePetscMat(),
                jnp.array([1.0, 2.0]),
                None,
                precond=False,
                method="not-a-method",
            )

    def test_linear_solver_forwards_jax_options(self):
        timing = {}

        result = jax_fem_solver.linear_solver(
            FakePetscMat(),
            jnp.array([1.0, 2.0]),
            None,
            {
                "jax_solver": {
                    "precond": False,
                    "method": "cg",
                    "tol": 1e-8,
                    "atol": 1e-8,
                    "maxiter": 20,
                    "check_residual": False,
                }
            },
            timing,
        )

        onp.testing.assert_allclose(onp.asarray(result), onp.array([1.0, 2.0]))
        self.assertNotIn("linear_residual_check", timing)


@unittest.skipIf(IMPORT_ERROR is not None, f"jax runtime unavailable: {IMPORT_ERROR}")
class ResidualOnlyConvergenceCheckTest(unittest.TestCase):
    """Newton loop assembly-call pattern under 'residual_only_check'."""

    def make_counting_problem(self, newton_norms, residual_norms):
        """Fake single-var problem whose assembly calls pop scripted norms.

        ``newton_norms`` / ``residual_norms`` are queues of residual vector
        magnitudes returned by successive ``newton_update`` /
        ``compute_residual`` calls (0.0 means converged).
        """
        problem = SimpleNamespace(
            num_vars=1,
            num_total_dofs_all_vars=6,
            fes=[SimpleNamespace(vec=1)],
            newton_update_calls=0,
            compute_residual_calls=0,
            newton_tangents=[],
        )
        problem.unflatten_fn_sol_list = lambda dofs: [dofs]

        def newton_update(_sol_list):
            problem.newton_update_calls += 1
            value = newton_norms.pop(0)
            tangent = f"A{problem.newton_update_calls}"
            problem.newton_tangents.append(tangent)
            problem._latest_tangent = tangent
            return [jnp.full((6, 1), value)]

        def compute_residual(_sol_list):
            problem.compute_residual_calls += 1
            return [jnp.full((6, 1), residual_norms.pop(0))]

        problem.newton_update = newton_update
        problem.compute_residual = compute_residual
        return problem

    def run_solver(self, problem, solver_options):
        stepped_tangents = []

        def fake_newton_step(_problem, _res_vec, A, dofs, _cfg, _timing):
            stepped_tangents.append(A)
            return dofs, 0.0

        with mock.patch.object(
            jax_fem_solver, "newton_step", side_effect=fake_newton_step
        ), mock.patch.object(
            jax_fem_solver,
            "get_A",
            side_effect=lambda p: p._latest_tangent,
        ), mock.patch.object(
            jax_fem_solver,
            "apply_bc_vec",
            side_effect=lambda res_vec, dofs, p, *a, **k: res_vec,
        ):
            jax_fem_solver.solver(problem, solver_options=solver_options)
        return stepped_tangents

    def test_default_check_assembles_jacobian_twice_per_converged_step(self):
        problem = self.make_counting_problem(
            newton_norms=[1.0, 0.0], residual_norms=[]
        )
        self.run_solver(problem, {"newton": {}})
        self.assertEqual(problem.newton_update_calls, 2)
        self.assertEqual(problem.compute_residual_calls, 0)

    def test_residual_only_check_skips_jacobian_on_converged_step(self):
        problem = self.make_counting_problem(
            newton_norms=[1.0], residual_norms=[0.0]
        )
        stepped = self.run_solver(
            problem, {"newton": {"residual_only_check": True}}
        )
        self.assertEqual(problem.newton_update_calls, 1)
        self.assertEqual(problem.compute_residual_calls, 1)
        self.assertEqual(stepped, ["A1"])

    def test_residual_only_check_rebuilds_tangent_when_not_converged(self):
        problem = self.make_counting_problem(
            newton_norms=[1.0, 1.0], residual_norms=[1.0, 0.0]
        )
        stepped = self.run_solver(
            problem, {"newton": {"residual_only_check": True}}
        )
        # initial full assembly, probe (not converged), full rebuild,
        # second step, probe (converged)
        self.assertEqual(problem.newton_update_calls, 2)
        self.assertEqual(problem.compute_residual_calls, 2)
        self.assertEqual(stepped, ["A1", "A2"])

    def test_flat_layout_forwards_residual_only_check(self):
        _method, cfg = jax_fem_solver._resolve_solver_options(
            {"residual_only_check": True}
        )
        self.assertTrue(cfg.get("residual_only_check"))


if __name__ == "__main__":
    unittest.main()
