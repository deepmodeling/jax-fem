# Change log


## JAX-FEM 0.0.12 (Jun 2026)

* Major updates
  * Unified `solver()` for Newton, arc-length (displacement & force control), and dynamic relaxation; legacy flat `solver_options` still work for Newton.
  * Newton: per-iteration residual/timing logs; PETSc global-matrix cache.
  * New/refactored applications: third medium contact, updated Lagrangian, differentiable mesh; updated arc-length and dynamic-relaxation examples.
  * Documentation: inverse-problem tutorials, notebook reorganization, `box_mesh_gmsh` API fixes in examples.

* Breaking changes (from 0.0.11)
  * `gauss_order` → `quadrature_order` (optional `quadrature_rule` added).
  * `umfpack_solver` → `spsolve_solver`.
  * Arc-length / dynamic relaxation: use `solver({'arc_length': ...})` and `solver({'dynamic_relax': ...})`.


## JAX-FEM 0.0.11 (Aug 31, 2025)
* Major updates
  * Fixed a [bug](https://github.com/deepmodeling/jax-fem/issues/69) in MMA solver.


## JAX-FEM 0.0.10 (Aug 08, 2025)
* Major updates
  * Use `jax.tree_util.tree_map()` to replace the deprecated `jax.tree_map()`. See JAX updates [jax 0.4.26 (April 3, 2024)](https://docs.jax.dev/en/latest/changelog.html).
  * Support [AMGX](https://github.com/NVIDIA/AMGX) linear solver for better performance on GPU.


## JAX-FEM 0.0.9 (May 05, 2025)

* Major updates
  * Support customized linear solvers

## JAX-FEM 0.0.8 (Oct 30, 2024)

* Major updates
  * Implement the arc-length solver

## JAX-FEM 0.0.7 (Aug 18, 2024)

* Major updates
  *  Update `jax_fem.solver` to include `scipy` solver


## Beta

