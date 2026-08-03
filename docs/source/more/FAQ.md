# Frequently asked questions (FAQ)

### General

* I saw JAX-FEM supports third-party non-differentiable solvers such as PETSc. Does that mean it is not differentiable if I use PETSc?

> It is still differentiable. We apply implicit differentiation based on the adjoint formulation to cover that. We never have to differentiate through the linear solver or Newton's method involved in the forward problem.

### Remarks

If your question isn't covered above, you can [search existing issues](https://github.com/deepmodeling/jax-fem/issues?q=) or [open a new issue](https://github.com/deepmodeling/jax-fem/issues/new/choose) on Github. 

